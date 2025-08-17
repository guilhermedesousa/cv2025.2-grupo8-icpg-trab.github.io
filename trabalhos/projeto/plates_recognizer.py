#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconhecimento de Placas MercoSul (tempo real + OCR paralelo)

Recursos:
- Câmera em tempo real (index da câmera configurável)
- OCR paralelo e assíncrono (multiprocessing) para não travar a UI
- Motores de OCR selecionáveis: Tesseract, EasyOCR e Dual (ambos)
- Presets "rápido" (padrão) e "ultra" (mais FPS, menos acurácia)
- Debounce temporal do OCR (ocr_every)
- Limite de regiões por frame
- Fallback do Haar Cascade
- Debounce de placa confirmada
- Calibração de câmera para correção de distorção

Atalhos na janela:
  q = sair | d = alterna debug | p = pausa
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import cv2
import time
import sys
import argparse
import numpy as np
import pytesseract
import subprocess
from typing import List, Tuple, Dict
from collections import deque, Counter
import multiprocessing as mp
from datetime import datetime

# --------------- Worker de OCR (Tesseract) ---------------
def _tesseract_worker(payload):
    """Executa OCR com Tesseract. Roda em um processo separado."""
    import cv2, re, pytesseract
    roi = payload["roi"]
    psm_configs = payload["psm_cfgs"]
    mode = payload["mode"]
    save_preprocessing = payload.get("save_preprocessing", False)
    output_dir = payload.get("output_dir", "")
    region_id = payload.get("region_id", 0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    variants = []
    variant_names = []
    
    # Variante original
    variants.append(gray)
    variant_names.append("01_original_gray")

    if mode == "ultra":
        eq = cv2.equalizeHist(gray)
        variants.append(eq)
        variant_names.append("02_equalized")
        
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        variants.append(adaptive)
        variant_names.append("03_adaptive_mean")
    else:
        eq = cv2.equalizeHist(gray)
        variants.append(eq)
        variant_names.append("02_equalized")
        
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        variants.append(blurred)
        variant_names.append("03_gaussian_blur")
        
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        variants.append(adaptive)
        variant_names.append("04_adaptive_gaussian")

    # Salvar etapas de pré-processamento se solicitado
    if save_preprocessing and output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Salvar ROI original
            cv2.imwrite(os.path.join(output_dir, f"region_{region_id:02d}_00_original_roi.png"), roi)
            
            # Salvar cada variante processada
            for i, (variant, name) in enumerate(zip(variants, variant_names)):
                filename = f"region_{region_id:02d}_{name}.png"
                cv2.imwrite(os.path.join(output_dir, filename), variant)
        except Exception as e:
            print(f"Erro ao salvar pré-processamento: {e}")

    outs = []
    for v_idx, v in enumerate(variants):
        for cfg_idx, cfg in enumerate(psm_configs):
            try:
                data = pytesseract.image_to_data(v, config=cfg, output_type=pytesseract.Output.DICT)
                
                # Salvar dados detalhados do OCR se solicitado
                if save_preprocessing and output_dir:
                    try:
                        ocr_info_file = os.path.join(output_dir, f"region_{region_id:02d}_ocr_variant_{v_idx:02d}_config_{cfg_idx:02d}.txt")
                        with open(ocr_info_file, 'w', encoding='utf-8') as f:
                            f.write(f"Configuração Tesseract: {cfg}\n")
                            f.write(f"Variante de pré-processamento: {variant_names[v_idx]}\n")
                            f.write("="*50 + "\n")
                            
                            n = len(data['text'])
                            for i in range(n):
                                text = (data['text'][i] or "").strip()
                                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                                if text:
                                    f.write(f"Texto: '{text}' | Confiança: {conf}% | "
                                           f"Posição: ({data['left'][i]}, {data['top'][i]}) | "
                                           f"Tamanho: {data['width'][i]}x{data['height'][i]}\n")
                    except Exception:
                        pass
                
            except Exception:
                continue
                
            n = len(data['text'])
            for i in range(n):
                text = (data['text'][i] or "").strip()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                if conf < 40 or not text:
                    continue
                clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean) >= 5:
                    outs.append(clean)
    return outs

# --------------- Worker de OCR (EasyOCR) ---------------
easyocr_reader = None
def _easyocr_worker(payload):
    """Executa OCR com EasyOCR. Roda em um processo separado."""
    global easyocr_reader
    import cv2, re
    # Inicializa o reader uma vez por processo para economizar tempo
    if easyocr_reader is None:
        import easyocr
        easyocr_reader = easyocr.Reader(['pt'], gpu=False, verbose=False)

    roi = payload["roi"]
    save_preprocessing = payload.get("save_preprocessing", False)
    output_dir = payload.get("output_dir", "")
    region_id = payload.get("region_id", 0)
    
    # Salvar ROI para EasyOCR se solicitado
    if save_preprocessing and output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, f"region_{region_id:02d}_easyocr_input.png"), roi)
        except Exception as e:
            print(f"Erro ao salvar entrada do EasyOCR: {e}")
    
    results = easyocr_reader.readtext(roi, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', paragraph=False)
    
    # Salvar resultados do EasyOCR se solicitado
    if save_preprocessing and output_dir:
        try:
            easyocr_info_file = os.path.join(output_dir, f"region_{region_id:02d}_easyocr_results.txt")
            with open(easyocr_info_file, 'w', encoding='utf-8') as f:
                f.write("Resultados EasyOCR\n")
                f.write("="*30 + "\n")
                for i, (bbox, text, prob) in enumerate(results):
                    f.write(f"Detecção {i+1}:\n")
                    f.write(f"  Texto: '{text}'\n")
                    f.write(f"  Probabilidade: {prob:.3f}\n")
                    f.write(f"  Bounding Box: {bbox}\n\n")
                    
            # Criar imagem com anotações do EasyOCR
            annotated = roi.copy()
            for (bbox, text, prob) in results:
                if prob > 0.3:
                    pts = np.array(bbox, np.int32)
                    cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                    cv2.putText(annotated, f"{text} ({prob:.2f})", 
                               tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imwrite(os.path.join(output_dir, f"region_{region_id:02d}_easyocr_annotated.png"), annotated)
        except Exception as e:
            print(f"Erro ao salvar resultados do EasyOCR: {e}")
    
    outs = []
    for (bbox, text, prob) in results:
        if prob > 0.3: # Limiar de confiança para EasyOCR
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(clean) >= 5:
                outs.append(clean)
    return outs


# --------------- Classe principal ---------------
class BrazilianPlateRecognizer:
    def __init__(self, use_haar: bool = True, target_w: int = 640,
                 max_regions_per_frame: int = 4, ocr_every: int = 6,
                 preset_ultra: bool = False, pool_size: int = None,
                 ocr_engine: str = 'easyocr', cam_calib_path: str = None,
                 save_preprocessing: bool = False, output_dir: str = "plate_processing"):
        """
        :param use_haar: ativa o Haar Cascade
        :param target_w: largura alvo para processamento
        :param max_regions_per_frame: ROIs com OCR por frame
        :param ocr_every: faz OCR a cada N frames
        :param preset_ultra: menos variantes/PSMs (mais FPS)
        :param pool_size: nº de processos de OCR
        :param ocr_engine: 'tesseract', 'easyocr' ou 'dual'
        :param cam_calib_path: caminho para o arquivo de calibração
        :param save_preprocessing: salva etapas de pré-processamento
        :param output_dir: diretório para salvar arquivos de processamento
        """
        self.preset_ultra = preset_ultra
        self.target_w = target_w
        self.max_regions_per_frame = max_regions_per_frame
        self.ocr_every = ocr_every
        self.ocr_engine = ocr_engine
        self.save_preprocessing = save_preprocessing
        self.output_dir = output_dir
        self.mtx, self.dist = None, None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Criar diretório de saída se necessário
        if self.save_preprocessing:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"📁 Salvando etapas de processamento em: {self.output_dir}")

        if cam_calib_path and os.path.exists(cam_calib_path):
            try:
                data = np.load(cam_calib_path)
                self.mtx = data['mtx']
                self.dist = data['dist']
                print(f"✅ Parâmetros de calibração carregados de {cam_calib_path}")
            except Exception as e:
                print(f"⚠️ Erro ao carregar calibração: {e}")
        
        if self.ocr_engine in ['tesseract', 'dual']:
            self.setup_tesseract()
        if self.ocr_engine in ['easyocr', 'dual']:
            try:
                import easyocr
            except ImportError:
                print("❌ EasyOCR não encontrado! Instale com: pip install easyocr")
                sys.exit(1)

        if use_haar:
            self.load_haar_cascade()
        else:
            self.plate_cascade = None

        if self.preset_ultra:
            self.psm_configs = [r'--oem 3 --psm 7', r'--oem 3 --psm 6']
            self.use_variants = "ultra"
        else:
            self.psm_configs = [
                r'--oem 3 --psm 6', r'--oem 3 --psm 7', r'--oem 3 --psm 8',
                r'--oem 3 --psm 11', r'--oem 3 --psm 12', r'--oem 3 --psm 13',
            ]
            self.use_variants = "fast"

        default_procs = max(1, min(4, (mp.cpu_count() or 2) - 1))
        self.pool = mp.Pool(processes=pool_size or default_procs)

    # ---------- Infra ----------
    def close(self):
        try:
            self.pool.terminate()
            self.pool.join()
        except Exception:
            pass

    def setup_tesseract(self):
        """Detecta binário do tesseract e configura pytesseract."""
        try:
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0:
                pytesseract.pytesseract.tesseract_cmd = result.stdout.strip()
                print(f"✅ Tesseract configurado: {result.stdout.strip()}")
            else:
                print("❌ Tesseract não encontrado! Ex.: brew install tesseract (macOS)")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Erro ao configurar tesseract: {e}")
            sys.exit(1)

    def load_haar_cascade(self):
        """Carrega Haar local ou fallback do OpenCV."""
        local = os.path.join(os.path.dirname(__file__), 'trabalhos/projeto', 'haarcascade_russian_plate_number.xml')
        if os.path.exists(local):
            self.plate_cascade = cv2.CascadeClassifier(local)
            print(f"✅ Haar Cascade carregado: {local}")
            return
        try:
            base = cv2.data.haarcascades
            alt = os.path.join(base, 'haarcascade_russian_plate_number.xml')
            if os.path.exists(alt):
                self.plate_cascade = cv2.CascadeClassifier(alt)
                print(f"✅ Haar Cascade (fallback) carregado: {alt}")
                return
        except Exception:
            pass
        self.plate_cascade = None
        print("⚠️ Haar Cascade não encontrado, usando apenas contornos")

    # ---------- Util ----------
    def _resize_for_speed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        if w <= self.target_w: return image, 1.0
        scale = self.target_w / float(w)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale

    # ---------- Regiões ----------
    def detect_plate_regions_haar(self, image: np.ndarray) -> List[Tuple[int,int,int,int]]:
        if self.plate_cascade is None: return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return [(x, y, w, h) for (x, y, w, h) in self.plate_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(80, 20), maxSize=(400, 140)
        )]

    def detect_plate_regions_contour(self, image: np.ndarray) -> List[Tuple[int,int,int,int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        regions = []
        edge_params = [(50, 150), (100, 200)] if self.preset_ultra else [(30, 100), (80, 180)]
        for low, high in edge_params:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, low, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if 1200 < cv2.contourArea(contour) < 50000:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        if 2.0 <= w / float(h) <= 5.2 and w > 60 and h > 15:
                            regions.append((x, y, w, h))
        return regions

    def remove_duplicate_regions(self, regions: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
        if not regions: return []
        regions = sorted(regions, key=lambda r: r[2]*r[3], reverse=True)
        unique = []
        for (x1, y1, w1, h1) in regions:
            dup = False
            for (x2, y2, w2, h2) in unique:
                if (abs(x1-x2) < 40 and abs(y1-y2) < 40 and abs(w1-w2) < 80 and abs(h1-h2) < 40):
                    dup = True
                    break
            if not dup: unique.append((x1, y1, w1, h1))
        return unique

    def save_detection_stages(self, frame: np.ndarray, regions: List[Tuple[int,int,int,int]], 
                            detected_plates: List[str], timestamp: str = None) -> str:
        """
        Salva as etapas de detecção: frame original, regiões detectadas, etc.
        Retorna o diretório onde os arquivos foram salvos.
        """
        if not self.save_preprocessing:
            return ""
            
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
        # Criar subdiretório para esta detecção
        detection_dir = os.path.join(self.output_dir, f"detection_{timestamp}")
        os.makedirs(detection_dir, exist_ok=True)
        
        try:
            # 1. Salvar frame original
            cv2.imwrite(os.path.join(detection_dir, "00_original_frame.png"), frame)
            
            # 2. Frame com correção de distorção (se aplicável)
            if self.mtx is not None and self.dist is not None:
                undistorted = cv2.undistort(frame, self.mtx, self.dist, None)
                cv2.imwrite(os.path.join(detection_dir, "01_undistorted_frame.png"), undistorted)
                frame_for_processing = undistorted
            else:
                frame_for_processing = frame
            
            # 3. Frame redimensionado para processamento
            proc_frame, scale = self._resize_for_speed(frame_for_processing)
            cv2.imwrite(os.path.join(detection_dir, "02_resized_for_processing.png"), proc_frame)
            
            # 4. Visualização da detecção Haar Cascade
            if self.plate_cascade is not None:
                haar_regions = self.detect_plate_regions_haar(proc_frame)
                haar_vis = proc_frame.copy()
                for i, (x, y, w, h) in enumerate(haar_regions):
                    cv2.rectangle(haar_vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(haar_vis, f"Haar {i+1}", (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imwrite(os.path.join(detection_dir, "03_haar_cascade_detection.png"), haar_vis)
            
            # 5. Visualização da detecção por contornos
            contour_regions = self.detect_plate_regions_contour(proc_frame)
            contour_vis = proc_frame.copy()
            for i, (x, y, w, h) in enumerate(contour_regions):
                cv2.rectangle(contour_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(contour_vis, f"Contour {i+1}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(detection_dir, "04_contour_detection.png"), contour_vis)
            
            # 6. Visualização de todas as regiões finais
            all_regions_vis = proc_frame.copy()
            for i, (x, y, w, h) in enumerate(regions):
                cv2.rectangle(all_regions_vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(all_regions_vis, f"Region {i+1}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(detection_dir, "05_final_regions.png"), all_regions_vis)
            
            # 7. Salvar informações de detecção
            info_file = os.path.join(detection_dir, "detection_info.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"Sessão: {self.session_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Motor OCR: {self.ocr_engine}\n")
                f.write(f"Preset Ultra: {self.preset_ultra}\n")
                f.write(f"Frame original: {frame.shape}\n")
                f.write(f"Frame processado: {proc_frame.shape}\n")
                f.write(f"Fator de escala: {scale:.3f}\n")
                f.write(f"Usar Haar Cascade: {self.plate_cascade is not None}\n")
                f.write(f"Regiões detectadas: {len(regions)}\n")
                f.write(f"Placas reconhecidas: {detected_plates}\n")
                f.write("="*50 + "\n")
                
                for i, (x, y, w, h) in enumerate(regions):
                    f.write(f"Região {i+1}: posição=({x},{y}) tamanho={w}x{h} área={w*h}\n")
            
            print(f"📁 Etapas de detecção salvas em: {detection_dir}")
            return detection_dir
            
        except Exception as e:
            print(f"❌ Erro ao salvar etapas de detecção: {e}")
            return ""

    # ---------- Validação ----------
    def validate_and_format_plate(self, text: str) -> Tuple[bool, str]:
        if not text or len(text) < 6: return False, text
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        corrections = [('0','O'),('O','0'),('1','I'),('I','1'),('5','S'),('S','5')]
        candidates = {clean}
        for a,b in corrections:
            for c in list(candidates): candidates.add(c.replace(a,b))
        for cand in candidates:
            if len(cand) == 7:
                if cand[:3].isalpha() and cand[3].isdigit() and cand[4].isalpha() and cand[5:].isdigit():
                    return True, f"{cand[:3]}-{cand[3:4]}{cand[4:5]}{cand[5:]}"
                if cand[:3].isalpha() and cand[3:].isdigit():
                    return True, f"{cand[:3]}-{cand[3:]}"
        return False, clean

    # ---------- Pipeline por frame (para imagens estáticas) ----------
    def detect_plate_in_frame(self, frame: np.ndarray, debug: bool = False) -> Tuple[List[str], List[Tuple[int,int,int,int]]]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if self.mtx is not None and self.dist is not None:
            frame = cv2.undistort(frame, self.mtx, self.dist, None)
        
        proc, scale = self._resize_for_speed(frame)
        regions = self.remove_duplicate_regions(
            self.detect_plate_regions_haar(proc) + self.detect_plate_regions_contour(proc)
        )
        regions_limited = regions[:self.max_regions_per_frame]
        rois = [proc[y:y+h, x:x+w] for (x,y,w,h) in regions_limited]

        plates = []
        if rois:
            # Preparar dados para OCR com informações de salvamento
            output_subdir = os.path.join(self.output_dir, f"detection_{timestamp}") if self.save_preprocessing else ""
            
            ocr_results = []
            if self.ocr_engine == 'easyocr':
                jobs = [{"roi": r, "save_preprocessing": self.save_preprocessing, 
                        "output_dir": output_subdir, "region_id": i} for i, r in enumerate(rois)]
                ocr_results = self.pool.map(_easyocr_worker, jobs)
            elif self.ocr_engine == 'tesseract':
                jobs = [{"roi": r, "psm_cfgs": self.psm_configs, "mode": self.use_variants,
                        "save_preprocessing": self.save_preprocessing, "output_dir": output_subdir, 
                        "region_id": i} for i, r in enumerate(rois)]
                ocr_results = self.pool.map(_tesseract_worker, jobs)
            elif self.ocr_engine == 'dual':
                jobs_easy = [{"roi": r, "save_preprocessing": self.save_preprocessing, 
                             "output_dir": output_subdir, "region_id": i} for i, r in enumerate(rois)]
                jobs_tess = [{"roi": r, "psm_cfgs": self.psm_configs, "mode": self.use_variants,
                             "save_preprocessing": self.save_preprocessing, "output_dir": output_subdir, 
                             "region_id": i} for i, r in enumerate(rois)]
                res_easy = self.pool.map_async(_easyocr_worker, jobs_easy)
                res_tess = self.pool.map_async(_tesseract_worker, jobs_tess)
                ocr_results = res_easy.get() + res_tess.get()

            detected_texts = [t for sub in ocr_results for t in sub]
            for t in detected_texts:
                ok, p = self.validate_and_format_plate(t)
                if ok and p not in plates: plates.append(p)

        # Salvar etapas de detecção se solicitado
        if self.save_preprocessing:
            self.save_detection_stages(frame, regions_limited, plates, timestamp)

        inv = 1.0 / scale
        boxes = [(int(x*inv), int(y*inv), int(w*inv), int(h*inv)) for (x, y, w, h) in regions_limited]
        if debug: print(f"[static] regiões={len(regions)} OCR em {len(regions_limited)} | placas={plates}")
        return plates, boxes

    # ---------- Loop da câmera (Lógica Assíncrona) ----------
    def run_camera(self, cam_index: int = 0, debug: bool = False):
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"❌ Não foi possível abrir a câmera {cam_index}")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        recent = deque(maxlen=20)
        confirmed_plate = None
        frame_id = 0
        t0 = time.time()
        fps = 0.0
        fps_update_every = 12
        paused = False
        ocr_job_tess, ocr_job_easy = None, None
        last_drawn_boxes = []

        print(f"🎥 Usando OCR Engine: {self.ocr_engine}. Pressione 'q' para sair.")

        try:
            while True:
                if not paused:
                    ok, frame = cap.read()
                    if not ok:
                        print("⚠️ Falha ao ler frame."); break
                    
                    if self.mtx is not None and self.dist is not None:
                        frame = cv2.undistort(frame, self.mtx, self.dist, None)
                    
                    # --- Lógica de OCR Assíncrono ---
                    should_start_job = frame_id % self.ocr_every == 0
                    if should_start_job and ocr_job_tess is None and ocr_job_easy is None:
                        proc_frame, scale = self._resize_for_speed(frame)
                        regions = self.remove_duplicate_regions(
                            self.detect_plate_regions_haar(proc_frame) + self.detect_plate_regions_contour(proc_frame)
                        )
                        regions_limited = regions[:self.max_regions_per_frame]
                        rois = [proc_frame[y:y+h, x:x+w] for (x,y,w,h) in regions_limited]
                        
                        inv_scale = 1.0 / scale
                        last_drawn_boxes = [(int(x*inv_scale), int(y*inv_scale), int(w*inv_scale), int(h*inv_scale)) for (x,y,w,h) in regions_limited]
                        
                        if rois:
                            output_subdir = os.path.join(self.output_dir, f"live_{self.session_id}_{frame_id:06d}") if self.save_preprocessing else ""
                            
                            if self.ocr_engine == 'easyocr':
                                jobs = [{"roi": r, "save_preprocessing": self.save_preprocessing, 
                                        "output_dir": output_subdir, "region_id": i} for i, r in enumerate(rois)]
                                ocr_job_easy = self.pool.map_async(_easyocr_worker, jobs)
                            elif self.ocr_engine == 'tesseract':
                                jobs = [{"roi": r, "psm_cfgs": self.psm_configs, "mode": self.use_variants,
                                        "save_preprocessing": self.save_preprocessing, "output_dir": output_subdir, 
                                        "region_id": i} for i, r in enumerate(rois)]
                                ocr_job_tess = self.pool.map_async(_tesseract_worker, jobs)
                            elif self.ocr_engine == 'dual':
                                jobs_easy = [{"roi": r, "save_preprocessing": self.save_preprocessing, 
                                             "output_dir": output_subdir, "region_id": i} for i, r in enumerate(rois)]
                                jobs_tess = [{"roi": r, "psm_cfgs": self.psm_configs, "mode": self.use_variants,
                                             "save_preprocessing": self.save_preprocessing, "output_dir": output_subdir, 
                                             "region_id": i} for i, r in enumerate(rois)]
                                ocr_job_easy = self.pool.map_async(_easyocr_worker, jobs_easy)
                                ocr_job_tess = self.pool.map_async(_tesseract_worker, jobs_tess)
                            
                            if debug: print(f"[live] Job OCR com {self.ocr_engine} iniciado para {len(rois)} regiões.")

                    tess_ready = ocr_job_tess is None or ocr_job_tess.ready()
                    easy_ready = ocr_job_easy is None or ocr_job_easy.ready()

                    if tess_ready and easy_ready:
                        try:
                            all_results = []
                            if ocr_job_tess is not None: all_results.extend(ocr_job_tess.get())
                            if ocr_job_easy is not None: all_results.extend(ocr_job_easy.get())
                            
                            if all_results:
                                detected_texts = [t for sublist in all_results for t in sublist]
                                plates = [p for t in detected_texts if (p := self.validate_and_format_plate(t)[1]) and self.validate_and_format_plate(t)[0]]
                                
                                if plates:
                                    recent.extend(list(set(plates)))
                                    confirmed_plate = plates[0]
                                    print(f"✅ PLACA CONFIRMADA: {confirmed_plate}")
                                    
                                    # Salvar etapas do frame onde a placa foi confirmada
                                    if self.save_preprocessing:
                                        self.save_detection_stages(frame, 
                                                                  [(int(x*inv_scale), int(y*inv_scale), 
                                                                    int(w*inv_scale), int(h*inv_scale)) 
                                                                   for (x,y,w,h) in regions_limited], 
                                                                  plates, f"live_{self.session_id}_{frame_id:06d}")
                                            
                                if debug: print(f"[live] Job OCR finalizado. Placas: {plates}")
                        except Exception as e:
                            if debug: print(f"Erro ao obter resultado do OCR: {e}")
                        finally:
                            ocr_job_tess, ocr_job_easy = None, None

                    # --- Desenho na tela ---
                    for (x, y, w, h) in last_drawn_boxes:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    overlay = frame.copy()
                    label = confirmed_plate if confirmed_plate else "Procurando placa..."
                    cv2.rectangle(overlay, (10, 10), (420, 80), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                    cv2.putText(frame, f"Placa: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)

                    if frame_id % fps_update_every == 0:
                        t1 = time.time()
                        fps = fps_update_every / max(t1 - t0, 1e-6)
                        t0 = t1
                    cv2.putText(frame, f"{fps:.1f} FPS", (frame.shape[1]-150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                    frame_id += 1

                cv2.imshow("Reconhecimento de Placas - Live", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('d'): debug = not debug; print(f"🔧 Debug: {debug}")
                elif key == ord('p'): paused = not paused; print("⏸️ Pausado" if paused else "▶️ Continuando")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.close()

    # ---------- Calibração de câmera ----------
    def calibrate_camera(self, cam_index: int, output_file: str, checkerboard: Tuple[int, int], num_images: int = 15):
        """
        Calibra a câmera usando um tabuleiro de xadrez e salva os parâmetros.
        :param cam_index: Índice da câmera para calibração.
        :param output_file: Caminho para salvar os parâmetros de calibração.
        :param checkerboard: Dimensões do tabuleiro de xadrez (e.g., (6, 9)).
        :param num_images: Número de imagens a serem capturadas para calibração.
        """
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"❌ Não foi possível abrir a câmera {cam_index}.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
        
        objpoints = []
        imgpoints = []
        count = 0
        print(f"👀 Prepare o tabuleiro. Capturando {num_images} imagens para calibração. Pressione 's' para salvar, 'q' para sair.")
        
        try:
            while count < num_images:
                ok, frame = cap.read()
                if not ok: break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
                
                if ret:
                    cv2.drawChessboardCorners(frame, checkerboard, corners, ret)
                    cv2.putText(frame, f"Cantos detectados!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(frame, f"Imagens capturadas: {count}/{num_images}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Calibracao de Camera", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') and ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    count += 1
                    print(f"📸 Imagem {count} capturada.")
                elif key == ord('q'):
                    print("❌ Calibração cancelada.")
                    return
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if len(objpoints) > 5:
            try:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                if ret:
                    np.savez(output_file, mtx=mtx, dist=dist)
                    print(f"✅ Calibração completa. Parâmetros salvos em {output_file}")
                    print("Matriz da Câmera (mtx):\n", mtx)
                    print("\nCoeficientes de Distorção (dist):\n", dist)
                else:
                    print("❌ Erro na calibração. Não foi possível calcular os parâmetros.")
            except Exception as e:
                print(f"❌ Erro ao calibrar a câmera: {e}")
        else:
            print("❌ Número insuficiente de imagens para calibração. Tente novamente.")

# --------------- CLI ---------------
def main():
    parser = argparse.ArgumentParser(description="Reconhecedor de Placas (tempo real + OCR paralelo)")
    parser.add_argument('--camera', action='store_true', help='Usar câmera em tempo real')
    parser.add_argument('--cam-index', type=int, default=0, help='Índice da câmera')
    parser.add_argument('--no-haar', action='store_true', help='Desativar Haar Cascade')
    parser.add_argument('--debug', action='store_true', help='Ativar logs de debug')
    parser.add_argument('--ultra', action='store_true', help='Preset ULTRA (mais FPS, menos acurácia)')
    parser.add_argument('--pool-size', type=int, default=None, help='Nº de processos de OCR')
    parser.add_argument('--ocr-engine', type=str, default='easyocr', choices=['tesseract', 'easyocr', 'dual'], help='Motor de OCR a ser usado')
    parser.add_argument('--cam-calib', type=str, default=None, help='Caminho para o arquivo de calibração de câmera (ex: camera_params.npz)')
    parser.add_argument('--calibrate', action='store_true', help='Modo de calibração de câmera. Use com --camera.')
    parser.add_argument('--save-preprocessing', action='store_true', help='Salvar todas as etapas de pré-processamento')
    parser.add_argument('--output-dir', type=str, default='plate_processing', help='Diretório para salvar arquivos de processamento')
    parser.add_argument('image_path', nargs='?', help='Caminho da imagem (modo estático)')
    args = parser.parse_args()

    try:
        cv2.setUseOptimized(True)
    except Exception: pass

    if args.calibrate and not args.camera:
        print("⚠️ O modo de calibração (--calibrate) deve ser usado com a câmera (--camera).")
        return
    
    if args.calibrate:
        recognizer = BrazilianPlateRecognizer()
        # Parâmetros para o tabuleiro (ex: 6x9, padrão para calibração)
        recognizer.calibrate_camera(cam_index=args.cam_index, output_file='camera_params.npz', checkerboard=(6, 8))
        recognizer.close()
        return

    params = dict(
        target_w=576 if args.ultra else 640,
        max_regions_per_frame=3 if args.ultra else 4,
        ocr_every=8 if args.ultra else 6,
        preset_ultra=args.ultra,
        save_preprocessing=args.save_preprocessing,
        output_dir=args.output_dir
    )
    recognizer = BrazilianPlateRecognizer(use_haar=not args.no_haar, pool_size=args.pool_size, ocr_engine=args.ocr_engine, cam_calib_path=args.cam_calib, **params)

    if args.camera:
        recognizer.run_camera(cam_index=args.cam_index, debug=args.debug)
    elif args.image_path:
        img = cv2.imread(args.image_path)
        if img is None:
            print(f"❌ Não foi possível abrir a imagem: {args.image_path}")
        else:
            plates, _ = recognizer.detect_plate_in_frame(img, debug=args.debug)
            print("🎯 Placas detectadas:", plates if plates else "Nenhuma")
            if args.save_preprocessing and plates:
                print(f"📁 Arquivos de processamento salvos em: {args.output_dir}")
    else:
        parser.print_help()
    
    recognizer.close()

if __name__ == "__main__":
    main()