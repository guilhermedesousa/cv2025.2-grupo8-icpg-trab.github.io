#!/usr/bin/env python3
"""
ESZA019 - Visão Computacional
Sistema de Detecção de Placas de Veículos
Autores: [Nome Completo] - RA: [RA]
         [Nome Completo] - RA: [RA]  
         [Nome Completo] - RA: [RA]
Data: Agosto 2025
Programa: license_plate_detector.py
Exemplo de execução: python3 license_plate_detector.py
"""

import cv2
import numpy as np
import pytesseract
import re
import time
import math
import os
from datetime import datetime
from PIL import Image
import json


class LicensePlateDetector:
    def __init__(self):
        """
        Inicializa o detector de placas com parâmetros otimizados
        """
        # Parâmetros para detecção de placas
        self.min_area = 2000
        self.max_area = 50000
        self.min_width = 80
        self.min_height = 10
        self.max_width = 600
        self.max_height = 300
        
        # Configuração do Tesseract
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        
        # Histórico de detecções
        self.detections = []
        
        # Métricas de desempenho
        self.total_frames = 0
        self.successful_detections = 0
        self.processing_times = []
        
        print("Sistema de Detecção de Placas - NOME_DA_EQUIPE")
        print("Inicializando sistema...")

    def preprocess_image(self, frame):
        """
        Bloco: Pré-processamento
        Entrada: Imagem/frame
        Processamento: Redimensionamento, conversão para escala de cinza, 
                      redução de ruído, operações morfológicas
        Saída: Imagem pré-processada
        """
        # Redimensionamento (transformação geométrica)
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), 
                             interpolation=cv2.INTER_AREA)
        
        # Conversão para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Filtragem de imagens - Filtro Gaussiano para redução de ruído
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Equalização de histograma para melhorar contraste
        # equalized = cv2.equalizeHist(blurred)
        
        # Detecção de bordas usando Canny
        edges = cv2.Canny(blurred, 50, 200)
        
        # Operações morfológicas para conectar componentes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return frame, gray, morph

    def detect_license_plate(self, processed_image, original_frame):
        """
        Bloco: Detecção de Placa (OpenCV)
        Entrada: Imagem pré-processada
        Processamento: Localização da placa no frame
        Saída: Imagem da placa recortada com o ROI (Region of Interest)
        """
        # Encontrar contornos
        contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        potential_plates = []
        
        for contour in contours:
            # Calcular área e perímetro
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # Aproximar contorno para polígono
            epsilon = 0.018 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Considerar apenas contornos com 4 vértices (retângulos)
            # if len(approx) != 4:
            #     continue
            
            # Calcular bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Verificar proporções típicas de placa
            aspect_ratio = w / float(h)
            print(f"Aspect Ratio: {aspect_ratio}")
            print(f"Width: {w}, Height: {h}")
            color = (0, 255, 0)
            if (aspect_ratio < 1.5 or aspect_ratio > 6.0 or
                w < self.min_width or w > self.max_width or
                h < self.min_height or h > self.max_height):
                continue
                # color = (0, 0, 255)
            
            # Calcular solidez (área do contorno / área do retângulo)
            rect_area = w * h
            if rect_area == 0:
                continue
            
            solidity = area / float(rect_area)
            if solidity < 0.2:
                continue
            
            # Adicionar verificação de ângulo usando minAreaRect
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            if angle < -45:
                angle = 90 + angle
            
            print(f"Angle: {angle}")
            # if abs(angle) > 25: # Ignorar placas muito inclinadas
            #     continue
            
            # cv2.rectangle(original_frame, (x, y), (x+w, y+h), color, 2)
            # cv2.putText(original_frame, str(aspect_ratio), (x, y-10), 
            #               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # cv2.imshow('Teste', original_frame)
            
            potential_plates.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity
            })
        
        # Ordenar por área (maior primeiro)
        potential_plates.sort(key=lambda x: x['area'], reverse=True)
        
        plates = []
        for plate_info in potential_plates[:5]:  # Considerar até 5 melhores candidatos
            x, y, w, h = plate_info['bbox']
            
            # Expandir ROI ligeiramente
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(original_frame.shape[1] - x, w + 2 * margin)
            h = min(original_frame.shape[0] - y, h + 2 * margin)
            
            # Extrair ROI
            plate_roi = original_frame[y:y+h, x:x+w]
            
            if plate_roi.size > 0:
                plates.append({
                    'roi': plate_roi,
                    'bbox': (x, y, w, h),
                    'info': plate_info
                })
        
        return plates

    def recognize_characters(self, plate_image):
        """
        Bloco: Reconhecimento de Caracteres (OCR)
        Entrada: Imagem da placa
        Processamento: Reconhecimento dos caracteres (pytesseract)
        Saída: Texto da placa
        """
        if plate_image is None or plate_image.size == 0:
            return ""
        
        # Pré-processamento específico para OCR
        gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar para melhorar OCR
        height, width = gray_plate.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray_plate = cv2.resize(gray_plate, (new_width, new_height), 
                                  interpolation=cv2.INTER_CUBIC)
        
        # Aplicar filtros para melhorar OCR
        blurred = cv2.GaussianBlur(gray_plate, (3, 3), 0)
        
        # Binarização adaptativa
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Operações morfológicas para limpar a imagem
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        cv2.imshow('Imagem Limpa', cleaned)
        
        # Configuração do Tesseract para placas brasileiras
        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        try:
            # Reconhecimento de texto
            text = pytesseract.image_to_string(cleaned, config=config)
            print(f"Reconhecimento OCR: {text}")
            text = text.strip().upper()
            
            # Filtrar apenas caracteres alfanuméricos
            text = re.sub(r'[^A-Z0-9]', '', text)
            
            # Validar padrão de placa brasileira (ABC1234 ou ABC1D23)
            if self.validate_brazilian_plate(text):
                return text
            
        except Exception as e:
            print(f"Erro no OCR: {e}")
        
        return ""

    def validate_brazilian_plate(self, text):
        """
        Valida se o texto corresponde ao padrão de placa brasileira
        """
        if len(text) != 7:
            return False
        
        # Padrão antigo: ABC1234
        old_pattern = re.match(r'^[A-Z]{3}[0-9]{4}$', text)
        
        # Padrão Mercosul: ABC1D23
        mercosul_pattern = re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', text)
        
        return old_pattern is not None or mercosul_pattern is not None

    def register_detection(self, plate_text, confidence=0.0):
        """
        Bloco: Registro/Exibição do Resultado
        Entrada: Texto da placa
        Processamento: Armazenamento ou exibição ao usuário
        Saída: Informação registrada/exibida
        """
        if plate_text:
            detection = {
                'timestamp': datetime.now().isoformat(),
                'plate': plate_text,
                'confidence': confidence
            }
            
            self.detections.append(detection)
            self.successful_detections += 1
            
            print(f"Placa detectada: {plate_text} - {detection['timestamp']}")
            
            # Salvar em arquivo JSON
            self.save_detections()

    def save_detections(self):
        """
        Salva as detecções em arquivo JSON
        """
        try:
            with open('detections.json', 'w') as f:
                json.dump(self.detections, f, indent=2)
        except Exception as e:
            print(f"Erro ao salvar detecções: {e}")

    def calculate_metrics(self):
        """
        Calcula métricas de desempenho do sistema
        """
        if self.total_frames == 0:
            return {
                'accuracy': 0.0,
                'avg_processing_time': 0.0,
                'total_detections': 0
            }
        
        accuracy = (self.successful_detections / self.total_frames) * 100
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'accuracy': accuracy,
            'avg_processing_time': avg_time,
            'total_detections': self.successful_detections,
            'total_frames': self.total_frames
        }

    def process_frame(self, frame):
        """
        Processa um frame completo seguindo o pipeline
        """
        start_time = time.time()
        
        # Bloco: Pré-processamento
        processed_frame, gray, edges = self.preprocess_image(frame)
        
        cv2.imshow('Imagem Pré-Processada', processed_frame)
        cv2.imshow('Imagem em Escala de Cinza', gray)
        cv2.imshow('Imagem com Bordas', edges)
        
        # Bloco: Detecção de Placa
        plates = self.detect_license_plate(edges, processed_frame)
        print(f"Placas detectadas: {len(plates)}")
        
        detected_text = ""
        
        # Processar cada placa candidata
        for plate in plates:
            # Bloco: Reconhecimento de Caracteres
            plate_text = self.recognize_characters(plate['roi'])
            
            if plate_text:
                detected_text = plate_text
                
                # Bloco: Registro/Exibição
                self.register_detection(plate_text)
                
                # Desenhar bounding box na imagem
                x, y, w, h = plate['bbox']
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(processed_frame, plate_text, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break
        
        # Atualizar métricas
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.total_frames += 1
        
        return processed_frame, detected_text

    def run_video_detection(self, source=0):
        """
        Executa detecção em tempo real usando webcam
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a câmera")
            return
        
        print("Iniciando detecção em tempo real...")
        print("Pressione 'q' para sair, 's' para capturar screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Processar frame
            processed_frame, detected_text = self.process_frame(frame)
            
            # Exibir informações na tela
            metrics = self.calculate_metrics()
            info_text = [
                f"Frames processados: {self.total_frames}",
                f"Deteccoes: {self.successful_detections}",
                f"Precisao: {metrics['accuracy']:.1f}%",
                f"Tempo medio: {metrics['avg_processing_time']:.3f}s"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(processed_frame, text, (10, 30 + i*25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if detected_text:
                cv2.putText(processed_frame, f"Ultima placa: {detected_text}", 
                          (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Exibir frame
            cv2.imshow('Detector de Placas - NOME_DA_EQUIPE', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Salvar screenshot
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Screenshot salvo: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Exibir métricas finais
        final_metrics = self.calculate_metrics()
        print("\n=== MÉTRICAS FINAIS ===")
        print(f"Total de frames processados: {final_metrics['total_frames']}")
        print(f"Total de detecções: {final_metrics['total_detections']}")
        print(f"Precisão: {final_metrics['accuracy']:.2f}%")
        print(f"Tempo médio de processamento: {final_metrics['avg_processing_time']:.3f}s")

    def test_with_image(self, image_path):
        """
        Testa o sistema com uma imagem estática
        """
        if not os.path.exists(image_path):
            print(f"Erro: Imagem não encontrada: {image_path}")
            return
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Erro: Não foi possível carregar a imagem: {image_path}")
            return
        
        print(f"Processando imagem: {image_path}")
        processed_frame, detected_text = self.process_frame(frame)
        
        if detected_text:
            print(f"Placa detectada: {detected_text}")
        else:
            print("Nenhuma placa detectada")
        
        # Exibir resultado
        cv2.imshow('Resultado - Detector de Placas - NOME_DA_EQUIPE', processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """
    Função principal do programa
    """
    detector = LicensePlateDetector()
    
    # Opções de execução
    print("\nOpções de execução:")
    print("1. Detecção em tempo real (webcam)")
    print("2. Teste com imagem")
    print("3. Detecção em vídeo")
    
    choice = input("\nEscolha uma opção (1-3): ").strip()
    
    if choice == "1":
        detector.run_video_detection()
    elif choice == "2":
        image_path = input("Digite o caminho da imagem: ").strip()
        detector.test_with_image(image_path)
    elif choice == "3":
        video_path = input("Digite o caminho do vídeo: ").strip()
        detector.run_video_detection(video_path)
    else:
        print("Opção inválida. Executando detecção em tempo real...")
        detector.run_video_detection()

if __name__ == "__main__":
    main()