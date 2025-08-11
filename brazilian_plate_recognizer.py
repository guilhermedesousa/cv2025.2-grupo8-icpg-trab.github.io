#!/usr/bin/env python3
"""
Script final para reconhecimento de placas brasileiras
Combina múltiplas técnicas para máxima eficácia
"""

import cv2
import numpy as np
import pytesseract
import argparse
import os
import re
import subprocess
import sys
from typing import List, Tuple, Optional


class BrazilianPlateRecognizer:
    """
    Reconhecedor de placas brasileiras (Mercosul e padrão antigo)
    Usa OpenCV para pré-processamento e pytesseract para OCR
    """
    
    def __init__(self):
        self.setup_tesseract()
        self.load_haar_cascade()
        
    def setup_tesseract(self):
        """Configura o path do tesseract automaticamente usando 'which'"""
        try:
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0:
                tesseract_path = result.stdout.strip()
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                print(f"✅ Tesseract configurado: {tesseract_path}")
            else:
                print("❌ Tesseract não encontrado! Instale com: brew install tesseract")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Erro ao configurar tesseract: {e}")
            sys.exit(1)
    
    def load_haar_cascade(self):
        """Carrega classificador Haar Cascade se disponível"""
        cascade_path = os.path.join(os.path.dirname(__file__), 'trabalhos', 'haarcascade_russian_plate_number.xml')
        if os.path.exists(cascade_path):
            self.plate_cascade = cv2.CascadeClassifier(cascade_path)
            print(f"✅ Haar Cascade carregado: {cascade_path}")
        else:
            self.plate_cascade = None
            print("⚠️ Haar Cascade não encontrado, usando apenas outras técnicas")
    
    def preprocess_image_variants(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Cria múltiplas versões processadas da imagem para melhorar OCR
        """
        variants = []
        
        # Converter para escala de cinza
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Variante 1: Original em escala de cinza
        variants.append(gray)
        
        # Variante 2: Equalização de histograma
        equalized = cv2.equalizeHist(gray)
        variants.append(equalized)
        
        # Variante 3: Filtro bilateral + OTSU
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        
        # Variante 4: Detectar áreas brancas (típicas de placas)
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 150])
            upper_white = np.array([180, 50, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            white_enhanced = cv2.bitwise_and(gray, gray, mask=white_mask)
            variants.append(white_enhanced)
        
        # Variante 5: Aumentar contraste e brilho
        alpha = 1.5  # Contraste
        beta = 20    # Brilho
        contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        variants.append(contrasted)
        
        # Variante 6: Filtro gaussiano + threshold adaptativo
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        variants.append(adaptive)
        
        return variants
    
    def detect_plate_regions_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detecta placas usando Haar Cascade"""
        if self.plate_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detectar em múltiplas escalas
        plates = self.plate_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=3, 
            minSize=(80, 20),
            maxSize=(400, 120)
        )
        
        return [(x, y, w, h) for (x, y, w, h) in plates]
    
    def detect_plate_regions_contour(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detecta placas usando contornos e filtros geométricos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        regions = []
        
        # Múltiplas abordagens de detecção de bordas
        edge_params = [(30, 100), (50, 150), (100, 200)]
        
        for low, high in edge_params:
            # Pré-processamento
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, low, high)
            
            # Morfologia para conectar bordas
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # Filtrar por área
                    
                    # Aproximar para polígono
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4:  # Aproximadamente retangular
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        # Filtros específicos para placas brasileiras
                        if 2.0 <= aspect_ratio <= 5.0 and w > 60 and h > 15:
                            regions.append((x, y, w, h))
        
        return regions
    
    def extract_text_comprehensive(self, image: np.ndarray) -> List[dict]:
        """
        Extrai todo texto da imagem com informações de posição e confiança
        """
        variants = self.preprocess_image_variants(image)
        all_detections = []
        
        # Configurações do tesseract para diferentes cenários
        configs = [
            r'--oem 3 --psm 6',  # Bloco uniforme de texto
            r'--oem 3 --psm 7',  # Linha de texto única
            r'--oem 3 --psm 8',  # Palavra única        
            r'--oem 3 --psm 11', # Texto esparso
            r'--oem 3 --psm 12', # Texto esparso com OSD
            r'--oem 3 --psm 13', # Linha bruta - trata a imagem como uma única linha de texto
        ]
        
        for variant_idx, variant in enumerate(variants):
            for config_idx, config in enumerate(configs):
                try:
                    # OCR com dados detalhados
                    data = pytesseract.image_to_data(variant, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Processar cada detecção
                    for i in range(len(data['text'])):
                        text = data['text'][i].strip()
                        conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                        
                        if text and conf > 30:  # Filtrar por confiança mínima
                            # Limpar e normalizar texto
                            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                            
                            if len(clean_text) >= 5:  # Mínimo para consideração
                                detection = {
                                    'text': clean_text,
                                    'original': text,
                                    'confidence': conf,
                                    'x': data['left'][i],
                                    'y': data['top'][i],
                                    'w': data['width'][i],
                                    'h': data['height'][i],
                                    'variant': variant_idx,
                                    'config': config_idx
                                }
                                all_detections.append(detection)
                
                except Exception:
                    continue
        
        return all_detections
    
    def validate_and_format_plate(self, text: str) -> Tuple[bool, str]:
        """
        Valida se o texto é uma placa brasileira válida e a formata corretamente
        """
        if not text or len(text) < 6:
            return False, f"Texto muito curto: '{text}'"
        
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Correções comuns de OCR
        ocr_corrections = [
            ('0', 'O'), ('O', '0'), ('Q', '0'),  # Zero/O confusão
            ('1', 'I'), ('I', '1'), ('L', '1'),  # Um/I/L confusão
            ('5', 'S'), ('S', '5'),              # Cinco/S confusão
            ('8', 'B'), ('B', '8'),              # Oito/B confusão
            ('2', 'Z'), ('Z', '2'),              # Dois/Z confusão
            ('6', 'G'), ('G', '6'),              # Seis/G confusão
        ]
        
        # Gerar candidatos aplicando correções
        candidates = [clean]
        
        for original, replacement in ocr_corrections:
            new_candidates = []
            for candidate in candidates:
                for i, char in enumerate(candidate):
                    if char == original:
                        corrected = candidate[:i] + replacement + candidate[i+1:]
                        if corrected not in candidates:
                            new_candidates.append(corrected)
            candidates.extend(new_candidates)
        
        # Validar cada candidato
        for candidate in candidates:
            # Tentar comprimentos diferentes (OCR pode falhar)
            for test_candidate in [candidate, candidate[:-1], candidate + '0', candidate + 'A']:
                if len(test_candidate) == 7:
                    # Padrão Mercosul: AAA0A00 (3 letras, 1 número, 1 letra, 2 números)
                    if (test_candidate[:3].isalpha() and 
                        test_candidate[3].isdigit() and 
                        test_candidate[4].isalpha() and 
                        test_candidate[5:].isdigit()):
                        formatted = f"{test_candidate[:3]}-{test_candidate[3:4]}{test_candidate[4:5]}{test_candidate[5:]}"
                        return True, formatted
                    
                    # Padrão antigo: AAA0000 (3 letras, 4 números)
                    if (test_candidate[:3].isalpha() and 
                        test_candidate[3:].isdigit()):
                        formatted = f"{test_candidate[:3]}-{test_candidate[3:]}"
                        return True, formatted
        
        return False, f"Padrão não reconhecido: '{clean}'"
    
    def detect_plate_in_image(self, image_path: str, debug: bool = False) -> List[str]:
        """
        Método principal para detectar placas em uma imagem
        """
        if not os.path.exists(image_path):
            print(f"❌ Arquivo não encontrado: {image_path}")
            return []
        
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Erro ao carregar imagem: {image_path}")
            return []
        
        print(f"\\n🔍 Analisando: {os.path.basename(image_path)}")
        
        # Abordagem 1: OCR direto em toda a imagem
        print("  📝 Executando OCR direto...")
        all_text_detections = self.extract_text_comprehensive(image)
        print(f"     Encontrados {len(all_text_detections)} textos")
        
        for detection in all_text_detections:
            print(f"     Texto: '{detection['text']}' (confiança: {detection['confidence']})")
        
        detected_plates = []
        
        # Verificar cada texto detectado
        for detection in all_text_detections:
            text = detection['text']
            confidence = detection['confidence']
            
            if debug:
                print(f"     Testando: '{text}' (confiança: {confidence})")
            
            # Validar se é placa
            is_valid, result = self.validate_and_format_plate(text)                    
            
            if is_valid:
                if result not in detected_plates:
                    detected_plates.append(result)
                    print(f"  ✅ PLACA DETECTADA: {result} (confiança: {confidence})")
                    
            # Para escolher somente uma placa, quanto mais o mesmo texto tiver aparecido, melhor
            # Contar ocorrências para priorizar

        # Abordagem 2: Detecção de regiões + OCR focado
        if not detected_plates or debug:
            print("  🎯 Detectando regiões específicas...")
            
            # Haar Cascade
            haar_regions = self.detect_plate_regions_haar(image)
            if haar_regions:
                print(f"     Haar Cascade: {len(haar_regions)} regiões")
            
            # Contornos
            contour_regions = self.detect_plate_regions_contour(image)
            if contour_regions:
                print(f"     Contornos: {len(contour_regions)} regiões")
            
            # Combinar e remover duplicatas
            all_regions = haar_regions + contour_regions
            unique_regions = self.remove_duplicate_regions(all_regions)
            
            if unique_regions:
                print(f"     Total de regiões únicas: {len(unique_regions)}")
                
                # OCR em cada região
                for i, (x, y, w, h) in enumerate(unique_regions[:10]):  # Limitar a 10 regiões
                    region = image[y:y+h, x:x+w]
                    
                    if debug:
                        print(f"     Região {i+1}: ({x},{y},{w}x{h})")
                    
                    # OCR na região específica
                    region_detections = self.extract_text_comprehensive(region)
                    
                    for detection in region_detections:
                        text = detection['text']
                        confidence = detection['confidence']
                        
                        is_valid, result = self.validate_and_format_plate(text)
                        
                        if is_valid and result not in detected_plates:
                            detected_plates.append(result)
                            print(f"  ✅ PLACA EM REGIÃO: {result} (confiança: {confidence})")
        
        return detected_plates
    
    def remove_duplicate_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Remove regiões sobrepostas"""
        if not regions:
            return []
        
        unique = []
        for region in regions:
            x1, y1, w1, h1 = region
            is_duplicate = False
            
            for existing in unique:
                x2, y2, w2, h2 = existing
                
                # Calcular sobreposição
                if (abs(x1 - x2) < 50 and abs(y1 - y2) < 50 and 
                    abs(w1 - w2) < 100 and abs(h1 - h2) < 50):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(region)
        
        # Ordenar por área (maiores primeiro)
        unique.sort(key=lambda r: r[2] * r[3], reverse=True)
        return unique


def test_all_images():
    """Testa o reconhecedor em todas as imagens de teste"""
    expected_results = {
        'car1.jpg': 'HEX-0049',
        'car3.jpg': 'PYN-4532', 
        'car4.jpg': 'ASU-3212',
        'screenshot_20250804_195353.jpg': 'BRA2E19'
    }
    
    recognizer = BrazilianPlateRecognizer()
    test_dir = '/Users/pedro/Desktop/projects/cv2025.2-grupo8-icpg-trab.github.io/trabalhos/teste'
    
    print("🚗 TESTE COMPLETO DO RECONHECEDOR DE PLACAS 🚗")
    print("="*70)
    
    results = {}
    
    for filename, expected_plate in expected_results.items():
        image_path = os.path.join(test_dir, filename)
        
        print(f"\\n📷 {filename}")
        print(f"🎯 Esperado: {expected_plate}")
        print("-" * 50)
        
        if not os.path.exists(image_path):
            print("❌ Arquivo não encontrado")
            results[filename] = 'ERROR'
            continue
        
        detected_plates = recognizer.detect_plate_in_image(image_path)
        
        if detected_plates:
            # Verificar se a placa esperada foi encontrada
            found_expected = False
            for plate in detected_plates:
                if plate.replace('-', '') == expected_plate.replace('-', ''):
                    found_expected = True
                    break
            
            if found_expected:
                print(f"\\n✅ SUCESSO! Placa esperada encontrada!")
                results[filename] = 'PASS'
            else:
                print(f"\\n⚠️ PARCIAL - Outras placas detectadas: {detected_plates}")
                results[filename] = 'PARTIAL'
        else:
            print(f"\\n❌ FALHA - Nenhuma placa detectada")
            results[filename] = 'FAIL'
    
    # Resumo final
    print("\\n" + "="*70)
    print("📊 RESUMO DOS RESULTADOS")
    print("="*70)
    
    for filename, result in results.items():
        icon = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌", "ERROR": "💥"}[result]
        print(f"{icon} {filename}: {result}")
    
    success_count = sum(1 for r in results.values() if r == 'PASS')
    partial_count = sum(1 for r in results.values() if r == 'PARTIAL')
    total = len(expected_results)
    
    print(f"\\n🏆 ESTATÍSTICAS:")
    print(f"   ✅ Sucessos completos: {success_count}/{total}")
    print(f"   ⚠️ Sucessos parciais: {partial_count}/{total}")
    print(f"   ❌ Falhas: {total - success_count - partial_count}/{total}")
    
    if success_count == total:
        print("\\n🎉 PERFEITO! Todas as placas foram detectadas corretamente! 🎉")
    elif success_count >= total * 0.75:
        print("\\n🌟 EXCELENTE! Maioria das placas detectadas!")
    elif success_count > 0:
        print("\\n👍 BOM PROGRESSO! Algumas placas detectadas!")
    else:
        print("\\n🔧 Necessário ajustar algoritmos...")


def main():
    """Função principal com CLI"""
    parser = argparse.ArgumentParser(
        description='Reconhecedor de placas brasileiras (Mercosul e padrão antigo)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Exemplos de uso:
  python %(prog)s imagem.jpg                    # Detectar placa em uma imagem
  python %(prog)s imagem.jpg --debug            # Modo debug com informações detalhadas
  python %(prog)s --test                        # Testar com todas as imagens de exemplo
        '''
    )
    
    parser.add_argument('image_path', nargs='?', 
                       help='Caminho para a imagem a ser analisada')
    parser.add_argument('--test', action='store_true', 
                       help='Executar teste com todas as imagens de exemplo')
    parser.add_argument('--debug', action='store_true', 
                       help='Ativar modo debug com informações detalhadas')
    
    args = parser.parse_args()
    
    if args.test:
        test_all_images()
    elif args.image_path:
        recognizer = BrazilianPlateRecognizer()
        plates = recognizer.detect_plate_in_image(args.image_path, args.debug)
        
        print(f"\\n{'='*50}")
        if plates:
            print(f"🎯 PLACAS DETECTADAS: {plates}")
        else:
            print("❌ NENHUMA PLACA DETECTADA")
        print('='*50)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
