#!/usr/bin/env python3
"""
Script de teste para demonstrar o sistema de exportação de etapas de pré-processamento
"""

import os
import sys
import cv2
from plates_recognizer import BrazilianPlateRecognizer

def test_preprocessing_export():
    """Testa o sistema de exportação em todas as imagens de teste"""
    
    # Imagens de teste
    test_images = [
        '/Users/pedro/Desktop/projects/cv2025.2-grupo8-icpg-trab.github.io/trabalhos/teste/car1.jpg',
        '/Users/pedro/Desktop/projects/cv2025.2-grupo8-icpg-trab.github.io/trabalhos/teste/car3.jpg',
        '/Users/pedro/Desktop/projects/cv2025.2-grupo8-icpg-trab.github.io/trabalhos/teste/car4.jpg',
        '/Users/pedro/Desktop/projects/cv2025.2-grupo8-icpg-trab.github.io/trabalhos/teste/screenshot_20250804_195353.jpg'
    ]
    
    print("🔬 TESTE DO SISTEMA DE EXPORTAÇÃO DE PRÉ-PROCESSAMENTO")
    print("="*60)
    
    # Criar reconhecedor com exportação ativada
    recognizer = BrazilianPlateRecognizer(
        save_preprocessing=True,
        output_dir="plate_processing_test",
        ocr_engine='dual',  # Usar ambos os motores para demonstração completa
        preset_ultra=False  # Usar modo completo para mais etapas
    )
    
    try:
        for i, image_path in enumerate(test_images):
            if not os.path.exists(image_path):
                print(f"⚠️ Imagem não encontrada: {image_path}")
                continue
                
            print(f"\n📷 Processando imagem {i+1}/4: {os.path.basename(image_path)}")
            print("-" * 50)
            
            # Carregar e processar imagem
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Erro ao carregar: {image_path}")
                continue
            
            # Detectar placas com exportação completa
            plates, boxes = recognizer.detect_plate_in_frame(img, debug=True)
            
            if plates:
                print(f"✅ Placas detectadas: {plates}")
                print(f"📊 Regiões processadas: {len(boxes)}")
            else:
                print("❌ Nenhuma placa detectada")
            
            print(f"📁 Arquivos salvos em: plate_processing_test/")
            
        print(f"\n🎉 TESTE CONCLUÍDO!")
        print(f"📁 Todos os arquivos de processamento estão em: plate_processing_test/")
        print("\n📋 Estrutura dos arquivos gerados:")
        print("   detection_YYYYMMDD_HHMMSS_mmm/")
        print("   ├── 00_original_frame.png          # Frame original")
        print("   ├── 01_undistorted_frame.png       # Correção de distorção (se aplicável)")
        print("   ├── 02_resized_for_processing.png  # Frame redimensionado")
        print("   ├── 03_haar_cascade_detection.png  # Detecção Haar Cascade")
        print("   ├── 04_contour_detection.png       # Detecção por contornos")
        print("   ├── 05_final_regions.png           # Regiões finais selecionadas")
        print("   ├── detection_info.txt             # Informações da detecção")
        print("   ├── region_XX_00_original_roi.png  # ROI original de cada região")
        print("   ├── region_XX_01_original_gray.png # Conversão para escala de cinza")
        print("   ├── region_XX_02_equalized.png     # Equalização de histograma")
        print("   ├── region_XX_03_gaussian_blur.png # Filtro gaussiano")
        print("   ├── region_XX_04_adaptive_gaussian.png # Threshold adaptativo")
        print("   ├── region_XX_easyocr_input.png    # Entrada do EasyOCR")
        print("   ├── region_XX_easyocr_results.txt  # Resultados do EasyOCR")
        print("   ├── region_XX_easyocr_annotated.png # EasyOCR com anotações")
        print("   └── region_XX_ocr_variant_XX_config_XX.txt # Detalhes do Tesseract")
        
    finally:
        recognizer.close()

def create_summary_report():
    """Cria um relatório resumido dos resultados"""
    
    output_dir = "plate_processing_test"
    if not os.path.exists(output_dir):
        print("❌ Diretório de processamento não encontrado. Execute o teste primeiro.")
        return
    
    # Encontrar todas as detecções
    detections = []
    for item in os.listdir(output_dir):
        if item.startswith("detection_") and os.path.isdir(os.path.join(output_dir, item)):
            detections.append(item)
    
    detections.sort()
    
    print(f"\n📊 RELATÓRIO RESUMIDO")
    print("="*50)
    print(f"Total de detecções processadas: {len(detections)}")
    
    for detection in detections:
        detection_path = os.path.join(output_dir, detection)
        info_file = os.path.join(detection_path, "detection_info.txt")
        
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extrair informações básicas
                lines = content.split('\n')
                timestamp = next((line.split(': ')[1] for line in lines if line.startswith('Timestamp:')), 'N/A')
                placas = next((line.split(': ')[1] for line in lines if line.startswith('Placas reconhecidas:')), 'N/A')
                regioes = next((line.split(': ')[1] for line in lines if line.startswith('Regiões detectadas:')), 'N/A')
                
                print(f"\n🔍 {detection}")
                print(f"   ⏰ Timestamp: {timestamp}")
                print(f"   🎯 Placas: {placas}")
                print(f"   📦 Regiões: {regioes}")
                
            except Exception as e:
                print(f"❌ Erro ao ler {info_file}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Teste do sistema de exportação de pré-processamento")
    parser.add_argument('--test', action='store_true', help='Executar teste completo')
    parser.add_argument('--report', action='store_true', help='Gerar relatório dos resultados')
    parser.add_argument('--clean', action='store_true', help='Limpar arquivos de teste anteriores')
    
    args = parser.parse_args()
    
    if args.clean:
        import shutil
        if os.path.exists("plate_processing_test"):
            shutil.rmtree("plate_processing_test")
            print("🧹 Arquivos de teste anteriores removidos.")
        else:
            print("ℹ️ Não há arquivos de teste para remover.")
    
    elif args.test:
        test_preprocessing_export()
        
    elif args.report:
        create_summary_report()
        
    else:
        parser.print_help()
        print("\nExemplos:")
        print("  python test_preprocessing_export.py --test    # Executar teste completo")
        print("  python test_preprocessing_export.py --report  # Gerar relatório")
        print("  python test_preprocessing_export.py --clean   # Limpar arquivos anteriores")
