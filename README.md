# Grupo 8 - ICPG - Sistema de Leitura Automática de Placas de Veículos

Este projeto implementa um sistema de reconhecimento de placas de veículos brasileiro em tempo real, utilizando visão computacional e Reconhecimento Óptico de Caracteres (OCR). Suporta múltiplos motores OCR, processamento paralelo e calibração de câmeras para melhorar a precisão.

## Features

- **Suporte de câmera real-time**: Processa fluxos de vídeo de uma câmera em tempo real.
- **OCR com processamento paralelo**: Usa processamento paralelo para realizar OCR sem bloquear a interface do usuário.
- **Motores OCR**: Suporta múltiplos motores OCR:
  - **Tesseract**: Um motor OCR de código aberto amplamente utilizado.
  - **EasyOCR**: Um motor OCR baseado em PyTorch, opcionalmente integrado.
  - **Dual Mode**: Permite o uso combinado de Tesseract e EasyOCR para maior flexibilidade.
- **Presets**: Escolha entre `fast` (default) e `ultra` para otimizar o desempenho ou a precisão.
- **Haar Cascade Fallback**: Detecta placas usando cascatas Haar como método de fallback.
- **Calibração da câmera**: 
  - Calibra a câmera para melhorar a precisão do reconhecimento.
  - Usa parâmetros de calibração para corrigir distorções de lente.
- **Mecanismos de debounce**: 
  - Implementa mecanismos de debounce para evitar leituras duplicadas.
  - Reduz a carga de processamento ao ignorar placas já reconhecidas recentemente.

## Requisitos

- Python 3.7+
- OpenCV
- NumPy
- Tesseract OCR
- EasyOCR (opcional, para EasyOCR ou dual mode)

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/your-repo/brazilian-plate-recognizer.git
   cd brazilian-plate-recognizer
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Instale o Tesseract OCR:
   - No Ubuntu:
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - No Windows, baixe o instalador do [site oficial](https://tesseract-ocr.github.io/tessdoc/Downloads.html) e instale
   - No macOS, use Homebrew:
     ```bash
     brew install tesseract
     ```
4. (Opcional) Instale o EasyOCR:
   ```bash
   pip install easyocr
   ```

## Exemplo de uso

Interface de linha de comando (CLI). Execute o script com as opções desejadas:

```bash
python3 plates_recognizer.py [OPTIONS] [IMAGE_PATH]
```

### Opções

- `--camera`: Usa a câmera para captura de vídeo (real-time).
- `--cam-index INTEGER`: Índice da câmera (padrão: `0`).
- `--no-haar`: Desativa o uso de cascata Haar.
- `--debug`: Ativa o modo de depuração para informações detalhadas.
- `--ultra`: Ativa o modo ultra para maior FPS, mas com menor acurácia.
- `pool-size INTEGER`: Tamanho do pool de threads para OCR.
- `--ocr-engine [tesseract|easyocr|dual]`: Motor OCR a ser usado (padrão: tesseract).
- `--cam-calib`: Caminho para o arquivo de calibração da câmera.
- `--calibrate`: Ativa o modo de calibração da câmera.

### Exemplos

**Reconhecimento em real-time**:
```bash
python3 plates_recognizer.py --camera
```

**Reconhecimento em uma imagem**:
```bash
python3 plates_recognizer.py path/to/image.jpg
```

**Usando EasyOCR**:
```bash
python3 plates_recognizer.py --ocr-engine easyocr --camera
```

**Calibração da câmera**:
```bash
python3 plates_recognizer.py --calibrate --camera
```

---

### Licença
Este projeto está licenciado sob a Licença MIT.