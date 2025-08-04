import cv2

# Caminho do classificador Haar cascade
cascade_path = "haarcascade_russian_plate_number.xml"

# Carregar o classificador
plate_cascade = cv2.CascadeClassifier(cascade_path)

# Iniciar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Criar a janela antes do loop, para evitar múltiplas janelas
cv2.nameWindow("Detecção de Placas (somente OpenCV)", cv2.WINDOW_NORMAL)d

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame")
        break

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar placas
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Desenhar retângulos em torno das placas detectadas
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar o frame na janela
    cv2.imshow("Detecção de Placas (somente OpenCV)", frame)

    # Interromper o loop ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
