import cv2

# ⚠️ Certifique-se de ter esse arquivo na mesma pasta:
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_russian_plate_number.xml
cascade_path = "haarcascade_russian_plate_number.xml"

# Carregando o classificador
plate_cascade = cv2.CascadeClassifier(cascade_path)

# Inicializa a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame")
        break

    # Pré-processamento: escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecção de placas
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in plates:
        # Desenha retângulo em volta da placa
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Exibe o resultado
    cv2.imshow("Detecção de Placas (somente OpenCV)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
