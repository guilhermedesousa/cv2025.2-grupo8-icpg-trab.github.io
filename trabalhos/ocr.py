import cv2
import numpy as np
import albumentations as A
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# 1. Hiperparâmetros
IMG_SIZE     = 28
BATCH_SIZE   = 64
EPOCHS       = 20
PATIENCE     = 3
NUM_CLASSES  = 10

# 2. Data Augmentation
augmenter = A.Compose([
    A.Rotate(limit=15, p=0.7),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
])

def augment(image):
    return augmenter(image=image)['image']

# 3. Carregar MNIST
print("🔄 Carregando MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X, y  = mnist['data'], mnist['target'].astype(int)

# 4. Pré-processamento
X = X.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('uint8')
X = cv2.normalize(X, None, 0, 1.0, cv2.NORM_MINMAX)  # float32 [0,1]
y = to_categorical(y, NUM_CLASSES)

# 5. Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Criar e treinar CNN
model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)),
    MaxPool2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True
)

print("🚀 Iniciando treinamento da CNN...")
model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)
print("🎉 Modelo treinado! Avaliando no conjunto de teste:")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Teste - Loss: {loss:.4f}, Accuracy: {acc:.4%}")

# 7. Função de Previsão de Dígitos na Imagem
def predict_digits_on_image(path_img):
    img = cv2.imread(path_img)
    orig = img.copy()
    
    # Pre-processamento adaptativo
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v     = hsv[:, :, 2]
    thresh = cv2.adaptiveThreshold(
        v, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    # Remover ruído e fechar gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 10:
            continue

        # Pad e extrai ROI
        pad = 5
        x0 = max(x-pad, 0)
        y0 = max(y-pad, 0)
        x1 = min(x+w+pad, img.shape[1])
        y1 = min(y+h+pad, img.shape[0])
        roi = thresh[y0:y1, x0:x1]

        # Redimensiona mantendo proporção
        h0, w0 = roi.shape
        scale  = 20.0 / max(h0, w0)
        new_w, new_h = int(w0*scale), int(h0*scale)
        resized = cv2.resize(roi, (new_w, new_h))

        # Centra em 28x28
        blank = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        dx = (IMG_SIZE - new_w) // 2
        dy = (IMG_SIZE - new_h) // 2
        blank[dy:dy+new_h, dx:dx+new_w] = resized

        # Normaliza e prediz
        sample = blank.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
        pred   = np.argmax(model.predict(sample), axis=1)[0]

        # Desenha resultado
        cv2.rectangle(orig, (x0,y0), (x1,y1), (0,255,0), 2)
        cv2.putText(
            orig, str(pred), (x0, y0-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2
        )

    cv2.imshow("OCR Customizado", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 8. Executa predição na sua imagem
predict_digits_on_image('image_com_numeros.png')
