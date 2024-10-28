 import os
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load and preprocess data
def load_data(data_dir):
    images, labels = [], []
    for label in os.listdir(data_dir):
        for image_file in os.listdir(os.path.join(data_dir, label)):
            img_path = os.path.join(data_dir, label, image_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            images.append(img)
            labels.append(label)
    images = np.array(images).reshape(-1, 48, 48, 1)
    labels = pd.get_dummies(pd.Series(labels)).values  # One-hot encoding
    return images / 255.0, labels  # Normalize images


data_dir = os.path.normpath('C:/Users/shris/OneDrive/Desktop/project 1/fer_dataset/data/test')

X, y = load_data(data_dir)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build a simple CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2)
model.save('emotion_detector_model.h5')

from sklearn.metrics import classification_report

# Evaluate model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Predict and print classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred_classes))
import cv2

# Load the trained model
model = keras.models.load_model('emotion_detector_model.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0
    emotion = emotion_labels[np.argmax(model.predict(face))]
    
    # Display the resulting frame with emotion label
    cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

