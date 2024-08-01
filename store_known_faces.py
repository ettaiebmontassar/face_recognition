import cv2
import numpy as np
from keras.models import load_model
import os
import pickle

def preprocess_image(image_path):
    print(f"Loading image from {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        print("No faces detected")
        return None
    
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    return np.expand_dims(face, axis=0)

def create_embeddings():
    model = load_model('models/facenet_keras.h5')
    known_faces = {}
    for filename in os.listdir('known_faces'):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join('known_faces', filename)
            face = preprocess_image(image_path)
            if face is None:
                print(f"Skipping {filename} due to preprocessing issues")
                continue
            embeddings = model.predict(face)
            if len(embeddings) == 0:
                print(f"No embeddings generated for {filename}")
                continue
            name = os.path.splitext(filename)[0]
            known_faces[name] = embeddings[0]
            print(f"Processed {filename}")

    with open('models/known_faces_embeddings.pkl', 'wb') as file:
        pickle.dump(known_faces, file)

if __name__ == "__main__":
    create_embeddings()
