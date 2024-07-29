import cv2
import numpy as np
import tensorflow as tf
import os
import pickle

def load_facenet_model():
    model_path = 'models/facenet_keras.h5'
    try:
        print(f"Attempting to load model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def resize_image(image, max_size=1024):
    """Resize image to have a maximum dimension of max_size."""
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scaling_factor = max_size / float(max(height, width))
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def load_known_faces_embeddings():
    embeddings_file = 'models/known_faces_embeddings.pkl'
    if not os.path.exists(embeddings_file):
        print(f"Error: Known faces embeddings file not found at {embeddings_file}")
        return None

    with open(embeddings_file, 'rb') as f:
        known_faces = pickle.load(f)
    return known_faces

def compare_faces(known_faces, face_embedding, threshold=0.6):
    for name, known_embedding in known_faces.items():
        distance = np.linalg.norm(known_embedding - face_embedding)
        if distance < threshold:
            return name, distance
    return "Unknown", None

def process_image(image_path, model):
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None, None

        print(f"Loading image from {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image at {image_path}")
            return None, None

        img = resize_image(img)
        print("Converting image to grayscale")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cascade_path = 'models/haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            print(f"Error: Haarcascade file not found at {cascade_path}")
            return None, None

        print(f"Loading Haarcascade model from {cascade_path}")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        print(f"Detected {len(faces)} faces")

        results = []
        for (x, y, w, h) in faces:
            try:
                print(f"Processing face at position {(x, y, w, h)}")
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=0)

                if model.input_shape == (None, 784):
                    face = face.reshape((1, -1))  # Flatten the image

                embedding = model.predict(face)
                results.append(embedding[0])  # Append only the embedding
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            except Exception as e:
                print(f"Error predicting embedding for face at {(x, y, w, h)}: {e}")
                continue

        result_path = image_path.replace('uploads', 'app/static/results')
        if not os.path.exists('app/static/results'):
            os.makedirs('app/static/results')
        cv2.imwrite(result_path, img)

        print(f"Processed image saved to {result_path}")
        return result_path, results
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None



if __name__ == "__main__":
    test_image_path = 'uploads/test_image.jpg'
    model = load_facenet_model()
    result_path, results = process_image(test_image_path, model)
    if result_path and results:
        print(f"Processed image saved to {result_path}")
        print(f"Results: {results}")
    else:
        print("Failed to process the image.")
