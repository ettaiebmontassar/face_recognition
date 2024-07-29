from flask import Blueprint, render_template, request, redirect, url_for, current_app
import os
import pickle
import numpy as np
from models import face_recognition  # Correct import from models directory

main = Blueprint('main', __name__)

@main.route('/')
def upload():
    return render_template('upload.html')

@main.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        print("No file part in the request.")
        return redirect(url_for('main.upload'))

    file = request.files['file']
    if file.filename == '':
        print("No file selected for uploading.")
        return redirect(url_for('main.upload'))

    if file:
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
        print(f"Saving file to {filepath}")
        file.save(filepath)
        
        # Load the FaceNet model
        model = face_recognition.load_facenet_model()
        if model is None:
            print("Error loading the model.")
            return "Error loading the model", 500
        
        # Process the uploaded image
        result_path, detected_embeddings = face_recognition.process_image(filepath, model)
        
        if result_path is None or detected_embeddings is None:
            print("Error processing image.")
            return "Error processing image", 500
        
        # Ensure detected embeddings are float32
        detected_embeddings = [embedding.astype(np.float32) for embedding in detected_embeddings]
        
        # Load known face embeddings
        with open('models/known_faces_embeddings.pkl', 'rb') as f:
            known_faces = pickle.load(f)
        
        identified_faces = []
        for embedding in detected_embeddings:
            min_distance = float('inf')
            identity = "Unknown"
            for known_name, known_embedding in known_faces.items():
                # Ensure known embeddings are float32
                known_embedding = np.array(known_embedding, dtype=np.float32)
                distance = np.linalg.norm(embedding - known_embedding)
                if distance < min_distance:
                    min_distance = distance
                    identity = known_name
            identified_faces.append((identity, min_distance))
        
        # Save or use embeddings as needed
        return redirect(url_for('main.result', filename=os.path.basename(result_path), results=str(identified_faces)))

@main.route('/result')
def result():
    filename = request.args.get('filename')
    results = eval(request.args.get('results'))
    return render_template('result.html', filename=filename, identified_faces=results)
