from flask import Blueprint, render_template, request, redirect, url_for, current_app
import os
import pickle
import numpy as np
from models import face_recognition

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
        
        model = face_recognition.load_facenet_model()
        if model is None:
            print("Error loading the model.")
            return "Error loading the model", 500
        
        result_path, detected_embeddings = face_recognition.process_image(filepath, model)
        
        if result_path is None or detected_embeddings is None:
            print("Error processing image.")
            return "Error processing image", 500
        
        detected_embeddings = [embedding.astype(np.float32) for embedding in detected_embeddings]
        
        with open('models/known_faces_embeddings.pkl', 'rb') as f:
            known_faces = pickle.load(f)
        
        identified_faces = []
        for embedding in detected_embeddings:
            identity, distance = face_recognition.compare_faces(known_faces, embedding)
            identified_faces.append((identity, distance))

        # Filtrer les résultats pour ne garder que les correspondances uniques les plus proches
        unique_identified_faces = {}
        for identity, distance in identified_faces:
            if identity not in unique_identified_faces or unique_identified_faces[identity] > distance:
                unique_identified_faces[identity] = distance
        
        # Convertir le dictionnaire en liste de tuples
        identified_faces = list(unique_identified_faces.items())

        return redirect(url_for('main.result', filename=os.path.basename(result_path), results=str(identified_faces)))

@main.route('/result')
def result():
    filename = request.args.get('filename')
    results = eval(request.args.get('results'))
    identified_faces = [(identity, distance) for identity, distance in results]
    return render_template('result.html', filename=filename, identified_faces=identified_faces)
