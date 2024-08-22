from flask import Flask, request, jsonify, render_template
import cv2
import face_recognition
import numpy as np
import base64

app = Flask(__name__)

# Load a sample picture and learn how to recognize it.
try:
    known_image = face_recognition.load_image_file("pg_pic.jpeg")
    face_encodings = face_recognition.face_encodings(known_image)
    if len(face_encodings) > 0:
        known_face_encoding = face_encodings[0]
        known_face_encodings = [known_face_encoding]
        known_face_names = ["Person 1"]  # You can change this to any label
    else:
        raise ValueError("No face detected in the image.")
except FileNotFoundError:
    raise FileNotFoundError("The image file could not be found.")
except Exception as e:
    raise Exception(f"An error occurred while processing the image: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    frame_data = data['frame']
    nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    recognitions = []
    person_count = 1
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match_results = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = f"Person {person_count}"
        if True in match_results:
            match_index = match_results.index(True)
            name = known_face_names[match_index]
        else:
            person_count += 1

        recognitions.append({
            "top": top,
            "right": right,
            "bottom": bottom,
            "left": left,
            "name": name
        })

    return jsonify({'recognitions': recognitions})

if __name__ == '__main__':
    app.run(debug=True)
