from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model
model = load_model('models/emotion_model.h5')
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def home():
    return '''
    <html>
    <body style="font-family: Arial; text-align: center; padding: 50px;">
        <h1>🎭 Facial Emotion Detection</h1>
        <p>Upload an image to detect emotions</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Detect Emotion</button>
        </form>
    </body>
    </html>
    '''

@app.route('/', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No image uploaded"
    
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if len(faces) == 0:
        return "No faces detected"
    
    results = []
    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        
        predictions = model.predict(face_roi, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(predictions)]
        confidence = float(predictions[np.argmax(predictions)])
        results.append(f"<p>Emotion: <b>{emotion}</b> ({confidence:.2%})</p>")
    
    return "".join(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)