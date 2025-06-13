from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import librosa
import logging
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import numpy as np

app = FastAPI()

# Setup
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = os.path.abspath('static')
MODEL_PATH = 'models/fcnn_bird_call_model2.keras'
LABEL_ENCODER_PATH = 'models/fcnn_label_encoder2.pkl'
ALLOWED_EXTENSIONS = {'.wav', '.mp3'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load models
try:
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

def preprocess_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        if len(audio) == 0:
            return None
        audio = librosa.util.normalize(audio)
        scores, embeddings, _ = yamnet_model(audio)
        embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()
        normalized = (embedding_mean - np.mean(embedding_mean)) / np.std(embedding_mean)
        return normalized
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        return None

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    file_path = os.path.join(UPLOAD_FOLDER, "live_audio.wav")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    features = preprocess_audio(file_path)
    if features is None:
        return {"bird_name": "No bird detected"}

    features = np.expand_dims(features, axis=0)
    try:
        logits = model.predict(features)
        probabilities = tf.nn.softmax(logits).numpy()[0]
        best_idx = np.argmax(probabilities)
        confidence = probabilities[best_idx] * 100
        threshold = 8

        predicted = label_encoder.inverse_transform([best_idx])[0] if confidence >= threshold else "No bird detected"
        if predicted == "Human":
            predicted = "No bird detected"

        return {"bird_name": predicted}
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

@app.get("/get_bird_image/{bird_name}/")
async def get_bird_image(bird_name: str, request: Request):
    bird_name = bird_name.lower().strip()
    for ext in ['.jpg', '.jpeg', '.png']:
        file_path = os.path.join(STATIC_FOLDER, f"{bird_name}{ext}")
        if os.path.exists(file_path):
            base_url = str(request.base_url).rstrip("/")
            return JSONResponse(content={"image_url": f"{base_url}/static/{os.path.basename(file_path)}"})
    return JSONResponse(content={"image_url": ""})

@app.get("/")
async def root():
    return {"status": "ok"}

# Serve static bird images
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
