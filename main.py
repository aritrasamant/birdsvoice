import os
import numpy as np
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Paths and constants
LABEL_ENCODER_PATH = 'models/fcnn_label_encoder2.pkl'
MODEL_PATH = 'models/fcnn_bird_call_model2.keras'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = os.path.abspath('static')
ALLOWED_EXTENSIONS = {'.wav', '.mp3'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load models
try:
    yamnet_model = hub.load("https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    logging.error(f"Model loading error: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

# Audio Preprocessing with silence trimming and normalization
def preprocess_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        audio, _ = librosa.effects.trim(audio, top_db=30)  # Remove silence
        if len(audio) == 0:
            logging.warning("Trimmed audio is empty.")
            return None

        audio = librosa.util.normalize(audio)  # Normalize amplitude
        scores, embeddings, _ = yamnet_model(audio)
        embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()
        normalized_embedding = (embedding_mean - np.mean(embedding_mean)) / np.std(embedding_mean)
        return normalized_embedding
    except Exception as e:
        logging.error(f"Audio preprocessing failed: {e}")
        return None

@app.post("/predict/")
async def prediction(file: UploadFile = File(...)):
    logging.info(f"Received file: {file.filename}")
    logging.info(f"Content type: {file.content_type}")

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    file_path = os.path.join(UPLOAD_FOLDER, 'live_audio.wav')
    content = await file.read()

    logging.info(f"Received file size: {len(content)} bytes")
    logging.info(f"First 20 bytes: {content[:20]}")

    with open(file_path, "wb") as f:
        f.write(content)

    features = preprocess_audio(file_path)
    if features is None:
        logging.warning("Preprocessing returned None (probably due to silence or error)")
        return {"bird_name": "No bird detected"}

    features = np.expand_dims(features, axis=0)
    try:
        logits = model.predict(features)
        probabilities = tf.nn.softmax(logits).numpy()[0]

        best_index = np.argmax(probabilities)
        best_confidence = probabilities[best_index] * 100

        threshold = 8  # Adjustable confidence threshold
        predicted_label = (
            label_encoder.inverse_transform([best_index])[0]
            if best_confidence >= threshold else "No bird detected"
        )

        # Log top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_labels = label_encoder.inverse_transform(top_indices)
        logging.info("Top predictions:")
        for i in range(3):
            label = top_labels[i]
            score = probabilities[top_indices[i]] * 100
            logging.info(f"{label}: {score:.2f}%")

        result = {"bird_name": predicted_label if predicted_label != "Human" else "No bird detected"}
        return result
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

# Static image API
@app.get("/get_bird_image/{bird_name}/")
async def get_bird_image(bird_name: str, request: Request):
    bird_name = bird_name.lower().strip()
    for ext in ['.jpg', '.jpeg', '.png']:
        file_path = os.path.join(STATIC_FOLDER, f"{bird_name}{ext}")
        if os.path.exists(file_path):
            base_url = str(request.base_url).rstrip("/")
            return JSONResponse(content={
                "image_url": f"{base_url}/static/{os.path.basename(file_path)}"
            })
    return JSONResponse(content={"image_url": ""})

# Serve static images
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
