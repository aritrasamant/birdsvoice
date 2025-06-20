from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware  # ✅ Add this
import os
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

# ✅ Add this CORS middleware for mobile app compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Add a simple root health check endpoint
@app.get("/")
def health_check():
    return {"status": "ok"}

# --- rest of your existing code continues unchanged ---

# Configure logging
logging.basicConfig(level=logging.INFO)

# Paths and constants
LABEL_ENCODER_PATH = 'models/fcnn_label_encoder2.pkl'
MODEL_PATH = 'models/fcnn_bird_call_model2.keras'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = os.path.abspath('static')
YAMNET_MODEL_PATH = os.path.abspath('yamnet_model')
ALLOWED_EXTENSIONS = {'.wav', '.mp3'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load models
try:
    yamnet_model = hub.load(YAMNET_MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    logging.error(f"Model loading error: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

def preprocess_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        if len(audio) == 0:
            logging.warning("Trimmed audio is empty.")
            return None

        audio = librosa.util.normalize(audio)
        scores, embeddings, _ = yamnet_model(audio)
        embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()
        normalized_embedding = (embedding_mean - np.mean(embedding_mean)) / np.std(embedding_mean)
        return normalized_embedding
    except Exception as e:
        logging.error(f"Audio preprocessing failed: {e}")
        return None

@app.post("/predict/")
async def prediction(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    file_path = os.path.join(UPLOAD_FOLDER, 'live_audio.wav')
    with open(file_path, "wb") as f:
        f.write(await file.read())

    features = preprocess_audio(file_path)
    if features is None:
        return {"bird_name": "No bird detected"}

    features = np.expand_dims(features, axis=0)
    try:
        logits = model.predict(features)
        probabilities = tf.nn.softmax(logits).numpy()[0]

        best_index = np.argmax(probabilities)
        best_confidence = probabilities[best_index] * 100

        threshold = 8
        predicted_label = (
            label_encoder.inverse_transform([best_index])[0]
            if best_confidence >= threshold else "No bird detected"
        )

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_labels = label_encoder.inverse_transform(top_indices)
        logging.info(f"Top predictions: {[(label, round(probabilities[i]*100, 2)) for i, label in zip(top_indices, top_labels)]}")

        result = {"bird_name": predicted_label if predicted_label != "Human" else "No bird detected"}
        return result
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

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

app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
