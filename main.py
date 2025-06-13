import os
import io
import asyncio
import numpy as np
import pickle
import librosa
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import tensorflow_hub as hub

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {'.wav', '.mp3'}
STATIC_FOLDER = 'static'
MODEL_PATH = 'models/fcnn_bird_call_model2.keras'
LABEL_ENCODER_PATH = 'models/fcnn_label_encoder2.pkl'
CONFIDENCE_THRESHOLD = 8  # Adjust based on your model's performance

# Ensure static folder exists
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Global variables for loaded models
yamnet_model = None
model = None
label_encoder = None

@app.on_event("startup")
async def load_models():
    """Load ML models during startup"""
    global yamnet_model, model, label_encoder
    
    try:
        # Load YAMNet model from TensorFlow Hub
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        
        # Load custom bird call classification model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load label encoder
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
            
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise RuntimeError("Model loading failed")

def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def preprocess_audio(audio_data: bytes):
    """
    Process audio data directly from memory
    Returns: Normalized audio embeddings or None if processing fails
    """
    try:
        # Load audio from bytes
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
        
        # Trim silence and normalize
        audio, _ = librosa.effects.trim(audio, top_db=30)
        if len(audio) == 0:
            logger.warning("Trimmed audio is empty")
            return None
            
        audio = librosa.util.normalize(audio)
        
        # Extract embeddings using YAMNet
        scores, embeddings, _ = yamnet_model(audio)
        embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()
        return (embedding_mean - np.mean(embedding_mean)) / np.std(embedding_mean)
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return None

@app.post("/predict/")
async def predict_bird(file: UploadFile = File(...)):
    """
    Handle bird sound prediction requests
    Processes audio entirely in memory without saving to disk
    """
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload WAV or MP3.")

    try:
        # Read file contents into memory
        contents = await file.read()
        
        # Process audio
        features = preprocess_audio(contents)
        if features is None:
            return {"bird_name": "No bird detected"}
            
        # Make prediction
        features = np.expand_dims(features, axis=0)
        logits = model.predict(features)
        probabilities = tf.nn.softmax(logits).numpy()[0]
        best_index = np.argmax(probabilities)
        best_confidence = probabilities[best_index] * 100

        # Apply confidence threshold
        predicted_label = (
            label_encoder.inverse_transform([best_index])[0]
            if best_confidence >= CONFIDENCE_THRESHOLD
            else "No bird detected"
        )

        # Log prediction results
        logger.info(f"Prediction: {predicted_label} (Confidence: {best_confidence:.2f}%)")
        
        # Filter out human sounds
        final_label = "No bird detected" if predicted_label == "Human" else predicted_label
        return {"bird_name": final_label}
        
    except asyncio.TimeoutError:
        logger.error("Prediction timeout")
        return {"bird_name": "No bird detected"}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/get_bird_image/{bird_name}/")
async def get_bird_image(bird_name: str, request: Request):
    """
    Serve bird images from static files
    """
    bird_name = bird_name.lower().strip()
    for ext in ['.jpg', '.jpeg', '.png']:
        file_path = os.path.join(STATIC_FOLDER, f"{bird_name}{ext}")
        if os.path.exists(file_path):
            base_url = str(request.base_url).rstrip("/")
            return JSONResponse(content={
                "image_url": f"{base_url}/static/{os.path.basename(file_path)}"
            })
    return JSONResponse(content={"image_url": ""})

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "models_loaded": all(
        [yamnet_model is not None, model is not None, label_encoder is not None]
    )}

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
