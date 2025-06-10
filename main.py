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

# Define constants
LABEL_ENCODER_PATH = 'models/fcnn_label_encoder.pkl'
MODEL_PATH = 'models/fcnn_bird_call_model.keras'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = os.path.abspath('static')  # Absolute path for reliability
ALLOWED_EXTENSIONS = {'.wav', '.mp3'}
TEMPERATURE = 0.25  # Adjust this value for best performance

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load YAMNet model and fine-tuned model
try:
    yamnet_model = hub.load("https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as file:
        label_encoder = pickle.load(file)
except Exception as e:
    logging.error(f"Error loading model or label encoder: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

# Function to preprocess audio and extract embeddings
def preprocess_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)  # YAMNet expects 16 kHz audio
        scores, embeddings, spectrogram = yamnet_model(audio)
        embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()
        # Normalize embeddings
        normalized_embedding = (embedding_mean - np.mean(embedding_mean)) / np.std(embedding_mean)
        return normalized_embedding
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

# MAIN EXISTING PREDICT ENDPOINT
@app.post("/predict/")
async def prediction(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    # Log file details
    contents = await file.read()
    logging.info(f"Received file size: {len(contents)} bytes")
    logging.info(f"File content type: {file.content_type}")

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, 'live_audio.wav')
    with open(file_path, "wb") as f:
        f.write(contents)

    # Log saved file details
    file_stats = os.stat(file_path)
    logging.info(f"Saved file size: {file_stats.st_size} bytes")

    try:
        # Preprocess audio
        features = preprocess_audio(file_path)

        if features is None:
            logging.error("No features extracted.")
            return {"error": "No bird detected"}

        # Reshape features for model input
        features = np.expand_dims(features, axis=0)

        # Predict with fine-tuned model
        logits = model.predict(features)
        probabilities = tf.nn.softmax(logits)

        predicted_class = np.argmax(probabilities, axis=1)
        best_probability = probabilities.numpy()[0][predicted_class[0]] * 100

        logging.info(f"Predicted class: {predicted_class}, Probability: {best_probability:.2f}%")
        
        threshold = 8  # Confidence threshold
        if best_probability < threshold:
            predicted_label = "No bird detected"
        else:
            predicted_label = label_encoder.inverse_transform(predicted_class)[0]

        if predicted_label == "Human":
            result = {"bird_name": "No bird detected"}
        else:
            result = {"bird_name": predicted_label}

        logging.info(f"Detected bird: {predicted_label}")
        return result
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing the audio file")

# Serve static folder at /static/
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")

# Bird image API
@app.get("/get_bird_image/{bird_name}/")
async def get_bird_image(bird_name: str):
    # Normalize bird name
    bird_name = bird_name.lower().strip()
    logging.info(f"Looking for image of bird: '{bird_name}'")

    # Check possible file extensions
    for ext in ['.jpg', '.jpeg', '.png']:
        filename = f"{bird_name}{ext}"
        file_path = os.path.join(STATIC_FOLDER, filename)
        logging.info(f"Checking file: {file_path}")
        if os.path.exists(file_path):
            # Use relative URL assuming your server is running at this IP
            base_url = str(request.base_url)
            image_url = f"/{base_url}static/{filename}"
            return JSONResponse(content={"image_url": image_url})

    # If no matching image found
    logging.info("No image found.")
    return JSONResponse(content={"image_url": ""})
