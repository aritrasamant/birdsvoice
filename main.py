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
STATIC_FOLDER = os.path.abspath('static')
ALLOWED_EXTENSIONS = {'.wav', '.mp3'}
TEMPERATURE = 0.25

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load YAMNet and fine-tuned model
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

def preprocess_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        scores, embeddings, spectrogram = yamnet_model(audio)
        embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()
        normalized_embedding = (embedding_mean - np.mean(embedding_mean)) / np.std(embedding_mean)
        return normalized_embedding
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

@app.post("/predict/")
async def prediction(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format")

    contents = await file.read()
    file_path = os.path.join(UPLOAD_FOLDER, 'live_audio.wav')
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        features = preprocess_audio(file_path)
        if features is None:
            return {"error": "No bird detected"}

        features = np.expand_dims(features, axis=0)
        logits = model.predict(features)
        probabilities = tf.nn.softmax(logits)
        predicted_class = np.argmax(probabilities, axis=1)
        best_probability = probabilities.numpy()[0][predicted_class[0]] * 100

        if best_probability < 8:
            predicted_label = "No bird detected"
        else:
            predicted_label = label_encoder.inverse_transform(predicted_class)[0]

        if predicted_label.lower() == "human":
            result = {"bird_name": "No bird detected"}
        else:
            result = {"bird_name": predicted_label}

        return result
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing the audio file")

app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")

@app.get("/get_bird_image/{bird_name}/")
async def get_bird_image(bird_name: str, request: Request):
    bird_name = bird_name.lower().strip()

    for ext in ['.jpg', '.jpeg', '.png']:
        filename = f"{bird_name}{ext}"
        file_path = os.path.join(STATIC_FOLDER, filename)
        if os.path.exists(file_path):
            base_url = str(request.base_url).rstrip('/')
            image_url = f"{base_url}/static/{filename}"
            return JSONResponse(content={"image_url": image_url})

    return JSONResponse(content={"image_url": ""})

# ✅ Necessary for Render to detect open port (8080)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
