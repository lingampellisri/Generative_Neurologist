 
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
from patchify import patchify
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

dcf = {
    "image_size": 224,  # Changed size to 224 to match common models like VGG16
    "num_channels": 3   # Ensure to use grayscale images
}

def preprocess_imagedp(image_path):
    # Read the image
    imagedp = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    if imagedp is None:
        raise ValueError(f"Image at path {image_path} could not be read.")
    
    # Resize the image to the desired input size
    imagedp = cv2.resize(imagedp, (dcf["image_size"], dcf["image_size"]))
    # Convert to float32 and normalize pixel values
    imagedp = imagedp.astype(np.float32) / 255.0
    # Reshape to match model input shape
    imagedp = np.expand_dims(imagedp, axis=-1)  # Add channel dimension
    imagedp = np.expand_dims(imagedp, axis=0)   # Add batch dimension
    return imagedp

def build_and_load_modeldp():
    # Path to the saved model
    model_path = r"C:\Users\DELL\Downloads\project1\project1\ProjectModels\Brain tumor clasification.keras"

    # Load the model
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return model

modeldp = build_and_load_modeldp()

@app.route('/')
def home():
    return "Brain Tumor Classification API"

@app.route('/predict-dp', methods=['POST'])
def predict_dp():
    if 'file' not in request.files:
        return jsonify(error="No file provided"), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No file provided"), 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join('static', filename)
    file.save(filepath)

    try:
        # Define imagedp within the try block
        imagedp = preprocess_imagedp(filepath)
        
        # Make predictions
        predictions = modeldp.predict(imagedp)
        print("Predictions:", predictions)

        # Convert the prediction to "yes" or "no"
        prediction_label = "YES" if predictions[0][0] > 0.5 else "NO"

        # Return the prediction result as JSON
        return jsonify(prediction=prediction_label)
    except Exception as e:
        return jsonify(error=f"Error during prediction: {str(e)}"), 500

if __name__ == "__main__":
    app.run(port=5003, debug=True)
