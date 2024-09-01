 

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import numpy as np
import cv2
import tensorflow as tf
import io
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# UNET Configuration
image_size = (256, 256)
smooth = 1e-15

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Register custom objects
tf.keras.utils.get_custom_objects().update({"dice_loss": dice_loss, "dice_coef": dice_coef})

print("Custom objects registered successfully.")
# Load the model
model_path = os.path.abspath("C:/Users/DELL/Downloads/project1/project1/ProjectModels/model/Brain segmentation.keras")
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    # Preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, image_size)  # Resize image to (256, 256)
    x = image / 255.0  # Normalize image
    x = np.expand_dims(x, axis=-1)  # Add channel dimension
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    return x, image

@app.route('/predictdpseg', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']

        # Check if the image is valid
        if file.filename == '':
            return jsonify({"error": "No file provided"}), 400

        static_dir = os.path.join(os.getcwd(), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        filename = secure_filename(file.filename)
        filepath = os.path.join(static_dir, filename)
        file.save(filepath)

        # Preprocess the image
        x, image = preprocess_image(filepath)

        # Prediction
        pred = model.predict(x, verbose=0)[0]

        # Ensure the prediction is in the correct shape and format if needed
        pred = np.squeeze(pred)  # Remove batch dimension if necessary

        # Create a figure to display the images
        plt.figure(figsize=(12, 4))

        # Display the input image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB))
        plt.title('Input Image')
        plt.axis('off')

        # Display the ground truth mask (if available)
        ground_truth_path = os.path.join("C:/Users/DELL/Desktop/Brain tumor segmentation/brain_tumor_segmentation/masks", os.path.splitext(os.path.basename(filepath))[0] + ".png")
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        if ground_truth is not None:
            ground_truth = cv2.resize(ground_truth, image_size)  # Resize ground truth to match the predicted mask size
            plt.subplot(1, 3, 2)
            plt.imshow(ground_truth, cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')
        else:
            plt.subplot(1, 3, 2)
            plt.text(0.5, 0.5, 'Ground Truth Mask Not Available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')

        # Display the predicted image
        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        pred_binary = (pred > 0.5).astype(np.uint8)
        if ground_truth is not None:
            accuracy = np.mean(pred_binary == ground_truth)
            plt.text(0.5, 0.1, f'Accuracy: {accuracy:.2f}', ha='center', va='center', transform=plt.gcf().transFigure, fontsize=12)
        else:
            plt.text(0.5, 0.1, 'Ground Truth Mask Not Available', ha='center', va='center', transform=plt.gcf().transFigure, fontsize=12)

        plt.tight_layout()

        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        return send_file(buf, mimetype='image/png')

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(tf.__version__)
    print(dir(tf))
    app.run(port=5002, debug=True)

 