from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize ViT model without loading default weights
pretrained_vit = models.vit_b_16(weights=None).to(device)

# Change the classifier head
class_names = ['no', 'yes']
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

# Load the saved model weights
saved_model_path ="C:/Users/DELL/Downloads/project1/project1/ProjectModels/model/trained_vit_model.pth"
pretrained_vit.load_state_dict(torch.load(saved_model_path, map_location=device))

# Define transformation pipeline
pretrained_vit_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to classify image
def classify_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_image = pretrained_vit_transforms(image).unsqueeze(0).to(device)

    # Set the model to evaluation mode and make prediction
    pretrained_vit.eval()
    with torch.no_grad():
        outputs = pretrained_vit(input_image)

    # Convert the output logits to probabilities
    probabilities = torch.softmax(outputs, dim=1)

    # Get the predicted class and its probability
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    predicted_prob = probabilities[0, predicted.item()].item()

    return predicted_class, predicted_prob

@app.route('/predictllm', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image_bytes = file.read()
        predicted_class, predicted_prob = classify_image(image_bytes)

        return jsonify({'prediction': predicted_class, 'probability': predicted_prob})
    except Exception as e:
        print(f'Error processing image: {str(e)}')
        return jsonify({'error': 'Error processing image'})

if __name__ == '__main__':
    app.run(port=5004, debug=True)