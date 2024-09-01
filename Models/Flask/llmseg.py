import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
from patchify import patchify
import matplotlib.pyplot as plt
import keras as k

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

cf = {
    "image_size": 256,
    "num_channels": 3,
    "num_layers": 12,
    "hidden_dim": 128,
    "mlp_dim": 32,
    "num_heads": 6,
    "dropout_rate": 0.1,
    "patch_size": 16,
    "num_patches": (256**2) // (16**2),
    "flat_patches_shape": ((256**2) // (16**2), 16 * 16 * 3)
}

smooth = 1e-15

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def load_model():
    model_path = os.path.abspath("C:/Users/DELL/Downloads/project1/project1/ProjectModels/model/llmseg.keras")
    model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})
    return model

model = load_model()

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(f"Original Image Shape: {image.shape}")
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    print(f"Resized Image Shape: {image.shape}")
    x = image / 255.0
    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(x, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"])
    patches = patches.astype(np.float32)
    patches = np.expand_dims(patches, axis=0)
    return patches, image

@app.route('/')
def home():
    return "Brain Tumor Segmentation API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error="No file provided"), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No file provided"), 400

    static_dir = os.path.join(os.getcwd(), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    filename = secure_filename(file.filename)
    filepath = os.path.join(static_dir, filename)
    file.save(filepath)

    print(f"Input image path: {filepath}")

    try:
        patches, original_image = preprocess_image(filepath)
        pred = model.predict(patches, verbose=0)[0]
        pred = np.concatenate([pred, pred, pred], axis=-1)

        mask = pred > 0.5
        mask = mask.astype(np.uint8) * 255

        # Construct the ground truth path based on the input image path
        ground_truth_path = os.path.join("C:/Users/DELL/Desktop/Brain tumor segmentation/brain_tumor_segmentation/masks", os.path.splitext(os.path.basename(filepath))[0] + ".png")

        print(f"Ground truth path: {ground_truth_path}")

        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

        if ground_truth is None:
            print(f"Error: Ground truth mask not found at {ground_truth_path}")
            return jsonify(error=f"Ground truth mask not found at {ground_truth_path}"), 500
        else:
            print(f"Ground truth mask loaded successfully from {ground_truth_path}")

        ground_truth = cv2.resize(ground_truth, (cf["image_size"], cf["image_size"]))
        pred_binary = (pred > 0.5).astype(np.uint8)[:, :, 0]

        accuracy = np.mean(pred_binary == ground_truth)
        print("Accuracy:", accuracy)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth, cmap='gray')
        plt.title(f'Ground Truth Mask\nAccuracy: {accuracy:.2f}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('Predicted Tumor Regions')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(static_dir, 'prediction_result.png'))

        return send_file(os.path.join(static_dir, 'prediction_result.png'), mimetype='image/png')

    except Exception as e:
        return jsonify(error=f"Error during prediction: {str(e)}"), 500

if __name__ == '__main__':
   app.run(port=5000, debug=True)




# import os
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import numpy as np
# import cv2
# import tensorflow as tf
# from werkzeug.utils import secure_filename
# from patchify import patchify
# import matplotlib.pyplot as plt

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# cf = {
#     "image_size": 256,
#     "num_channels": 3,
#     "num_layers": 12,
#     "hidden_dim": 128,
#     "mlp_dim": 32,
#     "num_heads": 6,
#     "dropout_rate": 0.1,
#     "patch_size": 16,
#     "num_patches": (256**2) // (16**2),
#     "flat_patches_shape": ((256**2) // (16**2), 16 * 16 * 3)
# }

# smooth = 1e-15

# def dice_coef(y_true, y_pred):
#     y_true = tf.keras.layers.Flatten()(y_true)
#     y_pred = tf.keras.layers.Flatten()(y_pred)
#     intersection = tf.keras.backend.sum(y_true * y_pred)
#     return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)

# def load_model():
#     model_path = os.path.abspath("C:/Users/DELL/Downloads/project1/project1/ProjectModels/model/llmseg.keras")
#     model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})
#     return model

# model = load_model()

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     print(f"Original Image Shape: {image.shape}")
#     image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
#     print(f"Resized Image Shape: {image.shape}")
#     x = image / 255.0
#     patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
#     patches = patchify(x, patch_shape, cf["patch_size"])
#     patches = np.reshape(patches, cf["flat_patches_shape"])
#     patches = patches.astype(np.float32)
#     patches = np.expand_dims(patches, axis=0)
#     return patches, image

# @app.route('/')
# def home():
#     return "Brain Tumor Segmentation API"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify(error="No file provided"), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify(error="No file provided"), 400

#     static_dir = os.path.join(os.getcwd(), 'static')
#     if not os.path.exists(static_dir):
#         os.makedirs(static_dir)

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(static_dir, filename)
#     file.save(filepath)

#     print(f"Input image path: {filepath}")

#     try:
#         patches, original_image = preprocess_image(filepath)
#         pred = model.predict(patches, verbose=0)[0]
#         pred = np.concatenate([pred, pred, pred], axis=-1)

#         mask = pred > 0.5
#         mask = mask.astype(np.uint8) * 255

#         # Construct the ground truth path based on the input image path
#         ground_truth_path = os.path.join("C:/Users/DELL/Desktop/Brain tumor segmentation/brain_tumor_segmentation/masks", os.path.splitext(os.path.basename(filepath))[0] + ".png")

#         print(f"Ground truth path: {ground_truth_path}")

#         ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

#         if ground_truth is None:
#             print(f"Error: Ground truth mask not found at {ground_truth_path}")
#             return jsonify(error=f"Ground truth mask not found at {ground_truth_path}"), 500
#         else:
#             print(f"Ground truth mask loaded successfully from {ground_truth_path}")

#         ground_truth = cv2.resize(ground_truth, (cf["image_size"], cf["image_size"]))
#         pred_binary = (pred > 0.5).astype(np.uint8)[:, :, 0]

#         accuracy = np.mean(pred_binary == ground_truth)
#         print("Accuracy:", accuracy)

#         plt.figure(figsize=(12, 4))

#         plt.subplot(1, 3, 1)
#         plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#         plt.title('Input Image')
#         plt.axis('off')

#         plt.subplot(1, 3, 2)
#         plt.imshow(ground_truth, cmap='gray')
#         plt.title(f'Ground Truth Mask\nAccuracy: {accuracy:.2f}')
#         plt.axis('off')

#         plt.subplot(1, 3, 3)
#         plt.imshow(mask, cmap='gray')
#         plt.title('Predicted Tumor Regions')
#         plt.axis('off')

#         plt.tight_layout()
#         plt.savefig(os.path.join(static_dir, 'prediction_result.png'))

#         return send_file(os.path.join(static_dir, 'prediction_result.png'), mimetype='image/png')

#     except Exception as e:
#         return jsonify(error=f"Error during prediction: {str(e)}"), 500

# if __name__ == '__main__':
#    app.run(port=5000, debug=True)




# import os
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import numpy as np
# import cv2
# import tensorflow as tf
# from werkzeug.utils import secure_filename
# from patchify import patchify
# import matplotlib.pyplot as plt

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# cf = {
#     "image_size": 256,
#     "num_channels": 3,
#     "num_layers": 12,
#     "hidden_dim": 128,
#     "mlp_dim": 32,
#     "num_heads": 6,
#     "dropout_rate": 0.1,
#     "patch_size": 16,
#     "num_patches": (256**2) // (16**2),
#     "flat_patches_shape": ((256**2) // (16**2), 16 * 16 * 3)
# }

# smooth = 1e-15

# def dice_coef(y_true, y_pred):
#     y_true = tf.keras.layers.Flatten()(y_true)
#     y_pred = tf.keras.layers.Flatten()(y_pred)
#     intersection = tf.keras.backend.sum(y_true * y_pred)
#     return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)

# def load_model():
#     model_path = os.path.abspath("C:/Users/DELL/Downloads/project1/project1/ProjectModels/model/llmseg.keras")
#     model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})
#     return model

# model = load_model()

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     print(f"Original Image Shape: {image.shape}")
#     image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
#     print(f"Resized Image Shape: {image.shape}")
#     x = image / 255.0
#     patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
#     patches = patchify(x, patch_shape, cf["patch_size"])
#     patches = np.reshape(patches, cf["flat_patches_shape"])
#     patches = patches.astype(np.float32)
#     patches = np.expand_dims(patches, axis=0)
#     return patches, image

# @app.route('/')
# def home():
#     return "Brain Tumor Segmentation API"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify(error="No file provided"), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify(error="No file provided"), 400

#     static_dir = os.path.join(os.getcwd(), 'static')
#     if not os.path.exists(static_dir):
#         os.makedirs(static_dir)

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(static_dir, filename)
#     file.save(filepath)

#     print(f"Input image path: {filepath}")

#     try:
#         patches, original_image = preprocess_image(filepath)
#         pred = model.predict(patches, verbose=0)[0]
#         pred = np.concatenate([pred, pred, pred], axis=-1)

#         mask = pred > 0.5
#         mask = mask.astype(np.uint8) * 255

#         # Construct the ground truth path based on the input image path
#         ground_truth_path = os.path.join("C:/Users/DELL/Desktop/Brain tumor segmentation/brain_tumor_segmentation/masks", os.path.splitext(os.path.basename(filepath))[0] + ".png")

#         print(f"Ground truth path: {ground_truth_path}")

#         ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

#         if ground_truth is None:
#             print(f"Error: Ground truth mask not found at {ground_truth_path}")
#             return jsonify(error=f"Ground truth mask not found at {ground_truth_path}"), 500
#         else:
#             print(f"Ground truth mask loaded successfully from {ground_truth_path}")

#         ground_truth = cv2.resize(ground_truth, (cf["image_size"], cf["image_size"]))
#         pred_binary = (pred > 0.5).astype(np.uint8)[:, :, 0]

#         accuracy = np.mean(pred_binary == ground_truth)
#         print("Accuracy:", accuracy)

#         plt.figure(figsize=(12, 4))

#         plt.subplot(1, 3, 1)
#         plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#         plt.title('Input Image')
#         plt.axis('off')

#         plt.subplot(1, 3, 2)
#         plt.imshow(ground_truth, cmap='gray')
#         plt.title(f'Ground Truth Mask\nAccuracy: {accuracy:.2f}')
#         plt.axis('off')

#         plt.subplot(1, 3, 3)
#         plt.imshow(mask, cmap='gray')
#         plt.title('Predicted Tumor Regions')
#         plt.axis('off')

#         plt.tight_layout()
#         plt.savefig(os.path.join(static_dir, 'prediction_result.png'))

#         return send_file(os.path.join(static_dir, 'prediction_result.png'), mimetype='image/png')

#     except Exception as e:
#         return jsonify(error=f"Error during prediction: {str(e)}"), 500

# if __name__ == '__main__':
#    app.run(port=5000, debug=True)
# import tensorflow.keras as keras
# from tensorflow.keras.models import load_model


# import os
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Flatten
# from werkzeug.utils import secure_filename
# from patchify import patchify
# import matplotlib.pyplot as plt

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# cf = {
#     "image_size": 256,
#     "num_channels": 3,
#     "num_layers": 12,
#     "hidden_dim": 128,
#     "mlp_dim": 32,
#     "num_heads": 6,
#     "dropout_rate": 0.1,
#     "patch_size": 16,
#     "num_patches": (256**2) // (16**2),
#     "flat_patches_shape": ((256**2) // (16**2), 16 * 16 * 3)
# }

# smooth = 1e-15

# def dice_coef(y_true, y_pred):
#     y_true = Flatten()(y_true)
#     y_pred = Flatten()(y_pred)
#     intersection = tf.keras.backend.sum(y_true * y_pred)
#     return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)

# def load_model():
#     model_path = os.path.abspath("C:/Users/DELL/Downloads/project1/project1/ProjectModels/model/llmseg.keras")
#     with tf.keras.utils.CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
#         model = load_model(model_path)
#     return model

# model = load_model()

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     print(f"Original Image Shape: {image.shape}")
#     image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
#     print(f"Resized Image Shape: {image.shape}")
#     x = image / 255.0
#     patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
#     patches = patchify(x, patch_shape, cf["patch_size"])
#     patches = np.reshape(patches, cf["flat_patches_shape"])
#     patches = patches.astype(np.float32)
#     patches = np.expand_dims(patches, axis=0)
#     return patches, image

# @app.route('/')
# def home():
#     return "Brain Tumor Segmentation API"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify(error="No file provided"), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify(error="No file provided"), 400

#     static_dir = os.path.join(os.getcwd(), 'static')
#     if not os.path.exists(static_dir):
#         os.makedirs(static_dir)

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(static_dir, filename)
#     file.save(filepath)

#     print(f"Input image path: {filepath}")

#     try:
#         patches, original_image = preprocess_image(filepath)
#         pred = model.predict(patches, verbose=0)[0]
#         pred = np.concatenate([pred, pred, pred], axis=-1)

#         mask = pred > 0.5
#         mask = mask.astype(np.uint8) * 255

#         # Construct the ground truth path based on the input image path
#         ground_truth_path = os.path.join("C:/Users/DELL/Desktop/Brain tumor segmentation/brain_tumor_segmentation/masks", os.path.splitext(os.path.basename(filepath))[0] + ".png")

#         print(f"Ground truth path: {ground_truth_path}")

#         ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

#         if ground_truth is None:
#             print(f"Error: Ground truth mask not found at {ground_truth_path}")
#             return jsonify(error=f"Ground truth mask not found at {ground_truth_path}"), 500
#         else:
#             print(f"Ground truth mask loaded successfully from {ground_truth_path}")

#         ground_truth = cv2.resize(ground_truth, (cf["image_size"], cf["image_size"]))
#         pred_binary = (pred > 0.5).astype(np.uint8)[:, :, 0]

#         accuracy = np.mean(pred_binary == ground_truth)
#         print("Accuracy:", accuracy)

#         plt.figure(figsize=(12, 4))

#         plt.subplot(1, 3, 1)
#         plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#         plt.title('Input Image')
#         plt.axis('off')

#         plt.subplot(1, 3, 2)
#         plt.imshow(ground_truth, cmap='gray')
#         plt.title(f'Ground Truth Mask\nAccuracy: {accuracy:.2f}')
#         plt.axis('off')

#         plt.subplot(1, 3, 3)
#         plt.imshow(mask, cmap='gray')
#         plt.title('Predicted Tumor Regions')
#         plt.axis('off')

#         plt.tight_layout()
#         plt.savefig(os.path.join(static_dir, 'prediction_result.png'))

#         return send_file(os.path.join(static_dir, 'prediction_result.png'), mimetype='image/png')

#     except Exception as e:
#         return jsonify(error=f"Error during prediction: {str(e)}"), 500

# if __name__ == '__main__':
#    app.run(port=5000, debug=True)


# import os
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Flatten
# from werkzeug.utils import secure_filename
# from patchify import patchify
# import matplotlib.pyplot as plt

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# cf = {
#     "image_size": 256,
#     "num_channels": 3,
#     "num_layers": 12,
#     "hidden_dim": 128,
#     "mlp_dim": 32,
#     "num_heads": 6,
#     "dropout_rate": 0.1,
#     "patch_size": 16,
#     "num_patches": (256**2) // (16**2),
#     "flat_patches_shape": ((256**2) // (16**2), 16 * 16 * 3)
# }

# smooth = 1e-15

# def dice_coef(y_true, y_pred):
#     y_true = Flatten()(y_true)
#     y_pred = Flatten()(y_pred)
#     intersection = tf.keras.backend.sum(y_true * y_pred)
#     return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)

# def load_model():
#     model_path = os.path.abspath("C:/Users/DELL/Downloads/project1/project1/ProjectModels/model/llmseg.keras")
#     with tf.keras.utils.CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
#         model = load_model(model_path)
#     return model

# model = load_model()

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     print(f"Original Image Shape: {image.shape}")
#     image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
#     print(f"Resized Image Shape: {image.shape}")
#     x = image / 255.0
#     patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
#     patches = patchify(x, patch_shape, cf["patch_size"])
#     patches = np.reshape(patches, cf["flat_patches_shape"])
#     patches = patches.astype(np.float32)
#     patches = np.expand_dims(patches, axis=0)
#     return patches, image

# @app.route('/')
# def home():
#     return "Brain Tumor Segmentation API"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify(error="No file provided"), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify(error="No file provided"), 400

#     static_dir = os.path.join(os.getcwd(), 'static')
#     if not os.path.exists(static_dir):
#         os.makedirs(static_dir)

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(static_dir, filename)
#     file.save(filepath)

#     print(f"Input image path: {filepath}")

#     try:
#         patches, original_image = preprocess_image(filepath)
#         pred = model.predict(patches, verbose=0)[0]
#         pred = np.concatenate([pred, pred, pred], axis=-1)

#         mask = pred > 0.5
#         mask = mask.astype(np.uint8) * 255

#         # Construct the ground truth path based on the input image path
#         ground_truth_path = os.path.join("C:/Users/DELL/Desktop/Brain tumor segmentation/brain_tumor_segmentation/masks", os.path.splitext(os.path.basename(filepath))[0] + ".png")

#         print(f"Ground truth path: {ground_truth_path}")

#         ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

#         if ground_truth is None:
#             print(f"Error: Ground truth mask not found at {ground_truth_path}")
#             return jsonify(error=f"Ground truth mask not found at {ground_truth_path}"), 500
#         else:
#             print(f"Ground truth mask loaded successfully from {ground_truth_path}")

#         ground_truth = cv2.resize(ground_truth, (cf["image_size"], cf["image_size"]))
#         pred_binary = (pred > 0.5).astype(np.uint8)[:, :, 0]

#         accuracy = np.mean(pred_binary == ground_truth)
#         print("Accuracy:", accuracy)

#         plt.figure(figsize=(12, 4))

#         plt.subplot(1, 3, 1)
#         plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#         plt.title('Input Image')
#         plt.axis('off')

#         plt.subplot(1, 3, 2)
#         plt.imshow(ground_truth, cmap='gray')
#         plt.title(f'Ground Truth Mask\nAccuracy: {accuracy:.2f}')
#         plt.axis('off')

#         plt.subplot(1, 3, 3)
#         plt.imshow(mask, cmap='gray')
#         plt.title('Predicted Tumor Regions')
#         plt.axis('off')

#         plt.tight_layout()
#         plt.savefig(os.path.join(static_dir, 'prediction_result.png'))

#         return send_file(os.path.join(static_dir, 'prediction_result.png'), mimetype='image/png')

#     except Exception as e:
#         return jsonify(error=f"Error during prediction: {str(e)}"), 500

# if __name__ == '__main__':
#    app.run(port=5000, debug=True)


# import os
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import numpy as np
# import cv2
# import tensorflow as tf
# from keras.models import load_model
# from keras.layers import Flatten
# from werkzeug.utils import secure_filename
# from patchify import patchify
# import matplotlib.pyplot as plt

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# cf = {
#     "image_size": 256,
#     "num_channels": 3,
#     "num_layers": 12,
#     "hidden_dim": 128,
#     "mlp_dim": 32,
#     "num_heads": 6,
#     "dropout_rate": 0.1,
#     "patch_size": 16,
#     "num_patches": (256**2) // (16**2),
#     "flat_patches_shape": ((256**2) // (16**2), 16 * 16 * 3)
# }

# smooth = 1e-15

# def dice_coef(y_true, y_pred):
#     y_true = Flatten()(y_true)
#     y_pred = Flatten()(y_pred)
#     intersection = tf.keras.backend.sum(y_true * y_pred)
#     return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)

# def load_model():
#     model_path = os.path.abspath("C:/Users/DELL/Downloads/project1/project1/ProjectModels/model/llmseg.keras")
#     with tf.keras.utils.CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
#         model = load_model(model_path)
#     return model

# model = load_model()

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     print(f"Original Image Shape: {image.shape}")
#     image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
#     print(f"Resized Image Shape: {image.shape}")
#     x = image / 255.0
#     patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
#     patches = patchify(x, patch_shape, cf["patch_size"])
#     patches = np.reshape(patches, cf["flat_patches_shape"])
#     patches = patches.astype(np.float32)
#     patches = np.expand_dims(patches, axis=0)
#     return patches, image

# @app.route('/')
# def home():
#     return "Brain Tumor Segmentation API"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify(error="No file provided"), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify(error="No file provided"), 400

#     static_dir = os.path.join(os.getcwd(), 'static')
#     if not os.path.exists(static_dir):
#         os.makedirs(static_dir)

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(static_dir, filename)
#     file.save(filepath)

#     print(f"Input image path: {filepath}")

#     try:
#         patches, original_image = preprocess_image(filepath)
#         pred = model.predict(patches, verbose=0)[0]
#         pred = np.concatenate([pred, pred, pred], axis=-1)

#         mask = pred > 0.5
#         mask = mask.astype(np.uint8) * 255

#         # Construct the ground truth path based on the input image path
#         ground_truth_path = os.path.join("C:/Users/DELL/Desktop/Brain tumor segmentation/brain_tumor_segmentation/masks", os.path.splitext(os.path.basename(filepath))[0] + ".png")

#         print(f"Ground truth path: {ground_truth_path}")

#         ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

#         if ground_truth is None:
#             print(f"Error: Ground truth mask not found at {ground_truth_path}")
#             return jsonify(error=f"Ground truth mask not found at {ground_truth_path}"), 500
#         else:
#             print(f"Ground truth mask loaded successfully from {ground_truth_path}")

#         ground_truth = cv2.resize(ground_truth, (cf["image_size"], cf["image_size"]))
#         pred_binary = (pred > 0.5).astype(np.uint8)[:, :, 0]

#         accuracy = np.mean(pred_binary == ground_truth)
#         print("Accuracy:", accuracy)

#         plt.figure(figsize=(12, 4))

#         plt.subplot(1, 3, 1)
#         plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#         plt.title('Input Image')
#         plt.axis('off')

#         plt.subplot(1, 3, 2)
#         plt.imshow(ground_truth, cmap='gray')
#         plt.title(f'Ground Truth Mask\nAccuracy: {accuracy:.2f}')
#         plt.axis('off')

#         plt.subplot(1, 3, 3)
#         plt.imshow(mask, cmap='gray')
#         plt.title('Predicted Tumor Regions')
#         plt.axis('off')

#         plt.tight_layout()
#         plt.savefig(os.path.join(static_dir, 'prediction_result.png'))

#         return send_file(os.path.join(static_dir, 'prediction_result.png'), mimetype='image/png')

#     except Exception as e:
#         return jsonify(error=f"Error during prediction: {str(e)}"), 500

# if __name__ == '__main__':
#    app.run(port=5000, debug=True)


# import os
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import numpy as np
# import cv2
# import tensorflow as tf
# from keras.models import load_model as keras_load_model
# from keras.layers import Flatten
# from werkzeug.utils import secure_filename
# from patchify import patchify
# import matplotlib.pyplot as plt

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# cf = {
#     "image_size": 256,
#     "num_channels": 3,
#     "num_layers": 12,
#     "hidden_dim": 128,
#     "mlp_dim": 32,
#     "num_heads": 6,
#     "dropout_rate": 0.1,
#     "patch_size": 16,
#     "num_patches": (256*2) // (16*2),
#     "flat_patches_shape": ((256*2) // (16*2), 16 * 16 * 3)
# }
# smooth = 1e-15

# def dice_coef(y_true, y_pred):
#     y_true = Flatten()(y_true)
#     y_pred = Flatten()(y_pred)
#     intersection = tf.keras.backend.sum(y_true * y_pred)
#     return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)

# def custom_load_model():
#     model_path = os.path.abspath("C:/Users/DELL/Downloads/project1/project1/ProjectModels/model/llmseg.keras")
#     with tf.keras.utils.CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
#         model = keras_load_model(model_path)
#     return model

# model = custom_load_model()

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     print(f"Original Image Shape: {image.shape}")
#     image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
#     print(f"Resized Image Shape: {image.shape}")
#     x = image / 255.0
#     patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
#     patches = patchify(x, patch_shape, cf["patch_size"])
#     patches = np.reshape(patches, cf["flat_patches_shape"])
#     patches = patches.astype(np.float32)
#     patches = np.expand_dims(patches, axis=0)
#     return patches, image

# @app.route('/')
# def home():
#     return "Brain Tumor Segmentation API"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify(error="No file provided"), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify(error="No file provided"), 400

#     static_dir = os.path.join(os.getcwd(), 'static')
#     if not os.path.exists(static_dir):
#         os.makedirs(static_dir)

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(static_dir, filename)
#     file.save(filepath)

#     print(f"Input image path: {filepath}")

#     try:
#         patches, original_image = preprocess_image(filepath)
#         pred = model.predict(patches, verbose=0)[0]
#         pred = np.concatenate([pred, pred, pred], axis=-1)

#         mask = pred > 0.5
#         mask = mask.astype(np.uint8) * 255
        
#         # Construct the ground truth path based on the input image path
#         ground_truth_path = os.path.join("C:/Users/DELL/Desktop/Brain tumor segmentation/brain_tumor_segmentation/masks", os.path.splitext(os.path.basename(filepath))[0] + ".png")

#         print(f"Ground truth path: {ground_truth_path}")

#         ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

#         if ground_truth is None:
#             print(f"Error: Ground truth mask not found at {ground_truth_path}")
#             return jsonify(error=f"Ground truth mask not found at {ground_truth_path}"), 500
#         else:
#             print(f"Ground truth mask loaded successfully from {ground_truth_path}")

#         ground_truth = cv2.resize(ground_truth, (cf["image_size"], cf["image_size"]))
#         pred_binary = (pred > 0.5).astype(np.uint8)[:, :, 0]

#         accuracy = np.mean(pred_binary == ground_truth)
#         print("Accuracy:", accuracy)

#         plt.figure(figsize=(12, 4))

#         plt.subplot(1, 3, 1)
#         plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#         plt.title('Input Image')
#         plt.axis('off')

#         plt.subplot(1, 3, 2)
#         plt.imshow(ground_truth, cmap='gray')
#         plt.title(f'Ground Truth Mask\nAccuracy: {accuracy:.2f}')
#         plt.axis('off')

#         plt.subplot(1, 3, 3)
#         plt.imshow(mask, cmap='gray')
#         plt.title('Predicted Tumor Regions')
#         plt.axis('off')

#         plt.tight_layout()
#         plt.savefig(os.path.join(static_dir, 'prediction_result.png'))

#         return send_file(os.path.join(static_dir, 'prediction_result.png'), mimetype='image/png')

#     except Exception as e:
#         return jsonify(error=f"Error during prediction: {str(e)}"), 500

# if __name__== '__main__':
#     app.run(port=5000, debug=True)



# import os
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import numpy as np
# import cv2
# import tensorflow as tf
# from keras.models import load_model as keras_load_model
# from keras.layers import Flatten
# from werkzeug.utils import secure_filename
# from patchify import patchify
# import matplotlib.pyplot as plt

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# cf = {
#     "image_size": 256,
#     "num_channels": 3,
#     "num_layers": 12,
#     "hidden_dim": 128,
#     "mlp_dim": 32,
#     "num_heads": 6,
#     "dropout_rate": 0.1,
#     "patch_size": 16,
#     "num_patches": (256*2) // (16*2),
#     "flat_patches_shape": ((256*2) // (16*2), 16 * 16 * 3)
# }
# smooth = 1e-15

# def dice_coef(y_true, y_pred):
#     y_true = Flatten()(y_true)
#     y_pred = Flatten()(y_pred)
#     intersection = tf.keras.backend.sum(y_true * y_pred)
#     return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)

# def custom_load_model():
#     model_path = os.path.abspath("C:/Users/DELL/Downloads/project1/project1/ProjectModels/model/llmseg.keras")
#     # C:\Users\DELL\Downloads\project1\project1\ProjectModels\model\llmseg.keras
#     print(f"Loading model from: {model_path}")
#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#         raise FileNotFoundError(f"Model file not found at {model_path}")
#     with tf.keras.utils.CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
#         model = keras_load_model(model_path)
#     return model

# model = custom_load_model()

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     print(f"Original Image Shape: {image.shape}")
#     image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
#     print(f"Resized Image Shape: {image.shape}")
#     x = image / 255.0
#     patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
#     patches = patchify(x, patch_shape, cf["patch_size"])
#     patches = np.reshape(patches, cf["flat_patches_shape"])
#     patches = patches.astype(np.float32)
#     patches = np.expand_dims(patches, axis=0)
#     return patches, image

# @app.route('/')
# def home():
#     return "Brain Tumor Segmentation API"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify(error="No file provided"), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify(error="No file provided"), 400

#     static_dir = os.path.join(os.getcwd(), 'static')
#     if not os.path.exists(static_dir):
#         os.makedirs(static_dir)

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(static_dir, filename)
#     file.save(filepath)

#     print(f"Input image path: {filepath}")

#     try:
#         patches, original_image = preprocess_image(filepath)
#         pred = model.predict(patches, verbose=0)[0]
#         pred = np.concatenate([pred, pred, pred], axis=-1)

#         mask = pred > 0.5
#         mask = mask.astype(np.uint8) * 255
        
#         # Construct the ground truth path based on the input image path
#         ground_truth_path = os.path.join("C:/Users/DELL/Desktop/Brain tumor segmentation/brain_tumor_segmentation/masks", os.path.splitext(os.path.basename(filepath))[0] + ".png")

#         print(f"Ground truth path: {ground_truth_path}")

#         ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

#         if ground_truth is None:
#             print(f"Error: Ground truth mask not found at {ground_truth_path}")
#             return jsonify(error=f"Ground truth mask not found at {ground_truth_path}"), 500
#         else:
#             print(f"Ground truth mask loaded successfully from {ground_truth_path}")

#         ground_truth = cv2.resize(ground_truth, (cf["image_size"], cf["image_size"]))
#         pred_binary = (pred > 0.5).astype(np.uint8)[:, :, 0]

#         accuracy = np.mean(pred_binary == ground_truth)
#         print("Accuracy:", accuracy)

#         plt.figure(figsize=(12, 4))

#         plt.subplot(1, 3, 1)
#         plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#         plt.title('Input Image')
#         plt.axis('off')

#         plt.subplot(1, 3, 2)
#         plt.imshow(ground_truth, cmap='gray')
#         plt.title(f'Ground Truth Mask\nAccuracy: {accuracy:.2f}')
#         plt.axis('off')

#         plt.subplot(1, 3, 3)
#         plt.imshow(mask, cmap='gray')
#         plt.title('Predicted Tumor Regions')
#         plt.axis('off')

#         plt.tight_layout()
#         plt.savefig(os.path.join(static_dir, 'prediction_result.png'))

#         return send_file(os.path.join(static_dir, 'prediction_result.png'), mimetype='image/png')

#     except Exception as e:
#         print(f"Error during prediction: {str(e)}")
#         return jsonify(error=f"Error during prediction: {str(e)}"), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)