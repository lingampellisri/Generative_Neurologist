
# 🧠 Generative Neurologist

Generative Neurologist is a deep learning-based web platform for **early detection and segmentation of brain tumors** using MRI images. It combines modern **machine learning (ML)** models with a robust **MERN stack interface**, and features **role-based access** for doctors and patients.

---

## 🗂️ Project Overview

This platform performs two primary tasks:

- 🧪 **Brain Tumor Detection:** Classifies MRI images as "Tumor" or "No Tumor".
- 🧠 **Brain Tumor Segmentation:** Accurately outlines the tumor area in MRI scans.

---

## 🔍 Models Used

### ✅ 1. VGG16 for Classification
- A deep CNN used to detect if a tumor is present.
- Binary classification: **Tumor / No Tumor**.

### ✅ 2. Vision Transformer (ViT) for Classification
- Transformer-based model that treats image patches as tokens.
- Captures complex visual features.

### ✅ 3. UNET for Segmentation
- Popular for biomedical segmentation tasks.
- Outputs pixel-wise tumor masks from MRI images.

### ✅ 4. UNETR for Advanced Segmentation
- Combines transformers with UNET for improved accuracy.
- Learns both local details and global context.

---

## 🚀 Features

- 🧠 Brain tumor classification and segmentation
- 🖼️ Color-coded tumor overlays on MRI scans
- 🧍 **Role-Based Dashboards:** Doctor 👨‍⚕️ & Patient 👩‍⚕️
- 📊 Diagnosis history for both user types
- 🔐 Secure login/signup (MongoDB Atlas)
- 🌐 MERN Stack frontend with integrated Python backend

---

## 🛠️ Tech Stack

| Layer         | Technology |
|---------------|------------|
| Frontend      | React.js, Tailwind CSS, Bootstrap |
| Backend (API) | Node.js, Express.js |
| Database      | MongoDB Atlas |
| ML Backend    | Python, TensorFlow, Keras, OpenCV, FastAPI |
| Deployment    | Localhost (Dev), Docker (Optional) |

---

## 📁 Folder Structure

```

Generative-Neurologist/
├── client/            # React frontend
│   └── src/components/
├── server/            # Node.js backend
│   └── routes/
│   └── controllers/
├── ml-models/         # Python ML backend
│   ├── classify\_vgg16.py
│   ├── segment\_unet.py
│   └── utils/
├── README.md
└── requirements.txt

````

---

## 🧑‍⚕️ How to Run the Project Locally

### 📌 Prerequisites:
- Node.js & npm
- Python 3.7+
- MongoDB Atlas account
- `uvicorn`, `fastapi`, and ML dependencies

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Generative-Neurologist.git
cd Generative-Neurologist
````

---

### 2️⃣ Run the Backend (Node.js + Express)

```bash
cd server
npm install
npm run dev
```

🟢 Server runs at `http://localhost:5000`

---

### 3️⃣ Run the Frontend (React)

```bash
cd client
npm install
npm start
```

🟢 React app runs at `http://localhost:3000`

---

### 4️⃣ Run the ML Backend (FastAPI)

```bash
cd ml-models
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

🟢 ML API is available at `http://localhost:8000`

---

## 🧪 Sample: VGG16 Classification Code

```python
# ml-models/classify_vgg16.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("vgg16_brain_tumor.h5")

def classify_mri(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    prediction = model.predict(img_tensor)[0][0]
    return "Tumor Detected" if prediction > 0.5 else "No Tumor"
```

---

## 🧾 requirements.txt

```txt
tensorflow==2.9.1
keras==2.9.0
numpy
pillow
fastapi
uvicorn
python-multipart
opencv-python
```

---

## 🌐 API Routes (Python ML Backend)

| Method | Endpoint    | Description                               |
| ------ | ----------- | ----------------------------------------- |
| POST   | `/classify` | Classifies MRI using VGG16 or ViT         |
| POST   | `/segment`  | Returns segmentation mask from UNET/UNETR |

---

## ✨ Future Improvements

* 📲 Flutter-based mobile app
* 🔐 JWT Authentication with refresh tokens
* 🧾 Report upload/download for patients
* 📈 Admin dashboard for model analytics
* 🧠 Auto-updating model with transfer learning

---

## 🤝 Contributing

We welcome contributions from developers, data scientists, and researchers.
Feel free to:

* Fork the repo
* Create a new branch
* Submit a pull request

---

## 📜 License

This project is licensed under the **MIT License**.



> Built with ❤️ to save lives through technology.

```
