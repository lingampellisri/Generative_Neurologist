Generative Neurologist
Generative Neurologist is a deep learning-based project that aims to detect the presence of brain tumors and perform tumor segmentation in MRI images. The project uses state-of-the-art models for both classification and segmentation tasks, leveraging deep learning and large language models (LLMs).

Project Overview
This project focuses on two main tasks:

Brain Tumor Detection: Identifying whether a brain tumor is present or not in MRI images.
Brain Tumor Segmentation: Precisely outlining the tumor area within the MRI images.
Models Used

1. VGG16 for Classification
VGG16 is a popular convolutional neural network (CNN) architecture known for its simplicity and effectiveness in image classification tasks.
In this project, VGG16 is used to classify MRI images into two categories: tumor and no tumor.

3. UNET for Segmentation
UNET is a well-known architecture for image segmentation that is particularly effective for biomedical image analysis.
It is used here to segment the brain tumor area from MRI images, providing a detailed map of the tumor location.

5. Vision Transformer (ViT) for Classification
Vision Transformer (ViT) is a transformer-based model that has shown great success in image classification by treating image patches as sequences.
In this project, ViT is used as an alternative to VGG16 for brain tumor detection, utilizing its ability to capture complex patterns in the data.

7. UNETR for Segmentation
UNETR (UNet with Transformers) combines the strengths of UNET and transformer architectures to perform segmentation tasks.
This model is used for more accurate segmentation of brain tumors, benefiting from both the localization capabilities of UNET and the global context understanding of transformers.
