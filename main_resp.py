import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image

# Set CUDNN benchmark and deactivate interactive mode for matplotlib
torch.backends.cudnn.benchmark = True
plt.ioff()

# Define data transforms
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet model and modify it for binary classification
@st.cache_resource
def load_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

model_conv = load_model()

# Define class names for prediction (update this based on your dataset)
class_names = ['Class 0', 'Class 1']  # Modify these based on actual classes

# Image preprocessing and visualization function
def preprocess_image(img):
    img = Image.open(img).convert('RGB')
    img = data_transforms['val'](img).unsqueeze(0).to(device)
    return img

def visualize_predictions(img, model):
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)

    # Convert tensor image back to numpy for displaying
    img_display = img_tensor.cpu().squeeze().numpy()
    img_display = np.transpose(img_display, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
    img_display = np.clip(img_display, 0, 1)  # Clip values to valid range

    # Plot the image and prediction
    plt.figure(figsize=(4, 4))
    plt.imshow(img_display)
    plt.title(f'Predicted: {class_names[preds[0]]}')
    plt.axis('off')
    st.pyplot(plt)

# Set Streamlit page configuration for mobile responsiveness
#st.set_page_config(page_title="Diagnostic Classification", layout="centered")

# Main header and image upload section
st.title("Sistema de Classificação Diagnóstico")

uploaded_file = st.file_uploader("Escolha uma imagem de diagnóstico...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption='Imagem Carregada', use_column_width=True)
    
    # Display prediction results
    st.write("Classificando a imagem...")
    visualize_predictions(uploaded_file, model_conv)
else:
    st.info("Faça o upload de uma imagem para começar.")

# Footer with lighter layout
st.write("---")
st.markdown("Feito com ❤️ usando Streamlit. Otimizado para dispositivos móveis.")

# Footer with lighter layout
st.write("---")
st.markdown("Feito com ❤️ usando Streamlit. Otimizado para dispositivos móveis.")
