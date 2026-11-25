
# 1.imports
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image

# 2.Streamlit page setup

# Page configuration
st.set_page_config(
    page_title="TrashNet Waste Classification",
    page_icon="♻️",
    layout="wide"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-left: 4rem;
            padding-right: 4rem;
            padding-bottom: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("♻️ Smart Waste Classification")

# Introduction
st.markdown(
    """ 
### AI-Powered Waste Detection
Upload an image of waste and let our MobileNetV2 model identify its category.

**Supported Categories:** Cardboard | Glass | Metal | Paper | Plastic | Trash
     """
)

st.divider()



st.subheader("Upload Waste Image")

st.caption("Upload one or more waste images in JPG, JPEG or PNG format to get predictions.")


# 3.Image uploader
uploaded_files = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# 4. Model path and classes
MODEL_PATH = "models/mobilenetv2_trashnet.pth"

CLASSES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

# 5.Load_model() funktion

@st.cache_resource
def load_model():

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASSES)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model

# 6.Load model

model = load_model()



# 7.Image preprocessing for MobileNetV2

weights = MobileNet_V2_Weights.DEFAULT
preprocess = weights.transforms()


# 8. preprocessing function
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")

    image_tensor = preprocess(image)

    return image_tensor


if not uploaded_files:
    st.info("👆 Upload one or more waste images to get started.")

# 9. Test Preprocessing and prediction
if uploaded_files:
    for uploaded_file in uploaded_files:

        st.divider()

        image_col, result_col= st.columns([1,1])

        with image_col:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption=f"Uploaded Image: {uploaded_file.name}",
                use_container_width=True
            )
        
        # Preprocess image  
        uploaded_file.seek(0)  
        processed_image = preprocess_image(uploaded_file)
        processed_image = processed_image.unsqueeze(0)


        # Make prediction
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_index = torch.argmax(probabilities, dim=1).item()
            predicted_class = CLASSES[predicted_index]
            confidence = probabilities[0][predicted_index].item()

      

        # Display prediction
        with result_col:
            st.subheader("♻️ Prediction Result")

            st.caption("The model predicts the waste category and shows its confidence level.")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Category", predicted_class)

            with col2:
                st.metric("Confidence", f"{confidence:.2%}") 

            st.progress(confidence)

            if confidence < 0.60:
                st.caption("The model has relatively low confidence in this prediction.")

            else:
                st.caption("The model is relatively confident in this prediction.")


        
        # Showing probabilities of all categories
        st.subheader("Probability of all categories")

        st.caption("See how confident the model is across all six waste categories.")

        class_probabilities = []

        for i, class_name in enumerate(CLASSES):
                probability = probabilities[0][i].item()

                class_probabilities.append((class_name,probability))
                
        class_probabilities.sort(key=lambda x: x[1], reverse=True)  

        for index, (class_name, probability) in enumerate(class_probabilities):  

            if index == 0:
                st.success(f"🏆 Top Prediction: {class_name} - {probability:.2%}")

            else:
                st.write(f"**{class_name}** - {probability:.2%}")
            
            st.progress(probability)  

#Footer
st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray; padding:15px;">
        <p><strong>Smart Waste Classification</strong><p>
        <p>Powered by MobileNetV2 | TrashNet Dataset<p>
        <p>AI-Powered Waste Detection | Powered by MobileNetV2</p> 
    <div>    

    """,
    unsafe_allow_html=True

)



    

   

