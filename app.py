from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator

# Load the model and tokenizer for clinical note analysis
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Function for analyzing and translating clinical notes
def analyze_clinical_notes(text, target_language='en'):
    # First, translate the input text to the target language (English by default)
    translated_text = GoogleTranslator(source='auto', target=target_language).translate(text)
    
    # Perform text analysis using Flan-T5 model
    inputs = tokenizer(translated_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
clinical_note = "Paciente presenta dolor en el pecho y antecedentes de enfermedad cardiovascular."
print("Text Analysis Output:", analyze_clinical_notes(clinical_note, target_language='en'))

from torchvision import models, transforms
from PIL import Image
import torch

# Load a pre-trained DenseNet model for image classification
densenet = models.densenet121(pretrained=True)
densenet.eval()

# Define the image transformation
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

# Function for analyzing medical images
def analyze_medical_image(image_path):
    img = Image.open(image_path)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        out = densenet(batch_t)
    return out.argmax().item()  # Returns the predicted class ID

# Example usage with a sample image (upload an X-ray image)
image_path = 'C:\Users\KIIT\Downloads\download.jpeg'  # Provide your own image path here
print("Image Analysis Output:", analyze_medical_image(image_path))
from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate synthetic images
def generate_synthetic_image(prompt):
    image = pipe(prompt).images[0]
    image.save("generated_medical_image.png")
    return "generated_medical_image.png"

# Example usage
prompt = "A chest X-ray showing pneumonia"
generated_image_path = generate_synthetic_image(prompt)
print("Generated Image saved at:", generated_image_path)
# Function for generating a personalized treatment plan
def generate_treatment_plan(patient_data):
    if patient_data['age'] > 50 and 'hypertension' in patient_data['conditions']:
        return "Recommend lifestyle changes and prescribe anti-hypertensive drugs."
    elif 'diabetes' in patient_data['conditions']:
        return "Suggest metformin and regular blood sugar monitoring."
    else:
        return "Further testing required."

# Example usage
patient_data = {'age': 55, 'conditions': ['hypertension']}
print("Treatment Plan:", generate_treatment_plan(patient_data))
# Function to generate a diagnostic report combining text and image analysis
def generate_diagnostic_report(clinical_notes, image_path, patient_data, target_language='en'):
    # Analyze clinical notes
    text_analysis = analyze_clinical_notes(clinical_notes, target_language)
    
    # Analyze medical image
    image_analysis = analyze_medical_image(image_path)
    
    # Generate personalized treatment plan
    treatment_plan = generate_treatment_plan(patient_data)
    
    # Combine results into a report
    report = {
        "Text Analysis": text_analysis,
        "Image Analysis": f"Predicted Class ID: {image_analysis}",
        "Treatment Plan": treatment_plan
    }
    return report

# Example usage
clinical_notes = "Paciente presenta tos persistente y dificultad para respirar."
image_path = "path_to_xray_image.jpg"  # Provide your own image path here
patient_data = {'age': 60, 'conditions': ['hypertension', 'diabetes']}
report = generate_diagnostic_report(clinical_notes, image_path, patient_data, target_language='en')

# Display the report
for key, value in report.items():
    print(f"{key}: {value}")
import streamlit as st

st.title("AI-Driven Multimodal Diagnostic Assistant")

# Input for clinical notes
text = st.text_area("Enter Clinical Notes (e.g., symptoms, patient history)")

# Upload for medical images
uploaded_image = st.file_uploader("Upload Medical Image (JPG/PNG)", type=["jpg", "png"])

# Input for patient data
age = st.number_input("Enter Patient Age", min_value=0, max_value=120)
conditions = st.text_area("Enter Known Conditions (e.g., diabetes, hypertension)")

# Generate Diagnostic Report
if st.button("Generate Diagnostic Report"):
    if text and uploaded_image:
        # Save uploaded image
        with open("uploaded_image.png", "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Create the patient data dictionary
        patient_data = {
            'age': age,
            'conditions': conditions.split(',')
        }
        
        # Generate report
        report = generate_diagnostic_report(text, "uploaded_image.png", patient_data)
        
        # Display report
        st.write("### Diagnostic Report")
        for key, value in report.items():
            st.write(f"**{key}:** {value}")
    else:
        st.error("Please provide both clinical notes and an image.")
