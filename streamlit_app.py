import streamlit as st

st.title("Diagnostic Assistant")

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
