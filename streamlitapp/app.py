import streamlit as st
from model_helper import predict

st.title("Vehcile Damage Detection")

uploaded_file = st.file_uploader("Upload image", type=['jpg', 'png'])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        predicted = predict(image_path)
        st.info(f"Predicted Class : {predicted}")
