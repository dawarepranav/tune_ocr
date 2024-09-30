
# Paste your updated Streamlit code here (same as above)
import streamlit as st
from transformers import AutoModel, AutoTokenizer
import os
import re
from PIL import Image

# Load the model and tokenizer from the local directory or huggingface
model_path = 'pranavdaware/web_ocr'  

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True)
model = model.eval().cuda()

# Function to extract text using OCR
def ocr_processing(image_file):
    try:
        # Perform OCR on the uploaded image
        result = model.chat(tokenizer, image_file, ocr_type='ocr')
        return result
    except Exception as e:
        return str(e)

# Function to search for keywords in extracted text
def search_keyword(ocr_text, keyword):
    try:
        # Use regex to search for the keyword and highlight matches
        matches = re.findall(rf"({keyword})", ocr_text, re.IGNORECASE)
        if matches:
            highlighted_text = re.sub(rf"({keyword})", r'<mark>\1</mark>', ocr_text, flags=re.IGNORECASE)
            return highlighted_text
        else:
            return f"No matches found for '{keyword}' in the extracted text."
    except Exception as e:
        return str(e)

# Streamlit app
def main():
    st.title("OCR and Keyword Search Application")

    # Upload image
    image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if image_file is not None:
        # Display uploaded image
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # OCR processing
        if st.button("Process Image for OCR"):
            ocr_output = ocr_processing(image_file)
            st.text_area("OCR Output", ocr_output, height=200)
        
        # Keyword search
        keyword = st.text_input("Enter keyword to search in OCR text")
        if keyword and st.button("Search Keyword in OCR Text"):
            search_output = search_keyword(ocr_output, keyword)
            st.markdown(search_output, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
