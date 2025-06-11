import streamlit as st
from PIL import Image
import pandas as pd
import json
import asyncio
from google import genai
import tempfile
import os
from pypdf import PdfReader
import fitz  # PyMuPDF
import pathlib
from google.genai import types

# Set page config
st.set_page_config(page_title="Invoice Extractor", layout="wide")

# Prompt for invoice extraction
invoice_extraction_prompt = """
You are an expert in document understanding and structured data extraction from financial documents like invoices.

Below is the image/text content of an invoice.

Your task is to extract the following key fields from the invoice and return the result in a valid JSON format. If any information is not available in the invoice, return its value as `null`.

Ensure all string values are clean and trimmed of unnecessary whitespace.

Expected JSON response:
{
  "invoice_date": "YYYY-MM-DD or null",
  "invoice_number": "string or null",
  "purchase_order_number": "string or null",
  "purchase_order_date": "YYYY-MM-DD or null",
  "phone_number": "string or null",
  "customer_name": "string or null",
  "supplier_name": "string or null",
  "supplier_address": "string or null",
  "customer_address": "string or null",
  "sub_total": "string or null",
  "total_tax_amount": "string or null",
  "total_amount": "string or null",
  "currency": "string or null"
}
"""


def pdf_to_images(path):
    images = []
    with fitz.open(path) as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=150)  # higher DPI = better quality
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images


# Load Google API key from Streamlit secrets
google_api_key = st.secrets["api_keys"]["google"]


async def extract_info_from_pdf(prompt, pdf_path):
    google_client = genai.Client(api_key=google_api_key)
    response = google_client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[
            types.Part.from_bytes(
                data=pdf_path.read_bytes(),
                mime_type="application/pdf",
            ),
            prompt,
        ],
    )
    return response


async def extract_info_from_image(prompt, image):
    google_client = genai.Client(api_key=google_api_key)
    response = google_client.models.generate_content(
        model="gemini-2.0-flash-lite", contents=[prompt] + image
    )
    return response


# Sidebar: Upload image
st.sidebar.title("Upload Invoice Image")
uploaded_files = st.sidebar.file_uploader(
    "Choose invoice images or pdf",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True,
)

if len(uploaded_files) > 0:
    # Display layout
    col1, col2 = st.columns([1, 2])

    # Store all extracted data
    all_extracted_data = []

    with col1:
        st.subheader("Document Preview")

    with col2:
        st.subheader("Extracted Invoice Data")

    # Process each file
    for uploaded_file in uploaded_files:
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Handle PDF files
            if uploaded_file.type == "application/pdf":
                # Extract text from PDF
                pdf_reader = PdfReader(tmp_file_path)
                text_content = ""
                # Get text from first page only
                if len(pdf_reader.pages) > 0:
                    text_content = pdf_reader.pages[0].extract_text()
                    images = pdf_to_images(tmp_file_path)

                with col1:
                    st.write(f"**PDF: {uploaded_file.name}**")
                    for img in images:
                        st.image(img, use_container_width=True)

                # Send to Gemini
                with st.spinner(f"Extracting data from {uploaded_file.name}..."):
                    try:
                        response = asyncio.run(
                            extract_info_from_pdf(
                                invoice_extraction_prompt, pathlib.Path(tmp_file_path)
                            )
                        )
                        info_text = response.text

                        try:
                            json_start = info_text.find("{")
                            json_end = info_text.rfind("}") + 1
                            info_formatted = json.loads(info_text[json_start:json_end])
                            all_extracted_data.append(
                                {"filename": uploaded_file.name, "data": info_formatted}
                            )

                        except json.JSONDecodeError:
                            st.error(
                                f"Could not parse valid JSON from model output for {uploaded_file.name}."
                            )
                    except Exception as e:
                        st.error(
                            f"Model failed to extract info from {uploaded_file.name}: {e}"
                        )

            else:
                # Handle regular image files
                image = Image.open(uploaded_file)

                with col1:
                    st.write(f"**Image: {uploaded_file.name}**")
                    st.image(image, use_container_width=True)

                # Send to Gemini
                with st.spinner(f"Extracting data from {uploaded_file.name}..."):
                    try:
                        response = asyncio.run(
                            extract_info_from_image(invoice_extraction_prompt, [image])
                        )
                        info_text = response.text

                        try:
                            json_start = info_text.find("{")
                            json_end = info_text.rfind("}") + 1
                            info_formatted = json.loads(info_text[json_start:json_end])
                            all_extracted_data.append(
                                {"filename": uploaded_file.name, "data": info_formatted}
                            )

                        except json.JSONDecodeError:
                            st.error(
                                f"Could not parse valid JSON from model output for {uploaded_file.name}."
                            )
                    except Exception as e:
                        st.error(
                            f"Model failed to extract info from {uploaded_file.name}: {e}"
                        )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    # Display all extracted data in a tabbed interface
    if all_extracted_data:
        with col2:
            # Create a container with increased height
            with st.container(height=1600):  # Increased height to 800 pixels
                # Create a list to store all DataFrames
                dfs = []
                for data in all_extracted_data:
                    # Create DataFrame and transpose it
                    df = pd.DataFrame(
                        data["data"].items(), columns=["Field", data["filename"]]
                    )
                    df.set_index("Field", inplace=True)
                    dfs.append(df)

                # Concatenate all DataFrames horizontally
                combined_df = pd.concat(dfs, axis=1)

                # Display the combined DataFrame
                st.dataframe(combined_df.transpose(), use_container_width=True)
