import streamlit as st
from PIL import Image
import pandas as pd
import json
import asyncio
from google import genai
import tempfile
import os
from pypdf import PdfReader
import base64

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
  "supplier_invoice_number": "string or null",
  "customer_name": "string or null",
  "supplier_name": "string or null",
  "supplier_address": "string or null",
  "customer_address": "string or null",
  "total_amount": "string or null",
  "tax_amount": "string or null",
  "currency": "string or null",
  "line_items": [
    {
      "description": "string or null",
      "quantity": "string or null",
      "unit_price": "string or null",
      "price": "string or null unit price x qty"
      "net_amount":"string or null"
    }
  ]
}
"""

# Load Google API key from Streamlit secrets
google_api_key = st.secrets["api_keys"]["google"]


async def extract_info_from_text(prompt, text):
    google_client = genai.Client(api_key=google_api_key)
    response = google_client.models.generate_content(
        model="gemini-2.0-flash-lite", contents=[prompt, text]
    )
    return response


async def extract_info_from_image(prompt, image):
    google_client = genai.Client(api_key=google_api_key)
    response = google_client.models.generate_content(
        model="gemini-2.0-flash-lite", contents=[prompt, image]
    )
    return response


# Sidebar: Upload image
st.sidebar.title("Upload Invoice Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an invoice image", type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file:
    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Display layout
        col1, col2 = st.columns(2)

        # Handle PDF files
        if uploaded_file.type == "application/pdf":
            # Extract text from PDF
            pdf_reader = PdfReader(tmp_file_path)
            text_content = ""
            # Get text from first page only
            if len(pdf_reader.pages) > 0:
                text_content = pdf_reader.pages[0].extract_text()

            with col1:
                st.subheader("PDF Preview")
                # Display PDF
                with open(tmp_file_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode("utf-8")
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)

            # Send to Gemini
            with st.spinner("Extracting invoice data..."):
                try:
                    response = asyncio.run(
                        extract_info_from_text(invoice_extraction_prompt, text_content)
                    )
                    info_text = response.text

                    try:
                        json_start = info_text.find("{")
                        json_end = info_text.rfind("}") + 1
                        info_formatted = json.loads(info_text[json_start:json_end])

                        with col2:
                            st.subheader("Extracted Invoice Data")

                            # Extract line_items if present
                            line_items = info_formatted.pop("line_items", [])

                            # Display other fields as a table (key-value pairs)
                            summary_df = pd.DataFrame(
                                info_formatted.items(), columns=["Field", "Value"]
                            )
                            edited_summary_df = st.data_editor(
                                summary_df,
                                use_container_width=True,
                                num_rows="dynamic",
                                key="summary_editor",
                            )

                            # Display line items if any
                            if line_items:
                                st.subheader("Line Items")
                                line_items_df = pd.DataFrame(line_items)
                                edited_line_items_df = st.data_editor(
                                    line_items_df,
                                    use_container_width=True,
                                    num_rows="dynamic",
                                    key="line_items_editor",
                                )
                    except json.JSONDecodeError:
                        st.error("Could not parse valid JSON from model output.")
                except Exception as e:
                    st.error(f"Model failed to extract info: {e}")

        else:
            # Handle regular image files
            image = Image.open(uploaded_file)

            # Show uploaded image
            with col1:
                st.subheader("Invoice Image")
                st.image(image, use_container_width=True)

            # Send to Gemini
            with st.spinner("Extracting invoice data..."):
                try:
                    response = asyncio.run(
                        extract_info_from_image(invoice_extraction_prompt, image)
                    )
                    info_text = response.text

                    try:
                        json_start = info_text.find("{")
                        json_end = info_text.rfind("}") + 1
                        info_formatted = json.loads(info_text[json_start:json_end])

                        with col2:
                            st.subheader("Extracted Invoice Data")

                            # Extract line_items if present
                            line_items = info_formatted.pop("line_items", [])

                            # Display other fields as a table (key-value pairs)
                            summary_df = pd.DataFrame(
                                info_formatted.items(), columns=["Field", "Value"]
                            )
                            edited_summary_df = st.data_editor(
                                summary_df,
                                use_container_width=True,
                                num_rows="dynamic",
                                key="summary_editor",
                            )

                            # Display line items if any
                            if line_items:
                                st.subheader("Line Items")
                                line_items_df = pd.DataFrame(line_items)
                                edited_line_items_df = st.data_editor(
                                    line_items_df,
                                    use_container_width=True,
                                    num_rows="dynamic",
                                    key="line_items_editor",
                                )
                    except json.JSONDecodeError:
                        st.error("Could not parse valid JSON from model output.")
                except Exception as e:
                    st.error(f"Model failed to extract info: {e}")

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
