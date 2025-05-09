import streamlit as st
from PIL import Image
import pandas as pd
import json
import asyncio
from google import genai

# Set page config
st.set_page_config(page_title="Invoice Extractor", layout="wide")

# Prompt for invoice extraction
invoice_extraction_prompt = """
You are an expert in document understanding and structured data extraction from financial documents like invoices.

Below is an image of an invoice.

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
      "total_price": "string or null"
    }
  ]
}
"""

# Load Google API key from Streamlit secrets
google_api_key = st.secrets["api_keys"]["google"]


async def extract_info_from_image(prompt, image):
    google_client = genai.Client(api_key=google_api_key)

    response = google_client.models.generate_content(
        model="gemini-2.0-flash-lite", contents=[prompt, image]
    )
    return response


# Sidebar: Upload image
st.sidebar.title("Upload Invoice Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an invoice image", type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    # Display layout
    col1, col2 = st.columns(2)

    # Show uploaded image
    with col1:
        st.subheader("Invoice Image")
        st.image(image, use_container_width=True)

    # Convert image to byte array if needed
    img_bytes = uploaded_file.read()

    # Send to Gemini
    with st.spinner("Extracting invoice data..."):
        try:
            response = asyncio.run(
                extract_info_from_image(invoice_extraction_prompt, image)
            )
            info_text = response.text
            st.subheader("Raw Model Output")
            st.code(info_text)

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
                    st.dataframe(summary_df, use_container_width=True)

                    # Display line items if any
                    if line_items:
                        st.subheader("Line Items")
                        line_items_df = pd.DataFrame(line_items)
                        st.dataframe(line_items_df, use_container_width=True)
            except json.JSONDecodeError:
                st.error("Could not parse valid JSON from model output.")
        except Exception as e:
            st.error(f"Model failed to extract info: {e}")
