import streamlit as st
from PIL import Image
import pandas as pd
import json
import asyncio
from google import genai

# Set page config
st.set_page_config(page_title="Invoice Extractor", layout="wide")

# Initialize session state for uploaded files and current index
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

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
st.sidebar.title("Upload Invoice Images")
uploaded_files = st.sidebar.file_uploader(
    "Choose invoice images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

# Update session state when new files are uploaded
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    st.session_state.current_index = 0

# Navigation controls
if st.session_state.uploaded_files:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous", disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
    with col2:
        st.write(
            f"Image {st.session_state.current_index + 1} of {len(st.session_state.uploaded_files)}"
        )
    with col3:
        if st.button(
            "Next",
            disabled=st.session_state.current_index
            == len(st.session_state.uploaded_files) - 1,
        ):
            st.session_state.current_index += 1

    # Get current file
    current_file = st.session_state.uploaded_files[st.session_state.current_index]
    image = Image.open(current_file)

    # Display layout
    col1, col2 = st.columns(2)

    # Show uploaded image
    with col1:
        st.subheader("Invoice Image")
        st.image(image, use_container_width=True)

    # Convert image to byte array if needed
    img_bytes = current_file.read()

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
                    edited_summary_df = st.data_editor(
                        summary_df,
                        use_container_width=True,
                        num_rows="dynamic",
                        key=f"summary_editor_{st.session_state.current_index}",
                    )

                    # Display line items if any
                    if line_items:
                        st.subheader("Line Items")
                        line_items_df = pd.DataFrame(line_items)
                        edited_line_items_df = st.data_editor(
                            line_items_df,
                            use_container_width=True,
                            num_rows="dynamic",
                            key=f"line_items_editor_{st.session_state.current_index}",
                        )
            except json.JSONDecodeError:
                st.error("Could not parse valid JSON from model output.")
        except Exception as e:
            st.error(f"Model failed to extract info: {e}")
else:
    st.info("Please upload one or more invoice images to begin.")
