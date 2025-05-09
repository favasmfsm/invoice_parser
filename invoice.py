import streamlit as st
from PIL import Image
import pandas as pd
import json
import asyncio
from google import genai

# Set page config
st.set_page_config(page_title="Invoice Extractor", layout="wide")

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "processed_data" not in st.session_state:
    st.session_state.processed_data = []
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False

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


# Sidebar for file upload
st.sidebar.title("Upload Invoice Images")
uploaded_files = st.sidebar.file_uploader(
    "Choose invoice images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

# If new files uploaded
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    st.session_state.current_index = 0
    st.session_state.processed_data = []
    st.session_state.show_summary = False

# If files exist
if st.session_state.uploaded_files:
    files = st.session_state.uploaded_files
    idx = st.session_state.current_index

    # Navigation Buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous", disabled=idx == 0):
            st.session_state.current_index -= 1
            st.rerun()
    with col2:
        st.write(f"Image {idx + 1} of {len(files)}")
    with col3:
        col3a, col3b = st.columns(2)
        with col3a:
            if st.button("Next", disabled=idx == len(files) - 1):
                st.session_state.current_index += 1
                st.rerun()
                process_button = True
        with col3b:
            if st.button("Finish"):
                if st.session_state.processed_data:
                    st.session_state.show_summary = True
                else:
                    st.warning("No data has been processed yet.")

    # Load and display current image
    current_file = files[idx]
    current_file.seek(0)
    image = Image.open(current_file)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Invoice Image")
        st.image(image, use_container_width=True)

    # Process button
    response_key = f"response_{idx}"
    if response_key not in st.session_state:
        if process_button:
            with st.spinner("Extracting invoice data..."):
                try:
                    response = asyncio.run(
                        extract_info_from_image(invoice_extraction_prompt, image)
                    )
                    st.session_state[response_key] = response.text
                    st.rerun()
                except Exception as e:
                    st.error(f"Model failed to extract info: {e}")
    else:
        st.success("Already processed.")
        info_text = st.session_state[response_key]
        st.subheader("Raw Model Output")
        st.code(info_text)

        try:
            json_start = info_text.find("{")
            json_end = info_text.rfind("}") + 1
            info_formatted = json.loads(info_text[json_start:json_end])

            with col_right:
                st.subheader("Extracted Invoice Data")

                # Line items
                line_items = info_formatted.pop("line_items", [])
                summary_df = pd.DataFrame(
                    info_formatted.items(), columns=["Field", "Value"]
                )
                edited_summary_df = st.data_editor(
                    summary_df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"summary_editor_{idx}",
                )

                if f"summary_editor_{idx}" in st.session_state:
                    current_data = st.session_state[f"summary_editor_{idx}"]
                    if current_data not in st.session_state.processed_data:
                        st.session_state.processed_data.append(current_data)

                if line_items:
                    st.subheader("Line Items")
                    line_items_df = pd.DataFrame(line_items)
                    st.data_editor(
                        line_items_df,
                        use_container_width=True,
                        num_rows="dynamic",
                        key=f"line_items_editor_{idx}",
                    )
        except json.JSONDecodeError:
            st.error("Could not parse valid JSON from model output.")
else:
    st.info("Please upload one or more invoice images to begin.")

# Show summary if finish was clicked
if st.session_state.show_summary:
    st.header("All Processed Invoices Summary")
    if st.session_state.processed_data:
        all_data = pd.DataFrame(st.session_state.processed_data)

        st.dataframe(all_data, use_container_width=True)
        csv = all_data.to_csv(index=False)
        st.download_button(
            "Download Summary as CSV",
            data=csv,
            file_name="invoice_summary.csv",
            mime="text/csv",
        )
