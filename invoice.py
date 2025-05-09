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
    client = genai.Client(api_key=google_api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite", contents=[prompt, image]
    )
    return response


# Sidebar - File uploader
st.sidebar.title("Upload Invoice Images")
new_uploads = st.sidebar.file_uploader(
    "Choose invoice images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

# Reset on new uploads
if new_uploads:
    st.session_state.uploaded_files = new_uploads
    st.session_state.current_index = 0
    st.session_state.processed_data = []
    st.session_state.show_summary = False

# Navigation logic
if st.session_state.uploaded_files:
    if "nav" in st.session_state:
        nav = st.session_state.pop("nav")
        if (
            nav == "next"
            and st.session_state.current_index
            < len(st.session_state.uploaded_files) - 1
        ):
            st.session_state.current_index += 1
        elif nav == "prev" and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
        st.experimental_rerun()

# Main UI
if st.session_state.uploaded_files:
    files = st.session_state.uploaded_files
    idx = st.session_state.current_index
    response_key = f"response_{idx}"

    # Navigation buttons and status
    col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
    with col1:
        if st.button("Previous", disabled=(idx == 0)):
            st.session_state["nav"] = "prev"
    with col2:
        total = len(files)
        status = (
            "✅ Processed" if response_key in st.session_state else "❌ Not Processed"
        )
        st.markdown(f"**Image {idx+1} of {total}** — {status}")
    with col3:
        if st.button("Next", disabled=(idx == len(files) - 1)):
            st.session_state["nav"] = "next"
    with col4:
        if st.button("Finish"):
            if st.session_state.processed_data:
                st.session_state.show_summary = True
            else:
                st.warning("No data has been processed yet.")

    # Display current image and process
    current_file = files[idx]
    current_file.seek(0)
    image = Image.open(current_file)
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Invoice Image")
        if response_key not in st.session_state:
            if st.button("Process This Invoice", key=f"process_btn_{idx}"):
                with st.spinner("Extracting invoice data..."):
                    try:
                        resp = asyncio.run(
                            extract_info_from_image(invoice_extraction_prompt, image)
                        )
                        st.session_state[response_key] = resp.text
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Model failed to extract info: {e}")
        else:
            st.success("Already processed.")
        st.image(image, use_container_width=True)

    # Show extracted data
    if response_key in st.session_state:
        raw = st.session_state[response_key]
        st.subheader("Raw Model Output")
        st.code(raw)
        try:
            obj = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
            with col_right:
                st.subheader("Extracted Invoice Data")
                line_items = obj.pop("line_items", [])
                df_summary = pd.DataFrame(obj.items(), columns=["Field", "Value"])
                edited = st.data_editor(
                    df_summary,
                    key=f"editor_sum_{idx}",
                    use_container_width=True,
                    num_rows="dynamic",
                )
                if edited not in st.session_state.processed_data:
                    st.session_state.processed_data.append(edited)
                if line_items:
                    st.subheader("Line Items")
                    df_items = pd.DataFrame(line_items)
                    st.data_editor(
                        df_items,
                        key=f"editor_items_{idx}",
                        use_container_width=True,
                        num_rows="dynamic",
                    )
        except json.JSONDecodeError:
            st.error("Could not parse valid JSON from model output.")
else:
    st.info("Please upload one or more invoice images to begin.")

# Show summary
if st.session_state.show_summary:
    st.header("All Processed Invoices Summary")
    all_data = pd.concat(st.session_state.processed_data, ignore_index=True)
    st.dataframe(all_data, use_container_width=True)
    csv = all_data.to_csv(index=False)
    st.download_button(
        "Download Summary as CSV",
        data=csv,
        file_name="invoice_summary.csv",
        mime="text/csv",
    )
