# extract.py

import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a given PDF file using PyMuPDF.

    Args:
        pdf_path: The file path to the PDF document.

    Returns:
        A single string containing all the text from the document.
    """
    try:
        document = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            full_text += page.get_text() + "\n"
        
        logger.info(f"Successfully extracted text from {len(document)} pages.")
        return full_text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text from PDF at {pdf_path}: {e}")
        # Return an empty string or raise the exception depending on desired error handling
        return ""