"""
PDF Tools MCP Server

A FastMCP-based PDF reading and manipulation tool server.
"""

__version__ = "0.1.1"
__author__ = "Junlong Li"
__email__ = "lockonlvange@gmail.com"

from .server import (
    read_pdf_pages,
    get_pdf_info,
    merge_pdfs,
    extract_pdf_pages,
    validate_path,
    validate_page_range,
    extract_text_from_pdf,
)

__all__ = [
    "read_pdf_pages",
    "get_pdf_info", 
    "merge_pdfs",
    "extract_pdf_pages",
    "validate_path",
    "validate_page_range",
    "extract_text_from_pdf",
] 