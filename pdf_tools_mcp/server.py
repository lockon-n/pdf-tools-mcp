from typing import Any, List
import PyPDF2
import io
import os
import argparse
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import logging
import sys

# Global variables
mcp = FastMCP("pdf-tools")
WORKSPACE_PATH = None

def setup_server(workspace_path: str = None):
    """Setup the MCP server with optional workspace path"""
    global WORKSPACE_PATH
    
    # change working dir to workspace_path
    if workspace_path: 
        os.chdir(workspace_path)
        sys.path.append(workspace_path)
    
    # Global workspace path
    WORKSPACE_PATH = Path(workspace_path).resolve() if workspace_path else None

def validate_path(file_path: str) -> tuple[bool, str]:
    """Validate if the file path is within the allowed workspace"""
    if WORKSPACE_PATH is None:
        return True, ""
    
    try:
        resolved_path = Path(file_path).resolve()
        if not resolved_path.is_relative_to(WORKSPACE_PATH):
            return False, f"Error: Path '{file_path}' is outside the allowed workspace '{WORKSPACE_PATH}'"
        return True, ""
    except Exception as e:
        return False, f"Error validating path: {str(e)}"

def validate_page_range(start_page: int, end_page: int, total_pages: int) -> tuple[int, int]:
    """Validate and correct page range to ensure it's within valid bounds"""
    # Handle negative page numbers
    if start_page < 1:
        start_page = 1
    if end_page < 1:
        end_page = 1
    
    # Handle pages exceeding total page count
    if start_page > total_pages:
        start_page = total_pages
    if end_page > total_pages:
        end_page = total_pages
    
    # Ensure start page is not greater than end page
    if start_page > end_page:
        start_page, end_page = end_page, start_page
    
    return start_page, end_page

def extract_text_from_pdf(pdf_content: bytes, start_page: int, end_page: int) -> str:
    """Extract text from PDF content for specified page range"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        total_pages = len(pdf_reader.pages)
        
        # Validate and correct page range
        start_page, end_page = validate_page_range(start_page, end_page, total_pages)
        
        # Extract text
        extracted_text = []
        for page_num in range(start_page - 1, end_page):  # PyPDF2 uses 0-index
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                extracted_text.append(f"=== Page {page_num + 1} ===\n{text}")
        
        if not extracted_text:
            return f"PDF total pages: {total_pages}\nSpecified page range ({start_page}-{end_page}) has no extractable text content."
        
        result = f"PDF total pages: {total_pages}\nExtracted page range: {start_page}-{end_page}\n\n"
        result += "\n\n".join(extracted_text)
        return result
        
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

@mcp.tool()
async def read_pdf_pages(pdf_file_path: str, start_page: int = 1, end_page: int = 1) -> str:
    """Read content from PDF file for specified page range.
    
    Note: Avoid reading too many pages at once (recommended: <50 pages) to prevent errors.

    Args:
        pdf_file_path: Path to the PDF file
        start_page: Starting page number (default: 1)
        end_page: Ending page number (default: 1)
    """
    # Validate path
    is_valid, error_msg = validate_path(pdf_file_path)
    if not is_valid:
        return error_msg
    
    # Warning for large page ranges
    if end_page - start_page > 50:
        warning = "Warning: Reading more than 50 pages at once may cause performance issues or errors.\n"
    else:
        warning = ""
    
    try:
        # Read PDF file
        with open(pdf_file_path, 'rb') as file:
            pdf_content = file.read()
        
        # Extract text
        result = extract_text_from_pdf(pdf_content, start_page, end_page)
        return warning + result if warning else result
        
    except FileNotFoundError:
        return f"Error: File not found '{pdf_file_path}'"
    except PermissionError:
        return f"Error: No permission to read file '{pdf_file_path}'"
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"

@mcp.tool()
async def get_pdf_info(pdf_file_path: str) -> str:
    """Get basic information about a PDF file including page count.

    Args:
        pdf_file_path: Path to the PDF file
    """
    # Validate path
    is_valid, error_msg = validate_path(pdf_file_path)
    if not is_valid:
        return error_msg
    
    try:
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # 在 with 块内部获取所有需要的信息
            total_pages = len(pdf_reader.pages)
            info = pdf_reader.metadata
            
            result = "PDF file information:\n"
            result += f"Total pages: {total_pages}\n"
            
            if info:
                result += f"Title: {info.get('/Title', 'Unknown')}\n"
                result += f"Author: {info.get('/Author', 'Unknown')}\n"
                result += f"Creator: {info.get('/Creator', 'Unknown')}\n"
                result += f"Creation date: {info.get('/CreationDate', 'Unknown')}\n"
            
            return result
        
    except FileNotFoundError:
        return f"Error: File not found '{pdf_file_path}'"
    except Exception as e:
        return f"Error getting PDF information: {str(e)}"

@mcp.tool()
async def merge_pdfs(pdf_paths: List[str], output_path: str) -> str:
    """Merge multiple PDF files into one.

    Args:
        pdf_paths: List of paths to PDF files to merge (in order)
        output_path: Path where the merged PDF will be saved
    """
    # Validate all paths
    for pdf_path in pdf_paths:
        is_valid, error_msg = validate_path(pdf_path)
        if not is_valid:
            return error_msg
    
    is_valid, error_msg = validate_path(output_path)
    if not is_valid:
        return error_msg
    
    try:
        pdf_writer = PyPDF2.PdfWriter()
        total_pages_merged = 0
        
        for pdf_path in pdf_paths:
            try:
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    pages_count = len(pdf_reader.pages)
                    
                    for page in pdf_reader.pages:
                        pdf_writer.add_page(page)
                    
                    total_pages_merged += pages_count
                    logging.info(f"Added {pages_count} pages from {pdf_path}")
            
            except Exception as e:
                return f"Error reading PDF '{pdf_path}': {str(e)}"
        
        # Write the merged PDF
        with open(output_path, 'wb') as output_file:
            pdf_writer.write(output_file)
        
        return f"Successfully merged {len(pdf_paths)} PDFs into '{output_path}'\nTotal pages: {total_pages_merged}"
        
    except Exception as e:
        return f"Error merging PDFs: {str(e)}"

@mcp.tool()
async def extract_pdf_pages(source_path: str, page_numbers: List[int], output_path: str) -> str:
    """Extract specific pages from a PDF and create a new PDF.

    Args:
        source_path: Path to the source PDF file
        page_numbers: List of page numbers to extract (1-indexed)
        output_path: Path where the new PDF will be saved
    """
    # Validate paths
    is_valid, error_msg = validate_path(source_path)
    if not is_valid:
        return error_msg
    
    is_valid, error_msg = validate_path(output_path)
    if not is_valid:
        return error_msg
    
    try:
        with open(source_path, 'rb') as source_file:
            pdf_reader = PyPDF2.PdfReader(source_file)
            total_pages = len(pdf_reader.pages)
            pdf_writer = PyPDF2.PdfWriter()
            
            extracted_pages = []
            
            for page_num in page_numbers:
                if 1 <= page_num <= total_pages:
                    pdf_writer.add_page(pdf_reader.pages[page_num - 1])
                    extracted_pages.append(page_num)
                else:
                    logging.warning(f"Page {page_num} is out of range (1-{total_pages}), skipping")
            
            if not extracted_pages:
                return f"Error: No valid pages to extract from PDF (total pages: {total_pages})"
            
            # Write the new PDF
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)
            
            return f"Successfully extracted {len(extracted_pages)} pages from '{source_path}' to '{output_path}'\nExtracted pages: {extracted_pages}\nSource PDF total pages: {total_pages}"
            
    except FileNotFoundError:
        return f"Error: File not found '{source_path}'"
    except Exception as e:
        return f"Error extracting pages: {str(e)}"

def main():
    """Main function to run the MCP server"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PDF Tools MCP Server')
    parser.add_argument('--workspace_path', type=str, default=None, 
                        help='Workspace directory path. All PDF operations will be restricted to this directory and its subdirectories.')
    args = parser.parse_args()
    
    # Setup server
    setup_server(args.workspace_path)
    
    # Log workspace restriction if set
    if WORKSPACE_PATH:
        logging.info(f"Workspace restricted to: {WORKSPACE_PATH}")
    
    # Initialize and run the server
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()