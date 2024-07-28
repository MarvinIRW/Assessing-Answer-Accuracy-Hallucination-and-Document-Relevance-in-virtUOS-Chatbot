import io
from PyPDF2 import PdfReader
import requests
import re
import concurrent.futures
from typing import Optional, Tuple

def read_pdf_from_url(response, num_pages=7):
        """
        Read the content of a pdf file from a given response object
        :param response: response object from the request
        :param num_pages: number of pages to process. If None, process all pages
        """
        pdf_stream = io.BytesIO(response.content)

        pdf_text = ''
        with pdf_stream as f:
            reader = PdfReader(f)
            if num_pages is None:
                num_pages = len(reader.pages)
            else:
                num_pages = min(num_pages, len(reader.pages))

            for page_num in range(num_pages):
                page = reader.pages[page_num]
                pdf_text += page.extract_text()

        return pdf_text



    

# def open_pdf_as_binary(pdf_url: str):
#     try:
#         # Send a GET request to the URL to download the PDF file
#         response = requests.get(pdf_url)
#         response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        
#         # Return the binary content of the PDF file
#         return response.content
#     except requests.exceptions.RequestException as e:
#         print(f"Error: Failed to retrieve the PDF file from the URL: {e}")
#         return None


def extract_pdf_url(text: str ):
    # Regular expression to detect links to PDF files
    pattern = r'(https?://[^/]+)?(/[^"]*\/([^"/]+\.pdf))'

    
    # Default domain
    default_domain = ['https://www.lili.uni-osnabrueck.de', 'https://www.uni-osnabrueck.de']
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    if matches:
        # TODO if there are multiple PDF files in the response, choose the most relevant one based on the context
        for match in matches:
            domain, path, filename = match
            break
    
        if domain is None:
            
            for d in default_domain:
                response = requests.get(d + path)
                if response.status_code == 404:
                    continue
                else:
                    break
                
        else:
            response = requests.get(domain + path)
    else:
        return None, None
    
    if response.status_code == 200:
        return response.content, filename
    else:
        return None, None
        
        
# Function to run the extract_pdf_url with a timeout
def extract_pdf_with_timeout(text: str, timeout: int) -> Tuple[Optional[bytes], Optional[str]]:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(extract_pdf_url, text)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            print(f"Operation timed out after {timeout} seconds")
            return None, None