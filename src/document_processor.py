import os
import re
from typing import List, Dict
import PyPDF2
import docx
from io import BytesIO

class DocumentProcessor:
    """Process various document formats and extract text"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx']
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error processing DOCX: {str(e)}")
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extract text from TXT file"""
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            raise Exception(f"Error processing TXT: {str(e)}")
    
    def process_document(self, file_content: bytes, filename: str) -> str:
        """Process document based on file extension"""
        file_ext = os.path.splitext(filename.lower())[1]
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_content)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_content)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            chunk = text[start:end]
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size * 0.5:
                end = start + break_point + 1
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks