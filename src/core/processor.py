"""
PDF Document Processor for Fin-Context RAG Engine
Handles PDF parsing and text splitting with page metadata preservation
"""

import os
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessor:
    """Processes PDF documents and splits them into chunks with page metadata."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and extract text with page metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects with page metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        documents = []
        
        try:
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                if text.strip():  # Only add non-empty pages
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(file_path),
                            "filename": os.path.basename(file_path),
                            "page": page_num,
                            "file_path": file_path
                        }
                    )
                    documents.append(doc)
                    
        except Exception as e:
            raise Exception(f"Error processing PDF {file_path}: {str(e)}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving page metadata.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of split Document objects with preserved metadata
        """
        if not documents:
            return []
        
        try:
            split_docs = self.text_splitter.split_documents(documents)
            
            # Ensure all split documents have the original metadata
            for doc in split_docs:
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 1
                if "source" not in doc.metadata:
                    doc.metadata["source"] = "unknown"
                    
            return split_docs
            
        except Exception as e:
            raise Exception(f"Error splitting documents: {str(e)}")
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Complete pipeline: load PDF and split into chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of processed Document chunks
        """
        documents = self.load_pdf(file_path)
        return self.split_documents(documents)
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all processed Document chunks
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_documents = []
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return []
        
        for pdf_file in pdf_files:
            file_path = os.path.join(directory_path, pdf_file)
            try:
                documents = self.process_pdf(file_path)
                all_documents.extend(documents)
                print(f"Processed {pdf_file}: {len(documents)} chunks")
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                continue
        
        return all_documents


if __name__ == "__main__":
    # Example usage
    processor = PDFProcessor()
    
    # Process a single PDF
    try:
        documents = processor.process_pdf("data/raw/example.pdf")
        print(f"Processed {len(documents)} chunks")
        for doc in documents[:3]:  # Show first 3 chunks
            print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
