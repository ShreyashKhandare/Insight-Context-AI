"""
Vector Store Manager for Fin-Context RAG Engine
Handles ChromaDB persistence and document retrieval
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import google.generativeai as genai
except ImportError:
    print("Warning: langchain_google_genai not available, using OpenAI embeddings as fallback")
    from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings


class VectorStoreManager:
    """Manages ChromaDB vector store for document storage and retrieval."""
    
    def __init__(self, persist_directory: str = "data/chroma", embedding_model: Optional[Embeddings] = None):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Custom embedding model (defaults to Google)
        """
        self.persist_directory = persist_directory
        
        # Use Google Generative AI embeddings with GEMINI_API_KEY
        if embedding_model is None:
            import os
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            try:
                # Use the discovered model from check_models.py
                self.embedding_model = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=gemini_api_key
                )
            except Exception as e:
                print(f"Error initializing Google embeddings: {e}")
                # Fallback to OpenAI if available
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    from langchain_openai import OpenAIEmbeddings
                    self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
                else:
                    raise Exception("No embedding model available")
        else:
            self.embedding_model = embedding_model
            
        self.vectorstore = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to store
            
        Returns:
            Chroma vector store instance
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        try:
            # Create vector store with documents
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory
            )
            
            print(f"Created vector store with {len(documents)} documents")
            return self.vectorstore
            
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def load_vectorstore(self) -> Chroma:
        """
        Load existing vector store from disk.
        
        Returns:
            Chroma vector store instance
        """
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
            
            # Test if vector store is accessible
            collection_count = self.vectorstore._collection.count()
            print(f"Loaded vector store with {collection_count} documents")
            
            return self.vectorstore
            
        except Exception as e:
            raise Exception(f"Error loading vector store: {str(e)}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if not self.vectorstore:
            self.load_vectorstore()
        
        if not documents:
            return
        
        try:
            self.vectorstore.add_documents(documents)
            print(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            raise Exception(f"Error adding documents to vector store: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar Document objects
        """
        if not self.vectorstore:
            self.load_vectorstore()
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
            
        except Exception as e:
            raise Exception(f"Error performing similarity search: {str(e)}")
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.vectorstore:
            self.load_vectorstore()
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return results
            
        except Exception as e:
            raise Exception(f"Error performing similarity search with scores: {str(e)}")
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        if not self.vectorstore:
            self.load_vectorstore()
        
        try:
            return self.vectorstore._collection.count()
            
        except Exception as e:
            raise Exception(f"Error getting document count: {str(e)}")
    
    def delete_collection(self) -> None:
        """
        Delete the entire vector store collection.
        """
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                self.vectorstore = None
                print("Vector store collection deleted")
            
            # Also delete the persist directory
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                os.makedirs(self.persist_directory, exist_ok=True)
                
        except Exception as e:
            raise Exception(f"Error deleting vector store collection: {str(e)}")
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Dict[str, Any] = None):
        """
        Get a retriever object for use in LangChain chains.
        
        Args:
            search_type: Type of search ("similarity", "mmr", etc.)
            search_kwargs: Additional search parameters
            
        Returns:
            Retriever object
        """
        if not self.vectorstore:
            self.load_vectorstore()
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )


if __name__ == "__main__":
    # Example usage
    from processor import PDFProcessor
    
    try:
        # Initialize processor and vector store
        processor = PDFProcessor()
        vector_store = VectorStoreManager()
        
        # Process documents (example)
        # documents = processor.process_directory("data/raw")
        # vector_store.create_vectorstore(documents)
        
        # Load existing vector store
        vector_store.load_vectorstore()
        
        # Perform search
        results = vector_store.similarity_search("financial analysis", k=3)
        print(f"Found {len(results)} results")
        
    except Exception as e:
        print(f"Error: {e}")
