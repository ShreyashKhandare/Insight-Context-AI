"""
RAG Engine for Fin-Context RAG Engine
Implements two-stage prompting with source citations
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from .vectorstore import VectorStoreManager


class RAGEngine:
    """RAG Engine with two-stage prompting and source citations."""
    
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize the RAG engine.
        
        Args:
            groq_api_key: Groq API key
            model_name: Groq model to use
        """
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model_name,
            temperature=0.1
        )
        
        self.vector_store = VectorStoreManager()
        
        # Two-stage prompts
        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial document analyst. Your task is to analyze retrieved context for relevance to the user's question.

Context:
{context}

User Question: {question}

Analyze the context and determine:
1. Is the context relevant to answering the question?
2. What specific information in the context is most relevant?
3. Are there any gaps or missing information needed to fully answer the question?

Provide a brief analysis focusing on relevance and completeness."""),
            ("human", "Please analyze the relevance of this context to the question.")
        ])
        
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial expert assistant. Based on the analyzed context, provide a comprehensive answer to the user's question.

IMPORTANT REQUIREMENTS:
1. Use ONLY information from the provided context
2. Cite sources explicitly using the format: "[Source: Page X]"
3. If information is insufficient, clearly state what additional information is needed
4. Be precise and professional in your financial analysis

Analyzed Context:
{context}

User Question: {question}

Relevance Analysis:
{relevance_analysis}

Provide your answer with proper source citations."""),
            ("human", "Please answer the question based on the context and analysis.")
        ])
        
        # Prompts are ready for direct LLM invocation
    
    def load_vectorstore(self) -> None:
        """Load the vector store."""
        self.vector_store.load_vectorstore()
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            page = doc.metadata.get('page', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            
            context_parts.append(
                f"[Document {i}] Source: {source}, Page: {page}\n"
                f"Content: {doc.page_content}\n"
            )
        
        return "\n".join(context_parts)
    
    def stage1_relevance_analysis(self, query: str, context: str) -> str:
        """
        Stage 1: Analyze context relevance.
        
        Args:
            query: User query
            context: Formatted context string
            
        Returns:
            Relevance analysis text
        """
        try:
            messages = self.relevance_prompt.format_messages(
                context=context,
                question=query
            )
            result = self.llm.invoke(messages)
            return result.content.strip()
            
        except Exception as e:
            raise Exception(f"Error in relevance analysis: {str(e)}")
    
    def stage2_generate_answer(self, query: str, context: str, relevance_analysis: str) -> str:
        """
        Stage 2: Generate final answer with citations.
        
        Args:
            query: User query
            context: Formatted context string
            relevance_analysis: Results from stage 1
            
        Returns:
            Final answer with citations
        """
        try:
            messages = self.answer_prompt.format_messages(
                context=context,
                question=query,
                relevance_analysis=relevance_analysis
            )
            result = self.llm.invoke(messages)
            return result.content.strip()
            
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Complete RAG query pipeline.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with query results
        """
        try:
            # Load vector store if not already loaded
            if not self.vector_store.vectorstore:
                self.load_vectorstore()
            
            # Stage 0: Retrieve relevant documents
            documents = self.retrieve_context(question, k)
            context = self.format_context(documents)
            
            # Stage 1: Relevance analysis
            relevance_analysis = self.stage1_relevance_analysis(question, context)
            
            # Stage 2: Generate answer
            answer = self.stage2_generate_answer(question, context, relevance_analysis)
            
            return {
                "question": question,
                "answer": answer,
                "relevance_analysis": relevance_analysis,
                "retrieved_documents": documents,
                "context": context
            }
            
        except Exception as e:
            raise Exception(f"Error in RAG query: {str(e)}")
    
    def get_retriever(self):
        """Get a retriever for use in other chains."""
        return self.vector_store.get_retriever()


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize RAG engine
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        rag_engine = RAGEngine(groq_api_key)
        
        # Example query
        question = "What are the key financial metrics mentioned in the documents?"
        result = rag_engine.query(question)
        
        print("Question:", result["question"])
        print("Answer:", result["answer"])
        print("Relevance Analysis:", result["relevance_analysis"])
        
    except Exception as e:
        print(f"Error: {e}")
