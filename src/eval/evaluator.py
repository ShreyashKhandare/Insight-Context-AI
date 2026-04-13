"""
MLOps Evaluator for Fin-Context RAG Engine
Integrates Ragas framework with W&B for experiment tracking
"""

import os
from typing import List, Dict, Any, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy, context_recall
import wandb
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI


class RAGEvaluator:
    """Evaluates RAG performance using Ragas and logs to W&B."""
    
    def __init__(self, wandb_project: str, wandb_api_key: str, gemini_api_key: str):
        """
        Initialize the RAG evaluator.
        
        Args:
            wandb_project: W&B project name
            wandb_api_key: W&B API key
            gemini_api_key: Gemini API key for Ragas evaluation
        """
        self.wandb_project = wandb_project
        self.wandb_api_key = wandb_api_key
        self.gemini_api_key = gemini_api_key
        
        # Initialize Gemini as critic_llm for Ragas
        self.critic_llm = ChatGoogleGenerativeAI(
            api_key=gemini_api_key,
            model="gemini-1.5-flash",
            temperature=0.1
        )
        
        # Initialize metrics with Gemini critic
        self.metrics = [
            context_precision,
            faithfulness, 
            answer_relevancy,
            context_recall
        ]
        
        # Initialize W&B
        wandb.login(key=wandb_api_key)
    
    def prepare_ragas_dataset(self, query_results: List[Dict[str, Any]]) -> Dataset:
        """
        Prepare query results for Ragas evaluation.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Dataset in Ragas format
        """
        ragas_data = []
        
        for result in query_results:
            # Extract relevant information
            question = result.get("question", "")
            answer = result.get("answer", "")
            contexts = []
            
            # Extract context from retrieved documents
            retrieved_docs = result.get("retrieved_documents", [])
            for doc in retrieved_docs:
                contexts.append(doc.page_content)
            
            # For ground truth, we'll use empty strings for now
            # In a real scenario, you'd have human-annotated ground truth
            ground_truth = result.get("ground_truth", "")
            
            ragas_data.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })
        
        return Dataset.from_dict({
            "question": [item["question"] for item in ragas_data],
            "answer": [item["answer"] for item in ragas_data],
            "contexts": [item["contexts"] for item in ragas_data],
            "ground_truth": [item["ground_truth"] for item in ragas_data]
        })
    
    def evaluate_single_query(self, query_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single query result.
        
        Args:
            query_result: Single query result dictionary
            
        Returns:
            Dictionary with evaluation scores
        """
        dataset = self.prepare_ragas_dataset([query_result])
        
        try:
            # Explicitly pass Gemini models to Ragas
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            gemini_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
            
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=gemini_llm,
                embeddings=gemini_embeddings
            )
            
            # Convert to dictionary
            scores = {
                "context_precision": result["context_precision"],
                "faithfulness": result["faithfulness"],
                "answer_relevancy": result["answer_relevancy"],
                "context_recall": result["context_recall"]
            }
            
            return scores
            
        except Exception as e:
            print(f"Error evaluating single query: {str(e)}")
            return {
                "context_precision": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": 0.0
            }
    
    def evaluate_batch(self, query_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate a batch of query results.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Dictionary with average evaluation scores
        """
        if not query_results:
            return {}
        
        dataset = self.prepare_ragas_dataset(query_results)
        
        try:
            # Explicitly pass Gemini models to Ragas
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            gemini_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
            
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=gemini_llm,
                embeddings=gemini_embeddings
            )
            
            # Convert to dictionary
            scores = {
                "context_precision": result["context_precision"],
                "faithfulness": result["faithfulness"],
                "answer_relevancy": result["answer_relevancy"],
                "context_recall": result["context_recall"]
            }
            
            return scores
            
        except Exception as e:
            print(f"Error evaluating batch: {str(e)}")
            return {
                "context_precision": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": 0.0
            }
    
    def log_to_wandb(self, query_result: Dict[str, Any], scores: Dict[str, float], 
                     experiment_name: str = "rag_evaluation", llm_hyperparameters: Dict[str, Any] = None) -> None:
        """
        Log evaluation results to W&B.
        
        Args:
            query_result: Query result dictionary
            scores: Evaluation scores
            experiment_name: Name for experiment run
            llm_hyperparameters: LLM hyperparameters to log
        """
        try:
            # Default LLM hyperparameters
            default_hyperparams = {
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.1,
                "max_tokens": 2048,
                "top_p": 0.95,
                "retrieval_k": len(query_result.get("retrieved_documents", [])),
                "context_length": len(query_result.get("context", "")),
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
            
            # Merge with provided hyperparameters
            if llm_hyperparameters:
                default_hyperparams.update(llm_hyperparameters)
            
            # Initialize W&B run
            run = wandb.init(
                project=self.wandb_project,
                name=experiment_name,
                config=default_hyperparams
            )
            
            # Log metrics
            wandb.log(scores)
            
            # Log additional information
            wandb.log({
                "answer_length": len(query_result.get("answer", "")),
                "relevance_analysis_length": len(query_result.get("relevance_analysis", "")),
                "num_retrieved_docs": len(query_result.get("retrieved_documents", []))
            })
            
            # Log actual answer and context as artifacts
            with open("temp_answer.txt", "w", encoding="utf-8") as f:
                f.write(query_result.get("answer", ""))
            
            with open("temp_context.txt", "w", encoding="utf-8") as f:
                f.write(query_result.get("context", ""))
            
            wandb.save("temp_answer.txt", base_path=".")
            wandb.save("temp_context.txt", base_path=".")
            
            # Clean up temp files
            os.remove("temp_answer.txt")
            os.remove("temp_context.txt")
            
            run.finish()
            
        except Exception as e:
            print(f"Error logging to W&B: {str(e)}")
    
    def evaluate_and_log(self, query_result: Dict[str, Any], 
                        experiment_name: Optional[str] = None, llm_hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate a query result and log to W&B.
        
        Args:
            query_result: Query result dictionary
            experiment_name: Optional custom experiment name
            llm_hyperparameters: Optional LLM hyperparameters to log
            
        Returns:
            Dictionary with evaluation scores
        """
        # Evaluate query
        scores = self.evaluate_single_query(query_result)
        
        # Generate experiment name if not provided
        if not experiment_name:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"rag_eval_{timestamp}"
        
        # Log to W&B with hyperparameters
        self.log_to_wandb(query_result, scores, experiment_name, llm_hyperparameters)
        
        return scores
    
    def create_evaluation_report(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Comprehensive evaluation report
        """
        if not query_results:
            return {"error": "No query results provided"}
        
        # Evaluate batch
        batch_scores = self.evaluate_batch(query_results)
        
        # Evaluate individual queries for detailed analysis
        individual_scores = []
        for i, result in enumerate(query_results):
            scores = self.evaluate_single_query(result)
            scores["query_index"] = i
            scores["question"] = result.get("question", "")
            individual_scores.append(scores)
        
        # Calculate statistics
        metric_names = ["context_precision", "faithfulness", "answer_relevancy", "context_recall"]
        stats = {}
        
        for metric in metric_names:
            values = [s.get(metric, 0) for s in individual_scores]
            stats[metric] = {
                "mean": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "std": self._calculate_std(values) if values else 0
            }
        
        return {
            "batch_scores": batch_scores,
            "individual_scores": individual_scores,
            "statistics": stats,
            "total_queries": len(query_results),
            "evaluation_metrics": metric_names
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize evaluator
        wandb_api_key = os.getenv("WANDB_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        wandb_project = os.getenv("WANDB_PROJECT", "fin-context-rag")
        
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY not found in environment variables")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        evaluator = RAGEvaluator(wandb_project, wandb_api_key, gemini_api_key)
        
        # Example query result
        example_result = {
            "question": "What are key financial metrics?",
            "answer": "According to [Source: Page 4], key financial metrics include revenue growth of 15% and profit margin of 12%.",
            "relevance_analysis": "The context is highly relevant and provides sufficient information.",
            "retrieved_documents": [
                Document(page_content="Revenue growth was 15% in Q3.", metadata={"page": 4}),
                Document(page_content="Profit margin increased to 12%.", metadata={"page": 4})
            ],
            "context": "Revenue growth was 15% in Q3. Profit margin increased to 12%."
        }
        
        # Evaluate and log
        scores = evaluator.evaluate_and_log(example_result)
        print("Evaluation scores:", scores)
        
    except Exception as e:
        print(f"Error: {e}")
