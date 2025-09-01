#!/usr/bin/env python3
"""
Retrieval accuracy evaluation script for the chatbot API.
Compares retrieved chunk IDs against expected golden chunk IDs.
"""

import json
import requests
from typing import List, Dict, Set, Tuple
from datetime import datetime
import argparse


class RetrievalAccuracyEvaluator:
    """Evaluates retrieval accuracy by comparing retrieved vs expected chunk IDs."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        
    def load_queries(self, queries_file: str) -> List[Dict]:
        """Load evaluation queries from JSON file."""
        with open(queries_file, 'r') as f:
            data = json.load(f)
        return data['queries']
    
    def query_api(self, query: str) -> Dict:
        """Send query to API and get response."""
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json={"query": query},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error querying API: {e}")
            return None
    
    def extract_chunk_ids(self, response: Dict) -> Set[str]:
        """Extract chunk IDs from API response."""
        if not response or response.get('status') != 'success':
            return set()
        
        chunk_ids = set()
        for citation in response.get('citations', []):
            if citation.get('chunk_id'):
                chunk_ids.add(citation['chunk_id'])
        
        return chunk_ids
    
    def calculate_metrics(self, retrieved: Set[str], expected: Set[str]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score."""
        if not retrieved and not expected:
            return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}
        
        if not retrieved or not expected:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        true_positives = len(retrieved & expected)
        
        precision = true_positives / len(retrieved) if retrieved else 0.0
        recall = true_positives / len(expected) if expected else 0.0
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3)
        }
    
    def evaluate(self, queries_file: str) -> Dict:
        """Run evaluation on all queries and compute metrics."""
        queries = self.load_queries(queries_file)
        results = []
        
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        successful_queries = 0
        
        for query_data in queries:
            print(f"Evaluating query {query_data['id']}: {query_data['query']}")
            
            # Query the API
            response = self.query_api(query_data['query'])
            
            if not response:
                results.append({
                    "query_id": query_data['id'],
                    "error": "Failed to query API",
                    "metrics": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
                })
                continue
            
            # Extract chunk IDs
            retrieved = self.extract_chunk_ids(response)
            expected = set(query_data['expected_chunk_ids'])
            
            # Calculate metrics
            metrics = self.calculate_metrics(retrieved, expected)
            
            # Store results
            result = {
                "query_id": query_data['id'],
                "query": query_data['query'],
                "category": query_data.get('category', 'unknown'),
                "retrieved_chunks": sorted(list(retrieved)),
                "expected_chunks": sorted(list(expected)),
                "metrics": metrics,
                "status": response.get('status', 'unknown')
            }
            results.append(result)
            
            # Update totals
            if response.get('status') == 'success' or (response.get('status') == 'insufficient_info' and not query_data['should_find']):
                successful_queries += 1
                total_precision += metrics['precision']
                total_recall += metrics['recall']
                total_f1 += metrics['f1_score']
        
        # Calculate aggregate metrics
        if successful_queries > 0:
            avg_precision = total_precision / successful_queries
            avg_recall = total_recall / successful_queries
            avg_f1 = total_f1 / successful_queries
        else:
            avg_precision = avg_recall = avg_f1 = 0.0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(queries),
            "successful_queries": successful_queries,
            "aggregate_metrics": {
                "avg_precision": round(avg_precision, 3),
                "avg_recall": round(avg_recall, 3),
                "avg_f1_score": round(avg_f1, 3)
            },
            "per_query_results": results
        }
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval accuracy of the chatbot API")
    parser.add_argument(
        "--queries", 
        default="eval_queries.json",
        help="Path to queries JSON file (default: eval_queries.json)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output results file (default: results/retrieval_accuracy_YYYYMMDD_HHMMSS.json)"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    # Generate timestamped output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/retrieval_accuracy_{timestamp}.json"
    
    # Run evaluation
    evaluator = RetrievalAccuracyEvaluator(api_url=args.api_url)
    results = evaluator.evaluate(args.queries)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Total queries: {results['total_queries']}")
    print(f"Successful queries: {results['successful_queries']}")
    print(f"Average Precision: {results['aggregate_metrics']['avg_precision']}")
    print(f"Average Recall: {results['aggregate_metrics']['avg_recall']}")
    print(f"Average F1 Score: {results['aggregate_metrics']['avg_f1_score']}")
    
    # Save results
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()