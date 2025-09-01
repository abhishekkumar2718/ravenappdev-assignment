# Evaluation Scripts

This directory contains evaluation scripts for measuring the performance of the chatbot API.

## Retrieval Accuracy

The `retrieval_accuracy.py` script evaluates how well the system retrieves relevant document chunks for given queries.

### Usage

1. Ensure the API server is running:
   ```bash
   fastapi dev src/main.py
   ```

2. Run the evaluation:
   ```bash
   cd evals
   python retrieval_accuracy.py
   ```

### Options

- `--queries`: Path to queries JSON file (default: `eval_queries.json`)
- `--output`: Path to output results file (default: `results/retrieval_accuracy_YYYYMMDD_HHMMSS.json`)
- `--api-url`: API URL (default: `http://localhost:8000`)

Note: If no output path is specified, results will be saved with a timestamp (e.g., `results/retrieval_accuracy_20250831_143052.json`). This allows you to track evaluation results over time as you make changes to the system.

### Query Categories

The evaluation includes different types of queries to test various aspects of the retrieval system:

1. **table_lookup**: Queries asking for tables or comparisons (e.g., "Show me the comparison of actuator types")
2. **technical_spec**: Queries about technical specifications or parameters (e.g., "What are the sizing factors for liquids?")
3. **figure_lookup**: Queries requesting diagrams or figures (e.g., "Do we have a figure that explains cavitation?")
4. **out_of_scope**: Queries outside the manual's domain to test handling of irrelevant queries (e.g., "Show me information about Mars rovers")

### Query Format

The `queries.json` file should contain queries with expected chunk IDs:

```json
{
  "queries": [
    {
      "id": "001",
      "query": "Show me the comparison of actuator types",
      "expected_chunk_ids": ["manual_chunk_042", "manual_chunk_043"],
      "should_find": true,
      "category": "table_lookup"
    }
  ]
}
```

### Metrics

The evaluation calculates:
- **Precision**: What fraction of retrieved chunks are relevant?
- **Recall**: What fraction of relevant chunks were retrieved?
- **F1 Score**: Harmonic mean of precision and recall

### Output

Results are saved to a JSON file containing:
- Aggregate metrics (average precision, recall, F1)
- Per-query results with retrieved vs expected chunks
- Timestamp and query success rate

## Future Evaluation Metrics

While this evaluation focuses on retrieval accuracy, we considered but deferred the following metrics for future implementation:

### 1. Citation Accuracy
- **Page Number Validation**: Verify that the page numbers in citations match the actual location of content
- **Bounding Box Accuracy**: Check if the bounding box coordinates correctly identify the table/figure location
- **Implementation Note**: Requires mapping between chunks and the `mmd_lines_data.json` file

### 2. Response Quality
- **Content Completeness**: Ensure the response includes all relevant information from retrieved chunks
- **Factual Accuracy**: Verify that facts in the response are supported by the citations
- **Relevance Scoring**: Measure how well the response addresses the user's query
- **Implementation Note**: Would require LLM-based evaluation or human annotation

### 3. Confidence Handling
- **Insufficient Info Detection**: Verify the system correctly returns "insufficient_info" for low-confidence matches
- **Confidence Calibration**: Check if confidence scores align with actual retrieval quality
- **Ambiguity Handling**: Test how well the system handles queries with multiple valid interpretations

### 4. Performance Metrics
- **Latency**: Measure response time for different query types
- **Throughput**: Test system capacity under load
- **Resource Usage**: Monitor memory and CPU usage during retrieval

These metrics would provide a more comprehensive evaluation of the system's capabilities beyond basic retrieval accuracy.