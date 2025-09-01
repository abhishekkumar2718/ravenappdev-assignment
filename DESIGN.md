# Design

Let's work through this problem incrementally, building more complex solutions as needed.

## Observations

1. The queries are complex and might require a multi-hop solution. 

For example: "Compare actuator types" can be broken down into a few different sub-queries: "list actuator types" and then for each actuator type, a summary and advantages/disadvantages. Then this will be summarized into a comparative table.

2. Getting page/bounding boxes

The data in `mmd_lines_data` is broken into individual lines, and I need some way to map chunks in `manual.mmd` to the appropriate line. Do-able, just annoying.

## Approach 1: Single-Hop

Let's try a single-hop approach to see if it works. Will be useful to understand the baseline approach.

For now, let's remove the limitation on getting the page number and bounding box. Just the original chunk is fine.

**Indexing**
- Chunk the paragraphs in `manual.mmd` and use a FAISS index.

The split chunks are really bad, using LatexTextSplitter and MarkdownTextSplitter. This is going to effect downstream quality, but I will come back to this problem once I get the pipeline working.

Possible solutions:
- Use Mistral/Mathpix to return markdown instead of Latex-markdown.
- 

**Retrieval**
- Take the user query and fetch from FAISS index. Let's use MMR with 10 results.

**Presentation**
- Present LLM with the results and ask it to implement it.

A few decisions:
- To reduce the time/cost, let's work on just the first chapter. I can re-run the scripts to generate index for the whole PDF later on.

## Approach 2: Section + Entity Aware Chunking

I need to fix chunking problem, because without good data - nothing else would work well.

### Indexing 

**Section-based chunking with special entity handling:**

1. **Chunk Types**:
   - `section`: Complete sections bounded by `\section*` markers
   - `table`: Full tables including captions, extracted as single units
   - `image`: Image references with captions (`\includegraphics` blocks)

2. **Chunking Strategy**:
   - Split document at section boundaries (`\section*` markers)
   - Extract tables (`\begin{table}` to `\end{table}`) as separate chunks
   - Extract figures (`\begin{figure}` to `\end{figure}`) as separate chunks
   - Maintain references between sections and their entities

3. **Metadata Structure**:
   ```python
   {
       "chunk_id": str,
       "chunk_type": "section|table|image",
       "section_title": str,
       "section_path": ["Chapter 1", "Control Valve Selection", "Ball Valves"],
       "entity_ids": ["table_1-2", "figure_1-6"],  # Entities referenced in this section
       "page_number": int,
       "line_numbers": [start, end]
   }
   ```

4. **Entity Registry**:
   - Maintain mapping of entity IDs (e.g., "table_1-2") to their chunks
   - Store relationships between sections and entities they reference

### Retrieval

**Multi-modal retrieval with special handling for tables and images:**

1. **Query Classification**:
   - `table_lookup`: Queries containing "table", "comparison", "chart", "specifications"
   - `image_lookup`: Queries containing "figure", "diagram", "show me", "illustration"
   - `section_search`: General information queries

2. **Section Retrieval**:
   - Dense search over section embeddings
   - For each retrieved section, fetch associated entities (tables/figures)
   - Return both section content and related entities

3. **Table-Specific Retrieval**:
   - Direct search within table content and captions
   - Pattern matching for table numbers (e.g., "Table 1-2")
   - Retrieve complete table even for partial matches
   - Include section that introduces/discusses the table

4. **Image-Specific Retrieval**:
   - Search figure captions and descriptions
   - Pattern matching for figure numbers (e.g., "Figure 1-6")
   - Return figure metadata with bounding box coordinates

---

I am not re-ranking right now, because I expect the LLM to filter irrelevant documents.

Observations:
- The chunks are okay, with images and tables taken out separately.
- The retrieval is terrible. I am thinking of switching to keyword search instead.

```
=== Evaluation Summary ===
Total queries: 35
Successful queries: 35
Average Precision: 0.023
Average Recall: 0.071
Average F1 Score: 0.034
Results saved to results/retrieval_accuracy_20250901_192350.json
```

Approach 2.1: Use vector search + keyword search:

The confidence is now 50% from vector search confidence, and 50% from keyword matches.

```
=== Evaluation Summary ===
Total queries: 35
Successful queries: 5
Average Precision: 0.64
Average Recall: 0.7
Average F1 Score: 0.657
Results saved to results/retrieval_accuracy_20250901_193357.json
```

Wait a second, I don't see any retrieved chunks in the result logs. I also wonder why it says only 5 successful queries.

I see that for most questions, it says no confident matches found. Let's remove the confidence score threshold (> 0.5) and try again.

```
=== Evaluation Summary ===
Total queries: 35
Successful queries: 30
Average Precision: 0.107
Average Recall: 0.117
Average F1 Score: 0.11
Results saved to results/retrieval_accuracy_20250901_194339.json
```

The results are somewhat better than vector search but still really, really bad.

I wonder if inverted search (keywords -> sections) would better than sections -> keywords. It's too late to rry new ideas now.

## Approach 3: Section + Entity + Multi-Hop Answering (Did not implement)

Use an LLM to break down the initial query into many smaller questions.

For example:

```
Compare actutator types?
-> List actutator types
-> Describe characteristics of actuator type A
-> Describe characterristics of actuator type B
...
```

This will get closer to implementing deep search: https://github.com/langchain-ai/open_deep_research with alternative rounds of search, and verification.

## Future Improvements

Noting down ideas that might be useful:
- Use markdown instead of LaTeX for manual? I don't know if the extra-noise added by LaTeX tags affects similarity score.
- Re-write document chunks with context to be self-explainable.
- Re-ranking results using Cohere or similar.
- Use semantic chunking instead of markdown text splitter: semantic chunking finds more natural boundaries between the different paragraphs.
- Add checks to validate that facts in the response is present in the citation and actually supports the response.
- Use keyword-search only: 

## Decision Log

- Picked FastAPI since it's pretty popular and accessible.
- Used pytest for writing tests.
- Used FAISS for vector store + 