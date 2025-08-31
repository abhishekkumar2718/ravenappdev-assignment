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

**Retrieval**
- Take the user query and fetch from FAISS index. Let's use MMR with 10 results.

**Presentation**
- Present LLM with the results and ask it to implement it.

## Future Improvements

Noting down ideas that might be useful:
- 

## Decision Log

- Picked FastAPI since it's pretty popular and accessible.
- Used pytest for writing tests.
- Used FAISS for vector store + 