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

## Future Improvements

Noting down ideas that might be useful:
- Use markdown instead of LaTeX for manual? I don't know if the extra-noise added by LaTeX tags affects similarity score.
- Re-write document chunks with context to be self-explainable.
- Re-ranking results using Cohere or similar.
- Use semantic chunking instead of markdown text splitter: semantic chunking finds more natural boundaries between the different paragraphs.
- Add checks to validate that facts in the response is present in the citation and actually supports the response.

## Decision Log

- Picked FastAPI since it's pretty popular and accessible.
- Used pytest for writing tests.
- Used FAISS for vector store + 