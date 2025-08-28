# Assignment — Backend: Locate & Extract Tables/Images

## Input

* `mmd_lines_data.json` — layout metadata (pages, blocks, coordinates).
* `manual.mmd` — Markdown manual (tables preserved, images referenced).

---

## Task

Build a backend service that can:

1. **Locate** the relevant table or image for a natural operator-style request (e.g., *“Can you show me the drawing of a typical control valve setup?”*).
2. **Return** the table/image content.
3. **Include citation data** with at least:

   * `page_no`
   * bounding box coordinates

Handle ambiguity (multiple matches) by returning top candidates; respond with `"insufficient_info"` if no confident match.

---

## Example Operator Queries

* *“Can you pull up the comparison of actuator types?”*
* *“Show me the sizing factors for liquids.”*
* *“I need the noise level reference values.”*
* *“Do we have a figure that explains cavitation?”*
* *“Show me the overall oil & gas value chain diagram.”*

---

## Deliverables

* API implementation (`/locate`, `/extract`).
* **DESIGN.md** (≤1 page) describing your ranking logic, tie/refusal policy, and modular design choices.
* Sample outputs for 3–4 queries.


---

## Note on Implementation

We prefer candidates who can **effectively leverage coding agents** (e.g. Cursor, Claude Code, or similar AI-assisted IDEs) to produce **modular, well-structured, and high-quality code**.

* This means clean separation of concerns (indexing, retrieval, response formatting).
* Meaningful function/module boundaries.
* Readable, maintainable code, not just a script that “works once.”

---
