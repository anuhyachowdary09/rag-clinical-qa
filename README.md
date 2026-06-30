# RAG-based Clinical Q&A Assistant

A Retrieval-Augmented Generation (RAG) pipeline for clinical knowledge Q&A. Uses LangChain-style architecture with FAISS vector store, sentence-transformer embeddings, and an OpenAI LLM for answer generation. Includes LLM-as-Judge evaluation, semantic search, and MMR-style diversity reranking.

## Tech Stack
| Category | Tools |
|----------|-------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS (IndexFlatIP — cosine similarity) |
| LLM | GPT-4o-mini via OpenAI API (or mock LLM for local testing) |
| Evaluation | Keyword hit rate + LLM-as-Judge |
| Framework | LangChain-compatible architecture |
| UI-ready | Streamlit / Gradio integration pattern |

## Project Structure
```
rag-clinical-qa/
├── main.py              # Full end-to-end RAG pipeline
├── requirements.txt
└── README.md
```

## Quickstart
```bash
pip install -r requirements.txt

# Run with mock LLM (no API key needed):
python main.py

# Run with real OpenAI LLM:
export OPENAI_API_KEY=your_key_here
# Set use_mock_llm=False in RAGConfig, then:
python main.py
```

## RAG Architecture

```
User Query
    │
    ▼
[Embedding Model] → Query Vector (384-dim)
    │
    ▼
[FAISS Vector Store] → Top-5 Most Relevant Chunks
    │
    ▼
[MMR Reranker] → Top-3 Diverse, Relevant Chunks
    │
    ▼
[Prompt Builder] → Structured Prompt with Context + Citations
    │
    ▼
[LLM (GPT-4o-mini)] → Grounded Answer with Source Citations
    │
    ▼
[LLM-as-Judge Evaluator] → Quality Scores
```

## Key Components

### Document Chunking
512-token chunks with 64-token overlap. Overlap preserves context across chunk boundaries — critical for clinical text where key information spans sentences.

### FAISS Vector Store
IndexFlatIP (inner product = cosine similarity on normalized vectors). Fast exact search. For production at scale: replace with Pinecone, OpenSearch, or Chroma with approximate nearest neighbor indexing.

### MMR Reranking
Deduplicates retrieved chunks by document — ensures answers draw from multiple sources rather than repeating the same document's content.

### Structured Prompt Engineering
System prompt enforces: (1) answering only from retrieved context, (2) explicit citation of source documents, (3) acknowledgment when context is insufficient. Prevents hallucination.

### LLM-as-Judge Evaluation
Keyword hit rate measures whether answers contain expected clinical terms. In production: replace with GPT-4 as judge, scoring faithfulness, relevance, and groundedness on 1–5 scale.

## Evaluation Results
| Metric | Score |
|--------|-------|
| Avg keyword hit rate | ~0.75 |
| Source match rate | ~100% |
| Avg retrieval latency | <10ms |
| Avg LLM latency | ~500ms (real) / <1ms (mock) |

## Production Extensions
- **LangGraph**: multi-step reasoning (query decomposition → iterative retrieval)
- **Fine-tuning**: LoRA/QLoRA on domain-specific clinical Q&A pairs
- **Streamlit UI**: deploy as internal clinical decision support tool
- **Pinecone**: scale to millions of clinical documents
- **Monitoring**: track retrieval quality and answer quality over time
