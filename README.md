# Industrial Safety Q&A System

A question-answering system built on industrial and machine safety documents, featuring hybrid reranking to improve answer quality over basic similarity search.

## Overview

This system processes 20 industrial safety PDF documents, creates a searchable knowledge base using embeddings, and provides accurate answers to safety-related questions. It implements both baseline similarity search and an improved hybrid reranker that combines vector similarity with BM25 keyword matching.

## Features

- **Document Processing**: Extracts and chunks text from industrial safety PDFs
- **Embeddings**: Uses `all-MiniLM-L6-v2` for semantic understanding
- **Search Modes**: 
  - Baseline: Pure cosine similarity search
  - Reranker: Hybrid approach combining vector and keyword scores
- **Answer Generation**: Extractive answers with confidence thresholding
- **API**: REST endpoint for programmatic access

## Architecture

```
PDFs → Text Extraction → Chunking → Embeddings → Chroma DB
                                                      ↓
Query → Embedding → Vector Search → Reranking → Answer Generation
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone [your-repo-url]
   cd industrial-safety-qa
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data directory**
   ```bash
   mkdir -p data/industrial-safety-pdfs
   ```

4. **Process documents and create embeddings**
   ```bash
   python ingest.py
   ```

### Running the System

1. **Start the API server**
   ```bash
   uvicorn api:app --reload
   ```
   The server will start at `http://127.0.0.1:8000`

2. **Access API documentation**
   Visit `http://127.0.0.1:8000/docs` for interactive API documentation

## Usage Examples

### Example 1: Easy Question
curl -X 'POST' \
  'http://127.0.0.1:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "q": "string",
  "k": 3,
  "mode": "rerank"
}'
### Example 2: Tricky Question  
curl -X 'POST' \
  'http://127.0.0.1:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
       "q": "What are the specific temperature requirements for chemical storage in woodworking facilities during winter months?",
       "k": 5,
       "mode": "rerank"
     }'

## API Reference

### Endpoint: `POST /ask`

**Request Format:**
```json
{
  "q": "your question here",
  "k": 3,
  "mode": "baseline"  // or "rerank"
}
```

**Response Format:**
```json
{
  "answer": "extracted answer with citation" | null,
  "contexts": [
    ["doc_id", "document_text", score],
    ...
  ],
  "reranker_used": true | false
}
```

**Parameters:**
- `q` (string): The question to ask
- `k` (integer): Number of documents to retrieve (default: 3)
- `mode` (string): Search mode - "baseline" or "rerank" (default: "baseline")

## Evaluation Results

### Before/After Comparison

<img width="845" height="488" alt="image" src="https://github.com/user-attachments/assets/33624216-7178-47cf-b2fe-6fda65d227c9" />


<img width="793" height="432" alt="image" src="https://github.com/user-attachments/assets/caec5955-27a2-4630-a5c5-b385774e3a54" />


## Implementation Details

### Document Processing
- **Text Extraction**: Uses PyPDF2 to extract text from PDF documents
- **Chunking Strategy**: Splits documents into ~300-word chunks with minimal overlap
- **Embeddings**: Generates 384-dimensional vectors using SentenceTransformers

### Search Methods

#### Baseline Search
- Pure cosine similarity between query and document embeddings
- Returns top-k most similar chunks
- Simple and fast but may miss important keyword matches

#### Hybrid Reranker
- Combines embedding similarity scores with BM25 keyword matching
- Formula: `final_score = α × embedding_score + (1-α) × bm25_score`
- Better handles both semantic similarity and exact keyword matches

### Answer Generation
- Extractive approach: selects relevant sentences from retrieved chunks
- Confidence thresholding: abstains when relevance scores are too low
- Citations: includes source document references



## Dependencies

Key libraries used:
- **FastAPI**: Web framework for API
- **Chroma**: Vector database for embeddings
- **SentenceTransformers**: Text embeddings
- **rank-bm25**: BM25 implementation for keyword matching
- **PyPDF2**: PDF text extraction
- **Pydantic**: Data validation

## What I Learned

Building this industrial safety Q&A system provided valuable insights into the practical challenges of retrieval-augmented generation systems. The most significant learning was that pure embedding-based similarity search, while effective for semantic understanding, often misses important keyword-specific queries that are crucial in technical domains like industrial safety. The hybrid reranker combining BM25 and vector similarity proved essential for handling both conceptual questions ("What is machine safety?") and specific technical queries ("What does EN ISO 13849-1 specify?"). However, I discovered that reranking improvements are highly dependent on document quality and chunking strategy - poorly segmented text can limit even the best reranking approaches.

The project also highlighted the importance of proper evaluation methodology and threshold tuning in production RAG systems. Initially, the system produced many low-quality extractive answers, but implementing confidence thresholding significantly improved reliability by enabling the system to abstain from answering when uncertain. This taught me that knowing when not to answer is as important as providing good answers. Additionally, working with real industrial safety documents revealed how domain-specific terminology and document structure (standards, regulations, technical specifications) require careful preprocessing to maintain context and ensure accurate retrieval. The experience reinforced that successful RAG systems require careful balance between retrieval precision, answer quality, and user trust through appropriate confidence mechanisms.





