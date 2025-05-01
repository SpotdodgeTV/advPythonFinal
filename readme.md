# Retrieval-Augmented Generation (RAG) Embedding Comparison Project

This project implements a system to test different embedding techniques for Retrieval-Augmented Generation (RAG) and compare their effectiveness.

## Overview

The implementation follows these steps:

1. Choose embedding models to compare:
   - Sentence-BERT (SBERT): `all-MiniLM-L6-v2`
   - OpenAI's text-embedding model: `text-embedding-ada-002`

2. Implement embedding generation functions for each model

3. Generate embeddings and store them in a vector database (Qdrant)

4. Implement a simple RAG pipeline for querying with each embedding type

5. Evaluate the performance of different embedding methods

## Requirements

```
sentence-transformers
openai
qdrant-client
python-dotenv
```

## Setup

1. Install dependencies:
```bash
pip install sentence-transformers openai qdrant-client python-dotenv
```

2. Create a `.env` file in the project root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

3. Start Qdrant:
You can run Qdrant using Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

Run the script to generate embeddings, store them in Qdrant, and evaluate the performance:

```bash
python RAG.py
```

## Customization

To use your own dataset:
- Replace the sample dataset in the script with your actual data
- Modify the `sample_queries` and `sample_ground_truth` variables for evaluation

## Output

The script outputs precision and Mean Reciprocal Rank (MRR) metrics for each embedding method, allowing you to compare their effectiveness for your specific RAG application.

## Further testing

This basic script was extended to be tested on a larger dataset (the SQuAD dataset) to test it on a practical application 
to run this extended script you must also pip install requests and tqdm
after which you can run the extended script "RAG_SQuAD.py" the same way