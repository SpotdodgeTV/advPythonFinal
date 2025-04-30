"""
Retrieval-Augmented Generation (RAG) Project
Comparing different embedding techniques for RAG
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Load environment variables from .env file
load_dotenv()

# Step 1: Choose Embedding Models
# We'll use two models as suggested in the requirements
# - Sentence-BERT (SBERT) model: all-MiniLM-L6-v2
# - OpenAI's text-embedding model: text-embedding-ada-002

# Step 2: Implement Embedding Generation
def sbert_embed(texts):
    """Generate embeddings using Sentence-BERT"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts)

def openai_embed(texts):
    """Generate embeddings using OpenAI's model"""
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    
    client = OpenAI(api_key=api_key)
    if isinstance(texts, str):
        texts = [texts]
    response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
    return [item.embedding for item in response.data]

# Step 3: Generate Embeddings
def generate_and_store_embeddings(dataset, embed_func, embed_type):
    """Generate embeddings and store them in Qdrant"""
    # Generate embeddings
    embeddings = embed_func(dataset["texts"])
    
    # Get embedding dimension
    emb_size = len(embeddings[0]) if len(embeddings) > 0 else 0
    
    # Set up Qdrant collection
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    client = QdrantClient(qdrant_host, port=qdrant_port)
    collection_name = f"{embed_type}_collection"
    
    # Check if collection exists and create if it doesn't
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=emb_size, distance=Distance.COSINE)
        )
    
    # Index embeddings
    points = [
        PointStruct(
            id=idx,
            vector=embedding,
            payload={"text": text, "metadata": metadata}
        )
        for idx, (embedding, text, metadata) in enumerate(zip(embeddings, dataset["texts"], dataset["metadata"]))
    ]
    
    client.upload_points(
        collection_name=collection_name,
        points=points
    )
    
    return collection_name

# Step 4: Implement RAG
def query_rag(query_text, embed_func, collection_name, top_k=5):
    """Perform RAG query using specified embedding method"""
    # Generate embedding for query
    query_embedding = embed_func([query_text])[0]
    
    # Connect to Qdrant
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    client = QdrantClient(qdrant_host, port=qdrant_port)
    
    # Search for similar documents
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    
    # Extract relevant text from search results
    retrieved_docs = [
        {
            "text": result.payload["text"],
            "metadata": result.payload["metadata"],
            "score": result.score
        }
        for result in search_results
    ]
    
    return retrieved_docs

# Step 5: Evaluate Performance
def evaluate_rag(queries, ground_truth, embed_funcs):
    """
    Evaluate RAG performance using different embedding methods
    
    Args:
        queries: List of query texts
        ground_truth: List of expected document IDs for each query
        embed_funcs: Dictionary of embedding functions to test
    
    Returns:
        Dictionary with evaluation metrics for each embedding method
    """
    results = {}
    
    for name, func in embed_funcs.items():
        collection_name = f"{name}_collection"
        
        total_queries = len(queries)
        correct_retrievals = 0
        mrr = 0  # Mean Reciprocal Rank
        
        for i, query in enumerate(queries):
            retrieved = query_rag(query, func, collection_name, top_k=10)
            retrieved_ids = [doc["metadata"]["id"] for doc in retrieved]
            
            # Check if the correct document is in the retrieved results
            expected_ids = ground_truth[i]
            
            # Calculate metrics
            found = False
            for j, doc_id in enumerate(retrieved_ids):
                if doc_id in expected_ids:
                    if not found:  # For MRR, use the first relevant doc position
                        mrr += 1 / (j + 1)
                        found = True
                    correct_retrievals += 1
                    break
        
        precision = correct_retrievals / total_queries if total_queries > 0 else 0
        mrr_score = mrr / total_queries if total_queries > 0 else 0
        
        results[name] = {
            "precision": precision,
            "mrr": mrr_score
        }
    
    return results

# Main execution
if __name__ == "__main__":
    # Sample dataset - replace with your actual data
    dataset = {
        "texts": [
            "Retrieval-Augmented Generation (RAG) combines retrieval with generative AI models.",
            "Vector databases store embeddings for efficient similarity search.",
            "Qdrant is a vector database optimized for similarity search.",
            "Embeddings capture semantic meaning of text in numerical form.",
            "Sentence-BERT models are efficient for generating text embeddings."
        ],
        "metadata": [
            {"id": 1, "source": "doc1"},
            {"id": 2, "source": "doc2"},
            {"id": 3, "source": "doc3"},
            {"id": 4, "source": "doc4"},
            {"id": 5, "source": "doc5"}
        ]
    }
    
    # Sample queries and ground truth for evaluation
    sample_queries = [
        "What is RAG?",
        "Tell me about vector databases",
        "How do embeddings work?"
    ]
    
    sample_ground_truth = [
        [1],  # For query 1, document 1 is relevant
        [2, 3],  # For query 2, documents 2 and 3 are relevant
        [4, 5]   # For query 3, documents 4 and 5 are relevant
    ]
    
    # Define embedding functions to test
    embed_funcs = {
        "sbert": sbert_embed,
        "openai": openai_embed
    }
    
    print("Step 1: Generating and storing embeddings")
    collections = {}
    for name, func in embed_funcs.items():
        print(f"Processing {name} embeddings...")
        collections[name] = generate_and_store_embeddings(dataset, func, name)
    
    print("\nStep 2: Evaluating RAG performance")
    eval_results = evaluate_rag(sample_queries, sample_ground_truth, embed_funcs)
    
    print("\nResults:")
    for name, metrics in eval_results.items():
        print(f"{name} embedding:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  MRR: {metrics['mrr']:.4f}")