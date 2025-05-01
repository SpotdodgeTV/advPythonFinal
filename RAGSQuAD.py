"""
Enhanced Retrieval-Augmented Generation (RAG) Project
Using SQuAD (Stanford Question Answering Dataset) to evaluate embedding techniques for RAG
"""

import os
import json
import random
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Load environment variables from .env file
load_dotenv()

# Function to download and process SQuAD dataset
def download_and_process_squad(max_passages=150, max_queries_per_passage=3):
    """
    Download and process the SQuAD dataset with strict limits
    
    Args:
        max_passages: Maximum number of passages to include
        max_queries_per_passage: Maximum number of questions to include per passage
        
    Returns:
        Dictionary containing processed dataset
    """
    print("Downloading SQuAD dataset...")
    
    # URL for SQuAD v1.1 training dataset
    squad_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    
    try:
        # Download the dataset
        response = requests.get(squad_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        squad_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading SQuAD dataset: {e}")
        # Fallback to a local file if present
        try:
            with open("train-v1.1.json", "r", encoding="utf-8") as f:
                squad_data = json.load(f)
            print("Using local SQuAD dataset file")
        except FileNotFoundError:
            print("Error: Could not access SQuAD dataset. Please download the file manually.")
            raise
    
    print("Processing SQuAD dataset (limited version)...")
    
    # Process the dataset
    passages = []
    metadata = []
    queries = []
    ground_truth = []
    
    passage_count = 0
    
    # Process each article in the dataset until we reach the limit
    for article in squad_data["data"]:
        title = article["title"]
        
        # Process each paragraph in the article
        for paragraph in article["paragraphs"]:
            if passage_count >= max_passages:
                break
                
            context = paragraph["context"]
            
            # Each paragraph becomes a passage in our RAG system
            passage_id_str = f"passage_{passage_count}"
            passages.append(context)
            metadata.append({"id": passage_id_str, "source": "squad", "title": title})
            
            # Get questions for this paragraph (limited)
            qa_count = 0
            for qa in paragraph["qas"]:
                if qa_count >= max_queries_per_passage:
                    break
                    
                question = qa["question"]
                queries.append(question)
                
                # The relevant passage for this question is the current paragraph
                ground_truth.append([passage_id_str])
                qa_count += 1
            
            passage_count += 1
            if passage_count >= max_passages:
                break
                
        if passage_count >= max_passages:
            break
    
    print(f"Processed {len(passages)} passages and {len(queries)} questions")
    
    # Return the processed dataset
    return {
        "texts": passages,
        "metadata": metadata,
        "queries": queries,
        "ground_truth": ground_truth
    }

# Step 1: Choose Embedding Models (keeping the same as the original code)
# We'll use two models as suggested:
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
    # Generate embeddings in batches to handle larger datasets
    print(f"Generating {embed_type} embeddings...")
    batch_size = 100  # Adjust based on your model and memory constraints
    all_embeddings = []
    
    for i in tqdm(range(0, len(dataset["texts"]), batch_size)):
        batch_texts = dataset["texts"][i:i+batch_size]
        batch_embeddings = embed_func(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    # Get embedding dimension
    emb_size = len(all_embeddings[0]) if len(all_embeddings) > 0 else 0
    
    # Set up Qdrant collection
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    client = QdrantClient(qdrant_host, port=qdrant_port)
    collection_name = f"{embed_type}_collection"
    
    # Check if collection exists and create if it doesn't
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name in collection_names:
        # Drop the existing collection to ensure fresh data
        client.delete_collection(collection_name=collection_name)
    
    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=emb_size, distance=Distance.COSINE)
    )
    
    # Index embeddings in batches
    for i in tqdm(range(0, len(all_embeddings), batch_size)):
        batch_points = [
            PointStruct(
                id=idx,
                vector=embedding,
                payload={"text": text, "metadata": metadata}
            )
            for idx, (embedding, text, metadata) in enumerate(
                zip(all_embeddings[i:i+batch_size], dataset["texts"][i:i+batch_size], dataset["metadata"][i:i+batch_size])
            )
        ]
        
        client.upload_points(
            collection_name=collection_name,
            points=batch_points
        )
    
    return collection_name

# Step 4: Implement RAG (keeping the same as the original code)
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

# Step 5: Enhanced Evaluation Metrics
def evaluate_rag(queries, ground_truth, embed_funcs, collection_prefix=""):
    """
    Evaluate RAG performance using different embedding methods
    
    Args:
        queries: List of query texts
        ground_truth: List of expected document IDs for each query
        embed_funcs: Dictionary of embedding functions to test
        collection_prefix: Optional prefix for collection names
    
    Returns:
        Dictionary with evaluation metrics for each embedding method
    """
    results = {}
    
    for name, func in embed_funcs.items():
        collection_name = f"{collection_prefix}{name}_collection" if collection_prefix else f"{name}_collection"
        
        total_queries = len(queries)
        precision_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        mrr = 0  # Mean Reciprocal Rank
        
        for i, query in enumerate(queries):
            retrieved = query_rag(query, func, collection_name, top_k=10)
            retrieved_ids = [doc["metadata"]["id"] for doc in retrieved]
            
            # Get expected document IDs for this query
            expected_ids = ground_truth[i]
            
            # Calculate metrics
            # MRR - position of first relevant document
            for j, doc_id in enumerate(retrieved_ids):
                if doc_id in expected_ids:
                    mrr += 1 / (j + 1)
                    break
            
            # Precision@k - percentage of retrieved documents that are relevant
            for k in precision_at_k.keys():
                retrieved_at_k = retrieved_ids[:k]
                relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in expected_ids)
                precision_at_k[k] += relevant_retrieved / k if k > 0 else 0
            
            # Recall@k - percentage of relevant documents that are retrieved
            for k in recall_at_k.keys():
                retrieved_at_k = retrieved_ids[:k]
                relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in expected_ids)
                total_relevant = len(expected_ids)
                recall_at_k[k] += relevant_retrieved / total_relevant if total_relevant > 0 else 0
        
        # Average metrics over all queries
        for k in precision_at_k.keys():
            precision_at_k[k] /= total_queries if total_queries > 0 else 1
            recall_at_k[k] /= total_queries if total_queries > 0 else 1
        
        mrr_score = mrr / total_queries if total_queries > 0 else 0
        
        results[name] = {
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
            "mrr": mrr_score
        }
    
    return results

# Main execution
if __name__ == "__main__":
    # Download and process the SQuAD dataset with a smaller size
    dataset = download_and_process_squad(max_passages=50, max_queries_per_passage=3)
    
    print(f"Dataset loaded with {len(dataset['texts'])} passages and {len(dataset['queries'])} queries")
    
    # Show a sample of the dataset
    print("\nSample Questions:")
    for i in range(min(5, len(dataset['queries']))):
        print(f"  Q{i+1}: {dataset['queries'][i]}")
        passage_id = dataset['ground_truth'][i][0]
        passage_idx = next((j for j, m in enumerate(dataset['metadata']) if m['id'] == passage_id), None)
        if passage_idx is not None:
            passage_text = dataset['texts'][passage_idx]
            print(f"  A{i+1}: {passage_text[:100]}...")
        print()
    
    # If OpenAI API key is not set, only use SBERT
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key not found. Using only SBERT embeddings.")
        embed_funcs = {"sbert": sbert_embed}
    else:
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
    eval_results = evaluate_rag(dataset["queries"], dataset["ground_truth"], embed_funcs)
    
    print("\nResults:")
    for name, metrics in eval_results.items():
        print(f"\n{name.upper()} Embedding Model:")
        
        print("  Precision@k:")
        for k, value in metrics["precision@k"].items():
            print(f"    P@{k}: {value:.4f}")
        
        print("  Recall@k:")
        for k, value in metrics["recall@k"].items():
            print(f"    R@{k}: {value:.4f}")
        
        print(f"  Mean Reciprocal Rank: {metrics['mrr']:.4f}")
    
    # Compare the performance of the two embedding models
    sbert_mrr = eval_results['sbert']['mrr']
    openai_mrr = eval_results['openai']['mrr']
    
    better_model = 'OpenAI' if openai_mrr > sbert_mrr else 'SBERT'
    print(f"\nConclusion: {better_model} embeddings performed better for this RAG task based on MRR.")