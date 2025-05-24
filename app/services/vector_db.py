from pinecone import Pinecone, ServerlessSpec
from app import config

# Initialize Pinecone client
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Ensure the index exists
def ensure_index_exists(index_name: str, dimension: int, metric: str = "cosine"):
    indexes = pc.list_indexes().names()
    if index_name not in indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD,
                region=config.PINECONE_REGION 
            )
        )

# Run this at module level or from startup logic
ensure_index_exists(config.PINECONE_INDEX_NAME, dimension=768)

# Get the index object to use for queries/upserts
index = pc.Index(config.PINECONE_INDEX_NAME)

def query_similar(embedding, threshold=0.90):
    try:
        result = index.query(vector=embedding, top_k=1, include_metadata=True)
        matches = result.get("matches", [])
        if matches and matches[0]["score"] >= threshold:
            return matches[0]["metadata"]["video_url"]
        return None
    except Exception as e:
        logger.error(f"Error querying vector database: {str(e)}")
        raise

def upsert_vector(id, embedding, prompt: str, video_url: str):
    metadata = {
        "prompt": prompt,
        "video_url": video_url
    }
    index.upsert(vectors=[{
        "id": id,
        "values": embedding,
        "metadata": metadata
    }])
