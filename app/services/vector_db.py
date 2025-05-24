import pinecone
from app import config

pinecone.init(api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENV)

# Create index if it doesn't exist
if config.PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=config.PINECONE_INDEX_NAME,
        dimension=768,  # for all-mpnet-base-v2
        metric="cosine"
    )

index = pinecone.Index(config.PINECONE_INDEX_NAME)

def query_similar(embedding: list[float], threshold: float):
    results = index.query(vector=embedding, top_k=1, include_metadata=True)
    if results.matches and results.matches[0].score >= threshold:
        return results.matches[0].metadata.get("video_url")
    return None

def upsert_vector(id_: str, embedding: list[float], prompt: str, video_url: str):
    index.upsert([
        (id_, embedding, {"prompt": prompt, "video_url": video_url})
    ])