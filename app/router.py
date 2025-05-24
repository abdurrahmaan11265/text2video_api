from fastapi import APIRouter
from app.schemas import PromptRequest, VideoResponse
from app.services.embedder import get_embeddings
from app.services.vector_db import query_similar, upsert_vector
from app.services.video_generator import generate_video
from app.services.uploader import upload_video
from app import config
import uuid

router = APIRouter()

@router.post("/generate", response_model=VideoResponse)
def generate_videos(request: PromptRequest):
    prompts = request.prompts
    embeddings = get_embeddings(prompts)
    results = []

    for prompt, emb in zip(prompts, embeddings):
        print(f"Processing prompt: {prompt}")
        existing_url = query_similar(emb, config.SIMILARITY_THRESHOLD)
        if existing_url:
            results.append({"prompt": prompt, "video_url": existing_url})
            continue

        video_path = generate_video(prompt)
        video_url = upload_video(video_path)
        uid = str(uuid.uuid4())
        upsert_vector(uid, emb, prompt, video_url)
        results.append({"prompt": prompt, "video_url": video_url})

    return {"results": results}

@router.get("/")
def health_check():
    return {"status": "ok"}