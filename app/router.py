from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.schemas import PromptRequest, VideoResponse
from app.services.embedder import get_embeddings
from app.services.vector_db import query_similar, upsert_vector
from app.services.video_generator import generate_video
from app.services.uploader import upload_video
from app import config
import uuid
import logging
from typing import Dict
from fastapi.responses import JSONResponse
import threading  # use thread-safe lock

logger = logging.getLogger(__name__)
router = APIRouter()

# Temporary in-memory task store
task_status: Dict[str, Dict] = {}

# Thread lock to prevent concurrent GPU access
gpu_lock = threading.Lock()

def process_prompt(task_id: str, prompt: str, emb: list):
    try:
        logger.info(f"Processing prompt for task {task_id}: {prompt}")

        existing_url = query_similar(emb, config.SIMILARITY_THRESHOLD)
        if existing_url:
            logger.info(f"Found existing video for task {task_id}")
            task_status[task_id] = {"status": "done", "video_url": existing_url, "prompt": prompt}
            return

        # Block other threads from using GPU
        with gpu_lock:
            logger.info(f"Acquired GPU lock for task {task_id}")
            video_path = generate_video(prompt)

        video_url = upload_video(video_path)
        upsert_vector(task_id, emb, prompt, video_url)
        task_status[task_id] = {"status": "done", "video_url": video_url, "prompt": prompt}
        logger.info(f"Successfully processed task {task_id}")
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        task_status[task_id] = {"status": "error", "message": str(e)}

@router.post("/generate", status_code=202)
async def generate_videos(request: PromptRequest, background_tasks: BackgroundTasks):
    try:
        prompts = request.prompts
        if not prompts:
            raise HTTPException(status_code=400, content={"message": "No prompts provided"})

        embeddings = get_embeddings(prompts)
        task_ids = []

        for prompt, emb in zip(prompts, embeddings):
            task_id = str(uuid.uuid4())
            task_status[task_id] = {"status": "processing", "prompt": prompt}
            background_tasks.add_task(process_prompt, task_id, prompt, emb)
            task_ids.append({"prompt": prompt, "task_id": task_id})

        return {"tasks": task_ids}
    except Exception as e:
        logger.error(f"Error generating videos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}", response_model=VideoResponse)
async def check_status(task_id: str):
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")

    status = task_status[task_id]
    if status["status"] == "done":
        return {
            "results": [{
                "prompt": status.get("prompt", "unknown"),
                "video_url": status["video_url"]
            }],
            "status": "done"
        }
    elif status["status"] == "error":
        raise HTTPException(status_code=500, detail=status["message"])
    else:
        return {"status": "processing"}

@router.get("/")
async def health_check():
    return {"status": "ok"}
