from pydantic import BaseModel
from typing import List, Dict, Optional

class PromptRequest(BaseModel):
    prompts: List[str]

class VideoResult(BaseModel):
    prompt: str
    video_url: str

class VideoResponse(BaseModel):
    results: Optional[List[VideoResult]] = None
    status: Optional[str] = None