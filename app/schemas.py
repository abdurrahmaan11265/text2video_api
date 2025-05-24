from pydantic import BaseModel
from typing import List, Dict

class PromptRequest(BaseModel):
    prompts: List[str]

class VideoResponse(BaseModel):
    results: List[Dict[str, str]]