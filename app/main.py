from fastapi import FastAPI
from app.router import router

app = FastAPI(title="Text-to-Video API")
app.include_router(router)