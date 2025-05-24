from fastapi import FastAPI
import os
import logging
from app.router import router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize FastAPI app
app = FastAPI(
    title="Text-to-Video API",
    description="API for generating videos from text prompts",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application...")

app.include_router(router)