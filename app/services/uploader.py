import cloudinary
import cloudinary.uploader
from app import config

cloudinary.config(
    cloud_name=config.CLOUDINARY_CLOUD_NAME,
    api_key=config.CLOUDINARY_API_KEY,
    api_secret=config.CLOUDINARY_API_SECRET,
    secure=True
)

def upload_video(path: str) -> str:
    response = cloudinary.uploader.upload_large(
        path,
        resource_type="video", 
        use_filename=True,
        unique_filename=True,
        overwrite=False
    )
    return response["secure_url"]