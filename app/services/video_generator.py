import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
import uuid
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
scheduler = UniPCMultistepScheduler(
    prediction_type='flow_prediction',
    use_flow_sigmas=True,
    num_train_timesteps=1000,
    flow_shift=5.0
)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = scheduler
pipe.to("cuda")

NEGATIVE_PROMPT = (
    "low resolution, blurry, distorted face, deformed hands, extra limbs, bad anatomy, "
    "text, watermark, logo, cartoonish, low quality, grainy, glitch, poorly rendered environment, "
    "unnatural lighting, bad shadows, out of frame, cropped, low detail"
)


def generate_video(prompt: str, out_path: str = None) -> str:
    if not out_path:
        out_path = f"/tmp/{uuid.uuid4()}.mp4"

    # Generate initial video
    temp_path = f"/tmp/temp_{uuid.uuid4()}.mp4"

    output = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=512,
        width=768,
        num_frames=24,
        guidance_scale=5.0,
    ).frames[0]

    # Save initial video
    export_to_video(output, temp_path, fps=16)

    # Load the video and concatenate it with itself
    video = VideoFileClip(temp_path)
    final_video = concatenate_videoclips([video, video])

    # Save the doubled video
    final_video.write_videofile(out_path, codec="libx264")

    # Clean up temporary file
    video.close()
    final_video.close()
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return out_path
