"""
FastAPI backend for Clipzy Phase 6.
Expose endpoints to trigger reel generation from a YouTube URL.
"""
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl

from src.reel_generator import ReelGenerator
import config


class GenerateRequest(BaseModel):
    youtube_url: HttpUrl = Field(..., description="Full YouTube URL")
    num_segments: int = Field(3, ge=1, le=10)
    min_duration: int = Field(config.MIN_REEL_DURATION, ge=5, le=120)
    max_duration: int = Field(config.MAX_REEL_DURATION, ge=10, le=300)
    add_captions: bool = Field(True, description="Add captions to reels")


class ReelInfo(BaseModel):
    reel_path: str
    reel_url: str  # URL to access the video file
    filename: str  # Just the filename
    start_time: float
    end_time: float
    duration: float
    viral_score: float
    preview_text: str


class GenerateResponse(BaseModel):
    message: str
    count: int
    output_dir: str
    reels: List[ReelInfo]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Clipzy API",
        description="Generate social reels from YouTube podcasts.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Get output directory path
    output_path = config.OUTPUT_DIR.resolve()
    
    generator = ReelGenerator()

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.get("/clips/{filename:path}")
    async def get_clip(filename: str, download: bool = False):
        """Serve video clip files"""
        clip_path = output_path / filename
        if not clip_path.exists() or not clip_path.is_file():
            raise HTTPException(status_code=404, detail="Clip not found")
        
        # Check if file is within output directory (security)
        try:
            clip_path.resolve().relative_to(output_path.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        headers = {"Accept-Ranges": "bytes"}
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{clip_path.name}"'
        
        return FileResponse(
            path=str(clip_path),
            media_type="video/mp4",
            filename=clip_path.name if download else None,
            headers=headers
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate_reels(payload: GenerateRequest):
        if payload.min_duration >= payload.max_duration:
            raise HTTPException(
                status_code=400,
                detail="min_duration must be smaller than max_duration.",
            )

        try:
            reels = generator.generate_reels(
                youtube_url=str(payload.youtube_url),
                num_segments=payload.num_segments,
                min_duration=payload.min_duration,
                max_duration=payload.max_duration,
                add_captions=payload.add_captions,
            )

            if not reels:
                raise HTTPException(
                    status_code=400,
                    detail="No interesting segments found. Try adjusting parameters.",
                )

            reel_infos = []
            for reel in reels:
                reel_path = Path(reel["reel_path"])
                # Get relative path from output directory
                try:
                    relative_path = reel_path.relative_to(output_path)
                    reel_url = f"/clips/{relative_path.as_posix()}"
                    filename = reel_path.name
                except ValueError:
                    # If path is not in output dir, use absolute path as fallback
                    reel_url = f"/clips/{reel_path.name}"
                    filename = reel_path.name
                
                reel_infos.append(
                    ReelInfo(
                        reel_path=str(reel_path),
                        reel_url=reel_url,
                        filename=filename,
                        start_time=reel["start_time"],
                        end_time=reel["end_time"],
                        duration=reel["duration"],
                        viral_score=reel["viral_score"],
                        preview_text=reel["text"],
                    )
                )

            return GenerateResponse(
                message="Reels generated successfully",
                count=len(reel_infos),
                output_dir=str(config.OUTPUT_DIR.resolve()),
                reels=reel_infos,
            )

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate reels: {str(exc)}",
            ) from exc

    return app


app = create_app()


