"""
FastAPI backend for Clipzy Phase 6.
Expose endpoints to trigger reel generation from a YouTube URL.
"""
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

    generator = ReelGenerator()

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

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

            reel_infos = [
                ReelInfo(
                    reel_path=reel["reel_path"],
                    start_time=reel["start_time"],
                    end_time=reel["end_time"],
                    duration=reel["duration"],
                    viral_score=reel["viral_score"],
                    preview_text=reel["text"],
                )
                for reel in reels
            ]

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


