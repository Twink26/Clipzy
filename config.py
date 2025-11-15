"""
Configuration file for Clipzy application
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Video processing settings
MAX_REEL_DURATION = int(os.getenv("MAX_REEL_DURATION", 60))  # seconds
MIN_REEL_DURATION = int(os.getenv("MIN_REEL_DURATION", 15))  # seconds
TARGET_ASPECT_RATIO = (9, 16)  # Vertical format for reels
TARGET_RESOLUTION = (1080, 1920)  # Full HD vertical

# Transcription settings
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
LANGUAGE = os.getenv("LANGUAGE", "en")

# NLP settings
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "viral_detector.pkl"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MIN_SEGMENT_SCORE = float(os.getenv("MIN_SEGMENT_SCORE", 0.6))  # Minimum score for viral detection

# Dataset settings
DATASET_PATH = os.getenv("DATASET_PATH", str(RAW_DATA_DIR / "viral_segments.csv"))
TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", 0.8))

# Caption settings
CAPTION_FONT_SIZE = int(os.getenv("CAPTION_FONT_SIZE", 60))
CAPTION_COLOR = os.getenv("CAPTION_COLOR", "white")
CAPTION_BACKGROUND = os.getenv("CAPTION_BACKGROUND", "black")
CAPTION_POSITION = os.getenv("CAPTION_POSITION", "bottom")  # top, center, bottom

# YouTube settings
DOWNLOAD_AUDIO_ONLY = os.getenv("DOWNLOAD_AUDIO_ONLY", "False").lower() == "true"
VIDEO_QUALITY = os.getenv("VIDEO_QUALITY", "best")

