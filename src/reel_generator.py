"""
Main reel generator module
Orchestrates the entire pipeline from YouTube URL to generated reels
"""
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from src.youtube_downloader import YouTubeDownloader
from src.transcriber import Transcriber
from src.nlp_analyzer import NLPAnalyzer
from src.video_editor import VideoEditor
from src.dataset_processor import DatasetProcessor

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReelGenerator:
    """Main class that orchestrates the reel generation pipeline"""

    def __init__(self):
        """Initialize reel generator with all required components"""
        logger.info("Initializing ReelGenerator...")

        self.youtube_downloader = YouTubeDownloader(
            output_dir=str(Path(config.DATA_DIR / "downloads"))
        )
        self.transcriber = Transcriber(model_name=config.WHISPER_MODEL)
        self.nlp_analyzer = NLPAnalyzer()
        self.video_editor = VideoEditor()
        self.dataset_processor = DatasetProcessor()

        # Try to load trained model
        try:
            if Path(config.MODEL_PATH).exists():
                self.nlp_analyzer.load_model()
                logger.info("Loaded existing trained model")
            else:
                logger.warning("No trained model found.")
        except Exception as e:
            logger.warning(f"Could not load model: {str(e)}")

    def download_and_transcribe(self, youtube_url: str) -> Dict:
        logger.info(f"Processing YouTube URL: {youtube_url}")

        # Step 1: Download
        logger.info("Step 1: Downloading video...")
        video_info = self.youtube_downloader.download_video(youtube_url, audio_only=False)
        video_path = video_info['file_path']

        # Step 2: Transcribe
        logger.info("Step 2: Transcribing audio...")
        transcription = self.transcriber.transcribe(video_path, language=config.LANGUAGE)

        result = {
            'video_info': video_info,
            'video_path': video_path,
            'transcription': transcription
        }

        logger.info(f"Transcription complete: {len(transcription['segments'])} segments")
        return result

    def detect_interesting_segments(
        self,
        transcription: Dict,
        num_segments: int = 5,
        min_duration: float = None,
        max_duration: float = None
    ) -> List[Dict]:

        min_duration = min_duration or config.MIN_REEL_DURATION
        max_duration = max_duration or config.MAX_REEL_DURATION

        logger.info(f"Detecting interesting segments (min: {min_duration}s, max: {max_duration}s)...")

        segments = []

        # FIX 1: Smaller window → more candidate chunks
        window_size = 15  # originally 30
        overlap = 5

        video_duration = transcription.get("duration", 0)
        current_time = 0

        while current_time < video_duration:
            end_time = min(current_time + window_size, video_duration)
            duration = end_time - current_time

            if duration < min_duration:
                current_time += window_size - overlap
                continue

            # Extract text for this window
            segment_text = self.transcriber.get_text_in_range(transcription, current_time, end_time)
            if not segment_text.strip():
                current_time += window_size - overlap
                continue

            # Score using NLP model
            try:
                prediction = self.nlp_analyzer.predict_segment(
                    segment_text,
                    duration=duration,
                    dataset_processor=self.dataset_processor
                )
                viral_score = prediction["viral_score"]
            except Exception as e:
                logger.warning(f"Error scoring segment at {current_time}s: {e}")
                viral_score = 0.0

            segments.append({
                "start_time": current_time,
                "end_time": end_time,
                "duration": duration,
                "text": segment_text,
                "viral_score": viral_score
            })

            current_time += window_size - overlap

        # Duration filtering
        segments = [
            s for s in segments
            if min_duration <= s["duration"] <= max_duration
        ]

        # FIX 2: Add viral score threshold (most important fix)
        segments = [s for s in segments if s["viral_score"] >= 0.40]

        if not segments:
            logger.warning("NO segments passed viral-score threshold (0.40).")
        else:
            logger.info(f"{len(segments)} segments passed viral-score threshold.")

        # Sort by score
        segments.sort(key=lambda x: x["viral_score"], reverse=True)

        # Top N
        top_segments = segments[:num_segments]

        logger.info(f"Returning {len(top_segments)} segments.")
        for s in top_segments:
            logger.info(
                f"Segment {s['start_time']:.1f}-{s['end_time']:.1f}s | "
                f"Score: {s['viral_score']:.3f}"
            )

        return top_segments

    def generate_reels(
        self,
        youtube_url: str,
        num_segments: int = 5,
        min_duration: float = None,
        max_duration: float = None,
        add_captions: bool = True
    ) -> List[Dict]:

        logger.info("=" * 60)
        logger.info("Starting Reel Generation Pipeline")
        logger.info("=" * 60)

        min_duration = min_duration or config.MIN_REEL_DURATION
        max_duration = max_duration or config.MAX_REEL_DURATION

        try:
            # Step 1: Download + transcribe
            video_data = self.download_and_transcribe(youtube_url)
            video_path = video_data["video_path"]
            transcription = video_data["transcription"]

            # Step 2: Detect segments
            segments = self.detect_interesting_segments(
                transcription,
                num_segments=num_segments,
                min_duration=min_duration,
                max_duration=max_duration
            )

            if not segments:
                logger.error("No interesting segments found!")
                return []

            # Step 3: Generate reels
            logger.info(f"Generating {len(segments)} reels...")
            reels = []

            for seg in tqdm(segments, desc="Creating reels"):
                reel_path = self.video_editor.create_reel(
                    video_path=video_path,
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    captions=[] if not add_captions else None  # captions handled inside editor
                )

                reels.append({
                    "reel_path": reel_path,
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "duration": seg["duration"],
                    "viral_score": seg["viral_score"],
                    "text": seg["text"],
                })

            logger.info("Reel generation complete.")
            return reels

        except Exception as e:
            logger.error(f"Error in reel generation: {e}")
            raise
