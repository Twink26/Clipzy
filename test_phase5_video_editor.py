"""
Phase 5 Test Script - Video Editing & Reel Generation
Allows manual testing of VideoEditor functionality with custom parameters.
"""
import argparse
import json
import logging
from pathlib import Path

from src.video_editor import VideoEditor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_captions(captions_path: str):
    """Load captions from JSON file."""
    path = Path(captions_path)
    if not path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Captions file must contain a list of caption objects.")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Test Phase 5 video editing pipeline with a local video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to source video file.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds for the reel segment.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=30.0,
        help="End time in seconds for the reel segment.",
    )
    parser.add_argument(
        "--captions-file",
        type=str,
        help="Optional path to JSON file containing captions.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output path for the generated reel.",
    )
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Skip adding captions even if a captions file is provided.",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    captions = None
    if not args.no_captions and args.captions_file:
        captions = load_captions(args.captions_file)

    editor = VideoEditor()
    reel_path = editor.create_reel(
        video_path=str(video_path),
        start_time=args.start,
        end_time=args.end,
        captions=captions,
        output_path=args.output,
    )

    logger.info("Reel generated successfully.")
    logger.info(f"Output file: {reel_path}")
    logger.info("Review the video to verify trimming, aspect ratio, captions, and branding.")


if __name__ == "__main__":
    main()

