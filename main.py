"""
Main entry point for Clipzy application
Command-line interface for generating reels from YouTube videos
"""
import argparse
import sys
import logging
from pathlib import Path

from src.reel_generator import ReelGenerator
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Clipzy - AI-Powered YouTube Podcast to Reels Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 reels from a YouTube video
  python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --segments 5
  
  # Generate reels with custom duration limits
  python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --segments 3 --min-duration 20 --max-duration 45
        """
    )
    
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='YouTube video URL'
    )
    
    parser.add_argument(
        '--segments',
        type=int,
        default=5,
        help='Number of reels to generate (default: 5)'
    )
    
    parser.add_argument(
        '--min-duration',
        type=float,
        default=None,
        help=f'Minimum reel duration in seconds (default: {config.MIN_REEL_DURATION})'
    )
    
    parser.add_argument(
        '--max-duration',
        type=float,
        default=None,
        help=f'Maximum reel duration in seconds (default: {config.MAX_REEL_DURATION})'
    )
    
    parser.add_argument(
        '--no-captions',
        action='store_true',
        help='Skip adding captions to reels'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for reels (default: output/)'
    )
    
    args = parser.parse_args()
    
    # Validate URL
    if 'youtube.com' not in args.url and 'youtu.be' not in args.url:
        logger.error("Invalid YouTube URL. Please provide a valid YouTube video URL.")
        sys.exit(1)
    
    # Set output directory if provided
    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not Path(config.MODEL_PATH).exists():
        logger.warning("=" * 60)
        logger.warning("WARNING: No trained model found!")
        logger.warning("=" * 60)
        logger.warning("The application requires a trained model to detect viral segments.")
        logger.warning("Please:")
        logger.warning("1. Collect a dataset of viral/not-viral segments")
        logger.warning("2. Run dataset_processor.py to process the dataset")
        logger.warning("3. Train the model using nlp_analyzer.py")
        logger.warning("=" * 60)
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Exiting...")
            sys.exit(0)
    
    try:
        # Initialize reel generator
        logger.info("Initializing Clipzy...")
        generator = ReelGenerator()
        
        # Generate reels
        reels = generator.generate_reels(
            youtube_url=args.url,
            num_segments=args.segments,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            add_captions=not args.no_captions
        )
        
        if reels:
            logger.info("\n" + "=" * 60)
            logger.info("SUCCESS! Reels generated:")
            logger.info("=" * 60)
            for i, reel in enumerate(reels, 1):
                logger.info(f"\nReel {i}:")
                logger.info(f"  Path: {reel['reel_path']}")
                logger.info(f"  Time: {reel['start_time']:.1f}s - {reel['end_time']:.1f}s")
                logger.info(f"  Duration: {reel['duration']:.1f}s")
                logger.info(f"  Viral Score: {reel['viral_score']:.3f}")
                logger.info(f"  Preview: {reel['text']}")
            logger.info("\n" + "=" * 60)
        else:
            logger.warning("No reels were generated. Try adjusting parameters or check the video content.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        logger.exception("Full error details:")
        sys.exit(1)


if __name__ == '__main__':
    main()

