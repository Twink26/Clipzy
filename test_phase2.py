"""
Test script for Phase 2: YouTube Video Processing
Tests YouTube downloader and transcription functionality
"""
import sys
import logging
from pathlib import Path

from src.youtube_downloader import YouTubeDownloader
from src.transcriber import Transcriber
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_youtube_downloader(url: str):
    """Test YouTube downloader"""
    logger.info("=" * 60)
    logger.info("Testing YouTube Downloader")
    logger.info("=" * 60)
    
    try:
        downloader = YouTubeDownloader(
            output_dir=str(config.DATA_DIR / "downloads")
        )
        
        # Test getting video info without downloading
        logger.info("Step 1: Getting video information...")
        info = downloader.get_video_info(url)
        logger.info(f"Video Title: {info['title']}")
        logger.info(f"Duration: {info['duration']} seconds ({info['duration']/60:.2f} minutes)")
        logger.info(f"Uploader: {info['uploader']}")
        logger.info(f"Views: {info.get('view_count', 'N/A')}")
        
        # Download video
        logger.info("\nStep 2: Downloading video...")
        result = downloader.download_video(url, audio_only=False)
        logger.info(f"Downloaded to: {result['file_path']}")
        logger.info(f"File exists: {Path(result['file_path']).exists()}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in YouTube downloader test: {str(e)}")
        raise


def test_transcriber(video_path: str):
    """Test transcription"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Transcriber")
    logger.info("=" * 60)
    
    try:
        transcriber = Transcriber(model_name=config.WHISPER_MODEL)
        
        # Transcribe
        logger.info("Transcribing video...")
        transcription = transcriber.transcribe(video_path, language=config.LANGUAGE)
        
        logger.info(f"\nTranscription Summary:")
        logger.info(f"  Language: {transcription['language']}")
        logger.info(f"  Duration: {transcription['duration']:.2f} seconds")
        logger.info(f"  Total Segments: {len(transcription['segments'])}")
        logger.info(f"  Full Text Length: {len(transcription['text'])} characters")
        
        # Show first few segments
        logger.info(f"\nFirst 3 segments:")
        for i, seg in enumerate(transcription['segments'][:3], 1):
            logger.info(f"  {i}. [{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text'][:80]}...")
        
        # Save transcript in multiple formats
        logger.info("\nSaving transcripts...")
        output_dir = config.DATA_DIR / "transcripts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = transcriber.save_transcript(transcription, output_dir, format="json")
        txt_path = transcriber.save_transcript(transcription, output_dir, format="txt")
        srt_path = transcriber.save_transcript(transcription, output_dir, format="srt")
        
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  TXT: {txt_path}")
        logger.info(f"  SRT: {srt_path}")
        
        # Test segment extraction
        logger.info("\nTesting segment extraction...")
        test_start = 10.0
        test_end = 30.0
        segments = transcriber.get_segments_in_range(transcription, test_start, test_end)
        text = transcriber.get_text_in_range(transcription, test_start, test_end)
        
        logger.info(f"  Segments in range [{test_start}s - {test_end}s]: {len(segments)}")
        logger.info(f"  Combined text: {text[:100]}...")
        
        return transcription
        
    except Exception as e:
        logger.error(f"Error in transcriber test: {str(e)}")
        raise


def test_integration(url: str):
    """Test full integration of downloader and transcriber"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Full Integration")
    logger.info("=" * 60)
    
    try:
        # Download
        downloader = YouTubeDownloader(
            output_dir=str(config.DATA_DIR / "downloads")
        )
        video_info = downloader.download_video(url, audio_only=False)
        video_path = video_info['file_path']
        
        # Transcribe
        transcriber = Transcriber(model_name=config.WHISPER_MODEL)
        transcription = transcriber.transcribe(video_path, language=config.LANGUAGE)
        
        # Save transcript
        output_dir = config.DATA_DIR / "transcripts"
        transcript_path = transcriber.save_transcript(
            transcription, 
            output_dir / f"{Path(video_path).stem}.json",
            format="json"
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("Integration Test Complete!")
        logger.info("=" * 60)
        logger.info(f"Video: {video_path}")
        logger.info(f"Transcript: {transcript_path}")
        logger.info(f"Segments: {len(transcription['segments'])}")
        
        return {
            'video_path': video_path,
            'transcript_path': transcript_path,
            'transcription': transcription
        }
        
    except Exception as e:
        logger.error(f"Error in integration test: {str(e)}")
        raise


def main():
    """Main test function"""
    if len(sys.argv) < 2:
        logger.error("Usage: python test_phase2.py <youtube_url>")
        logger.info("\nExample:")
        logger.info('  python test_phase2.py "https://www.youtube.com/watch?v=VIDEO_ID"')
        sys.exit(1)
    
    url = sys.argv[1]
    
    # Validate URL
    if 'youtube.com' not in url and 'youtu.be' not in url:
        logger.error("Invalid YouTube URL")
        sys.exit(1)
    
    try:
        # Run tests
        logger.info("Starting Phase 2 Tests...")
        logger.info(f"URL: {url}\n")
        
        # Test 1: YouTube Downloader
        video_result = test_youtube_downloader(url)
        
        # Test 2: Transcriber
        transcription = test_transcriber(video_result['file_path'])
        
        # Test 3: Full Integration
        integration_result = test_integration(url)
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED! ✅")
        logger.info("=" * 60)
        logger.info("\nPhase 2 is working correctly!")
        logger.info("You can now proceed to Phase 3 (Dataset Processing)")
        
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTest failed: {str(e)}")
        logger.exception("Full error details:")
        sys.exit(1)


if __name__ == '__main__':
    main()

