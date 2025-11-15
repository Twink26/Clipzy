"""
Main reel generator module
Orchestrates the entire pipeline from YouTube URL to generated reels
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional
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
        
        # Try to load trained model if it exists
        try:
            if Path(config.MODEL_PATH).exists():
                self.nlp_analyzer.load_model()
                logger.info("Loaded existing trained model")
            else:
                logger.warning("No trained model found. Model will need to be trained first.")
        except Exception as e:
            logger.warning(f"Could not load model: {str(e)}")
    
    def download_and_transcribe(self, youtube_url: str) -> Dict:
        """
        Download YouTube video and transcribe it
        
        Args:
            youtube_url: YouTube video URL
            
        Returns:
            Dictionary with video info and transcription
        """
        logger.info(f"Processing YouTube URL: {youtube_url}")
        
        # Download video
        logger.info("Step 1: Downloading video...")
        video_info = self.youtube_downloader.download_video(youtube_url, audio_only=False)
        video_path = video_info['file_path']
        
        # Transcribe
        logger.info("Step 2: Transcribing audio...")
        transcription = self.transcriber.transcribe(video_path, language=config.LANGUAGE)
        
        result = {
            'video_info': video_info,
            'video_path': video_path,
            'transcription': transcription
        }
        
        logger.info(f"Transcription complete: {len(transcription['segments'])} segments")
        return result
    
    def detect_interesting_segments(self, transcription: Dict, 
                                   num_segments: int = 5,
                                   min_duration: float = None,
                                   max_duration: float = None) -> List[Dict]:
        """
        Detect interesting/viral segments from transcription
        
        Args:
            transcription: Transcription dictionary with segments
            num_segments: Number of top segments to return
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            
        Returns:
            List of segment dictionaries with scores and timestamps
        """
        min_duration = min_duration or config.MIN_REEL_DURATION
        max_duration = max_duration or config.MAX_REEL_DURATION
        
        logger.info(f"Detecting interesting segments (min: {min_duration}s, max: {max_duration}s)...")
        
        # Split transcription into candidate segments
        segments = []
        window_size = 30  # seconds
        overlap = 5  # seconds
        
        current_time = 0
        video_duration = transcription.get('duration', 0)
        
        while current_time < video_duration:
            end_time = min(current_time + window_size, video_duration)
            duration = end_time - current_time
            
            # Skip if too short
            if duration < min_duration:
                current_time += window_size - overlap
                continue
            
            # Get text for this segment
            segment_text = self.transcriber.get_text_in_range(
                transcription, current_time, end_time
            )
            
            if not segment_text.strip():
                current_time += window_size - overlap
                continue
            
            # Score segment
            try:
                prediction = self.nlp_analyzer.predict_segment(
                    segment_text, 
                    duration=duration,
                    dataset_processor=self.dataset_processor
                )
                viral_score = prediction['viral_score']
            except Exception as e:
                logger.warning(f"Error scoring segment at {current_time}s: {str(e)}")
                viral_score = 0.0
            
            segments.append({
                'start_time': current_time,
                'end_time': end_time,
                'duration': duration,
                'text': segment_text,
                'viral_score': viral_score
            })
            
            current_time += window_size - overlap
        
        # Filter by duration
        segments = [s for s in segments 
                   if min_duration <= s['duration'] <= max_duration]
        
        # Sort by viral score
        segments.sort(key=lambda x: x['viral_score'], reverse=True)
        
        # Take top N
        top_segments = segments[:num_segments]
        
        logger.info(f"Found {len(top_segments)} interesting segments")
        for i, seg in enumerate(top_segments, 1):
            logger.info(f"  Segment {i}: {seg['start_time']:.1f}s-{seg['end_time']:.1f}s "
                       f"(score: {seg['viral_score']:.3f})")
        
        return top_segments
    
    def create_captions_from_segments(self, segments: List[Dict], 
                                     transcription: Dict) -> List[Dict]:
        """
        Create caption data from transcription segments
        
        Args:
            segments: List of detected segments
            transcription: Full transcription with word timestamps
            
        Returns:
            List of caption dictionaries
        """
        captions_list = []
        
        for segment in segments:
            start_time = segment['start_time']
            end_time = segment['end_time']
            
            # Get word-level timestamps for this segment
            segment_captions = []
            for trans_seg in transcription['segments']:
                if (trans_seg['start'] >= start_time and 
                    trans_seg['end'] <= end_time):
                    
                    # Create caption for this sub-segment
                    caption_text = trans_seg['text'].strip()
                    if caption_text:
                        segment_captions.append({
                            'text': caption_text,
                            'start': trans_seg['start'],
                            'end': trans_seg['end']
                        })
            
            captions_list.append(segment_captions)
        
        return captions_list
    
    def generate_reels(self, youtube_url: str, 
                      num_segments: int = 5,
                      min_duration: float = None,
                      max_duration: float = None,
                      add_captions: bool = True) -> List[Dict]:
        """
        Main method to generate reels from YouTube URL
        
        Args:
            youtube_url: YouTube video URL
            num_segments: Number of reels to generate
            min_duration: Minimum reel duration in seconds
            max_duration: Maximum reel duration in seconds
            add_captions: Whether to add captions to reels
            
        Returns:
            List of dictionaries with reel information
        """
        logger.info("=" * 60)
        logger.info("Starting Reel Generation Pipeline")
        logger.info("=" * 60)
        
        min_duration = min_duration or config.MIN_REEL_DURATION
        max_duration = max_duration or config.MAX_REEL_DURATION
        
        try:
            # Step 1: Download and transcribe
            video_data = self.download_and_transcribe(youtube_url)
            video_path = video_data['video_path']
            transcription = video_data['transcription']
            
            # Step 2: Detect interesting segments
            segments = self.detect_interesting_segments(
                transcription,
                num_segments=num_segments,
                min_duration=min_duration,
                max_duration=max_duration
            )
            
            if not segments:
                logger.warning("No interesting segments found!")
                return []
            
            # Step 3: Generate reels
            logger.info(f"Step 3: Generating {len(segments)} reels...")
            reels = []
            
            for i, segment in enumerate(tqdm(segments, desc="Creating reels"), 1):
                logger.info(f"\nCreating reel {i}/{len(segments)}...")
                
                # Prepare captions
                captions = None
                if add_captions:
                    # Get captions for this segment
                    segment_captions = []
                    for trans_seg in transcription['segments']:
                        if (trans_seg['start'] >= segment['start_time'] and 
                            trans_seg['end'] <= segment['end_time']):
                            segment_captions.append({
                                'text': trans_seg['text'].strip(),
                                'start': trans_seg['start'],
                                'end': trans_seg['end']
                            })
                    captions = segment_captions
                
                # Create reel
                reel_path = self.video_editor.create_reel(
                    video_path=video_path,
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    captions=captions
                )
                
                reels.append({
                    'reel_path': reel_path,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'duration': segment['duration'],
                    'viral_score': segment['viral_score'],
                    'text': segment['text'][:100] + "..." if len(segment['text']) > 100 else segment['text']
                })
            
            logger.info("=" * 60)
            logger.info("Reel Generation Complete!")
            logger.info("=" * 60)
            logger.info(f"Generated {len(reels)} reels:")
            for i, reel in enumerate(reels, 1):
                logger.info(f"  {i}. {reel['reel_path']} "
                           f"({reel['start_time']:.1f}s-{reel['end_time']:.1f}s, "
                           f"score: {reel['viral_score']:.3f})")
            
            return reels
            
        except Exception as e:
            logger.error(f"Error in reel generation: {str(e)}")
            raise

