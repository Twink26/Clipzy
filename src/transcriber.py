"""
Speech-to-text transcription module using Whisper
"""
import whisper
from typing import List, Dict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribes audio/video to text with timestamps"""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize Whisper model
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        logger.info("Model loaded successfully")
    
    def transcribe(self, audio_path: str, language: str = "en") -> Dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio/video file
            language: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Dictionary with transcription and segments
        """
        try:
            logger.info(f"Transcribing: {audio_path}")
            result = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True
            )
            
            # Process segments with timestamps
            segments = []
            for segment in result['segments']:
                segments.append({
                    'id': segment['id'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'words': segment.get('words', [])
                })
            
            transcription = {
                'text': result['text'],
                'language': result['language'],
                'segments': segments,
                'duration': result.get('duration', 0)
            }
            
            logger.info(f"Transcription complete. Found {len(segments)} segments")
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise
    
    def get_segments_in_range(self, transcription: Dict, start_time: float, end_time: float) -> List[Dict]:
        """
        Get transcription segments within a time range
        
        Args:
            transcription: Full transcription dictionary
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of segments within the time range
        """
        segments = []
        for segment in transcription['segments']:
            if segment['start'] >= start_time and segment['end'] <= end_time:
                segments.append(segment)
        return segments
    
    def get_text_in_range(self, transcription: Dict, start_time: float, end_time: float) -> str:
        """
        Get combined text for segments within a time range
        
        Args:
            transcription: Full transcription dictionary
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Combined text string
        """
        segments = self.get_segments_in_range(transcription, start_time, end_time)
        return ' '.join([seg['text'] for seg in segments])

