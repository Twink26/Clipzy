"""
Speech-to-text transcription module using Whisper
"""
import whisper
import json
from typing import List, Dict, Optional
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
        # ---------------------------------------
        # FIX: Inject FFmpeg paths (important!)
        # ---------------------------------------
        import os

        ffmpeg_path = os.getenv("FFMPEG_PATH")
        ffprobe_path = os.getenv("FFPROBE_PATH")

        if ffmpeg_path and os.path.exists(ffmpeg_path):
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            logger.info(f"FFmpeg found and added to PATH: {ffmpeg_path}")
        else:
            logger.warning("FFMPEG_PATH not set or invalid — Whisper may fail.")

        if ffprobe_path and os.path.exists(ffprobe_path):
            ffprobe_dir = os.path.dirname(ffprobe_path)
            os.environ["PATH"] = ffprobe_dir + os.pathsep + os.environ.get("PATH", "")
            logger.info(f"FFprobe found and added to PATH: {ffprobe_path}")
        else:
            logger.warning("FFPROBE_PATH not set or invalid — Whisper may fail.")

        # ---------------------------------------
        # Load Whisper Model (after FFmpeg patch)
        # ---------------------------------------
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
    
    def save_transcript(self, transcription: Dict, output_path: str, format: str = "json") -> str:
        """
        Save transcription to file
        
        Args:
            transcription: Transcription dictionary
            output_path: Path to save transcript. If directory, generates filename
            format: Format to save ('json', 'txt', 'srt', 'vtt')
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        
        # If directory, generate filename
        if output_path.is_dir() or not output_path.suffix:
            video_name = "transcript"
            ext_map = {
                'json': '.json',
                'txt': '.txt',
                'srt': '.srt',
                'vtt': '.vtt'
            }
            output_path = output_path / f"{video_name}{ext_map.get(format, '.json')}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, indent=2, ensure_ascii=False)
        
        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Language: {transcription.get('language', 'unknown')}\n")
                f.write(f"Duration: {transcription.get('duration', 0):.2f} seconds\n")
                f.write(f"Segments: {len(transcription.get('segments', []))}\n")
                f.write("\n" + "="*60 + "\n\n")
                for segment in transcription.get('segments', []):
                    start = segment['start']
                    end = segment['end']
                    text = segment['text']
                    f.write(f"[{start:.2f}s - {end:.2f}s]\n{text}\n\n")
        
        elif format == "srt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, segment in enumerate(transcription.get('segments', []), 1):
                    start = self._format_timestamp(segment['start'])
                    end = self._format_timestamp(segment['end'])
                    text = segment['text']
                    f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
        
        elif format == "vtt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                for segment in transcription.get('segments', []):
                    start = self._format_timestamp_vtt(segment['start'])
                    end = self._format_timestamp_vtt(segment['end'])
                    text = segment['text']
                    f.write(f"{start} --> {end}\n{text}\n\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json', 'txt', 'srt', or 'vtt'")
        
        logger.info(f"Transcript saved to: {output_path} (format: {format})")
        return str(output_path)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Format timestamp for VTT format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
