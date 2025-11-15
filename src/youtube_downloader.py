"""
YouTube video downloader module
"""
import os
import yt_dlp
from pathlib import Path
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """Downloads YouTube videos and extracts metadata"""
    
    def __init__(self, output_dir: str = "data/downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_video(self, url: str, audio_only: bool = False) -> Dict[str, any]:
        """
        Download video or audio from YouTube
        
        Args:
            url: YouTube video URL
            audio_only: If True, download only audio
            
        Returns:
            Dictionary with file path and metadata
        """
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
                'format': 'bestaudio/best' if audio_only else 'best',
            }
            
            if audio_only:
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }]
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info
                info = ydl.extract_info(url, download=True)
                
                # Get downloaded file path
                filename = ydl.prepare_filename(info)
                if audio_only:
                    filename = filename.rsplit('.', 1)[0] + '.mp3'
                
                result = {
                    'file_path': filename,
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', ''),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'url': url,
                    'video_id': info.get('id', ''),
                }
                
                logger.info(f"Successfully downloaded: {result['title']}")
                return result
                
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise
    
    def get_video_info(self, url: str) -> Dict[str, any]:
        """
        Get video information without downloading
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary with video metadata
        """
        try:
            ydl_opts = {'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', ''),
                    'view_count': info.get('view_count', 0),
                    'video_id': info.get('id', ''),
                }
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            raise

