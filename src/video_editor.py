"""
Video editing module for Clipzy
Handles video trimming, resizing, and caption overlay for reel generation
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx import resize
import numpy as np

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoEditor:
    """Handles video editing operations for reel generation"""
    
    def __init__(self):
        """Initialize video editor"""
        pass
    
    def trim_video(self, video_path: str, start_time: float, end_time: float, 
                   output_path: str = None) -> str:
        """
        Trim video segment from start to end time
        
        Args:
            video_path: Path to input video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save trimmed video. If None, creates temp file
            
        Returns:
            Path to trimmed video file
        """
        logger.info(f"Trimming video: {start_time}s to {end_time}s")
        
        if output_path is None:
            output_path = str(Path(video_path).parent / f"trimmed_{start_time}_{end_time}.mp4")
        
        try:
            video = VideoFileClip(video_path)
            trimmed = video.subclip(start_time, end_time)
            trimmed.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            trimmed.close()
            video.close()
            
            logger.info(f"Trimmed video saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error trimming video: {str(e)}")
            raise
    
    def resize_to_reel(self, video_path: str, output_path: str = None) -> str:
        """
        Resize video to reel format (9:16 aspect ratio)
        
        Args:
            video_path: Path to input video
            output_path: Path to save resized video. If None, overwrites input
            
        Returns:
            Path to resized video file
        """
        logger.info("Resizing video to 9:16 aspect ratio...")
        
        if output_path is None:
            output_path = video_path
        
        target_width, target_height = config.TARGET_RESOLUTION
        
        try:
            video = VideoFileClip(video_path)
            original_width, original_height = video.size
            
            # Calculate aspect ratios
            target_aspect = target_width / target_height
            original_aspect = original_width / original_height
            
            # Crop and resize
            if original_aspect > target_aspect:
                # Video is wider - crop width
                new_width = int(original_height * target_aspect)
                x_center = original_width / 2
                x1 = int(x_center - new_width / 2)
                x2 = int(x_center + new_width / 2)
                cropped = video.crop(x1=x1, y1=0, x2=x2, y2=original_height)
            else:
                # Video is taller - crop height
                new_height = int(original_width / target_aspect)
                y_center = original_height / 2
                y1 = int(y_center - new_height / 2)
                y2 = int(y_center + new_height / 2)
                cropped = video.crop(x1=0, y1=y1, x2=original_width, y2=y2)
            
            # Resize to target resolution
            resized = cropped.resize((target_width, target_height))
            
            resized.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            resized.close()
            cropped.close()
            video.close()
            
            logger.info(f"Resized video saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error resizing video: {str(e)}")
            raise
    
    def create_caption_clip(self, text: str, duration: float, 
                           start_time: float = 0, position: str = None) -> TextClip:
        """
        Create a text caption clip
        
        Args:
            text: Caption text
            duration: Duration of caption in seconds
            start_time: Start time in video
            position: Position ('top', 'center', 'bottom'). If None, uses config
            
        Returns:
            TextClip object
        """
        position = position or config.CAPTION_POSITION
        
        # Determine vertical position
        target_height = config.TARGET_RESOLUTION[1]
        if position == 'top':
            y_pos = 100
        elif position == 'center':
            y_pos = target_height / 2
        else:  # bottom
            y_pos = target_height - 150
        
        # Create text clip
        txt_clip = TextClip(
            text,
            fontsize=config.CAPTION_FONT_SIZE,
            color=config.CAPTION_COLOR,
            font='Arial-Bold',
            stroke_color=config.CAPTION_BACKGROUND,
            stroke_width=2,
            method='caption',
            size=(config.TARGET_RESOLUTION[0] - 100, None),
            align='center'
        ).set_duration(duration).set_start(start_time).set_position(('center', y_pos))
        
        return txt_clip
    
    def add_captions(self, video_path: str, captions: List[Dict], 
                    output_path: str = None) -> str:
        """
        Add captions to video
        
        Args:
            video_path: Path to input video
            captions: List of caption dicts with 'text', 'start', 'end' keys
            output_path: Path to save video with captions. If None, overwrites input
            
        Returns:
            Path to video with captions
        """
        logger.info(f"Adding {len(captions)} captions to video...")
        
        if output_path is None:
            output_path = video_path
        
        try:
            video = VideoFileClip(video_path)
            
            # Create caption clips
            caption_clips = []
            for caption in captions:
                text = caption.get('text', '')
                start = caption.get('start', 0)
                end = caption.get('end', start + 2)
                duration = end - start
                position = caption.get('position', None)
                
                if text:
                    txt_clip = self.create_caption_clip(text, duration, start, position)
                    caption_clips.append(txt_clip)
            
            # Composite video with captions
            final_video = CompositeVideoClip([video] + caption_clips)
            
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            final_video.close()
            video.close()
            for clip in caption_clips:
                clip.close()
            
            logger.info(f"Video with captions saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding captions: {str(e)}")
            raise
    
    def create_reel(self, video_path: str, start_time: float, end_time: float,
                   captions: List[Dict] = None, output_path: str = None) -> str:
        """
        Create a complete reel: trim, resize, and add captions
        
        Args:
            video_path: Path to input video
            start_time: Start time in seconds
            end_time: End time in seconds
            captions: List of caption dicts (optional)
            output_path: Path to save final reel. If None, generates filename
            
        Returns:
            Path to final reel file
        """
        logger.info(f"Creating reel from {start_time}s to {end_time}s")
        
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = str(config.OUTPUT_DIR / f"reel_{video_name}_{start_time}_{end_time}.mp4")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Trim video
            temp_trimmed = str(Path(output_path).parent / f"temp_trimmed_{start_time}.mp4")
            self.trim_video(video_path, start_time, end_time, temp_trimmed)
            
            # Step 2: Resize to reel format
            temp_resized = str(Path(output_path).parent / f"temp_resized_{start_time}.mp4")
            self.resize_to_reel(temp_trimmed, temp_resized)
            
            # Step 3: Add captions if provided
            if captions:
                # Adjust caption timestamps relative to start_time
                adjusted_captions = []
                for caption in captions:
                    adj_caption = caption.copy()
                    adj_caption['start'] = caption.get('start', 0) - start_time
                    adj_caption['end'] = caption.get('end', end_time) - start_time
                    # Ensure timestamps are within clip duration
                    adj_caption['start'] = max(0, adj_caption['start'])
                    adj_caption['end'] = min(end_time - start_time, adj_caption['end'])
                    adjusted_captions.append(adj_caption)
                
                self.add_captions(temp_resized, adjusted_captions, output_path)
                
                # Clean up temp files
                Path(temp_trimmed).unlink(missing_ok=True)
                Path(temp_resized).unlink(missing_ok=True)
            else:
                # No captions, just rename resized file
                Path(temp_resized).rename(output_path)
                Path(temp_trimmed).unlink(missing_ok=True)
            
            logger.info(f"Reel created successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating reel: {str(e)}")
            # Clean up temp files on error
            Path(temp_trimmed).unlink(missing_ok=True)
            Path(temp_resized).unlink(missing_ok=True)
            raise

