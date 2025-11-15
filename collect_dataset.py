"""
Dataset Collection Helper Script
Helps collect and label viral/not-viral podcast segments for training
"""
import pandas as pd
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

from src.youtube_downloader import YouTubeDownloader
from src.transcriber import Transcriber
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetCollector:
    """Helps collect dataset of viral/not-viral segments"""
    
    def __init__(self):
        self.downloader = YouTubeDownloader(
            output_dir=str(config.DATA_DIR / "downloads")
        )
        self.transcriber = Transcriber(model_name=config.WHISPER_MODEL)
        self.dataset_path = config.DATASET_PATH
        self.segments = []
    
    def load_existing_dataset(self) -> pd.DataFrame:
        """Load existing dataset if it exists"""
        if Path(self.dataset_path).exists():
            logger.info(f"Loading existing dataset from {self.dataset_path}")
            return pd.read_csv(self.dataset_path)
        else:
            logger.info("No existing dataset found. Creating new one.")
            return pd.DataFrame(columns=[
                'transcript', 'start_time', 'end_time', 'duration',
                'is_viral', 'views', 'likes', 'shares', 'topic', 'source_video'
            ])
    
    def save_dataset(self, df: pd.DataFrame):
        """Save dataset to CSV"""
        Path(self.dataset_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.dataset_path, index=False)
        logger.info(f"Dataset saved to {self.dataset_path}")
        logger.info(f"Total segments: {len(df)}")
        logger.info(f"  Viral: {df['is_viral'].sum()}")
        logger.info(f"  Not Viral: {len(df) - df['is_viral'].sum()}")
    
    def collect_from_youtube(self, url: str, segments: List[Dict]) -> List[Dict]:
        """
        Collect segments from a YouTube video
        
        Args:
            url: YouTube video URL
            segments: List of segment dicts with 'start_time', 'end_time', 'is_viral', 'topic'
            
        Returns:
            List of collected segment data
        """
        logger.info(f"Processing YouTube video: {url}")
        
        # Download and transcribe
        video_info = self.downloader.download_video(url, audio_only=False)
        transcription = self.transcriber.transcribe(video_info['file_path'], language=config.LANGUAGE)
        
        collected_segments = []
        
        for seg_info in segments:
            start = seg_info['start_time']
            end = seg_info['end_time']
            is_viral = seg_info.get('is_viral', 0)
            topic = seg_info.get('topic', 'unknown')
            
            # Get transcript for this segment
            segment_text = self.transcriber.get_text_in_range(transcription, start, end)
            duration = end - start
            
            segment_data = {
                'transcript': segment_text,
                'start_time': start,
                'end_time': end,
                'duration': duration,
                'is_viral': 1 if is_viral else 0,
                'views': seg_info.get('views', 0),
                'likes': seg_info.get('likes', 0),
                'shares': seg_info.get('shares', 0),
                'topic': topic,
                'source_video': url
            }
            
            collected_segments.append(segment_data)
            logger.info(f"Collected segment: {start}s-{end}s (viral={is_viral}, topic={topic})")
        
        return collected_segments
    
    def add_segment_manual(self, transcript: str, start_time: float, end_time: float,
                          is_viral: bool, topic: str = "unknown", 
                          views: int = 0, likes: int = 0, shares: int = 0,
                          source_video: str = "") -> Dict:
        """
        Manually add a segment to the dataset
        
        Args:
            transcript: Text transcript of the segment
            start_time: Start time in seconds
            end_time: End time in seconds
            is_viral: Whether this segment is viral (True/False)
            topic: Topic of the segment
            views: View count (optional)
            likes: Like count (optional)
            shares: Share count (optional)
            source_video: Source video URL
            
        Returns:
            Segment data dictionary
        """
        duration = end_time - start_time
        
        segment_data = {
            'transcript': transcript,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'is_viral': 1 if is_viral else 0,
            'views': views,
            'likes': likes,
            'shares': shares,
            'topic': topic,
            'source_video': source_video
        }
        
        return segment_data
    
    def interactive_collect(self):
        """Interactive mode for collecting segments"""
        logger.info("=" * 60)
        logger.info("Interactive Dataset Collection")
        logger.info("=" * 60)
        
        df = self.load_existing_dataset()
        
        while True:
            print("\n" + "-" * 60)
            print("Dataset Collection Menu:")
            print("1. Add segment from YouTube video")
            print("2. Add segment manually")
            print("3. View current dataset")
            print("4. Save and exit")
            print("5. Exit without saving")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                url = input("Enter YouTube URL: ").strip()
                if not url:
                    print("Invalid URL")
                    continue
                
                print("\nEnter segment information:")
                start = float(input("Start time (seconds): "))
                end = float(input("End time (seconds): "))
                is_viral = input("Is viral? (y/n): ").strip().lower() == 'y'
                topic = input("Topic: ").strip() or "unknown"
                views = int(input("Views (optional, press Enter for 0): ") or "0")
                likes = int(input("Likes (optional, press Enter for 0): ") or "0")
                shares = int(input("Shares (optional, press Enter for 0): ") or "0")
                
                try:
                    segments = self.collect_from_youtube(url, [{
                        'start_time': start,
                        'end_time': end,
                        'is_viral': is_viral,
                        'topic': topic,
                        'views': views,
                        'likes': likes,
                        'shares': shares
                    }])
                    
                    for seg in segments:
                        df = pd.concat([df, pd.DataFrame([seg])], ignore_index=True)
                    print(f"\nAdded {len(segments)} segment(s) to dataset")
                    
                except Exception as e:
                    logger.error(f"Error collecting segment: {str(e)}")
            
            elif choice == '2':
                print("\nEnter segment information:")
                transcript = input("Transcript text: ").strip()
                start = float(input("Start time (seconds): "))
                end = float(input("End time (seconds): "))
                is_viral = input("Is viral? (y/n): ").strip().lower() == 'y'
                topic = input("Topic: ").strip() or "unknown"
                views = int(input("Views (optional, press Enter for 0): ") or "0")
                likes = int(input("Likes (optional, press Enter for 0): ") or "0")
                shares = int(input("Shares (optional, press Enter for 0): ") or "0")
                source = input("Source video URL (optional): ").strip()
                
                if not transcript:
                    print("Transcript cannot be empty")
                    continue
                
                seg = self.add_segment_manual(
                    transcript, start, end, is_viral, topic,
                    views, likes, shares, source
                )
                df = pd.concat([df, pd.DataFrame([seg])], ignore_index=True)
                print("\nSegment added to dataset")
            
            elif choice == '3':
                print(f"\nCurrent Dataset Statistics:")
                print(f"  Total segments: {len(df)}")
                if len(df) > 0:
                    print(f"  Viral: {df['is_viral'].sum()}")
                    print(f"  Not Viral: {len(df) - df['is_viral'].sum()}")
                    print(f"\nRecent segments:")
                    print(df[['transcript', 'is_viral', 'topic']].tail(5).to_string())
            
            elif choice == '4':
                self.save_dataset(df)
                print("\nDataset saved. Exiting...")
                break
            
            elif choice == '5':
                print("\nExiting without saving...")
                break
            
            else:
                print("Invalid choice")


def main():
    """Main entry point"""
    collector = DatasetCollector()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        collector.interactive_collect()
    else:
        print("Dataset Collection Helper")
        print("\nUsage:")
        print("  python collect_dataset.py --interactive")
        print("\nThis will start an interactive session to collect dataset segments.")
        print("\nAlternatively, you can:")
        print("1. Manually create data/raw/viral_segments.csv using the template")
        print("2. Use the DatasetCollector class programmatically")
        
        # Show template location
        template_path = config.RAW_DATA_DIR / "dataset_template.csv"
        if template_path.exists():
            print(f"\nTemplate available at: {template_path}")


if __name__ == '__main__':
    main()

