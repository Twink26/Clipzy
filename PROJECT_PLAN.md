# Clipzy - AI-Powered YouTube Podcast to Reels Generator

## Project Overview
An AI-powered application that automatically identifies interesting/viral segments from YouTube podcasts and generates social media reels with captions.

## Technology Stack
- **Python 3.8+**
- **YouTube Download**: `yt-dlp` or `pytube`
- **Video Processing**: `moviepy`, `opencv-python`
- **NLP**: `transformers`, `sentence-transformers`, `nltk`, `spacy`
- **Speech-to-Text**: `whisper` (OpenAI) or `speech_recognition`
- **Dataset Processing**: `pandas`, `numpy`
- **ML Framework**: `scikit-learn`, `torch`

## Step-by-Step Implementation Plan

### Phase 1: Project Setup & Infrastructure
1. **Set up project structure**
   - Create virtual environment
   - Install dependencies
   - Set up configuration files

2. **Create core modules structure**
   - `youtube_downloader.py` - Download videos from YouTube
   - `transcriber.py` - Convert audio to text
   - `dataset_processor.py` - Process and prepare datasets
   - `nlp_analyzer.py` - NLP model for segment detection
   - `video_editor.py` - Video trimming and caption overlay
   - `reel_generator.py` - Main orchestration module

### Phase 2: YouTube Video Processing
3. **YouTube Downloader Module**
   - Extract video URL
   - Download video/audio
   - Extract metadata (title, duration, etc.)

4. **Transcription Module**
   - Extract audio from video
   - Use Whisper or similar for transcription
   - Generate timestamps for each segment
   - Save transcript with timestamps

### Phase 3: Dataset Processing (MANDATORY)
5. **Dataset Collection & Preparation**
   - Collect dataset of viral podcast segments
   - Label segments as "viral" or "not viral"
   - Features to extract:
     - Transcript text
     - Duration
     - Engagement metrics (if available)
     - Topic keywords
     - Sentiment scores
     - Energy/pace indicators

6. **Dataset Processing Pipeline**
   - Load and clean dataset
   - Feature engineering:
     - Text embeddings
     - Topic modeling
     - Sentiment analysis
     - Keyword extraction
     - Engagement score calculation
   - Train/validation/test split
   - Data normalization

### Phase 4: NLP Model Development
7. **Interesting Segment Detection Model**
   - Use pre-trained models (BERT, RoBERTa) for text understanding
   - Train classifier to identify "viral" segments
   - Features:
     - Topic relevance
     - Engagement potential
     - Hook detection (interesting openings)
     - Controversy/trending topics
     - Emotional intensity
   - Implement scoring system

8. **Model Training & Evaluation**
   - Train model on processed dataset
   - Evaluate with metrics (precision, recall, F1)
   - Fine-tune hyperparameters
   - Save trained model

### Phase 5: Video Editing & Reel Generation
9. **Video Editor Module**
   - Trim video segments based on detected timestamps
   - Add captions/subtitles overlay
   - Apply transitions
   - Optimize for social media (9:16 aspect ratio)
   - Add branding/watermarks (optional)

10. **Caption Generation**
    - Generate engaging captions from transcript
    - Style captions for readability
    - Add emojis/formatting (optional)
    - Position captions dynamically

### Phase 6: Main Application
11. **Main Pipeline Integration**
    - Connect all modules
    - Create CLI/API interface
    - Add error handling
    - Add progress tracking

12. **Testing & Optimization**
    - Test with various YouTube videos
    - Optimize processing speed
    - Handle edge cases
    - Improve model accuracy

## Dataset Requirements
- **Viral Podcast Segments Dataset**: 
  - Transcripts of viral podcast clips
  - Timestamps
  - Engagement metrics (views, likes, shares)
  - Labels (viral/not viral)
  
- **Processing Steps**:
  1. Data collection (scraping or manual)
  2. Data cleaning and preprocessing
  3. Feature extraction
  4. Label encoding
  5. Train/validation/test split

## Key Features to Detect "Viral" Segments
1. **Hook Detection**: Strong opening statements
2. **Topic Relevance**: Trending or controversial topics
3. **Emotional Intensity**: High sentiment variance
4. **Engagement Triggers**: Questions, surprising facts, personal stories
5. **Duration**: Optimal length (15-60 seconds for reels)
6. **Pace**: Fast-paced, energetic segments
7. **Keywords**: Trending keywords and hashtags

## Output Specifications
- **Reel Format**: 
  - Duration: 15-60 seconds
  - Aspect Ratio: 9:16 (vertical)
  - Resolution: 1080x1920 (or similar)
  - Captions: Auto-generated, styled, positioned
  - Format: MP4

## Future Enhancements
- Multi-language support
- Custom branding
- Multiple output formats (TikTok, Instagram, YouTube Shorts)
- Batch processing
- Web interface
- Cloud deployment

