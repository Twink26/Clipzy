# Clipzy - AI-Powered YouTube Podcast to Reels Generator

An intelligent application that automatically identifies viral-worthy segments from YouTube podcasts and generates social media reels with captions.

## Features

- 🎥 **YouTube Video Processing**: Download and process YouTube podcast videos
- 🎤 **Automatic Transcription**: Convert speech to text with timestamps
- 🤖 **AI-Powered Detection**: Identify interesting/viral segments using NLP
- ✂️ **Smart Video Editing**: Automatically trim and create reels
- 📝 **Caption Generation**: Auto-generate and overlay captions
- 📊 **Dataset Processing**: Train models on viral content datasets

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd clipzy
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

5. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

## Project Structure

```
clipzy/
├── src/
│   ├── __init__.py
│   ├── youtube_downloader.py      # YouTube video download
│   ├── transcriber.py               # Speech-to-text transcription
│   ├── dataset_processor.py        # Dataset processing pipeline
│   ├── nlp_analyzer.py             # NLP model for segment detection
│   ├── video_editor.py             # Video editing and caption overlay
│   └── reel_generator.py           # Main orchestration
├── data/
│   ├── raw/                        # Raw datasets
│   ├── processed/                  # Processed datasets
│   └── models/                     # Saved models
├── output/                         # Generated reels
├── config.py                       # Configuration
├── main.py                         # Entry point
├── requirements.txt
└── README.md
```

## Usage

### Basic Usage

```python
from src.reel_generator import ReelGenerator

generator = ReelGenerator()
reels = generator.generate_reels(
    youtube_url="https://www.youtube.com/watch?v=VIDEO_ID",
    num_segments=5,
    min_duration=15,
    max_duration=60
)
```

### Command Line

```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --segments 5
```

## Dataset Processing

The project requires dataset processing for training the NLP model:

1. **Prepare Dataset**: Place your dataset in `data/raw/`
2. **Process Dataset**: Run `python src/dataset_processor.py`
3. **Train Model**: The model will be trained automatically on first run

## Configuration

Create a `.env` file for configuration:

```env
OUTPUT_DIR=output
MAX_REEL_DURATION=60
MIN_REEL_DURATION=15
MODEL_PATH=data/models/viral_detector.pkl
```

## Requirements

- Python 3.8+
- FFmpeg (for video processing)
- Sufficient disk space for video downloads

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

