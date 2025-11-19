# Clipzy - AI-Powered YouTube Podcast to Reels Generator

An intelligent application that automatically identifies viral-worthy segments from YouTube podcasts and generates social media reels with captions.

## Features

- 🎥 **YouTube Video Processing**: Download and process YouTube podcast videos
- 🎤 **Automatic Transcription**: Convert speech to text with timestamps
- 🤖 **AI-Powered Detection**: Identify interesting/viral segments using NLP
- ✂️ **Smart Video Editing**: Automatically trim and create reels with branding
- 📝 **Caption Generation**: Auto-generate and overlay captions
- 📊 **Dataset Processing**: Train models on viral content datasets
- 🌐 **Web Interface**: FastAPI backend + lightweight frontend for self-serve reels

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
├── backend/
│   └── app.py                     # FastAPI service (Phase 6)
├── frontend/
│   └── index.html                 # Lightweight UI (Phase 6)
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
├── main.py                         # CLI entry point
├── collect_dataset.py              # Interactive dataset helper
├── process_dataset.py              # Dataset processing pipeline
├── train_model.py                  # Model training helper
├── test_phase2.py                  # Phase 2 validation script
├── test_phase5_video_editor.py     # Phase 5 validation script
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

### Web API (Phase 6)

1. **Start the API server**
   ```bash
   uvicorn backend.app:app --reload --port 8000
   ```

2. **Open the frontend**
   - Option A: Double-click `frontend/index.html`
   - Option B: Serve it locally:
     ```bash
     cd frontend
     python -m http.server 5173
     ```
     Then visit `http://localhost:5173`

3. **Use the UI**
   - Paste a YouTube link
   - Set segment/min/max duration
   - Click **Generate Reels**
   - The UI calls `POST /generate` and displays saved reel paths

> Tip: If your backend runs on a different host/port, set `window.CLIPZY_API_BASE` in the browser console before submitting the form.

## Dataset Processing

The project requires dataset processing for training the NLP model:

1. **Collect Dataset**: `python collect_dataset.py --interactive`
2. **Process Dataset**: `python process_dataset.py`
3. **Train Model**: `python train_model.py`

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

