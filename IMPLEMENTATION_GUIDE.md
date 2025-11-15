# Clipzy - Complete Implementation Guide

## 📋 Project Overview
Build an AI-powered system that:
1. Downloads YouTube podcast videos
2. Transcribes audio to text with timestamps
3. **Processes datasets** (MANDATORY) to train models
4. Uses NLP to detect viral/interesting segments
5. Generates social media reels with captions

---

## 🚀 Step-by-Step Implementation

### **STEP 1: Environment Setup** ✅ (Partially Done)

**What to do:**
1. Ensure Python 3.8+ is installed
2. Create/activate virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install FFmpeg (required for video processing):
   - Windows: Download from https://ffmpeg.org/download.html
   - Add to PATH
   - Or use: `choco install ffmpeg` (if Chocolatey installed)
5. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
6. Download NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
   ```

**Status:** Basic setup exists, verify all dependencies are installed.

---

### **STEP 2: Dataset Collection & Preparation** (MANDATORY)

**Why:** You need training data to teach the model what makes a segment "viral"

**What to do:**

#### 2.1 Create Dataset Structure
Create `data/raw/viral_segments.csv` with columns:
- `transcript`: Text content of the segment
- `start_time`: Start timestamp (seconds)
- `end_time`: End timestamp (seconds)
- `duration`: Duration in seconds
- `is_viral`: Label (1 = viral, 0 = not viral)
- `views`: View count (if available)
- `likes`: Like count (if available)
- `shares`: Share count (if available)
- `topic`: Main topic discussed
- `source_video`: YouTube URL or video ID

#### 2.2 Dataset Collection Options:

**Option A: Manual Collection**
1. Find popular podcast clips on TikTok/Instagram Reels
2. Note the timestamps from original videos
3. Download those segments
4. Transcribe them
5. Label as viral (1) or not viral (0)
6. Collect at least 200-500 examples

**Option B: Scrape YouTube Shorts/Reels**
1. Scrape viral podcast clips from YouTube Shorts
2. Extract transcripts and metadata
3. Use engagement metrics (views/likes) as labels

**Option C: Synthetic Dataset**
1. Use existing podcast transcripts
2. Manually label interesting segments
3. Use heuristics (high engagement, trending topics) to label

**Sample Dataset Format:**
```csv
transcript,start_time,end_time,duration,is_viral,views,likes,shares,topic,source_video
"I can't believe what happened next...",120.5,145.2,24.7,1,50000,5000,2000,story,https://youtube.com/watch?v=xyz
"Let me explain the basics...",300.0,350.0,50.0,0,1000,50,10,education,https://youtube.com/watch?v=xyz
```

**Minimum Dataset Size:** 100-200 labeled segments (more is better)

---

### **STEP 3: Implement Dataset Processor** (MANDATORY)

**File:** `src/dataset_processor.py`

**What it should do:**
1. Load raw dataset from CSV
2. Clean and preprocess text
3. Extract features:
   - Text embeddings (using sentence-transformers)
   - Sentiment scores (positive/negative/neutral)
   - Keyword extraction
   - Topic modeling
   - Engagement scores
   - Text length, word count
   - Question detection
   - Emotional intensity
4. Save processed dataset for model training

**Key Features to Extract:**
- **Text Embeddings**: Vector representation of text (using sentence-transformers)
- **Sentiment Analysis**: Positive/negative/neutral scores
- **Keyword Extraction**: Important keywords and phrases
- **Topic Modeling**: Main topics discussed
- **Engagement Indicators**: Questions, exclamations, personal pronouns
- **Pace Indicators**: Words per second, pause frequency
- **Hook Detection**: Strong opening statements

**Implementation Checklist:**
- [ ] Load CSV dataset
- [ ] Text cleaning (remove special chars, normalize)
- [ ] Feature extraction functions
- [ ] Save processed features to `data/processed/`
- [ ] Create train/test split

---

### **STEP 4: Implement NLP Analyzer**

**File:** `src/nlp_analyzer.py`

**What it should do:**
1. Load processed dataset
2. Train a classifier to predict "viral" segments
3. Use features from dataset processor
4. Save trained model
5. Provide scoring function for new segments

**Model Options:**

**Option A: Traditional ML (Recommended for Start)**
- Use scikit-learn: RandomForest, XGBoost, or SVM
- Train on extracted features
- Faster training, easier to debug

**Option B: Deep Learning**
- Fine-tune BERT/RoBERTa for classification
- Better accuracy but slower
- Requires more data

**Implementation Steps:**
1. Load processed dataset
2. Split into train/validation/test (80/10/10)
3. Train classifier model
4. Evaluate with metrics (accuracy, precision, recall, F1)
5. Save model to `data/models/viral_detector.pkl`
6. Implement `predict_viral_score()` function

**Scoring Function:**
- Input: Transcript segment with timestamps
- Output: Score (0-1) indicating viral potential
- Features: All extracted features from dataset processor

---

### **STEP 5: Implement Video Editor**

**File:** `src/video_editor.py`

**What it should do:**
1. Trim video segments based on timestamps
2. Convert to vertical format (9:16 aspect ratio)
3. Add captions overlay
4. Style captions (font, color, position)
5. Export as MP4

**Key Functions:**
- `trim_video()`: Cut video segment
- `resize_to_reel()`: Convert to 9:16 format
- `add_captions()`: Overlay text captions
- `export_reel()`: Save final reel

**Technical Details:**
- Use MoviePy for video editing
- Use OpenCV for advanced processing (optional)
- Caption positioning: Bottom (default), can be top/center
- Caption styling: White text, black background/border for readability
- Sync captions with audio (word-level timing)

**Aspect Ratio Conversion:**
- Original video → Crop/zoom to fit 9:16
- Center crop or smart crop (detect faces/subjects)

---

### **STEP 6: Implement Reel Generator (Main Orchestrator)**

**File:** `src/reel_generator.py`

**What it should do:**
1. Coordinate all modules
2. Main pipeline:
   - Download YouTube video
   - Transcribe audio
   - Split into segments (e.g., 30-second chunks)
   - Score each segment using NLP analyzer
   - Select top N segments
   - Generate reels for each segment
   - Return list of generated reel paths

**Pipeline Flow:**
```
YouTube URL 
  → Download Video
  → Transcribe (with timestamps)
  → Split into segments
  → Extract features for each segment
  → Score segments (viral potential)
  → Rank segments
  → Select top N segments
  → For each segment:
      → Trim video
      → Resize to 9:16
      → Add captions
      → Export reel
  → Return reel paths
```

**Key Functions:**
- `generate_reels(youtube_url, num_segments=5, min_duration=15, max_duration=60)`
- `detect_interesting_segments(transcription)`
- `create_reel(segment, video_path)`

---

### **STEP 7: Create Main Entry Point**

**File:** `main.py`

**What it should do:**
1. CLI interface to run the application
2. Accept YouTube URL as input
3. Accept parameters (number of segments, duration limits)
4. Run the pipeline
5. Display progress
6. Show output reel paths

**Example Usage:**
```bash
python main.py --url "https://youtube.com/watch?v=xyz" --segments 5 --min-duration 15 --max-duration 60
```

**Implementation:**
- Use `argparse` for CLI
- Add progress bars (tqdm)
- Error handling
- Logging

---

### **STEP 8: Testing & Refinement**

**What to do:**
1. **Test with sample videos:**
   - Use different podcast types
   - Test with various durations
   - Verify transcription accuracy

2. **Evaluate model performance:**
   - Check if segments detected are actually interesting
   - Adjust model thresholds
   - Fine-tune feature weights

3. **Optimize video quality:**
   - Test caption readability
   - Verify aspect ratio conversion
   - Check audio sync

4. **Handle edge cases:**
   - Very long videos
   - Poor audio quality
   - Multiple speakers
   - Background music

---

## 📁 Complete File Structure

```
clipzy/
├── src/
│   ├── __init__.py ✅
│   ├── youtube_downloader.py ✅
│   ├── transcriber.py ✅
│   ├── dataset_processor.py ⚠️ TO IMPLEMENT
│   ├── nlp_analyzer.py ⚠️ TO IMPLEMENT
│   ├── video_editor.py ⚠️ TO IMPLEMENT
│   └── reel_generator.py ⚠️ TO IMPLEMENT
├── data/
│   ├── raw/
│   │   └── viral_segments.csv ⚠️ TO CREATE
│   ├── processed/
│   │   └── (processed features) ⚠️ GENERATED
│   ├── models/
│   │   └── viral_detector.pkl ⚠️ GENERATED
│   └── downloads/ ⚠️ GENERATED
├── output/ ⚠️ GENERATED
├── config.py ✅
├── main.py ⚠️ TO IMPLEMENT
├── requirements.txt ✅
├── README.md ✅
└── IMPLEMENTATION_GUIDE.md ✅ (this file)
```

---

## 🎯 Implementation Priority Order

1. **Dataset Collection** (MANDATORY) - Start here!
   - Collect at least 100-200 labeled examples
   - Create `data/raw/viral_segments.csv`

2. **Dataset Processor** (MANDATORY)
   - Implement feature extraction
   - Process your dataset

3. **NLP Analyzer**
   - Train model on processed dataset
   - Test prediction accuracy

4. **Video Editor**
   - Implement trimming and resizing
   - Add caption overlay

5. **Reel Generator**
   - Connect all modules
   - Test end-to-end pipeline

6. **Main Entry Point**
   - Create CLI interface
   - Add user-friendly features

---

## 🔧 Quick Start Checklist

- [ ] Install all dependencies
- [ ] Install FFmpeg
- [ ] Download spaCy and NLTK models
- [ ] Collect/create dataset (100+ examples)
- [ ] Implement dataset_processor.py
- [ ] Implement nlp_analyzer.py
- [ ] Train model and save it
- [ ] Implement video_editor.py
- [ ] Implement reel_generator.py
- [ ] Create main.py
- [ ] Test with sample YouTube video
- [ ] Refine and optimize

---

## 💡 Tips for Success

1. **Start Small**: Begin with a small dataset (50-100 examples) to test the pipeline
2. **Iterate**: Test each module independently before integrating
3. **Monitor Performance**: Track model accuracy and adjust features
4. **User Feedback**: Test generated reels and refine based on quality
5. **Optimize Speed**: Video processing can be slow; consider caching transcriptions

---

## 🐛 Common Issues & Solutions

**Issue:** FFmpeg not found
- **Solution:** Install FFmpeg and add to PATH

**Issue:** Whisper model download fails
- **Solution:** Check internet connection, try smaller model (tiny/base)

**Issue:** Low model accuracy
- **Solution:** Collect more training data, adjust features, try different models

**Issue:** Video processing too slow
- **Solution:** Use lower resolution, process in background, optimize code

**Issue:** Captions not syncing
- **Solution:** Use word-level timestamps from Whisper, adjust timing offsets

---

## 📚 Resources

- **Whisper Documentation**: https://github.com/openai/whisper
- **MoviePy Documentation**: https://zulko.github.io/moviepy/
- **Sentence Transformers**: https://www.sbert.net/
- **scikit-learn**: https://scikit-learn.org/

---

## Next Steps

1. Start with **STEP 2** (Dataset Collection) - this is mandatory
2. Then implement **STEP 3** (Dataset Processor)
3. Continue with remaining steps in order

Good luck! 🚀

