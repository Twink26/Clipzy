# Phase 3: Dataset Processing - Complete Guide

## Overview
Phase 3 is **MANDATORY** for training the viral segment detection model. This phase involves collecting labeled data and processing it to extract features for machine learning.

---

## Step-by-Step Process

### Step 1: Collect Dataset

You need to collect at least **100-200 labeled segments** (more is better) with:
- Transcript text
- Start/end timestamps
- Label: viral (1) or not viral (0)
- Optional: views, likes, shares, topic

#### Option A: Interactive Collection Tool

```bash
python collect_dataset.py --interactive
```

This will guide you through:
1. Adding segments from YouTube videos (auto-transcribes)
2. Adding segments manually
3. Viewing current dataset
4. Saving dataset

#### Option B: Manual CSV Creation

1. Copy the template:
   ```bash
   cp data/raw/dataset_template.csv data/raw/viral_segments.csv
   ```

2. Edit `data/raw/viral_segments.csv` with your data

**Required Columns:**
- `transcript`: Text content of the segment
- `start_time`: Start time in seconds
- `end_time`: End time in seconds
- `duration`: Duration in seconds (auto-calculated if missing)
- `is_viral`: 1 for viral, 0 for not viral
- `views`: View count (optional, can be 0)
- `likes`: Like count (optional, can be 0)
- `shares`: Share count (optional, can be 0)
- `topic`: Topic/category (e.g., "story", "education", "reaction")
- `source_video`: YouTube URL or video ID

**Example:**
```csv
transcript,start_time,end_time,duration,is_viral,views,likes,shares,topic,source_video
"This is amazing! You won't believe what happened...",120.5,145.2,24.7,1,50000,5000,2000,story,https://youtube.com/watch?v=xyz
"Let me explain how this works. First...",300.0,350.0,50.0,0,1000,50,10,education,https://youtube.com/watch?v=xyz
```

#### Option C: Programmatic Collection

```python
from collect_dataset import DatasetCollector

collector = DatasetCollector()

# Collect from YouTube
segments = collector.collect_from_youtube(
    url="https://youtube.com/watch?v=VIDEO_ID",
    segments=[
        {
            'start_time': 120.5,
            'end_time': 145.2,
            'is_viral': True,
            'topic': 'story',
            'views': 50000,
            'likes': 5000,
            'shares': 2000
        }
    ]
)

# Or add manually
segment = collector.add_segment_manual(
    transcript="Amazing story text...",
    start_time=120.5,
    end_time=145.2,
    is_viral=True,
    topic="story"
)
```

---

### Step 2: Process Dataset

Once you have collected your dataset, process it to extract features:

```bash
python process_dataset.py
```

**What this does:**
1. Loads raw dataset from `data/raw/viral_segments.csv`
2. Cleans and normalizes text
3. Extracts features:
   - Text embeddings (sentence-transformers)
   - Sentiment scores (positive/negative/neutral)
   - Keyword extraction
   - Engagement indicators (questions, exclamations, personal pronouns)
   - Text statistics (length, word count, etc.)
4. Splits into train/test sets
5. Saves processed data to `data/processed/`

**Output Files:**
- `data/processed/processed_features.csv` - Full processed dataset
- `data/processed/train_features.csv` - Training set
- `data/processed/test_features.csv` - Test set

---

### Step 3: Train Model

After processing, train the model:

```bash
python train_model.py
```

**What this does:**
1. Loads processed dataset
2. Trains RandomForest classifier
3. Evaluates performance
4. Saves trained model to `data/models/viral_detector.pkl`

**Model Performance Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- Cross-validation scores

---

## Complete Workflow

```bash
# 1. Collect dataset (interactive)
python collect_dataset.py --interactive

# 2. Process dataset
python process_dataset.py

# 3. Train model
python train_model.py

# 4. Now you can generate reels!
python main.py --url "https://youtube.com/watch?v=VIDEO_ID" --segments 5
```

---

## Dataset Collection Tips

### What Makes a Segment "Viral"?

Look for segments with:
- ✅ Strong hooks/opening statements
- ✅ Emotional intensity
- ✅ Surprising or controversial content
- ✅ Personal stories
- ✅ Questions that engage viewers
- ✅ Fast-paced, energetic delivery
- ✅ Trending topics
- ✅ High engagement (views, likes, shares)

### What Makes a Segment "Not Viral"?

Avoid segments with:
- ❌ Slow, explanatory content
- ❌ Technical details without context
- ❌ Long pauses or filler words
- ❌ Boring introductions
- ❌ Low energy
- ❌ Generic information

### Data Collection Strategies

1. **From Existing Viral Clips:**
   - Find viral podcast clips on TikTok/Instagram
   - Note timestamps from original videos
   - Transcribe and label as viral

2. **From Your Own Videos:**
   - Use videos you know performed well
   - Label high-performing segments as viral
   - Label low-performing segments as not viral

3. **Manual Labeling:**
   - Watch segments and judge viral potential
   - Use engagement metrics if available
   - Get multiple opinions if possible

4. **Balanced Dataset:**
   - Aim for ~50/50 split (viral/not viral)
   - Or slightly more viral examples (60/40)
   - Include diverse topics and styles

---

## Dataset Requirements

### Minimum Requirements:
- **100 segments** (50 viral, 50 not viral) - Minimum viable
- **200 segments** (100 viral, 100 not viral) - Recommended
- **500+ segments** - Better accuracy

### Quality Requirements:
- Accurate transcripts
- Correct timestamps
- Consistent labeling
- Diverse topics
- Various durations (15-60 seconds)

---

## Feature Extraction Details

The dataset processor extracts:

### Text Features:
- Text length, word count, character count
- Sentiment scores (positive, negative, neutral, compound)
- Question/exclamation counts
- Personal pronoun counts
- Keyword density

### Embeddings:
- Sentence transformer embeddings (384 dimensions by default)
- Captures semantic meaning of text

### Engagement Features:
- Questions detected
- Exclamations detected
- Personal pronouns (I, you, we, etc.)
- Keyword vs stopword ratio

### Calculated Features:
- Words per second (pace indicator)
- Sentiment variance
- Text complexity

---

## Troubleshooting

### Issue: "Dataset not found"
**Solution:** Create `data/raw/viral_segments.csv` using the template

### Issue: "Not enough data"
**Solution:** Collect more segments. Minimum 50-100 for each class

### Issue: "Poor model performance"
**Solutions:**
- Collect more data (especially viral examples)
- Improve data quality (accurate transcripts, correct labels)
- Check for class imbalance
- Try different features or models

### Issue: "NLTK data not found"
**Solution:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### Issue: "Embedding model download slow"
**Solution:** The first run downloads the model. It's a one-time download (~80MB)

---

## File Structure

```
data/
├── raw/
│   ├── dataset_template.csv      # Template for dataset
│   └── viral_segments.csv        # Your collected dataset (CREATE THIS)
├── processed/
│   ├── processed_features.csv     # Full processed dataset (GENERATED)
│   ├── train_features.csv         # Training set (GENERATED)
│   └── test_features.csv          # Test set (GENERATED)
└── models/
    └── viral_detector.pkl         # Trained model (GENERATED)
```

---

## Next Steps After Phase 3

Once Phase 3 is complete:
1. ✅ Model is trained and saved
2. ✅ Ready for Phase 4 (NLP Model Development) - Already done!
3. ✅ Ready for Phase 5 (Video Editing) - Already done!
4. ✅ Ready to generate reels!

You can now use the full pipeline:
```bash
python main.py --url "YOUTUBE_URL" --segments 5
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Collect dataset | `python collect_dataset.py --interactive` |
| Process dataset | `python process_dataset.py` |
| Train model | `python train_model.py` |
| Generate reels | `python main.py --url "URL" --segments 5` |

---

## Phase 3 Status

✅ Dataset collection tools created
✅ Dataset processing pipeline ready
✅ Model training script ready
✅ Documentation complete

**Ready to collect your dataset!** 🚀

