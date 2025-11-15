# Phase 3 Quick Start Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Collect Your Dataset

**Option A - Interactive (Recommended for beginners):**
```bash
python collect_dataset.py --interactive
```

**Option B - Manual CSV:**
1. Copy template: `data/raw/dataset_template.csv` → `data/raw/viral_segments.csv`
2. Edit CSV with your segments
3. Minimum: 100 segments (50 viral, 50 not viral)

### Step 2: Process Dataset
```bash
python process_dataset.py
```

### Step 3: Train Model
```bash
python train_model.py
```

**Done!** Your model is now trained and ready to use.

---

## 📋 What You Need

- **Dataset**: At least 100 labeled segments
  - Viral segments (interesting, engaging)
  - Not viral segments (boring, explanatory)
  
- **Format**: CSV with columns:
  - `transcript` - Text content
  - `start_time`, `end_time` - Timestamps
  - `is_viral` - 1 or 0
  - `topic` - Category (optional)
  - Other fields optional

---

## 📁 Files Created

- ✅ `data/raw/dataset_template.csv` - Template
- ✅ `collect_dataset.py` - Collection tool
- ✅ `process_dataset.py` - Processing script
- ✅ `train_model.py` - Training script
- ✅ `PHASE3_GUIDE.md` - Full documentation

---

## ⚡ Quick Commands

```bash
# Full pipeline
python collect_dataset.py --interactive  # Collect data
python process_dataset.py                 # Process data
python train_model.py                     # Train model
python main.py --url "URL" --segments 5   # Generate reels!
```

---

## 💡 Tips

1. **Start Small**: Begin with 50-100 segments to test
2. **Quality > Quantity**: Accurate labels are crucial
3. **Balance**: Try for ~50/50 viral/not viral split
4. **Diversity**: Include different topics and styles

---

## 🆘 Need Help?

See `PHASE3_GUIDE.md` for detailed documentation.

