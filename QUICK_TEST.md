# Quick Test Guide

## 🚀 Fastest Way to Test Everything

### 1. Start Backend (Terminal 1)
```bash
uvicorn backend.app:app --reload --port 8000
```
✅ You should see: `Uvicorn running on http://127.0.0.1:8000`

### 2. Test Backend (Terminal 2)
```bash
python test_backend.py
```
✅ Should show all tests passing

### 3. Open Frontend
**Option A:** Double-click `frontend/index.html`  
**Option B:** 
```bash
cd frontend
python -m http.server 8080
```
Then visit: http://localhost:8080

### 4. Test Full Workflow
1. In the frontend, paste a YouTube URL
2. Click "Generate Reels"
3. Wait for processing (5-15 minutes)
4. Verify:
   - ✅ Videos appear in cards
   - ✅ Videos play when clicked
   - ✅ Download button works
   - ✅ Copy link works

## 🔍 Quick Health Checks

### Backend Health
```bash
curl http://localhost:8000/health
```
Expected: `{"status":"ok"}`

### API Docs
Open: http://localhost:8000/docs

### Frontend Console
Open browser DevTools (F12) → Console  
Should see no errors

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| Backend won't start | `pip install -r requirements.txt` |
| Port 8000 in use | Change port: `--port 8001` |
| Frontend can't connect | Check backend is running |
| Videos don't play | Check `output/` directory has files |

## 📋 Checklist

- [ ] Backend starts without errors
- [ ] Health endpoint returns OK
- [ ] Frontend loads in browser
- [ ] No console errors
- [ ] Can submit form
- [ ] Clips generate successfully
- [ ] Videos play in browser
- [ ] Download works
- [ ] Copy link works

## 🎯 Success = All Checkboxes ✅

For detailed testing instructions, see `TESTING_GUIDE.md`

