# Testing Guide for Clipzy

This guide will help you test if everything is working correctly.

## Prerequisites Check

Before testing, ensure you have:

1. **Python 3.8+** installed
   ```bash
   python --version
   ```

2. **All dependencies installed**
   ```bash
   pip install -r requirements.txt
   ```

3. **FFmpeg installed** (required for video processing)
   ```bash
   ffmpeg -version
   ```
   If not installed, download from: https://ffmpeg.org/download.html

4. **Output directory exists**
   ```bash
   # The output directory should be created automatically, but verify:
   ls output/
   ```

## Step-by-Step Testing

### Step 1: Test Backend Health Endpoint

1. **Start the backend server:**
   ```bash
   uvicorn backend.app:app --reload --port 8000
   ```

2. **In a new terminal, test the health endpoint:**
   ```bash
   curl http://localhost:8000/health
   ```
   
   Or open in browser: http://localhost:8000/health
   
   **Expected response:**
   ```json
   {"status": "ok"}
   ```

3. **Check API documentation:**
   - Open: http://localhost:8000/docs
   - You should see the Swagger UI with all available endpoints

### Step 2: Test Frontend Connection

1. **Open the frontend:**
   - **Option A (Simple):** Double-click `frontend/index.html` to open in browser
   - **Option B (Recommended):** Serve it with a local server:
     ```bash
     cd frontend
     python -m http.server 8080
     ```
     Then visit: http://localhost:8080

2. **Check browser console:**
   - Open Developer Tools (F12)
   - Go to Console tab
   - You should see no errors
   - The form should be visible and functional

3. **Verify API connection:**
   - In browser console, type:
     ```javascript
     fetch('http://localhost:8000/health')
       .then(r => r.json())
       .then(console.log)
     ```
   - Should return: `{status: "ok"}`

### Step 3: Test Full Workflow (Generate Clips)

**Important:** This will download and process a YouTube video, which can take several minutes.

1. **Prepare a test YouTube URL:**
   - Use a short podcast video (5-15 minutes) for faster testing
   - Example: Any YouTube podcast video URL

2. **In the frontend:**
   - Paste the YouTube URL
   - Set segments to 2-3 (for faster testing)
   - Set min duration: 15 seconds
   - Set max duration: 60 seconds
   - Check "Add captions"
   - Click "Generate Reels"

3. **Monitor the process:**
   - Status box should show "Processing..." with a spinner
   - Backend terminal will show progress logs
   - This can take 5-15 minutes depending on video length

4. **Expected results:**
   - Status changes to "Success! Generated X reels."
   - Video cards appear with:
     - Video preview player
     - Viral score badge
     - Duration and time range
     - Preview text
     - Download and Copy Link buttons

5. **Test video playback:**
   - Click play on any video card
   - Video should play in the browser

6. **Test download:**
   - Click "Download" button
   - Video file should download

7. **Test copy link:**
   - Click "Copy Link" button
   - Button should change to "Copied!"
   - Paste in new tab to verify link works

### Step 4: Test API Endpoints Directly

You can also test the API using curl or Postman:

1. **Test /generate endpoint:**
   ```bash
   curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "youtube_url": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
       "num_segments": 2,
       "min_duration": 15,
       "max_duration": 60,
       "add_captions": true
     }'
   ```

2. **Test /clips endpoint (after generating):**
   ```bash
   # Replace FILENAME with actual generated clip filename
   curl http://localhost:8000/clips/FILENAME.mp4
   ```

## Quick Test Script

Run this Python script to quickly test the backend:

```bash
python test_backend.py
```

## Troubleshooting

### Backend won't start
- **Error: "Module not found"**
  - Solution: Install dependencies: `pip install -r requirements.txt`

- **Error: "Address already in use"**
  - Solution: Change port: `uvicorn backend.app:app --port 8001`

### Frontend can't connect to backend
- **CORS errors in console**
  - Solution: Ensure backend is running and CORS is enabled (it should be)

- **404 errors**
  - Solution: Check `API_BASE` in frontend matches backend URL
  - Default is `http://localhost:8000`

### Videos not playing
- **Video player shows error**
  - Check: Backend is running
  - Check: Video file exists in `output/` directory
  - Check: Browser console for errors

- **Download doesn't work**
  - Check: Browser allows downloads
  - Check: File path is correct in API response

### Generation fails
- **"No interesting segments found"**
  - Try: Different YouTube video
  - Try: Adjust min/max duration
  - Try: Increase number of segments

- **Transcription errors**
  - Check: FFmpeg is installed
  - Check: Internet connection (for YouTube download)
  - Check: Video is not private/restricted

## Expected File Structure After Testing

```
clipzy/
├── output/
│   ├── reel_VIDEO_ID_START_END.mp4  (generated clips)
│   └── ...
├── data/
│   └── downloads/
│       └── VIDEO_ID.mp4  (downloaded video)
└── ...
```

## Performance Notes

- **First run:** May take longer (model loading, dependencies)
- **Video download:** Depends on video length and internet speed
- **Transcription:** ~1-2 minutes per 10 minutes of video
- **Clip generation:** ~30 seconds per clip

## Success Criteria

✅ Backend health endpoint returns `{"status": "ok"}`  
✅ Frontend loads without errors  
✅ Can submit form and see loading state  
✅ Clips are generated successfully  
✅ Video previews work in browser  
✅ Download buttons work  
✅ Copy link button works  
✅ No console errors in browser  
✅ No errors in backend terminal  

If all these pass, everything is working correctly! 🎉

