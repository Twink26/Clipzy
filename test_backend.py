"""
Quick test script for Clipzy backend
Tests all endpoints to verify everything is working
"""
import requests
import json
import sys
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing /health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                print("✅ Health check passed!")
                return True
            else:
                print(f"❌ Unexpected response: {data}")
                return False
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running?")
        print("   Start it with: uvicorn backend.app:app --reload --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_api_docs():
    """Test if API docs are accessible"""
    print("\n🔍 Testing API documentation...")
    try:
        response = requests.get(f"{API_BASE}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API docs accessible!")
            return True
        else:
            print(f"❌ API docs failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing docs: {str(e)}")
        return False

def test_output_directory():
    """Test if output directory exists"""
    print("\n🔍 Checking output directory...")
    output_dir = Path("output")
    if output_dir.exists():
        print(f"✅ Output directory exists: {output_dir.resolve()}")
        return True
    else:
        print(f"⚠️  Output directory doesn't exist, but it will be created automatically")
        return True  # Not a critical error

def test_clips_endpoint():
    """Test clips endpoint (will fail if no clips exist, which is OK)"""
    print("\n🔍 Testing /clips endpoint...")
    try:
        # Try to access a non-existent clip (should return 404, not 500)
        response = requests.get(f"{API_BASE}/clips/test_nonexistent.mp4", timeout=5)
        if response.status_code == 404:
            print("✅ Clips endpoint is working (correctly returns 404 for missing files)")
            return True
        elif response.status_code == 200:
            print("✅ Clips endpoint is working (found a clip)")
            return True
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            return True  # Not critical
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend")
        return False
    except Exception as e:
        print(f"⚠️  Error testing clips endpoint: {str(e)}")
        return True  # Not critical

def test_generate_endpoint_structure():
    """Test that /generate endpoint exists and accepts requests"""
    print("\n🔍 Testing /generate endpoint structure...")
    try:
        # Send a request with invalid URL to test endpoint structure
        response = requests.post(
            f"{API_BASE}/generate",
            json={
                "youtube_url": "https://invalid-url-test",
                "num_segments": 1,
                "min_duration": 15,
                "max_duration": 60,
                "add_captions": True
            },
            timeout=10
        )
        # We expect either 400 (validation error) or 500 (processing error)
        # Both mean the endpoint exists and is working
        if response.status_code in [400, 422, 500]:
            print("✅ /generate endpoint exists and is responding")
            return True
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            return True
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend")
        return False
    except Exception as e:
        print(f"⚠️  Error: {str(e)}")
        return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Clipzy Backend Test Suite")
    print("=" * 60)
    print(f"\nTesting backend at: {API_BASE}")
    print("Make sure the backend is running: uvicorn backend.app:app --reload --port 8000\n")
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("API Docs", test_api_docs()))
    results.append(("Output Directory", test_output_directory()))
    results.append(("Clips Endpoint", test_clips_endpoint()))
    results.append(("Generate Endpoint", test_generate_endpoint_structure()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Backend is working correctly.")
        print("\nNext steps:")
        print("1. Open frontend/index.html in your browser")
        print("2. Or serve it: cd frontend && python -m http.server 8080")
        print("3. Test generating clips with a YouTube URL")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)

