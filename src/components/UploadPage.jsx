import React, { useState, useRef } from 'react'

const UploadPage = () => {
  const [videoUrl, setVideoUrl] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileSelect = (e) => {
    const files = e.target.files
    if (files.length > 0) {
      setSelectedFile(files[0])
      setVideoUrl('') 
    }
  }

  const handleBrowseClick = () => {
    fileInputRef.current?.click()
  }

  const handleSubmit = async () => {
    if (!videoUrl && !selectedFile) {
      alert('Please provide either a video URL or select a file')
      return
    }

    setIsSubmitting(true)
    
    // Simulate API call
    setTimeout(() => {
      setIsSubmitting(false)
      if (videoUrl) {
        alert(`Processing video from URL: ${videoUrl}`)
      } else {
        alert(`Processing uploaded file: ${selectedFile.name}`)
      }
    }, 2000)
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-slate-900 to-slate-950 flex flex-col items-center justify-center p-4">
      {/* Welcome Message */}
      <div className="text-center mb-12">
        <h1 className="text-6xl font-bold text-shadow-black font-serif text-shadow-2xs text-white mb-4 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          Welcome to Clipzy
        </h1>
        <p className="text-gray-300 text-xl">Paste your video URL to get started</p>
      </div>

      {/* URL Input Section */}
      <div className="w-full max-w-2xl">
        <div className="relative">
          <input
            type="url"
            id="videoUrl"
            value={videoUrl}
            onChange={(e) => {
              setVideoUrl(e.target.value)
              if (e.target.value) setSelectedFile(null) // Clear file when URL is entered
            }}
            placeholder="Paste your video URL here (YouTube, Vimeo, etc.)"
            className="w-full px-6 py-4 text-lg bg-gray-800/50 border border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-white placeholder-gray-400 backdrop-blur-sm hover:bg-gray-800/70"
          />
          <div className="absolute inset-y-0 right-0 flex items-center pr-4">
            <svg className="h-6 w-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
            </svg>
          </div>
        </div>
      </div>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept="video/*,.mp4,.avi,.mov,.wmv,.mkv,.webm"
        onChange={handleFileSelect}
      />
    </div>
  )
}

export default UploadPage
