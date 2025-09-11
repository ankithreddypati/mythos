// src/App.jsx
import React, { useState, useEffect, useRef } from 'react';
import AudioInput from './components/AudioInput';
import VideoPlayer from './components/VideoPlayer';
import RobotSpace from './components/RobotSpace';
import OverlayPolys from './components/OverlayPolys';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';

export default function App() {
  const [transcription, setTranscription] = useState('');
  const [videoUrl, setVideoUrl] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');
  const [robotActions, setRobotActions] = useState([]);

  const handleAudioRecorded = async (audioBlob) => {
    setIsProcessing(true);
    setError('');
    setTranscription('Processing...');

    try {
      // Transcribe audio
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.wav');

      const transcribeResponse = await fetch(`${BACKEND_URL}/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!transcribeResponse.ok) {
        throw new Error('Transcription failed');
      }

      const transcribeData = await transcribeResponse.json();
      setTranscription(transcribeData.text);

      // Generate presentation
      const renderResponse = await fetch(`${BACKEND_URL}/render`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: transcribeData.text,
          voice: 'af_heart',
          lang_code: 'a',
          speed: 1.0,
          reasoning_effort: 'high'
        }),
      });

      if (!renderResponse.ok) {
        throw new Error('Rendering failed');
      }

      const renderData = await renderResponse.json();
      
      if (renderData.audio_url) {
        setAudioUrl(`${BACKEND_URL}${renderData.audio_url}`);
      }
      
      if (renderData.video_url) {
        setVideoUrl(`${BACKEND_URL}${renderData.video_url}`);
      }

      if (renderData.robot_actions) {
        setRobotActions(renderData.robot_actions);
      }

    } catch (err) {
      setError(`Error: ${err.message}`);
      console.error('Processing error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div 
      style={{ 
        width: '100vw',
        height: '100vh',
        backgroundImage: `url('img.png')`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gridTemplateRows: '1fr 1fr',
        margin: 0,
        padding: 0,
        overflow: 'hidden'
      }}
    >
      {/* Top Left */}
      <div style={{ position: 'relative' }}>
        <AudioInput 
          onAudioRecorded={handleAudioRecorded}
          transcription={transcription}
          isProcessing={isProcessing}
          error={error}
        />
      </div>

      {/* Top Right */}
      <div style={{ position: 'relative' }}>
        <VideoPlayer 
          videoUrl={videoUrl}
          audioUrl={audioUrl}
        />
      </div>

      {/* Bottom Left */}
      <div style={{ position: 'relative' }}>
        <RobotSpace 
          robotActions={robotActions}
          isActive={isProcessing}
        />
      </div>

      {/* Bottom Right */}
      <div style={{ position: 'relative' }}>
        {/* Empty space for projection mapping */}
      </div>

      {/* Overlay Projection Mapping */}
      <OverlayPolys 
        visible={true}
        opacity={0.8}
        zIndex={2147483647}
      />
    </div>
  );
}