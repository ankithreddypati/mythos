// src/components/VideoPlayer.jsx
import React, { useEffect, useRef } from 'react';

export default function VideoPlayer({ videoUrl, audioUrl }) {
  const videoRef = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    if (audioUrl && audioRef.current) {
      audioRef.current.play().catch(console.error);
    }
  }, [audioUrl]);

  useEffect(() => {
    if (videoUrl && videoRef.current) {
      videoRef.current.play().catch(console.error);
    }
  }, [videoUrl]);

  return (
    <div style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center'
    }}>
      
      {videoUrl ? (
        <video
          ref={videoRef}
          src={videoUrl}
          style={{ width: '90%', height: '90%' }}
          controls={false}
          autoPlay
          loop
          muted
          playsInline
        />
      ) : (
        <div style={{ color: 'white', fontSize: '24px' }}>
        </div>
      )}

      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          autoPlay
        />
      )}
    </div>
  );
}