// src/components/AudioInput.jsx
import React, { useState, useEffect, useRef } from 'react';

export default function AudioInput({ onAudioRecorded, transcription, isProcessing, error }) {
  const [isListening, setIsListening] = useState(false);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key.toLowerCase() === 'h') {
        e.preventDefault();
        if (isListening) {
          stopRecording();
        } else {
          startRecording();
        }
      }
    };

    window.addEventListener('keypress', handleKeyPress);
    return () => window.removeEventListener('keypress', handleKeyPress);
  }, [isListening]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        onAudioRecorded(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsListening(true);
    } catch (err) {
      console.error('Error accessing microphone:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isListening) {
      mediaRecorderRef.current.stop();
      setIsListening(false);
    }
  };

  return (
    <div style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      padding: '20px',
      color: 'black',
      fontSize: '28px'
    }}>
      
      {isListening && <div> LISTENING</div>}
      {isProcessing && <div> PROCESSING</div>}
      {!isListening && !isProcessing && <div>Press H to talk</div>}
      
      {transcription && (
        <div style={{ marginTop: '20px', textAlign: 'center' }}>
          {transcription}
        </div>
      )}
      
      {error && (
        <div style={{ marginTop: '20px', color: 'red' }}>
          {error}
        </div>
      )}
    </div>
  );
}