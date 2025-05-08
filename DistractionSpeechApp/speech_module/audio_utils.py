import os
import wave
import queue
import pyaudio
import tempfile
import logging
import threading
from typing import Optional, Callable, BinaryIO

# Configure logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('audio_utils')

class AudioProcessor:
    """
    Handles audio capture, processing and temporary storage
    """
    
    def __init__(self, 
                 rate: int = 16000, 
                 chunk_size: Optional[int] = None,
                 channels: int = 1):
        """Initialize audio processor
        
        Args:
            rate: Audio sample rate (Hz)
            chunk_size: Audio chunk size (if None, calculated from rate)
            channels: Number of audio channels
        """
        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size if chunk_size else int(rate * 0.5)
        self.format = pyaudio.paInt16
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        
        logger.info(f"AudioProcessor initialized: rate={rate}, chunk_size={self.chunk_size}")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function for streaming audio data
        
        Args:
            in_data: Recorded audio data
            frame_count: Number of frames
            time_info: Timing information
            status: Status flag
        
        Returns:
            tuple: (None, pyaudio.paContinue)
        """
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def start_stream(self):
        """Start the audio stream and begin processing audio
        
        Returns:
            AudioProcessor: self for chaining
        """
        if self.stream is not None and self.stream.is_active():
            logger.warning("Audio stream already active")
            return self
        
        try:
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            self.is_recording = True
            logger.info("Audio stream started")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
        
        return self
    
    def stop_stream(self):
        """Stop the audio stream
        
        Returns:
            AudioProcessor: self for chaining
        """
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            self.is_recording = False
            logger.info("Audio stream stopped")
        
        return self
    
    def close(self):
        """Clean up resources"""
        self.stop_stream()
        self.pyaudio.terminate()
        logger.info("Audio processor closed")
    
    def get_audio_chunk(self, timeout=1.0):
        """Get a chunk of audio from the queue
        
        Args:
            timeout: Queue timeout in seconds
        
        Returns:
            bytes: Audio data or None if queue is empty
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def save_audio_to_wav(self, audio_data, filename=None):
        """Save audio data to a WAV file
        
        Args:
            audio_data: Audio data bytes
            filename: Output filename (if None, creates a temp file)
        
        Returns:
            str: Path to saved WAV file
        """
        if filename is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                filename = tmpfile.name
        
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(audio_data)
            
            logger.info(f"Audio saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving audio to {filename}: {e}")
            return None
    
    def record_until_silence(self, silence_threshold=3.0, max_duration=30.0):
        """Record audio until a period of silence
        
        Args:
            silence_threshold: Seconds of silence to stop recording
            max_duration: Maximum recording duration in seconds
        
        Returns:
            bytes: Recorded audio data
        """
        if not self.is_recording:
            self.start_stream()
        
        buffer = b""
        last_audio = time.time()
        start_time = time.time()
        
        while time.time() - start_time < max_duration:
            chunk = self.get_audio_chunk()
            if chunk:
                buffer += chunk
                last_audio = time.time()
            elif time.time() - last_audio > silence_threshold:
                break
        
        return buffer