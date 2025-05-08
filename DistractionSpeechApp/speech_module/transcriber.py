import os
import time
import queue
import logging
import threading
from typing import List, Optional, Callable
from faster_whisper import WhisperModel
from .audio_utils import AudioProcessor

# Configure logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('transcriber')

class SpeechTranscriber:
    """
    Transcribes speech from audio using Whisper model
    """
    
    def __init__(self, 
                 model_size: str = "tiny", 
                 compute_type: str = "int8",
                 language: str = "en",
                 audio_processor: Optional[AudioProcessor] = None):
        """Initialize the speech transcriber
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            compute_type: Computation type ("int8", "float16", "float32")
            language: Language code
            audio_processor: Optional AudioProcessor instance
        """
        self.model_size = model_size
        self.compute_type = compute_type
        self.language = language
        self.model = None
        self.audio_processor = audio_processor
        self.transcripts = []
        self.is_running = False
        self.transcription_thread = None
        
        # Load the model
        self._load_model()
        
        logger.info(f"SpeechTranscriber initialized: model={model_size}, compute={compute_type}")
    
    def _load_model(self):
        """Load the Whisper model"""
        try:
            self.model = WhisperModel(self.model_size, compute_type=self.compute_type)
            logger.info(f"Whisper model '{self.model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_file(self, audio_file_path: str):
        """Transcribe audio from a file
        
        Args:
            audio_file_path: Path to audio file
        
        Returns:
            str: Transcribed text
        """
        try:
            logger.info(f"Transcribing file: {audio_file_path}")
            segments, _ = self.model.transcribe(
                audio_file_path, 
                language=self.language,
                vad_filter=True
            )
            
            segments_list = list(segments)  # Force evaluate the generator
            full_text = " ".join([seg.text.strip() for seg in segments_list])
            
            self.transcripts.append(full_text)
            logger.info(f"Transcription completed: {len(full_text)} chars")
            
            return full_text
        except Exception as e:
            logger.error(f"Error transcribing file {audio_file_path}: {e}")
            return ""
    
    def transcribe_audio(self, audio_data: bytes):
        """Transcribe audio from bytes data
        
        Args:
            audio_data: Raw audio data
        
        Returns:
            str: Transcribed text
        """
        if not audio_data:
            return ""
            
        if not self.audio_processor:
            self.audio_processor = AudioProcessor()
            
        # Save audio to temporary file
        audio_file = self.audio_processor.save_audio_to_wav(audio_data)
        if not audio_file:
            return ""
            
        # Transcribe the audio file
        result = self.transcribe_file(audio_file)
        
        # Clean up temporary file
        try:
            os.remove(audio_file)
        except:
            pass
            
        return result
    
    def start_continuous_transcription(self, 
                                      collection_interval: float = 3.0,
                                      callback: Optional[Callable[[str], None]] = None):
        """Start continuous transcription in a background thread
        
        Args:
            collection_interval: Seconds between processing audio chunks
            callback: Function to call with each transcription
            
        Returns:
            SpeechTranscriber: self for chaining
        """
        if self.is_running:
            logger.warning("Transcription already running")
            return self
            
        if not self.audio_processor:
            self.audio_processor = AudioProcessor()
            self.audio_processor.start_stream()
        
        self.is_running = True
        self.transcription_thread = threading.Thread(
            target=self._transcription_loop,
            args=(collection_interval, callback),
            daemon=True
        )
        self.transcription_thread.start()
        
        logger.info("Started continuous transcription")
        return self
    
    def stop_continuous_transcription(self):
        """Stop the continuous transcription thread
        
        Returns:
            SpeechTranscriber: self for chaining
        """
        self.is_running = False
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=1.0)
        
        logger.info("Stopped continuous transcription")
        return self
    
    def _transcription_loop(self, collection_interval, callback):
        """Background thread for continuous transcription
        
        Args:
            collection_interval: Seconds between processing
            callback: Function to call with each transcription
        """
        buffer = b""
        last_processed = time.time()
        
        while self.is_running:
            try:
                # Get audio chunk from queue
                chunk = self.audio_processor.get_audio_chunk(timeout=0.1)
                if chunk:
                    buffer += chunk
                
                # Process accumulated audio after interval
                if time.time() - last_processed > collection_interval and buffer:
                    # Transcribe the accumulated audio
                    text = self.transcribe_audio(buffer)
                    
                    # Call the callback if provided
                    if text and callback:
                        callback(text)
                    
                    # Reset buffer and timer
                    buffer = b""
                    last_processed = time.time()
            except Exception as e:
                logger.error(f"Error in transcription loop: {e}")
                time.sleep(1)  # Avoid tight loop on errors
    
    def get_recent_transcripts(self, count: int = 5) -> List[str]:
        """Get the most recent transcripts
        
        Args:
            count: Number of recent transcripts to return
            
        Returns:
            List[str]: Recent transcripts
        """
        return self.transcripts[-count:] if self.transcripts else []
    
    def clear_transcripts(self):
        """Clear the stored transcripts"""
        self.transcripts.clear()
    
    def close(self):
        """Clean up resources"""
        self.stop_continuous_transcription()
        if self.audio_processor:
            self.audio_processor.close()