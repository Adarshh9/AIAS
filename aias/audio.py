"""
AIAS Audio Pipeline
Handles hotword detection, voice activity detection, and speech-to-text
"""

import os
import time
import queue
import struct
import logging
import threading
import numpy as np
from typing import Optional, Callable, List
from collections import deque

logger = logging.getLogger(__name__)


class HotwordDetector:
    """
    Hotword detection using OpenWakeWord (open source, MIT license)
    Falls back to Porcupine if available, or VAD-based activation
    
    OpenWakeWord: https://github.com/dscripka/openWakeWord
    - 100% free and open source (MIT license)
    - Works fully offline
    - Supports custom wake words (train your own!)
    - Pre-trained models: "hey jarvis", "alexa", "hey mycroft", etc.
    """
    
    # OpenWakeWord built-in models
    OPENWAKEWORD_MODELS = [
        "hey_jarvis",        # Best for custom assistant
        "alexa",             # Amazon-style
        "hey_mycroft",       # Mycroft assistant
        "hey_rhasspy",       # Rhasspy assistant
        "current_weather",   # Command detection
        "timers",            # Command detection
    ]
    
    # Porcupine fallback keywords (requires API key for most)
    PORCUPINE_KEYWORDS = [
        "porcupine", "bumblebee", "alexa", "hey google", 
        "hey siri", "jarvis", "picovoice", "computer",
    ]
    
    def __init__(
        self,
        keyword: str = "hey_jarvis",
        sensitivity: float = 0.5,
        access_key: Optional[str] = None,
        prefer_openwakeword: bool = True
    ):
        self.keyword = keyword
        self.sensitivity = sensitivity
        self.access_key = access_key or os.environ.get("PICOVOICE_ACCESS_KEY", "")
        self.prefer_openwakeword = prefer_openwakeword
        
        self._oww_model = None
        self._porcupine = None
        self._backend = None  # 'openwakeword', 'porcupine', or 'fallback'
        self._frame_buffer = []
        
        self._initialize()
    
    def _initialize(self):
        """Initialize wake word detector (OpenWakeWord preferred)"""
        
        # Try OpenWakeWord first (recommended - fully open source)
        if self.prefer_openwakeword:
            if self._try_openwakeword():
                return
        
        # Fallback to Porcupine
        if self._try_porcupine():
            return
        
        # Final fallback: VAD-based detection
        logger.warning("No wake word engine available. Using VAD-based fallback.")
        self._backend = "fallback"
    
    def _try_openwakeword(self) -> bool:
        """Try to initialize OpenWakeWord"""
        try:
            import openwakeword
            from openwakeword.model import Model
            
            # Map common keywords to OpenWakeWord model names
            keyword_mapping = {
                "jarvis": "hey_jarvis",
                "hey jarvis": "hey_jarvis",
                "hey_jarvis": "hey_jarvis",
                "alexa": "alexa",
                "mycroft": "hey_mycroft",
                "hey mycroft": "hey_mycroft",
            }
            
            model_name = keyword_mapping.get(self.keyword.lower(), self.keyword)
            
            # Download models if needed (first run only)
            logger.info(f"Loading OpenWakeWord model: {model_name}")
            
            # Initialize model
            self._oww_model = Model(
                wakeword_models=[model_name],
                inference_framework="onnx"  # CPU-friendly
            )
            
            self._backend = "openwakeword"
            logger.info(f"✅ OpenWakeWord initialized with '{model_name}' (open source, MIT license)")
            return True
            
        except ImportError:
            logger.info("OpenWakeWord not installed. Install with: pip install openwakeword")
            return False
        except Exception as e:
            logger.warning(f"OpenWakeWord initialization failed: {e}")
            return False
    
    def _try_porcupine(self) -> bool:
        """Try to initialize Porcupine (fallback)"""
        try:
            import pvporcupine
            
            # Porcupine requires access key for most functionality
            if not self.access_key:
                logger.info("Porcupine requires access key. Get free key at https://picovoice.ai/")
                return False
            
            keyword = self.keyword.lower().replace("_", " ")
            if keyword not in [k.lower() for k in self.PORCUPINE_KEYWORDS]:
                keyword = "jarvis"  # Default fallback
            
            self._porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=[keyword],
                sensitivities=[self.sensitivity]
            )
            
            self._backend = "porcupine"
            logger.info(f"Porcupine initialized with keyword: {keyword}")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"Porcupine initialization failed: {e}")
            return False
    
    @property
    def frame_length(self) -> int:
        """Required audio frame length for processing"""
        if self._backend == "openwakeword":
            return 1280  # OpenWakeWord uses 80ms frames at 16kHz
        elif self._backend == "porcupine" and self._porcupine:
            return self._porcupine.frame_length
        return 512  # Default
    
    @property
    def sample_rate(self) -> int:
        """Required sample rate"""
        return 16000  # All backends use 16kHz
    
    def process(self, pcm: List[int]) -> bool:
        """
        Process audio frame and detect hotword
        
        Args:
            pcm: Audio frame as list of 16-bit integers
            
        Returns:
            True if hotword detected, False otherwise
        """
        if self._backend == "openwakeword":
            return self._process_openwakeword(pcm)
        elif self._backend == "porcupine":
            return self._process_porcupine(pcm)
        else:
            return self._process_fallback(pcm)
    
    def _process_openwakeword(self, pcm: List[int]) -> bool:
        """Process with OpenWakeWord"""
        try:
            # Convert to numpy array
            audio = np.array(pcm, dtype=np.int16)
            
            # OpenWakeWord expects specific chunk sizes, buffer if needed
            self._frame_buffer.extend(audio.tolist())
            
            # Process when we have enough samples (80ms = 1280 samples)
            if len(self._frame_buffer) >= 1280:
                chunk = np.array(self._frame_buffer[:1280], dtype=np.int16)
                self._frame_buffer = self._frame_buffer[1280:]
                
                # Run prediction
                prediction = self._oww_model.predict(chunk)
                
                # Check if any model triggered
                for model_name, scores in prediction.items():
                    # scores is typically the latest prediction score
                    score = scores if isinstance(scores, float) else scores[-1] if len(scores) > 0 else 0
                    if score > self.sensitivity:
                        logger.info(f"🎤 Wake word detected: {model_name} (score: {score:.2f})")
                        self._oww_model.reset()  # Reset for next detection
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"OpenWakeWord processing error: {e}")
            return False
    
    def _process_porcupine(self, pcm: List[int]) -> bool:
        """Process with Porcupine"""
        if self._porcupine:
            keyword_index = self._porcupine.process(pcm)
            if keyword_index >= 0:
                logger.info("🎤 Wake word detected (Porcupine)")
                return True
        return False
    
    def _process_fallback(self, pcm: List[int]) -> bool:
        """Fallback: simple energy-based voice detection"""
        # Calculate RMS energy
        audio = np.array(pcm, dtype=np.float32)
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Very basic threshold - not a real wake word detector
        # This just detects loud sounds as a last resort
        threshold = 3000 * self.sensitivity
        return rms > threshold
    
    def cleanup(self):
        """Release resources"""
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None
        
        if self._oww_model:
            self._oww_model = None
        
        self._frame_buffer = []
    
    @property
    def backend_name(self) -> str:
        """Get the name of the active backend"""
        return self._backend or "none"


class VoiceActivityDetector:
    """Voice Activity Detection using WebRTC VAD"""
    
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        """
        Args:
            aggressiveness: 0-3, higher = more aggressive filtering
            sample_rate: Must be 8000, 16000, 32000, or 48000
        """
        self.aggressiveness = aggressiveness
        self.sample_rate = sample_rate
        self._vad = None
        self._initialize()
    
    def _initialize(self):
        """Initialize WebRTC VAD"""
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.aggressiveness)
            logger.info(f"WebRTC VAD initialized (aggressiveness={self.aggressiveness})")
        except ImportError:
            logger.warning("webrtcvad not installed. VAD disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize VAD: {e}")
    
    def is_speech(self, audio_frame: bytes, sample_rate: Optional[int] = None) -> bool:
        """
        Check if audio frame contains speech
        
        Args:
            audio_frame: Audio data as bytes (10, 20, or 30ms of audio)
            sample_rate: Sample rate (uses default if not provided)
            
        Returns:
            True if speech detected
        """
        if not self._vad:
            return True  # Assume speech if VAD unavailable
        
        try:
            return self._vad.is_speech(audio_frame, sample_rate or self.sample_rate)
        except Exception:
            return True


class SpeechToText:
    """Speech-to-text using Faster-Whisper"""
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = "en"
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        
        self._model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Faster-Whisper model"""
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("Whisper model loaded successfully")
            
        except ImportError:
            logger.error("faster-whisper not installed. STT unavailable.")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio: Audio data as numpy array (float32, normalized)
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        if not self._model:
            return ""
        
        try:
            # Ensure correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0
            
            segments, info = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Combine all segments
            text = " ".join([segment.text.strip() for segment in segments])
            
            logger.debug(f"Transcribed: {text[:100]}...")
            return text
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""


class AudioCapture:
    """Continuous audio capture with circular buffer"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,
        buffer_seconds: float = 30.0
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.buffer_seconds = buffer_seconds
        
        self._stream = None
        self._audio_queue = queue.Queue()
        self._buffer = deque(maxlen=int(sample_rate * buffer_seconds / chunk_size))
        self._is_recording = False
        self._recording_buffer = []
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio capture"""
        import pyaudio
        
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        self._audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def start(self):
        """Start audio capture"""
        try:
            import pyaudio
            
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self._stream.start_stream()
            logger.info("Audio capture started")
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            raise
    
    def stop(self):
        """Stop audio capture"""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        if hasattr(self, '_pa'):
            self._pa.terminate()
        
        logger.info("Audio capture stopped")
    
    def get_chunk(self, timeout: float = 0.1) -> Optional[bytes]:
        """Get next audio chunk from queue"""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def start_recording(self):
        """Start recording speech"""
        self._is_recording = True
        self._recording_buffer = []
    
    def add_to_recording(self, audio_chunk: bytes):
        """Add audio chunk to recording buffer"""
        if self._is_recording:
            self._recording_buffer.append(audio_chunk)
    
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return audio as numpy array"""
        self._is_recording = False
        
        if not self._recording_buffer:
            return np.array([], dtype=np.float32)
        
        # Combine all chunks
        audio_bytes = b''.join(self._recording_buffer)
        self._recording_buffer = []
        
        # Convert to numpy array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        return audio_float32


class AudioPipeline:
    """
    Complete audio pipeline combining hotword detection, VAD, and STT
    """
    
    def __init__(
        self,
        hotword_keyword: str = "porcupine",
        hotword_sensitivity: float = 0.5,
        stt_model_size: str = "base",
        stt_device: str = "cuda",
        stt_compute_type: str = "float16",
        stt_language: str = "en",
        sample_rate: int = 16000,
        silence_threshold_ms: int = 1000,
        max_recording_seconds: int = 30,
        on_query_callback: Optional[Callable[[str], None]] = None
    ):
        self.sample_rate = sample_rate
        self.silence_threshold_ms = silence_threshold_ms
        self.max_recording_seconds = max_recording_seconds
        self.on_query_callback = on_query_callback
        
        # Initialize components
        self.hotword = HotwordDetector(
            keyword=hotword_keyword,
            sensitivity=hotword_sensitivity
        )
        
        self.vad = VoiceActivityDetector(
            aggressiveness=2,
            sample_rate=sample_rate
        )
        
        self.stt = SpeechToText(
            model_size=stt_model_size,
            device=stt_device,
            compute_type=stt_compute_type,
            language=stt_language
        )
        
        self.capture = AudioCapture(
            sample_rate=sample_rate,
            channels=1,
            chunk_size=self.hotword.frame_length
        )
        
        self._running = False
        self._thread = None
        self._state = "listening"  # listening, recording, processing
    
    def start(self):
        """Start the audio pipeline"""
        self._running = True
        self.capture.start()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("Audio pipeline started")
    
    def stop(self):
        """Stop the audio pipeline"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.capture.stop()
        self.hotword.cleanup()
        logger.info("Audio pipeline stopped")
    
    def _process_loop(self):
        """Main processing loop"""
        silence_frames = 0
        frames_per_ms = self.sample_rate / 1000
        silence_threshold_frames = int(self.silence_threshold_ms * frames_per_ms / self.hotword.frame_length)
        max_recording_frames = int(self.max_recording_seconds * self.sample_rate / self.hotword.frame_length)
        recording_frames = 0
        
        while self._running:
            audio_chunk = self.capture.get_chunk()
            if audio_chunk is None:
                continue
            
            # Convert to PCM for Porcupine
            pcm = struct.unpack_from("h" * self.hotword.frame_length, audio_chunk)
            
            if self._state == "listening":
                # Check for hotword
                if self.hotword.process(list(pcm)):
                    logger.info("Hotword detected! Starting recording...")
                    self._state = "recording"
                    self.capture.start_recording()
                    silence_frames = 0
                    recording_frames = 0
                    
            elif self._state == "recording":
                self.capture.add_to_recording(audio_chunk)
                recording_frames += 1
                
                # Check for speech
                is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
                
                if is_speech:
                    silence_frames = 0
                else:
                    silence_frames += 1
                
                # Check if we should stop recording
                should_stop = (
                    silence_frames >= silence_threshold_frames or
                    recording_frames >= max_recording_frames
                )
                
                if should_stop:
                    self._state = "processing"
                    audio = self.capture.stop_recording()
                    
                    if len(audio) > 0:
                        logger.info(f"Recording complete. Processing {len(audio) / self.sample_rate:.1f}s of audio...")
                        text = self.stt.transcribe(audio, self.sample_rate)
                        
                        if text.strip():
                            logger.info(f"Query: {text}")
                            if self.on_query_callback:
                                self.on_query_callback(text)
                        else:
                            logger.info("No speech detected in recording")
                    
                    self._state = "listening"
    
    @property
    def state(self) -> str:
        """Current pipeline state"""
        return self._state
    
    def manual_trigger(self):
        """Manually trigger recording (bypass hotword)"""
        if self._state == "listening":
            logger.info("Manual trigger activated")
            self._state = "recording"
            self.capture.start_recording()


# Alternative: Simple always-listening mode without hotword
class SimpleAudioPipeline:
    """
    Simplified audio pipeline using keyboard shortcut activation
    No hotword detection - uses push-to-talk style activation
    """
    
    def __init__(
        self,
        stt_model_size: str = "base",
        stt_device: str = "cuda",
        stt_compute_type: str = "float16",
        stt_language: str = "en",
        sample_rate: int = 16000,
        on_query_callback: Optional[Callable[[str], None]] = None
    ):
        self.sample_rate = sample_rate
        self.on_query_callback = on_query_callback
        
        self.stt = SpeechToText(
            model_size=stt_model_size,
            device=stt_device,
            compute_type=stt_compute_type,
            language=stt_language
        )
        
        self.vad = VoiceActivityDetector(
            aggressiveness=2,
            sample_rate=sample_rate
        )
        
        self._recording = False
        self._audio_buffer = []
        self._stream = None
    
    def start_recording(self):
        """Start recording audio"""
        import pyaudio
        
        self._recording = True
        self._audio_buffer = []
        
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        # Recording thread
        def record():
            while self._recording:
                try:
                    data = self._stream.read(1024, exception_on_overflow=False)
                    self._audio_buffer.append(data)
                except Exception as e:
                    logger.error(f"Recording error: {e}")
                    break
        
        self._record_thread = threading.Thread(target=record, daemon=True)
        self._record_thread.start()
        logger.info("Recording started")
    
    def stop_recording(self) -> str:
        """Stop recording and transcribe"""
        self._recording = False
        
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        
        if hasattr(self, '_pa'):
            self._pa.terminate()
        
        if not self._audio_buffer:
            return ""
        
        # Convert to numpy
        audio_bytes = b''.join(self._audio_buffer)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Transcribe
        text = self.stt.transcribe(audio_float32, self.sample_rate)
        logger.info(f"Transcribed: {text}")
        
        if text.strip() and self.on_query_callback:
            self.on_query_callback(text)
        
        return text
