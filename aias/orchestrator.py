"""
AIAS Orchestrator
Main controller that coordinates all components
"""

import time
import signal
import threading
from typing import Optional, List, Union
from pathlib import Path

from PIL import Image

from .config import Config, get_config
from .audio import AudioPipeline, SimpleAudioPipeline
from .screen import ScreenCapture, ScreenAnalyzer
from .llm import VisionLLM, LLMWorker, LLMResponse
from .groq_llm import GroqVisionLLM
from .overlay import OverlayWindow, OverlayTheme
from .memory import PersonalMemory
from .logger import (
    logger, 
    log_query, 
    save_query_screenshots, 
    log_llm_request, 
    log_llm_response,
    get_run_dir
)


class AIASOrchestrator:
    """
    Main orchestrator that coordinates all AIAS components:
    - Audio pipeline (hotword detection + STT)
    - Screen capture (continuous screenshot buffer)
    - Vision LLM (multimodal processing)
    - Overlay display (response output)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AIAS Orchestrator
        
        Args:
            config_path: Path to config.yaml (optional, auto-detected)
        """
        # Load configuration
        self.config = get_config(config_path)
        
        # Setup logging
        self._setup_logging()
        
        # Components (initialized lazily)
        self._audio: Optional[AudioPipeline] = None
        self._screen: Optional[ScreenCapture] = None
        self._llm: Optional[VisionLLM] = None
        self._llm_worker: Optional[LLMWorker] = None
        self._overlay: Optional[OverlayWindow] = None
        
        # State
        self._running = False
        self._processing = False
        self._query_lock = threading.Lock()
        
        logger.info("AIAS Orchestrator initialized")
    
    def _setup_logging(self):
        """Configure logging based on config"""
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.config.logging.file,
                    encoding='utf-8'
                )
            ]
        )
        
        # Reduce noise from third-party loggers
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
    
    def _initialize_components(self):
        """Initialize all components"""
        logger.info("Initializing AIAS components...")
        
        # 1. Initialize Overlay (first, for visual feedback)
        self._initialize_overlay()
        
        # 2. Initialize Screen Capture
        self._initialize_screen_capture()
        
        # 3. Initialize LLM
        self._initialize_llm()
        
        # 4. Initialize Audio Pipeline (last, starts listening)
        self._initialize_audio()
        
        logger.info("All components initialized")
    
    def _initialize_overlay(self):
        """Initialize the overlay window with interactive callbacks"""
        theme = OverlayTheme(
            background_color=self.config.overlay.appearance.background_color,
            text_color=self.config.overlay.appearance.text_color,
            font_family=self.config.overlay.appearance.font_family,
            font_size=self.config.overlay.appearance.font_size,
            opacity=self.config.overlay.appearance.opacity
        )
        
        self._overlay = OverlayWindow(
            width=420,
            height=500,
            position=self.config.overlay.window.position,
            margin=self.config.overlay.window.margin,
            theme=theme,
            auto_hide_seconds=0,  # Never auto-hide (permanent)
            on_query_callback=self._on_overlay_query,
            on_voice_start_callback=self._on_voice_start,
            on_voice_stop_callback=self._on_voice_stop,
            on_close_callback=None  # Don't close whole app, just hide
        )
        
        self._overlay.start()
    
    def _initialize_screen_capture(self):
        """Initialize screen capture"""
        self._screen = ScreenCapture(
            interval_seconds=self.config.screen.capture.interval_seconds,
            max_buffer_size=self.config.screen.capture.max_buffer_size,
            monitor=self.config.screen.capture.monitor,
            max_width=self.config.screen.processing.max_width,
            jpeg_quality=self.config.screen.processing.jpeg_quality
        )
        
        self._screen.start()
    
    def _initialize_llm(self):
        """Initialize the vision LLM"""
        self._llm = VisionLLM(
            model_name=self.config.llm.model.name,
            device_map=self.config.llm.model.device_map,
            torch_dtype=self.config.llm.model.torch_dtype,
            max_new_tokens=self.config.llm.generation.max_new_tokens,
            temperature=self.config.llm.generation.temperature,
            top_p=self.config.llm.generation.top_p,
            system_prompt=self.config.llm.system_prompt
        )
        
        # Pre-load model (shows loading in overlay)
        self._overlay.show("Loading AI model... This may take a minute.")
        self._llm.load_model()
        self._overlay.hide()
        
        # Create worker for async processing
        self._llm_worker = LLMWorker(
            llm=self._llm,
            on_response_callback=self._on_llm_response
        )
        self._llm_worker.start()
    
    def _initialize_audio(self):
        """Initialize audio pipeline"""
        self._audio = AudioPipeline(
            hotword_keyword=self.config.audio.hotword.keyword,
            hotword_sensitivity=self.config.audio.hotword.sensitivity,
            stt_model_size=self.config.audio.stt.model_size,
            stt_device=self.config.audio.stt.device,
            stt_compute_type=self.config.audio.stt.compute_type,
            stt_language=self.config.audio.stt.language,
            sample_rate=self.config.audio.capture.sample_rate,
            silence_threshold_ms=self.config.audio.capture.silence_threshold_ms,
            max_recording_seconds=self.config.audio.capture.max_recording_seconds,
            on_query_callback=self._on_query_received
        )
        
        self._audio.start()
    
    def _on_query_received(self, query: str):
        """
        Callback when a voice query is received
        
        Args:
            query: Transcribed speech text
        """
        if not query.strip():
            return
        
        with self._query_lock:
            if self._processing:
                logger.warning("Already processing a query, ignoring")
                return
            self._processing = True
        
        # Log the query and get query ID
        query_id = log_query(query, source="voice")
        self._current_query_id = query_id
        
        # Show processing state
        self._overlay.show_processing()
        
        # Get recent screenshot (just 1 for speed)
        screenshots = self._screen.get_screenshots_as_images(1)
        
        # Save screenshots for this query
        if screenshots:
            save_query_screenshots(query_id, screenshots)
        
        # Get screen context
        context = ScreenAnalyzer.get_screen_context()
        
        # Enhance query with context
        enhanced_query = self._enhance_query(query, context)
        
        # Log LLM request
        log_llm_request(
            query_id=query_id,
            query=enhanced_query,
            num_images=len(screenshots) if screenshots else 0,
            system_prompt=self._llm_worker._llm.system_prompt if self._llm_worker else "N/A"
        )
        
        # Submit to LLM
        self._llm_worker.submit_query(
            query=enhanced_query,
            images=screenshots if screenshots else None
        )
    
    def _on_overlay_query(self, query: str):
        """Callback when user submits query from overlay UI"""
        self._on_query_received(query)
    
    def _on_voice_start(self):
        """Callback when user starts voice recording from overlay"""
        if self._audio:
            self._audio.manual_trigger()
    
    def _on_voice_stop(self):
        """Callback when user stops voice recording from overlay"""
        # Voice stop is handled by VAD automatically
        self._overlay.stop_recording_ui()
    
    def _enhance_query(self, query: str, context: dict) -> str:
        """Enhance query with screen context"""
        enhanced = query
        
        if context.get("active_window"):
            enhanced = f"[Active window: {context['active_window']}]\n\n{query}"
        
        return enhanced
    
    def _on_llm_response(self, response: LLMResponse):
        """
        Callback when LLM generates a response
        
        Args:
            response: LLMResponse object
        """
        self._processing = False
        
        # Log LLM response
        query_id = getattr(self, '_current_query_id', 0)
        log_llm_response(
            query_id=query_id,
            response=response.text,
            generation_time=response.generation_time,
            tokens=response.tokens_generated
        )
        
        if response.text.startswith("Error:"):
            self._overlay.show_error(response.text)
        else:
            self._overlay.show_response(response.text)
        
        logger.info(
            f"Response generated in {response.generation_time:.1f}s "
            f"({response.images_processed} images processed)"
        )
    
    def start(self):
        """Start AIAS"""
        if self._running:
            logger.warning("AIAS already running")
            return
        
        logger.info("Starting AIAS...")
        self._running = True
        
        # Initialize all components
        self._initialize_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("AIAS started successfully")
        logger.info(f"Listening for wake word: '{self.config.audio.hotword.keyword}'")
        
        # Overlay is now always visible with chat interface
    
    def stop(self):
        """Stop AIAS"""
        if not self._running:
            return
        
        logger.info("Stopping AIAS...")
        self._running = False
        
        # Stop components in reverse order
        if self._audio:
            self._audio.stop()
        
        if self._llm_worker:
            self._llm_worker.stop()
        
        if self._llm:
            self._llm.unload_model()
        
        if self._screen:
            self._screen.stop()
        
        if self._overlay:
            self._overlay.stop()
        
        logger.info("AIAS stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
    
    def run_forever(self):
        """Run AIAS until interrupted"""
        self.start()
        
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def manual_query(self, query: str, include_screenshots: bool = True):
        """
        Process a query manually (without voice)
        
        Args:
            query: The query text
            include_screenshots: Whether to include recent screenshots
        """
        if not self._running:
            logger.error("AIAS not running")
            return
        
        logger.info(f"Manual query: {query}")
        
        self._overlay.show_processing()
        
        screenshots = None
        if include_screenshots and self._screen:
            screenshots = self._screen.get_screenshots_as_images(1)
        
        self._llm_worker.submit_query(query=query, images=screenshots)
    
    def trigger_listening(self):
        """Manually trigger listening mode (bypass hotword)"""
        if self._audio:
            self._overlay.show_listening()
            self._audio.manual_trigger()
    
    @property
    def is_running(self) -> bool:
        """Check if AIAS is running"""
        return self._running
    
    @property
    def status(self) -> dict:
        """Get current status of all components"""
        return {
            "running": self._running,
            "processing": self._processing,
            "audio_state": self._audio.state if self._audio else "not initialized",
            "screen_buffer_size": self._screen.buffer_size if self._screen else 0,
            "llm_loaded": self._llm.is_loaded if self._llm else False,
            "llm_queue_size": self._llm_worker.queue_size if self._llm_worker else 0,
            "overlay_visible": self._overlay.is_visible if self._overlay else False
        }


class AIASLite:
    """
    Lightweight version of AIAS without audio pipeline
    Uses interactive overlay for text/voice input
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        
        self._screen: Optional[ScreenCapture] = None
        self._llm: Optional[Union[VisionLLM, GroqVisionLLM]] = None
        self._overlay: Optional[OverlayWindow] = None
        self._memory: Optional[PersonalMemory] = None
        self._stt = None  # Speech-to-text for voice input
        self._running = False
        self._processing = False
        self._recording = False
        self._audio_data = []
        self._query_count = 0
    
    def _on_query(self, query: str):
        """Handle query from overlay"""
        if not query.strip() or self._processing:
            return
        
        self._processing = True
        self._query_count += 1
        query_id = self._query_count
        
        # Log the query
        log_query(query, source="overlay")
        
        try:
            # ===== Check for memory commands first =====
            if self._memory:
                memory_response = self._memory.handle_memory_command(query)
                if memory_response:
                    logger.info(f"Memory command handled: {query[:50]}")
                    self._overlay.add_response(memory_response)
                    return
            
            screenshots = self._screen.get_screenshots_as_images(1)
            
            # Save screenshots for debugging
            if screenshots:
                save_query_screenshots(query_id, screenshots)
            
            # Get memory context
            memory_context = self._memory.get_context_for_query(query) if self._memory else ""
            
            # Build enhanced query with memory
            if memory_context:
                enhanced_query = f"{memory_context}\n\n=== CURRENT QUERY ===\n{query}"
            else:
                enhanced_query = query
            
            # Log LLM request
            log_llm_request(
                query_id=query_id,
                query=enhanced_query,
                num_images=len(screenshots) if screenshots else 0,
                system_prompt=self._llm.system_prompt
            )
            
            response = self._llm.generate(enhanced_query, screenshots)
            
            # Log LLM response
            log_llm_response(
                query_id=query_id,
                response=response.text,
                generation_time=response.generation_time,
                tokens=response.tokens_generated
            )
            
            # Store memory and extract facts
            if self._memory:
                # Check if this is a "remember this" command - extract from RESPONSE
                if self._memory.is_remember_command(query):
                    extracted_facts = self._memory.extract_facts_from_response(query, response.text)
                    logger.info(f"Remember command: extracted {len(extracted_facts)} facts from screen")
                else:
                    # Normal extraction from query
                    extracted_facts = self._memory.extract_facts_from_conversation(query, response.text)
                
                self._memory.add_memory(query, response.text, extracted_facts)
                if extracted_facts:
                    logger.info(f"Learned {len(extracted_facts)} new facts from conversation")
            
            self._overlay.add_response(response.text)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            self._overlay.show_error(str(e))
        finally:
            self._processing = False
    
    def _on_voice_start(self):
        """Start recording voice"""
        self._recording = True
        self._audio_data = []
        
        # Start audio capture in thread
        threading.Thread(target=self._record_audio, daemon=True).start()
    
    def _on_voice_stop(self):
        """Stop recording and transcribe"""
        self._recording = False
        self._overlay.set_status("Transcribing...", "#4a9eff")
        
        # Transcribe in separate thread
        threading.Thread(target=self._transcribe_audio, daemon=True).start()
    
    def _record_audio(self):
        """Record audio from microphone"""
        try:
            import pyaudio
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=512
            )
            
            while self._recording:
                data = stream.read(512, exception_on_overflow=False)
                self._audio_data.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            logger.error(f"Audio recording failed: {e}")
            self._overlay.show_error(f"Microphone error: {e}")
    
    def _transcribe_audio(self):
        """Transcribe recorded audio"""
        try:
            if not self._audio_data:
                self._overlay.stop_recording_ui()
                return
            
            # Load STT model if needed
            if self._stt is None:
                from faster_whisper import WhisperModel
                self._stt = WhisperModel(
                    self.config.audio.stt.model_size,
                    device=self.config.audio.stt.device,
                    compute_type=self.config.audio.stt.compute_type
                )
            
            # Convert audio data to numpy array
            import numpy as np
            audio_bytes = b''.join(self._audio_data)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe
            segments, _ = self._stt.transcribe(audio_np, language="en")
            text = " ".join([s.text for s in segments]).strip()
            
            if text:
                self._overlay.set_voice_text(text)
            else:
                self._overlay.stop_recording_ui()
                self._overlay.set_status("No speech detected", "#ff4757")
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self._overlay.show_error(f"Transcription error: {e}")
            self._overlay.stop_recording_ui()
    
    def start(self):
        """Start AIAS Lite with interactive overlay"""
        self._running = True
        
        # Initialize personal memory
        self._memory = PersonalMemory(memory_dir="memory")
        logger.info(f"Memory loaded: {len(self._memory.memories)} past conversations")
        
        # Initialize overlay with callbacks
        theme = OverlayTheme(
            background_color=self.config.overlay.appearance.background_color,
            text_color=self.config.overlay.appearance.text_color
        )
        self._overlay = OverlayWindow(
            width=420,
            height=500,
            theme=theme,
            auto_hide_seconds=0,  # Never auto-hide
            on_query_callback=self._on_query,
            on_voice_start_callback=self._on_voice_start,
            on_voice_stop_callback=self._on_voice_stop
        )
        self._overlay.start()
        
        # Initialize screen capture
        self._screen = ScreenCapture(
            interval_seconds=2.0,
            max_buffer_size=5
        )
        self._screen.start()
        
        # Initialize LLM based on provider
        self._overlay.set_status("Loading AI model...", "#ffaa00")
        
        if self.config.llm.provider == "groq":
            logger.info(f"Using Groq provider: {self.config.llm.groq.model}")
            self._llm = GroqVisionLLM(
                model_name=self.config.llm.groq.model,
                max_tokens=self.config.llm.groq.max_tokens,
                temperature=self.config.llm.groq.temperature
            )
        else:
            logger.info(f"Using local provider: {self.config.llm.model.name}")
            self._llm = VisionLLM(
                model_name=self.config.llm.model.name
            )
        
        self._llm.load_model()
        
        # Show personalized greeting if we know the user
        if self._memory.profile.name:
            self._overlay.set_status(f"Ready - Hi {self._memory.profile.name}!", "#00ff88")
        else:
            self._overlay.set_status("Ready - Watching your screen", "#00ff88")
        logger.info("AIAS Lite started")
    
    def query(self, text: str) -> str:
        """
        Process a query with current screen context
        
        Args:
            text: Query text
            
        Returns:
            Response text
        """
        if not self._running:
            return "AIAS not running"
        
        self._query_count += 1
        query_id = self._query_count
        
        # Log the query
        log_query(text, source="api")
        
        self._overlay.show_processing()
        self._overlay.add_user_message(text)
        
        screenshots = self._screen.get_screenshots_as_images(1)
        
        # Save screenshots
        if screenshots:
            save_query_screenshots(query_id, screenshots)
        
        # Log LLM request
        log_llm_request(
            query_id=query_id,
            query=text,
            num_images=len(screenshots) if screenshots else 0,
            system_prompt=self._llm.system_prompt
        )
        
        response = self._llm.generate(text, screenshots)
        
        # Log LLM response
        log_llm_response(
            query_id=query_id,
            response=response.text,
            generation_time=response.generation_time,
            tokens=response.tokens_generated
        )
        
        self._overlay.add_response(response.text)
        return response.text
    
    def stop(self):
        """Stop AIAS Lite"""
        self._running = False
        self._recording = False
        
        if self._screen:
            self._screen.stop()
        if self._llm:
            self._llm.unload_model()
        if self._overlay:
            self._overlay.stop()
        
        logger.info("AIAS Lite stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIAS - AI Assistant System")
    parser.add_argument(
        "-c", "--config",
        help="Path to config.yaml",
        default=None
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Run in lite mode (no hotword, overlay-only interaction)"
    )
    
    args = parser.parse_args()
    
    if args.lite:
        aias = AIASLite(config_path=args.config)
        aias.start()
        
        print("\nAIAS Lite ready. Type your questions below.")
        print("Commands: 'quit' to exit, 'status' for status\n")
        
        # Interactive mode - the overlay handles GUI input
        # Terminal is optional fallback
        try:
            while aias._running:
                query = input("You: ")
                if query.lower() in ('quit', 'exit', 'q'):
                    break
                elif query.lower() == 'status':
                    print(f"Running: {aias._running}, Processing: {aias._processing}")
                elif query.strip():
                    response = aias.query(query)
                    print(f"\nAIAS: {response}\n")
        except KeyboardInterrupt:
            print("\nShutting down...")
        except EOFError:
            # Handle non-interactive mode
            try:
                while aias._running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        finally:
            aias.stop()
    else:
        aias = AIASOrchestrator(config_path=args.config)
        aias.run_forever()


if __name__ == "__main__":
    main()
