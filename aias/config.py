"""
AIAS Configuration Manager
Handles loading and accessing configuration from YAML
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class AudioHotwordConfig:
    keyword: str = "porcupine"
    sensitivity: float = 0.5


@dataclass
class AudioSTTConfig:
    model_size: str = "base"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"


@dataclass
class AudioCaptureConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 30
    silence_threshold_ms: int = 1000
    max_recording_seconds: int = 30


@dataclass
class AudioConfig:
    hotword: AudioHotwordConfig = field(default_factory=AudioHotwordConfig)
    stt: AudioSTTConfig = field(default_factory=AudioSTTConfig)
    capture: AudioCaptureConfig = field(default_factory=AudioCaptureConfig)


@dataclass
class ScreenCaptureConfig:
    interval_seconds: float = 2.5
    max_buffer_size: int = 5
    monitor: int = 0


@dataclass
class ScreenProcessingConfig:
    max_width: int = 1280
    jpeg_quality: int = 85


@dataclass
class ScreenConfig:
    capture: ScreenCaptureConfig = field(default_factory=ScreenCaptureConfig)
    processing: ScreenProcessingConfig = field(default_factory=ScreenProcessingConfig)


@dataclass
class LLMModelConfig:
    name: str = "Qwen/Qwen2-VL-7B-Instruct"
    device_map: str = "auto"
    torch_dtype: str = "auto"


@dataclass
class LLMGenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class GroqConfig:
    model: str = "llama-3.2-11b-vision-preview"
    max_tokens: int = 1024
    temperature: float = 0.3


@dataclass
class LLMConfig:
    provider: str = "groq"  # "local" or "groq"
    groq: GroqConfig = field(default_factory=GroqConfig)
    model: LLMModelConfig = field(default_factory=LLMModelConfig)
    generation: LLMGenerationConfig = field(default_factory=LLMGenerationConfig)
    system_prompt: str = "You are AIAS, an intelligent AI assistant."


@dataclass
class OverlayWindowConfig:
    width: int = 450
    height: int = 180
    position: str = "bottom-right"
    margin: int = 20


@dataclass
class OverlayAppearanceConfig:
    background_color: str = "#1a1a2e"
    text_color: str = "#00ff88"
    font_family: str = "Consolas"
    font_size: int = 11
    opacity: float = 0.92


@dataclass
class OverlayBehaviorConfig:
    auto_hide_seconds: int = 8
    fade_animation: bool = True
    show_on_startup: bool = False


@dataclass
class OverlayConfig:
    window: OverlayWindowConfig = field(default_factory=OverlayWindowConfig)
    appearance: OverlayAppearanceConfig = field(default_factory=OverlayAppearanceConfig)
    behavior: OverlayBehaviorConfig = field(default_factory=OverlayBehaviorConfig)


@dataclass
class PerformanceConfig:
    audio_thread_priority: str = "high"
    inference_thread_priority: str = "normal"
    clear_cuda_cache_interval: int = 10
    screenshot_buffer_in_memory: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "aias.log"
    max_size_mb: int = 50
    backup_count: int = 3


class Config:
    """Main configuration class that loads and manages all settings"""
    
    _instance: Optional['Config'] = None
    
    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if self._initialized:
            return
            
        self._initialized = True
        self._config_path = config_path or self._find_config()
        self._raw_config: Dict[str, Any] = {}
        
        # Initialize config sections with defaults
        self.audio = AudioConfig()
        self.screen = ScreenConfig()
        self.llm = LLMConfig()
        self.overlay = OverlayConfig()
        self.performance = PerformanceConfig()
        self.logging = LoggingConfig()
        
        # Load from file if exists
        if self._config_path and os.path.exists(self._config_path):
            self._load_config()
    
    def _find_config(self) -> Optional[str]:
        """Find config.yaml in common locations"""
        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path.home() / ".aias" / "config.yaml",
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        return None
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f) or {}
            
            self._parse_audio_config()
            self._parse_screen_config()
            self._parse_llm_config()
            self._parse_overlay_config()
            self._parse_performance_config()
            self._parse_logging_config()
            
        except Exception as e:
            logging.error(f"Failed to load config from {self._config_path}: {e}")
    
    def _parse_audio_config(self):
        """Parse audio configuration section"""
        audio_cfg = self._raw_config.get('audio', {})
        
        if hotword := audio_cfg.get('hotword'):
            self.audio.hotword = AudioHotwordConfig(
                keyword=hotword.get('keyword', 'porcupine'),
                sensitivity=hotword.get('sensitivity', 0.5)
            )
        
        if stt := audio_cfg.get('stt'):
            self.audio.stt = AudioSTTConfig(
                model_size=stt.get('model_size', 'base'),
                device=stt.get('device', 'cuda'),
                compute_type=stt.get('compute_type', 'float16'),
                language=stt.get('language', 'en')
            )
        
        if capture := audio_cfg.get('capture'):
            self.audio.capture = AudioCaptureConfig(
                sample_rate=capture.get('sample_rate', 16000),
                channels=capture.get('channels', 1),
                chunk_duration_ms=capture.get('chunk_duration_ms', 30),
                silence_threshold_ms=capture.get('silence_threshold_ms', 1000),
                max_recording_seconds=capture.get('max_recording_seconds', 30)
            )
    
    def _parse_screen_config(self):
        """Parse screen configuration section"""
        screen_cfg = self._raw_config.get('screen', {})
        
        if capture := screen_cfg.get('capture'):
            self.screen.capture = ScreenCaptureConfig(
                interval_seconds=capture.get('interval_seconds', 2.5),
                max_buffer_size=capture.get('max_buffer_size', 5),
                monitor=capture.get('monitor', 0)
            )
        
        if processing := screen_cfg.get('processing'):
            self.screen.processing = ScreenProcessingConfig(
                max_width=processing.get('max_width', 1280),
                jpeg_quality=processing.get('jpeg_quality', 85)
            )
    
    def _parse_llm_config(self):
        """Parse LLM configuration section"""
        llm_cfg = self._raw_config.get('llm', {})
        
        # Provider setting
        self.llm.provider = llm_cfg.get('provider', 'groq')
        
        # Groq settings
        if groq := llm_cfg.get('groq'):
            self.llm.groq = GroqConfig(
                model=groq.get('model', 'llama-3.2-11b-vision-preview'),
                max_tokens=groq.get('max_tokens', 1024),
                temperature=groq.get('temperature', 0.3)
            )
        
        # Local model settings
        if model := llm_cfg.get('model'):
            self.llm.model = LLMModelConfig(
                name=model.get('name', 'Qwen/Qwen2-VL-7B-Instruct'),
                device_map=model.get('device_map', 'auto'),
                torch_dtype=model.get('torch_dtype', 'auto')
            )
        
        if generation := llm_cfg.get('generation'):
            self.llm.generation = LLMGenerationConfig(
                max_new_tokens=generation.get('max_new_tokens', 512),
                temperature=generation.get('temperature', 0.7),
                top_p=generation.get('top_p', 0.9),
                do_sample=generation.get('do_sample', True)
            )
        
        if system_prompt := llm_cfg.get('system_prompt'):
            self.llm.system_prompt = system_prompt
    
    def _parse_overlay_config(self):
        """Parse overlay configuration section"""
        overlay_cfg = self._raw_config.get('overlay', {})
        
        if window := overlay_cfg.get('window'):
            self.overlay.window = OverlayWindowConfig(
                width=window.get('width', 450),
                height=window.get('height', 180),
                position=window.get('position', 'bottom-right'),
                margin=window.get('margin', 20)
            )
        
        if appearance := overlay_cfg.get('appearance'):
            self.overlay.appearance = OverlayAppearanceConfig(
                background_color=appearance.get('background_color', '#1a1a2e'),
                text_color=appearance.get('text_color', '#00ff88'),
                font_family=appearance.get('font_family', 'Consolas'),
                font_size=appearance.get('font_size', 11),
                opacity=appearance.get('opacity', 0.92)
            )
        
        if behavior := overlay_cfg.get('behavior'):
            self.overlay.behavior = OverlayBehaviorConfig(
                auto_hide_seconds=behavior.get('auto_hide_seconds', 8),
                fade_animation=behavior.get('fade_animation', True),
                show_on_startup=behavior.get('show_on_startup', False)
            )
    
    def _parse_performance_config(self):
        """Parse performance configuration section"""
        perf_cfg = self._raw_config.get('performance', {})
        
        self.performance = PerformanceConfig(
            audio_thread_priority=perf_cfg.get('audio_thread_priority', 'high'),
            inference_thread_priority=perf_cfg.get('inference_thread_priority', 'normal'),
            clear_cuda_cache_interval=perf_cfg.get('clear_cuda_cache_interval', 10),
            screenshot_buffer_in_memory=perf_cfg.get('screenshot_buffer_in_memory', True)
        )
    
    def _parse_logging_config(self):
        """Parse logging configuration section"""
        log_cfg = self._raw_config.get('logging', {})
        
        self.logging = LoggingConfig(
            level=log_cfg.get('level', 'INFO'),
            file=log_cfg.get('file', 'aias.log'),
            max_size_mb=log_cfg.get('max_size_mb', 50),
            backup_count=log_cfg.get('backup_count', 3)
        )
    
    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing)"""
        cls._instance = None
    
    def reload(self):
        """Reload configuration from file"""
        self._initialized = False
        self.__init__(self._config_path)


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance"""
    return Config(config_path)
