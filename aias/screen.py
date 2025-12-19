"""
AIAS Screen Capture
High-performance screenshot capture with circular buffer
"""

import io
import time
import threading
from typing import Optional, List, Tuple
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from PIL import Image
from loguru import logger


@dataclass
class Screenshot:
    """Container for a screenshot with metadata"""
    image: Image.Image
    timestamp: datetime
    monitor_index: int
    resolution: Tuple[int, int]
    
    def to_bytes(self, format: str = "JPEG", quality: int = 85) -> bytes:
        """Convert image to bytes"""
        buffer = io.BytesIO()
        self.image.save(buffer, format=format, quality=quality)
        return buffer.getvalue()
    
    def resize(self, max_width: int) -> 'Screenshot':
        """Return a resized copy of the screenshot"""
        if self.image.width <= max_width:
            return self
        
        ratio = max_width / self.image.width
        new_height = int(self.image.height * ratio)
        resized = self.image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        return Screenshot(
            image=resized,
            timestamp=self.timestamp,
            monitor_index=self.monitor_index,
            resolution=(max_width, new_height)
        )


class ScreenCapture:
    """
    High-performance screen capture using MSS
    Maintains a circular buffer of recent screenshots
    """
    
    def __init__(
        self,
        interval_seconds: float = 2.5,
        max_buffer_size: int = 5,
        monitor: int = 0,
        max_width: int = 1280,
        jpeg_quality: int = 85
    ):
        """
        Args:
            interval_seconds: Time between captures
            max_buffer_size: Maximum screenshots to keep in buffer
            monitor: Monitor index (0 = primary, or specific monitor)
            max_width: Resize images to this width for performance
            jpeg_quality: JPEG compression quality
        """
        self.interval_seconds = interval_seconds
        self.max_buffer_size = max_buffer_size
        self.monitor = monitor
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        
        self._buffer: deque[Screenshot] = deque(maxlen=max_buffer_size)
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._sct = None
        
        self._capture_count = 0
        self._last_capture_time = None
    
    def _initialize_mss(self):
        """Initialize MSS screen capture"""
        try:
            import mss
            self._sct = mss.mss()
            logger.info(f"MSS initialized. Available monitors: {len(self._sct.monitors)}")
        except ImportError:
            logger.error("MSS not installed. Screen capture unavailable.")
            raise
    
    def _get_monitor(self) -> dict:
        """Get monitor configuration"""
        if not self._sct:
            self._initialize_mss()
        
        monitors = self._sct.monitors
        
        if self.monitor == 0:
            # Primary monitor (combined view of all monitors)
            return monitors[0]
        elif self.monitor < len(monitors):
            return monitors[self.monitor]
        else:
            logger.warning(f"Monitor {self.monitor} not found, using primary")
            return monitors[0]
    
    def capture_now(self) -> Screenshot:
        """Capture a screenshot immediately"""
        if not self._sct:
            self._initialize_mss()
        
        monitor = self._get_monitor()
        
        # Capture
        start_time = time.perf_counter()
        sct_img = self._sct.grab(monitor)
        
        # Convert to PIL Image
        img = Image.frombytes(
            "RGB",
            (sct_img.width, sct_img.height),
            sct_img.rgb
        )
        
        capture_time = (time.perf_counter() - start_time) * 1000
        
        screenshot = Screenshot(
            image=img,
            timestamp=datetime.now(),
            monitor_index=self.monitor,
            resolution=(img.width, img.height)
        )
        
        logger.debug(f"Screenshot captured in {capture_time:.1f}ms ({img.width}x{img.height})")
        
        return screenshot
    
    def _capture_loop(self):
        """Continuous capture loop"""
        self._initialize_mss()
        
        while self._running:
            try:
                screenshot = self.capture_now()
                
                # Resize for performance
                if self.max_width and screenshot.image.width > self.max_width:
                    screenshot = screenshot.resize(self.max_width)
                
                # Add to buffer
                with self._lock:
                    self._buffer.append(screenshot)
                
                self._capture_count += 1
                self._last_capture_time = time.time()
                
            except Exception as e:
                logger.error(f"Capture error: {e}")
            
            # Wait for next capture
            time.sleep(self.interval_seconds)
    
    def start(self):
        """Start continuous capture"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Screen capture started (interval={self.interval_seconds}s, buffer={self.max_buffer_size})")
    
    def stop(self):
        """Stop continuous capture"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        
        if self._sct:
            self._sct.close()
            self._sct = None
        
        logger.info("Screen capture stopped")
    
    def get_recent_screenshots(self, count: Optional[int] = None) -> List[Screenshot]:
        """
        Get recent screenshots from buffer
        
        Args:
            count: Number of screenshots to return (default: all)
            
        Returns:
            List of Screenshot objects (oldest to newest)
        """
        with self._lock:
            if count is None or count >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-count:]
    
    def get_latest_screenshot(self) -> Optional[Screenshot]:
        """Get the most recent screenshot"""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None
    
    def get_screenshots_as_images(self, count: Optional[int] = None) -> List[Image.Image]:
        """Get screenshots as PIL Image objects"""
        screenshots = self.get_recent_screenshots(count)
        return [ss.image for ss in screenshots]
    
    def clear_buffer(self):
        """Clear the screenshot buffer"""
        with self._lock:
            self._buffer.clear()
        logger.info("Screenshot buffer cleared")
    
    @property
    def buffer_size(self) -> int:
        """Current number of screenshots in buffer"""
        return len(self._buffer)
    
    @property
    def is_running(self) -> bool:
        """Whether capture is running"""
        return self._running
    
    @property
    def stats(self) -> dict:
        """Get capture statistics"""
        return {
            "capture_count": self._capture_count,
            "buffer_size": self.buffer_size,
            "max_buffer_size": self.max_buffer_size,
            "interval_seconds": self.interval_seconds,
            "last_capture_time": self._last_capture_time,
            "is_running": self._running
        }


class ScreenAnalyzer:
    """
    Helper class for additional screen analysis
    Can be extended with OCR, element detection, etc.
    """
    
    @staticmethod
    def get_active_window_title() -> str:
        """Get the title of the currently active window"""
        try:
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            
            # Get foreground window handle
            hwnd = user32.GetForegroundWindow()
            
            # Get window title length
            length = user32.GetWindowTextLengthW(hwnd)
            
            # Get window title
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            
            return buffer.value
            
        except Exception as e:
            logger.error(f"Failed to get window title: {e}")
            return ""
    
    @staticmethod
    def get_active_window_process() -> str:
        """Get the process name of the active window"""
        try:
            import ctypes
            from ctypes import wintypes
            import psutil
            
            user32 = ctypes.windll.user32
            
            # Get foreground window handle
            hwnd = user32.GetForegroundWindow()
            
            # Get process ID
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            
            # Get process name
            process = psutil.Process(pid.value)
            return process.name()
            
        except Exception as e:
            logger.error(f"Failed to get process name: {e}")
            return ""
    
    @staticmethod
    def detect_text_regions(image: Image.Image) -> List[dict]:
        """
        Detect text regions in image using simple heuristics
        For production, use proper OCR like Tesseract
        """
        # Placeholder - implement with OCR if needed
        return []
    
    @staticmethod
    def get_screen_context() -> dict:
        """Get context about current screen state"""
        return {
            "active_window": ScreenAnalyzer.get_active_window_title(),
            "active_process": ScreenAnalyzer.get_active_window_process(),
            "timestamp": datetime.now().isoformat()
        }


# Convenience function for quick capture
def quick_capture(monitor: int = 0, max_width: int = 1280) -> Image.Image:
    """Quick one-shot screenshot"""
    capture = ScreenCapture(monitor=monitor, max_width=max_width)
    screenshot = capture.capture_now()
    
    if max_width and screenshot.image.width > max_width:
        screenshot = screenshot.resize(max_width)
    
    return screenshot.image
