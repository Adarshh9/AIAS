"""
AIAS Interactive Overlay Window
Permanent overlay with text/voice input, conversation display
Thread-safe implementation using queue-based communication
"""

import queue
import time
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from loguru import logger


class OverlayPosition(Enum):
    """Overlay window position presets"""
    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_LEFT = "bottom-left"
    TOP_RIGHT = "top-right"
    TOP_LEFT = "top-left"
    CENTER = "center"


@dataclass
class OverlayTheme:
    """Theme configuration for overlay"""
    background_color: str = "#1a1a2e"
    text_color: str = "#e0e0e0"
    accent_color: str = "#4a9eff"
    user_color: str = "#00ff88"
    error_color: str = "#ff4757"
    input_bg: str = "#252542"
    button_bg: str = "#333366"
    button_hover: str = "#4a4a7a"
    font_family: str = "Segoe UI"
    font_size: int = 10
    opacity: float = 0.95
    border_color: str = "#333366"


@dataclass
class Message:
    """Chat message"""
    text: str
    is_user: bool
    timestamp: datetime = field(default_factory=datetime.now)


class OverlayWindow:
    """
    Interactive always-on-top overlay window
    Features:
    - Permanent display with minimize/close
    - Text input field
    - Voice input with auto-detection
    - Conversation history
    - Draggable window
    """
    
    def __init__(
        self,
        width: int = 420,
        height: int = 500,
        position: str = "bottom-right",
        margin: int = 20,
        theme: Optional[OverlayTheme] = None,
        auto_hide_seconds: int = 0,  # 0 = never auto-hide
        on_query_callback: Optional[Callable[[str], None]] = None,
        on_voice_start_callback: Optional[Callable[[], None]] = None,
        on_voice_stop_callback: Optional[Callable[[], None]] = None,
        on_close_callback: Optional[Callable[[], None]] = None
    ):
        self.width = width
        self.height = height
        self.position = position
        self.margin = margin
        self.theme = theme or OverlayTheme()
        self.auto_hide_seconds = auto_hide_seconds
        self.on_query_callback = on_query_callback
        self.on_voice_start_callback = on_voice_start_callback
        self.on_voice_stop_callback = on_voice_stop_callback
        self.on_close_callback = on_close_callback
        
        # Window elements
        self._root: Optional[tk.Tk] = None
        self._chat_frame: Optional[tk.Frame] = None
        self._chat_canvas: Optional[tk.Canvas] = None
        self._chat_scrollable: Optional[tk.Frame] = None
        self._input_entry: Optional[tk.Entry] = None
        self._voice_btn: Optional[tk.Button] = None
        self._status_label: Optional[tk.Label] = None
        self._status_indicator = None
        
        # State
        self._messages: List[Message] = []
        self._is_visible = True
        self._is_minimized = False
        self._is_recording = False
        self._is_processing = False
        self._thread = None
        self._running = False
        self._ready = threading.Event()
        
        # Thread-safe command queue
        self._command_queue: queue.Queue = queue.Queue()
        
        # Drag data
        self._drag_data = {"x": 0, "y": 0}
    
    def _create_window(self):
        """Create the interactive overlay window"""
        self._root = tk.Tk()
        self._root.title("AIAS Assistant")
        
        # Window attributes
        self._root.attributes('-topmost', True)
        self._root.attributes('-alpha', self.theme.opacity)
        self._root.overrideredirect(True)  # Remove window decorations
        
        # Calculate position
        x, y = self._calculate_position()
        self._root.geometry(f"{self.width}x{self.height}+{x}+{y}")
        
        # Configure window
        self._root.configure(bg=self.theme.background_color)
        
        # Create main frame with border effect
        self._main_frame = tk.Frame(
            self._root,
            bg=self.theme.background_color,
            highlightbackground=self.theme.border_color,
            highlightthickness=2
        )
        self._main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Title Bar ===
        self._create_title_bar()
        
        # === Chat Display Area ===
        self._create_chat_area()
        
        # === Input Area ===
        self._create_input_area()
        
        # === Status Bar ===
        self._create_status_bar()
        
        logger.info("Overlay window created")
    
    def _create_title_bar(self):
        """Create custom title bar"""
        self._title_frame = tk.Frame(
            self._main_frame,
            bg=self.theme.button_bg,
            height=32
        )
        self._title_frame.pack(fill=tk.X)
        self._title_frame.pack_propagate(False)
        
        # Make title bar draggable
        self._title_frame.bind("<Button-1>", self._start_drag)
        self._title_frame.bind("<B1-Motion>", self._on_drag)
        
        # App icon/name
        self._title_label = tk.Label(
            self._title_frame,
            text="  🤖 AIAS Assistant",
            bg=self.theme.button_bg,
            fg=self.theme.text_color,
            font=(self.theme.font_family, 10, "bold")
        )
        self._title_label.pack(side=tk.LEFT, padx=4)
        self._title_label.bind("<Button-1>", self._start_drag)
        self._title_label.bind("<B1-Motion>", self._on_drag)
        
        # Window controls
        btn_frame = tk.Frame(self._title_frame, bg=self.theme.button_bg)
        btn_frame.pack(side=tk.RIGHT)
        
        # Minimize button
        self._min_btn = tk.Label(
            btn_frame,
            text=" ─ ",
            bg=self.theme.button_bg,
            fg=self.theme.text_color,
            font=(self.theme.font_family, 10),
            cursor="hand2"
        )
        self._min_btn.pack(side=tk.LEFT, padx=2)
        self._min_btn.bind("<Button-1>", lambda e: self._do_minimize())
        self._min_btn.bind("<Enter>", lambda e: self._min_btn.configure(bg=self.theme.button_hover))
        self._min_btn.bind("<Leave>", lambda e: self._min_btn.configure(bg=self.theme.button_bg))
        
        # Close button
        self._close_btn = tk.Label(
            btn_frame,
            text=" × ",
            bg=self.theme.button_bg,
            fg=self.theme.text_color,
            font=(self.theme.font_family, 12, "bold"),
            cursor="hand2"
        )
        self._close_btn.pack(side=tk.LEFT, padx=2)
        self._close_btn.bind("<Button-1>", lambda e: self._do_close())
        self._close_btn.bind("<Enter>", lambda e: self._close_btn.configure(bg=self.theme.error_color))
        self._close_btn.bind("<Leave>", lambda e: self._close_btn.configure(bg=self.theme.button_bg))
    
    def _create_chat_area(self):
        """Create scrollable chat display area"""
        self._chat_frame = tk.Frame(
            self._main_frame,
            bg=self.theme.background_color
        )
        self._chat_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        
        # Create canvas for scrolling
        self._chat_canvas = tk.Canvas(
            self._chat_frame,
            bg=self.theme.background_color,
            highlightthickness=0,
            borderwidth=0
        )
        
        # Scrollbar
        scrollbar = tk.Scrollbar(
            self._chat_frame,
            orient=tk.VERTICAL,
            command=self._chat_canvas.yview
        )
        
        # Scrollable frame inside canvas
        self._chat_scrollable = tk.Frame(
            self._chat_canvas,
            bg=self.theme.background_color
        )
        
        self._chat_scrollable.bind(
            "<Configure>",
            lambda e: self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all"))
        )
        
        self._chat_canvas.create_window((0, 0), window=self._chat_scrollable, anchor="nw")
        self._chat_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mousewheel
        self._chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Welcome message
        self._add_welcome_message()
    
    def _create_input_area(self):
        """Create input area with text field and voice button"""
        self._input_frame = tk.Frame(
            self._main_frame,
            bg=self.theme.background_color
        )
        self._input_frame.pack(fill=tk.X, padx=8, pady=4)
        
        # Text input
        self._input_entry = tk.Entry(
            self._input_frame,
            bg=self.theme.input_bg,
            fg=self.theme.text_color,
            insertbackground=self.theme.text_color,
            font=(self.theme.font_family, 11),
            relief=tk.FLAT,
            borderwidth=0
        )
        self._input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 4))
        self._input_entry.insert(0, "Ask me anything...")
        self._input_entry.bind("<FocusIn>", self._on_input_focus_in)
        self._input_entry.bind("<FocusOut>", self._on_input_focus_out)
        self._input_entry.bind("<Return>", self._on_submit)
        
        # Add a frame wrapper for input styling
        input_wrapper = tk.Frame(self._input_frame, bg=self.theme.input_bg, padx=8)
        
        # Voice button
        self._voice_btn = tk.Button(
            self._input_frame,
            text="🎤",
            bg=self.theme.button_bg,
            fg=self.theme.text_color,
            font=(self.theme.font_family, 12),
            relief=tk.FLAT,
            cursor="hand2",
            command=self._toggle_voice,
            width=3
        )
        self._voice_btn.pack(side=tk.LEFT, padx=2, ipady=4)
        self._voice_btn.bind("<Enter>", lambda e: self._voice_btn.configure(bg=self.theme.button_hover) if not self._is_recording else None)
        self._voice_btn.bind("<Leave>", lambda e: self._voice_btn.configure(bg=self.theme.button_bg) if not self._is_recording else None)
        
        # Send button
        self._send_btn = tk.Button(
            self._input_frame,
            text="➤",
            bg=self.theme.accent_color,
            fg="white",
            font=(self.theme.font_family, 12),
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._on_submit(None),
            width=3
        )
        self._send_btn.pack(side=tk.LEFT, padx=2, ipady=4)
    
    def _create_status_bar(self):
        """Create status bar at bottom"""
        self._status_frame = tk.Frame(
            self._main_frame,
            bg=self.theme.button_bg,
            height=24
        )
        self._status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_frame.pack_propagate(False)
        
        # Status indicator (circle)
        self._status_canvas = tk.Canvas(
            self._status_frame,
            width=12,
            height=12,
            bg=self.theme.button_bg,
            highlightthickness=0
        )
        self._status_canvas.pack(side=tk.LEFT, padx=(8, 4), pady=6)
        self._status_indicator = self._status_canvas.create_oval(
            2, 2, 10, 10,
            fill=self.theme.accent_color,
            outline=""
        )
        
        # Status text
        self._status_label = tk.Label(
            self._status_frame,
            text="Ready - Watching your screen",
            bg=self.theme.button_bg,
            fg=self.theme.text_color,
            font=(self.theme.font_family, 8)
        )
        self._status_label.pack(side=tk.LEFT)
    
    def _add_welcome_message(self):
        """Add welcome message to chat"""
        welcome = (
            "👋 Hi! I'm AIAS, your AI assistant.\n\n"
            "I can see your screen and help you with:\n"
            "• Answering questions about what's on screen\n"
            "• Explaining code, documents, or images\n"
            "• Providing suggestions and guidance\n\n"
            "Type below or click 🎤 to speak!"
        )
        self._add_message_bubble(welcome, is_user=False)
    
    def _add_message_bubble(self, text: str, is_user: bool = False):
        """Add a message bubble to the chat"""
        if not self._chat_scrollable:
            return
        
        # Message container
        msg_frame = tk.Frame(
            self._chat_scrollable,
            bg=self.theme.background_color
        )
        msg_frame.pack(fill=tk.X, pady=4, padx=4)
        
        # Alignment
        anchor = tk.E if is_user else tk.W
        
        # Bubble
        bubble_bg = self.theme.accent_color if is_user else self.theme.input_bg
        text_color = "white" if is_user else self.theme.text_color
        
        if is_user:
            # User messages: simple label (no need to copy)
            bubble = tk.Label(
                msg_frame,
                text=text,
                bg=bubble_bg,
                fg=text_color,
                font=(self.theme.font_family, self.theme.font_size),
                wraplength=self.width - 80,
                justify=tk.LEFT,
                padx=12,
                pady=8,
                anchor=tk.W
            )
            bubble.pack(anchor=anchor, padx=8)
        else:
            # AI responses: use Text widget for copy support
            # Calculate approximate height based on text length
            lines = text.count('\n') + 1
            char_per_line = (self.width - 100) // (self.theme.font_size - 2)
            wrapped_lines = max(lines, len(text) // char_per_line + 1)
            height = min(wrapped_lines + 1, 20)  # Cap at 20 lines visible
            
            bubble = tk.Text(
                msg_frame,
                bg=bubble_bg,
                fg=text_color,
                font=(self.theme.font_family, self.theme.font_size),
                wrap=tk.WORD,
                width=40,
                height=height,
                padx=12,
                pady=8,
                relief=tk.FLAT,
                cursor="arrow",
                selectbackground="#4a9eff",
                selectforeground="white",
                borderwidth=0,
                highlightthickness=0
            )
            bubble.insert(tk.END, text)
            bubble.configure(state=tk.DISABLED)  # Read-only but selectable
            bubble.pack(anchor=anchor, padx=8, fill=tk.X)
            
            # Enable copy with Ctrl+C
            bubble.bind("<Control-c>", lambda e: self._copy_selection(bubble))
            # Right-click context menu
            bubble.bind("<Button-3>", lambda e: self._show_copy_menu(e, bubble))
        
        # Store message
        self._messages.append(Message(text=text, is_user=is_user))
        
        # Scroll to bottom
        self._chat_canvas.update_idletasks()
        self._chat_canvas.yview_moveto(1.0)
    
    def _calculate_position(self) -> tuple:
        """Calculate window position based on screen size"""
        screen_width = self._root.winfo_screenwidth()
        screen_height = self._root.winfo_screenheight()
        taskbar_height = 48
        
        if self.position == "bottom-right":
            x = screen_width - self.width - self.margin
            y = screen_height - self.height - self.margin - taskbar_height
        elif self.position == "bottom-left":
            x = self.margin
            y = screen_height - self.height - self.margin - taskbar_height
        elif self.position == "top-right":
            x = screen_width - self.width - self.margin
            y = self.margin
        elif self.position == "top-left":
            x = self.margin
            y = self.margin
        elif self.position == "center":
            x = (screen_width - self.width) // 2
            y = (screen_height - self.height) // 2
        else:
            x = screen_width - self.width - self.margin
            y = screen_height - self.height - self.margin - taskbar_height
        
        return x, y
    
    def _start_drag(self, event):
        """Start window drag"""
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
    
    def _on_drag(self, event):
        """Handle window drag"""
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        x = self._root.winfo_x() + dx
        y = self._root.winfo_y() + dy
        self._root.geometry(f"+{x}+{y}")
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scroll"""
        self._chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_input_focus_in(self, event):
        """Handle input focus in"""
        if self._input_entry.get() == "Ask me anything...":
            self._input_entry.delete(0, tk.END)
            self._input_entry.configure(fg=self.theme.text_color)
    
    def _on_input_focus_out(self, event):
        """Handle input focus out"""
        if not self._input_entry.get():
            self._input_entry.insert(0, "Ask me anything...")
            self._input_entry.configure(fg="#666688")
    
    def _on_submit(self, event):
        """Handle query submission"""
        query = self._input_entry.get().strip()
        if query and query != "Ask me anything...":
            # Clear input
            self._input_entry.delete(0, tk.END)
            
            # Add user message
            self._add_message_bubble(query, is_user=True)
            
            # Show processing
            self._set_processing(True)
            
            # Callback
            if self.on_query_callback:
                threading.Thread(
                    target=self.on_query_callback,
                    args=(query,),
                    daemon=True
                ).start()
    
    def _toggle_voice(self):
        """Toggle voice recording"""
        if self._is_recording:
            self._stop_voice_recording()
        else:
            self._start_voice_recording()
    
    def _copy_selection(self, text_widget):
        """Copy selected text to clipboard"""
        try:
            text_widget.configure(state=tk.NORMAL)
            selected = text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            text_widget.configure(state=tk.DISABLED)
            self._root.clipboard_clear()
            self._root.clipboard_append(selected)
        except tk.TclError:
            pass  # No selection
    
    def _copy_all(self, text_widget):
        """Copy all text from widget to clipboard"""
        text_widget.configure(state=tk.NORMAL)
        all_text = text_widget.get("1.0", tk.END).strip()
        text_widget.configure(state=tk.DISABLED)
        self._root.clipboard_clear()
        self._root.clipboard_append(all_text)
        self._set_status("Copied to clipboard!", self.theme.user_color)
    
    def _show_copy_menu(self, event, text_widget):
        """Show right-click context menu for copying"""
        menu = tk.Menu(self._root, tearoff=0, bg=self.theme.input_bg, fg=self.theme.text_color)
        menu.add_command(label="Copy Selection", command=lambda: self._copy_selection(text_widget))
        menu.add_command(label="Copy All", command=lambda: self._copy_all(text_widget))
        menu.tk_popup(event.x_root, event.y_root)
    
    def _start_voice_recording(self):
        """Start voice recording"""
        self._is_recording = True
        self._voice_btn.configure(bg=self.theme.error_color, text="⏹")
        self._set_status("Listening... Speak now!", self.theme.error_color)
        
        if self.on_voice_start_callback:
            threading.Thread(target=self.on_voice_start_callback, daemon=True).start()
    
    def _stop_voice_recording(self):
        """Stop voice recording"""
        self._is_recording = False
        self._voice_btn.configure(bg=self.theme.button_bg, text="🎤")
        self._set_status("Processing speech...", self.theme.accent_color)
        
        if self.on_voice_stop_callback:
            threading.Thread(target=self.on_voice_stop_callback, daemon=True).start()
    
    def _set_processing(self, processing: bool):
        """Set processing state"""
        self._is_processing = processing
        if processing:
            self._set_status("Thinking...", self.theme.accent_color)
            self._add_typing_indicator()
        else:
            self._set_status("Ready - Watching your screen", self.theme.user_color)
    
    def _add_typing_indicator(self):
        """Add typing indicator"""
        # This will be replaced by actual response
        pass
    
    def _set_status(self, text: str, color: Optional[str] = None):
        """Set status bar text"""
        if self._status_label:
            self._status_label.configure(text=text)
        if color and self._status_canvas and self._status_indicator:
            self._status_canvas.itemconfigure(self._status_indicator, fill=color)
    
    def _do_minimize(self):
        """Minimize window to small bar"""
        if self._is_minimized:
            # Restore
            self._root.geometry(f"{self.width}x{self.height}")
            self._chat_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
            self._input_frame.pack(fill=tk.X, padx=8, pady=4)
            self._is_minimized = False
        else:
            # Minimize to title bar only
            self._chat_frame.pack_forget()
            self._input_frame.pack_forget()
            self._root.geometry(f"{self.width}x{56}")
            self._is_minimized = True
    
    def _do_close(self):
        """Close/hide window"""
        if self.on_close_callback:
            self.on_close_callback()
        self._root.withdraw()
        self._is_visible = False
    
    def _do_show(self):
        """Show window"""
        self._root.deiconify()
        self._root.lift()
        self._is_visible = True
    
    def _process_commands(self):
        """Process pending commands from queue"""
        try:
            while True:
                cmd, args = self._command_queue.get_nowait()
                
                if cmd == "show":
                    self._do_show()
                elif cmd == "hide":
                    self._do_close()
                elif cmd == "add_response":
                    self._add_message_bubble(args.get("text", ""), is_user=False)
                    self._set_processing(False)
                elif cmd == "add_user_message":
                    self._add_message_bubble(args.get("text", ""), is_user=True)
                elif cmd == "set_status":
                    self._set_status(args.get("text", ""), args.get("color"))
                elif cmd == "show_error":
                    self._add_message_bubble(f"⚠️ {args.get('text', 'Error')}", is_user=False)
                    self._set_processing(False)
                elif cmd == "voice_text":
                    # Voice transcription received
                    text = args.get("text", "")
                    if text:
                        self._input_entry.delete(0, tk.END)
                        self._input_entry.insert(0, text)
                        self._on_submit(None)
                elif cmd == "stop_recording":
                    self._is_recording = False
                    self._voice_btn.configure(bg=self.theme.button_bg, text="🎤")
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        if self._running and self._root:
            self._root.after(50, self._process_commands)
    
    def start(self):
        """Start the overlay in a separate thread"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
        # Wait for window to be ready
        self._ready.wait(timeout=3.0)
        logger.info("Overlay started")
    
    def _run(self):
        """Main Tkinter loop"""
        self._create_window()
        
        # Start processing command queue
        self._root.after(50, self._process_commands)
        
        # Signal ready
        self._ready.set()
        
        # Custom mainloop
        while self._running:
            try:
                self._root.update()
                time.sleep(0.016)  # ~60fps
            except tk.TclError:
                break
            except Exception as e:
                logger.error(f"Overlay error: {e}")
                break
    
    def stop(self):
        """Stop the overlay"""
        self._running = False
        
        if self._root:
            try:
                self._root.quit()
                self._root.destroy()
            except:
                pass
        
        logger.info("Overlay stopped")
    
    # === Public thread-safe methods ===
    
    def show(self, text: Optional[str] = None):
        """Show the overlay window (thread-safe)"""
        self._command_queue.put(("show", {}))
        if text:
            self.add_response(text)
    
    def hide(self):
        """Hide the overlay window (thread-safe)"""
        self._command_queue.put(("hide", {}))
    
    def add_response(self, text: str):
        """Add AI response to chat (thread-safe)"""
        self._command_queue.put(("add_response", {"text": text}))
    
    def add_user_message(self, text: str):
        """Add user message to chat (thread-safe)"""
        self._command_queue.put(("add_user_message", {"text": text}))
    
    def set_status(self, text: str, color: Optional[str] = None):
        """Set status bar text (thread-safe)"""
        self._command_queue.put(("set_status", {"text": text, "color": color}))
    
    def show_error(self, text: str):
        """Show error message (thread-safe)"""
        self._command_queue.put(("show_error", {"text": text}))
    
    def set_voice_text(self, text: str):
        """Set transcribed voice text (thread-safe)"""
        self._command_queue.put(("voice_text", {"text": text}))
    
    def stop_recording_ui(self):
        """Stop recording UI state (thread-safe)"""
        self._command_queue.put(("stop_recording", {}))
    
    # === Convenience methods (for backward compatibility) ===
    
    def show_listening(self):
        """Show listening state"""
        self.set_status("Listening...", "#ffaa00")
    
    def show_processing(self):
        """Show processing state"""
        self.set_status("Analyzing screen and processing...", "#4a9eff")
    
    def show_response(self, text: str):
        """Show AI response"""
        self.add_response(text)
        self.set_status("Ready - Watching your screen", "#00ff88")
    
    @property
    def is_visible(self) -> bool:
        """Check if overlay is visible"""
        return self._is_visible


class SystemTrayIcon:
    """System tray icon for AIAS"""
    
    def __init__(
        self,
        on_activate: Optional[Callable] = None,
        on_quit: Optional[Callable] = None
    ):
        self.on_activate = on_activate
        self.on_quit = on_quit
        self._icon = None
    
    def start(self):
        """Start the system tray icon"""
        try:
            import pystray
            from PIL import Image, ImageDraw
            
            image = self._create_icon_image()
            
            menu = pystray.Menu(
                pystray.MenuItem("Show AIAS", self._on_activate),
                pystray.MenuItem("Quit", self._on_quit)
            )
            
            self._icon = pystray.Icon(
                "AIAS",
                image,
                "AIAS - AI Assistant",
                menu
            )
            
            threading.Thread(target=self._icon.run, daemon=True).start()
            logger.info("System tray icon started")
            
        except ImportError:
            logger.warning("pystray not installed. System tray icon disabled.")
        except Exception as e:
            logger.error(f"Failed to create system tray icon: {e}")
    
    def _create_icon_image(self):
        """Create icon image"""
        from PIL import Image, ImageDraw
        
        size = 64
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse([4, 4, size-4, size-4], fill='#4a9eff')
        draw.text((size//2 - 10, size//2 - 14), "A", fill='white')
        
        return image
    
    def _on_activate(self, icon, item):
        """Handle activate"""
        if self.on_activate:
            self.on_activate()
    
    def _on_quit(self, icon, item):
        """Handle quit"""
        if self._icon:
            self._icon.stop()
        if self.on_quit:
            self.on_quit()
    
    def stop(self):
        """Stop the system tray icon"""
        if self._icon:
            self._icon.stop()
            logger.info("System tray icon stopped")
