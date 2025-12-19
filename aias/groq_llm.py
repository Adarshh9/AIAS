"""
AIAS Groq Vision LLM
Fast cloud-based vision LLM using Groq API with Llama 4 Scout/Maverick
"""

import os
import io
import base64
import time
from typing import Optional, List
from dataclasses import dataclass

from PIL import Image
from loguru import logger

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not installed. Run: pip install groq")


@dataclass
class LLMResponse:
    """Container for LLM response"""
    text: str
    generation_time: float
    tokens_generated: int
    images_processed: int


class GroqVisionLLM:
    """
    Groq-based Vision LLM for fast cloud inference
    
    Best vision models (Llama 4 family):
    - meta-llama/llama-4-scout-17b-16e-instruct (RECOMMENDED - 128K context, fast)
    - meta-llama/llama-4-maverick-17b-128e-instruct (larger, slower)
    
    Limits:
    - Max 5 images per request
    - Max 4MB for base64 encoded images  
    - Max 33 megapixels (33177600 total pixels) per image
    """
    
    # Vision models (can process images)
    VISION_MODELS = {
        "llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",      # Best for speed + quality
        "llama-4-maverick": "meta-llama/llama-4-maverick-17b-128e-instruct", # Larger model
        "llama-3.2-90b-vision": "llama-3.2-90b-vision-preview",            # Legacy
        "llama-3.2-11b-vision": "llama-3.2-11b-vision-preview",            # Legacy
    }
    
    # Text-only models (fallback)
    TEXT_MODELS = {
        "llama-3.3-70b": "llama-3.3-70b-versatile",
        "llama-3.1-8b": "llama-3.1-8b-instant",
        "mixtral-8x7b": "mixtral-8x7b-32768",
    }
    
    def __init__(
        self,
        model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize Groq Vision LLM
        
        Args:
            model_name: Groq model name (use Llama 4 Scout for best vision)
            api_key: Groq API key (or set GROQ_API_KEY env var)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (lower = more focused)
            system_prompt: Custom system prompt
        """
        if not GROQ_AVAILABLE:
            raise ImportError("Groq not installed. Run: pip install groq")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Initialize Groq client
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter. Get key at: https://console.groq.com/keys"
            )
        
        self._client = Groq(api_key=self._api_key)
        
        # Check if vision model
        self._is_vision_model = any(x in model_name.lower() for x in ["vision", "llava", "scout", "maverick"])
        
        logger.info(f"Groq LLM initialized: {model_name} (vision={self._is_vision_model})")
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for screen-aware assistant with memory"""
        return """You are AIAS, a personalized screen-aware AI assistant. You can SEE the user's screen through screenshots.

PERSONALIZATION:
- You remember information about the user from past conversations
- If USER PROFILE or RECENT CONVERSATION context is provided, use it to personalize responses
- Address the user by name when you know it
- Reference their interests, contacts, and preferences when relevant
- Build on previous conversations naturally

SCREEN READING:
- READ ALL TEXT visible in the screenshots carefully!
- When you see messaging apps (WhatsApp, Discord, Telegram, etc.):
  - READ the actual message text in the conversation
  - Note who sent what message (look at message bubbles, names, timestamps)
  - If asked "what to reply", suggest an actual contextual reply based on the conversation
- When you see code editors: READ and explain the actual code visible
- When you see documents/webpages: READ the actual text content

BE SPECIFIC: Always reference actual text, names, or content you see on screen.
NEVER give vague responses - read and analyze the visible content.
Remember: You're a personal assistant who knows and remembers the user.

BE SPECIFIC: Always reference actual text, names, or content you see on screen.
NEVER give vague responses - read and analyze the visible content."""
    
    def _image_to_base64(self, image: Image.Image, max_size: int = 1920) -> str:
        """
        Convert PIL Image to base64 string, resizing if needed
        
        Groq limits:
        - Max 4MB for base64 encoded images
        - Max 33 megapixels per image
        """
        # Calculate current megapixels
        current_mp = (image.width * image.height) / 1_000_000
        
        # Resize if exceeds 33MP or max_size
        if current_mp > 30 or image.width > max_size or image.height > max_size:
            # Scale down to fit within limits
            ratio = min(max_size / image.width, max_size / image.height)
            if current_mp > 30:
                # Also check megapixel limit
                mp_ratio = (30_000_000 / (image.width * image.height)) ** 0.5
                ratio = min(ratio, mp_ratio)
            
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to {new_size[0]}x{new_size[1]}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Encode to base64 with quality adjustment to stay under 4MB
        quality = 90
        while quality >= 50:
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Check size (base64 is ~33% larger than binary)
            size_mb = len(base64_data) / (1024 * 1024)
            if size_mb < 3.5:  # Leave some margin under 4MB
                logger.debug(f"Image encoded: {size_mb:.2f}MB, quality={quality}")
                return base64_data
            
            quality -= 10
        
        return base64_data
    
    def load_model(self):
        """No-op for API-based model (already initialized)"""
        logger.info(f"Groq model ready: {self.model_name}")
    
    def generate(
        self,
        query: str,
        images: Optional[List[Image.Image]] = None,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate response for query with optional images
        
        Args:
            query: User query text
            images: List of PIL Images (screenshots) - max 5
            system_prompt: Override default system prompt
            
        Returns:
            LLMResponse with generated text and metadata
        """
        start_time = time.time()
        
        try:
            # Build messages
            messages = []
            
            # System message
            sys_prompt = system_prompt or self.system_prompt
            messages.append({
                "role": "system",
                "content": sys_prompt
            })
            
            # User message with images
            if images and self._is_vision_model:
                # Vision model - include images (max 5)
                images_to_send = images[:5]  # Groq limit: max 5 images
                content = []
                
                for i, img in enumerate(images_to_send):
                    base64_img = self._image_to_base64(img)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    })
                
                content.append({
                    "type": "text",
                    "text": f"Screenshot of user's current screen above.\n\nUser's question: {query}"
                })
                
                messages.append({
                    "role": "user",
                    "content": content
                })
                
                logger.debug(f"Sending {len(images_to_send)} images to Groq")
            else:
                # Text-only
                messages.append({
                    "role": "user",
                    "content": query
                })
            
            # Call Groq API
            logger.debug(f"Calling Groq API: {self.model_name}")
            
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                top_p=0.9,
                stream=False
            )
            
            response_text = completion.choices[0].message.content
            tokens = completion.usage.completion_tokens if completion.usage else len(response_text.split())
            
            generation_time = time.time() - start_time
            
            logger.info(f"Groq response in {generation_time:.2f}s ({tokens} tokens)")
            
            return LLMResponse(
                text=response_text,
                generation_time=generation_time,
                tokens_generated=tokens,
                images_processed=len(images) if images else 0
            )
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            generation_time = time.time() - start_time
            return LLMResponse(
                text=f"Error: {str(e)}",
                generation_time=generation_time,
                tokens_generated=0,
                images_processed=0
            )
    
    def generate_stream(
        self,
        query: str,
        images: Optional[List[Image.Image]] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Generate streaming response
        
        Yields:
            Text chunks as they arrive
        """
        # Build messages
        messages = []
        sys_prompt = system_prompt or self.system_prompt
        messages.append({"role": "system", "content": sys_prompt})
        
        if images and self._is_vision_model:
            content = []
            for img in images[:5]:  # Max 5 images
                base64_img = self._image_to_base64(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                })
            content.append({"type": "text", "text": f"Screenshot above.\n\nUser: {query}"})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": query})
        
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            yield f"Error: {str(e)}"
    
    def unload_model(self):
        """No-op for API-based model"""
        pass
    
    @classmethod
    def list_models(cls) -> dict:
        """List available models"""
        return {
            "vision": cls.VISION_MODELS,
            "text": cls.TEXT_MODELS
        }
