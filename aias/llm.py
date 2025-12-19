"""
AIAS Vision LLM
Multimodal LLM processing with Qwen2-VL for screen understanding
"""

import gc
import time
import threading
from typing import Optional, List, Union, Generator
from queue import Queue
from dataclasses import dataclass

from PIL import Image
from loguru import logger


@dataclass
class LLMResponse:
    """Container for LLM response"""
    text: str
    generation_time: float
    tokens_generated: int
    images_processed: int


class VisionLLM:
    """
    Multimodal LLM for vision + text processing
    Supports Qwen2-VL, LLaVA, and other vision models
    """
    
    SUPPORTED_MODELS = {
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
        "qwen2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct",
        "llava-1.6-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-1.6-13b": "llava-hf/llava-v1.6-vicuna-13b-hf",
    }
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._inference_count = 0
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for AIAS"""
        return """You are AIAS, a screen-aware AI assistant. You can SEE the user's screen through screenshots.

CRITICAL: READ ALL TEXT visible in the screenshots carefully!

When you see messaging apps (WhatsApp, Discord, Telegram, etc.):
- READ the actual message text in the conversation
- Note who sent what (look at message bubbles, names, timestamps)
- If asked "what to reply", suggest an actual reply based on the conversation context

When you see code editors:
- READ the actual code visible
- Explain what the code does

When you see documents/webpages:
- READ the actual text content
- Summarize or explain based on what you see

NEVER give vague responses like "the user has not replied yet" - instead READ what's on screen and give specific, helpful answers based on the actual visible content."""
    
    def load_model(self):
        """Load the model and processor"""
        if self._model is not None:
            return
        
        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()
        
        try:
            import torch
            from transformers import AutoProcessor
            
            # Determine torch dtype
            if self.torch_dtype == "auto":
                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            elif self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Check if it's a Qwen model
            if "qwen" in self.model_name.lower():
                self._load_qwen_model(dtype)
            elif "llava" in self.model_name.lower():
                self._load_llava_model(dtype)
            else:
                # Generic vision model loading
                self._load_generic_model(dtype)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_qwen_model(self, dtype):
        """Load Qwen2-VL model"""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device_map,
            attn_implementation="flash_attention_2" if self._check_flash_attn() else "eager"
        )
        
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model_type = "qwen"
    
    def _load_llava_model(self, dtype):
        """Load LLaVA model"""
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device_map
        )
        
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model_type = "llava"
    
    def _load_generic_model(self, dtype):
        """Load generic vision-language model"""
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device_map
        )
        
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model_type = "generic"
    
    def _check_flash_attn(self) -> bool:
        """Check if Flash Attention 2 is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
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
            images: List of PIL Images (screenshots)
            system_prompt: Override default system prompt
            
        Returns:
            LLMResponse with generated text and metadata
        """
        if self._model is None:
            self.load_model()
        
        with self._lock:
            start_time = time.time()
            
            try:
                if self._model_type == "qwen":
                    response = self._generate_qwen(query, images, system_prompt)
                elif self._model_type == "llava":
                    response = self._generate_llava(query, images, system_prompt)
                else:
                    response = self._generate_generic(query, images, system_prompt)
                
                generation_time = time.time() - start_time
                self._inference_count += 1
                
                # Clear CUDA cache periodically
                if self._inference_count % 10 == 0:
                    self._clear_cache()
                
                return LLMResponse(
                    text=response,
                    generation_time=generation_time,
                    tokens_generated=len(response.split()),  # Approximate
                    images_processed=len(images) if images else 0
                )
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return LLMResponse(
                    text=f"Error: {str(e)}",
                    generation_time=time.time() - start_time,
                    tokens_generated=0,
                    images_processed=0
                )
    
    def _generate_qwen(
        self,
        query: str,
        images: Optional[List[Image.Image]],
        system_prompt: Optional[str]
    ) -> str:
        """Generate using Qwen2-VL"""
        import torch
        
        # Build conversation
        messages = []
        
        # System message
        sys_prompt = system_prompt or self.system_prompt
        if sys_prompt:
            messages.append({
                "role": "system",
                "content": sys_prompt
            })
        
        # User message with images
        content = []
        
        if images:
            for i, img in enumerate(images):
                content.append({
                    "type": "image",
                    "image": img
                })
            content.append({
                "type": "text",
                "text": f"The screenshots above show the user's current screen. The user's question is:\n\n{query}\n\nAnalyze what you see and answer the question directly. Do NOT give instructions about using this assistant."
            })
        else:
            content.append({
                "type": "text",
                "text": query
            })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Process with Qwen format
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Prepare inputs
        if images:
            inputs = self._processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt"
            ).to(self._model.device)
        else:
            inputs = self._processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self._model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0
            )
        
        # Decode output
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()
    
    def _generate_llava(
        self,
        query: str,
        images: Optional[List[Image.Image]],
        system_prompt: Optional[str]
    ) -> str:
        """Generate using LLaVA"""
        import torch
        
        # Build prompt
        sys_prompt = system_prompt or self.system_prompt
        
        if images:
            # LLaVA uses <image> tokens
            image_tokens = "<image>" * len(images)
            prompt = f"{sys_prompt}\n\n{image_tokens}\nUser: {query}\nAssistant:"
        else:
            prompt = f"{sys_prompt}\n\nUser: {query}\nAssistant:"
        
        # Process
        if images:
            inputs = self._processor(
                text=prompt,
                images=images,
                return_tensors="pt"
            ).to(self._model.device)
        else:
            inputs = self._processor(
                text=prompt,
                return_tensors="pt"
            ).to(self._model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0
            )
        
        # Decode
        response = self._processor.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        # Extract assistant response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
    def _generate_generic(
        self,
        query: str,
        images: Optional[List[Image.Image]],
        system_prompt: Optional[str]
    ) -> str:
        """Generate using generic vision model"""
        import torch
        
        sys_prompt = system_prompt or self.system_prompt
        prompt = f"{sys_prompt}\n\n{query}"
        
        if images:
            inputs = self._processor(
                text=prompt,
                images=images[0],  # Most generic models take single image
                return_tensors="pt"
            ).to(self._model.device)
        else:
            inputs = self._processor(
                text=prompt,
                return_tensors="pt"
            ).to(self._model.device)
        
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens
            )
        
        response = self._processor.decode(output_ids[0], skip_special_tokens=True)
        return response
    
    def generate_stream(
        self,
        query: str,
        images: Optional[List[Image.Image]] = None,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream response token by token
        
        Yields:
            Individual tokens as they're generated
        """
        if self._model is None:
            self.load_model()
        
        # TODO: Implement proper streaming with TextIteratorStreamer
        # For now, return full response
        response = self.generate(query, images, system_prompt)
        
        # Simulate streaming by yielding words
        for word in response.text.split():
            yield word + " "
    
    def _clear_cache(self):
        """Clear CUDA cache to free memory"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug("CUDA cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def unload_model(self):
        """Unload model from memory"""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._clear_cache()
            logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None
    
    @property
    def device(self) -> str:
        """Get model device"""
        if self._model is not None:
            return str(self._model.device)
        return "not loaded"


class LLMWorker:
    """
    Background worker for LLM inference
    Handles query queue and async processing
    """
    
    def __init__(
        self,
        llm: VisionLLM,
        on_response_callback=None
    ):
        self.llm = llm
        self.on_response_callback = on_response_callback
        
        self._query_queue = Queue()
        self._running = False
        self._thread = None
        self._current_query = None
    
    def start(self):
        """Start the worker thread"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("LLM worker started")
    
    def stop(self):
        """Stop the worker thread"""
        self._running = False
        if self._thread:
            self._query_queue.put(None)  # Sentinel to unblock
            self._thread.join(timeout=5.0)
        logger.info("LLM worker stopped")
    
    def submit_query(
        self,
        query: str,
        images: Optional[List[Image.Image]] = None
    ):
        """Submit a query for processing"""
        self._query_queue.put({
            "query": query,
            "images": images,
            "timestamp": time.time()
        })
        logger.info(f"Query submitted: {query[:50]}...")
    
    def _process_loop(self):
        """Main processing loop"""
        while self._running:
            try:
                item = self._query_queue.get(timeout=1.0)
                
                if item is None:
                    continue
                
                self._current_query = item
                
                logger.info(f"Processing query: {item['query'][:50]}...")
                response = self.llm.generate(
                    query=item['query'],
                    images=item.get('images')
                )
                
                logger.info(f"Generated response in {response.generation_time:.1f}s")
                
                if self.on_response_callback:
                    self.on_response_callback(response)
                
                self._current_query = None
                
            except Exception as e:
                if self._running:
                    logger.error(f"LLM worker error: {e}")
    
    @property
    def is_processing(self) -> bool:
        """Check if currently processing a query"""
        return self._current_query is not None
    
    @property
    def queue_size(self) -> int:
        """Number of pending queries"""
        return self._query_queue.qsize()


# Convenience function for quick inference
def quick_inference(
    query: str,
    images: Optional[List[Image.Image]] = None,
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
) -> str:
    """Quick one-shot inference"""
    llm = VisionLLM(model_name=model_name)
    response = llm.generate(query, images)
    return response.text
