"""
AIAS Logging System
Comprehensive logging with loguru - saves logs and screenshots per run
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

# Global run directory for this session
_run_dir: Optional[Path] = None
_screenshots_dir: Optional[Path] = None
_query_count: int = 0


def setup_logging(base_dir: str = "logs") -> Path:
    """
    Setup loguru logging for a new AIAS run
    Creates a timestamped directory for this session
    
    Returns:
        Path to the run directory
    """
    global _run_dir, _screenshots_dir, _query_count
    
    # Create base logs directory
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _run_dir = base_path / f"run_{timestamp}"
    _run_dir.mkdir(exist_ok=True)
    
    # Create screenshots subdirectory
    _screenshots_dir = _run_dir / "screenshots"
    _screenshots_dir.mkdir(exist_ok=True)
    
    # Reset query counter
    _query_count = 0
    
    # Remove default logger
    logger.remove()
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="DEBUG",
        colorize=True
    )
    
    # Add file handler for all logs
    logger.add(
        _run_dir / "aias.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="50 MB",
        encoding="utf-8"
    )
    
    # Add separate file for queries only
    logger.add(
        _run_dir / "queries.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        filter=lambda record: record["extra"].get("query_log", False),
        encoding="utf-8"
    )
    
    # Add separate file for LLM interactions
    logger.add(
        _run_dir / "llm.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="DEBUG",
        filter=lambda record: record["extra"].get("llm_log", False),
        encoding="utf-8"
    )
    
    logger.info(f"AIAS logging initialized - Run directory: {_run_dir}")
    
    return _run_dir


def get_run_dir() -> Optional[Path]:
    """Get current run directory"""
    return _run_dir


def get_screenshots_dir() -> Optional[Path]:
    """Get screenshots directory for this run"""
    return _screenshots_dir


def log_query(query: str, source: str = "unknown") -> int:
    """
    Log a user query
    
    Args:
        query: The user's query text
        source: Source of query (voice, keyboard, overlay)
        
    Returns:
        Query ID for this session
    """
    global _query_count
    _query_count += 1
    
    query_logger = logger.bind(query_log=True)
    query_logger.info(f"[Query #{_query_count}] Source: {source} | Query: {query}")
    
    logger.info(f"Query #{_query_count} received from {source}: {query[:100]}...")
    
    return _query_count


def save_query_screenshots(query_id: int, screenshots: list, prefix: str = "screenshot") -> list:
    """
    Save screenshots used for a specific query
    
    Args:
        query_id: The query number
        screenshots: List of PIL Images
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    if not _screenshots_dir or not screenshots:
        return []
    
    saved_paths = []
    query_dir = _screenshots_dir / f"query_{query_id:04d}"
    query_dir.mkdir(exist_ok=True)
    
    for i, img in enumerate(screenshots):
        filepath = query_dir / f"{prefix}_{i+1}.png"
        try:
            img.save(filepath, "PNG")
            saved_paths.append(filepath)
            logger.debug(f"Saved screenshot: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
    
    logger.info(f"Query #{query_id}: Saved {len(saved_paths)} screenshots to {query_dir}")
    
    return saved_paths


def log_llm_request(query_id: int, query: str, num_images: int, system_prompt: str):
    """Log LLM request details"""
    llm_logger = logger.bind(llm_log=True)
    llm_logger.info("=" * 80)
    llm_logger.info(f"LLM REQUEST - Query #{query_id}")
    llm_logger.info(f"Images: {num_images}")
    llm_logger.info(f"System Prompt:\n{system_prompt}")
    llm_logger.info(f"User Query: {query}")
    llm_logger.info("-" * 80)


def log_llm_response(query_id: int, response: str, generation_time: float, tokens: int):
    """Log LLM response details"""
    llm_logger = logger.bind(llm_log=True)
    llm_logger.info(f"LLM RESPONSE - Query #{query_id}")
    llm_logger.info(f"Generation Time: {generation_time:.2f}s | Tokens: ~{tokens}")
    llm_logger.info(f"Response:\n{response}")
    llm_logger.info("=" * 80 + "\n")
    
    logger.info(f"Query #{query_id}: LLM responded in {generation_time:.2f}s ({tokens} tokens)")


def log_screenshot_captured(resolution: tuple, capture_time_ms: float):
    """Log screenshot capture event"""
    logger.debug(f"Screenshot captured: {resolution[0]}x{resolution[1]} in {capture_time_ms:.1f}ms")


# Create a module-level logger for easy import
def get_logger(name: str):
    """Get a named logger instance"""
    return logger.bind(name=name)
