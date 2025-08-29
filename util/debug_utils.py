#!/usr/bin/env python3
"""
Debug utilities for conditional logging and debug output
"""

import os
import logging
import time
from typing import Optional, Callable, Any

class DebugLogger:
    """
    Conditional debug logger that can be enabled/disabled via config file
    """
    
    def __init__(self, name: str = None, default_level: str = "INFO", config: dict = None):
        """
        Initialize debug logger
        
        Args:
            name: Logger name (defaults to module name)
            default_level: Default log level if not specified in config
            config: Config dictionary containing debug settings
        """
        self.name = name or __name__
        self.default_level = default_level
        self.config = config or {}
        self._logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger based on config settings"""
        # Check global debug state first - this is the primary control
        from util.debug_utils import get_global_debug_enabled
        global_debug_enabled = get_global_debug_enabled()
        
        # If global debug is disabled, always disable regardless of any other settings
        if not global_debug_enabled:
            # Create a null logger that does nothing
            self._logger = logging.getLogger(f"{self.name}_null")
            self._logger.addHandler(logging.NullHandler())
            self._logger.setLevel(logging.CRITICAL)  # Disable all output
            return
        
        # Only proceed with logging setup if global debug is enabled
        # Check config settings (fallback to environment variables for backward compatibility)
        debug_enabled = self.config.get('debug_enabled', True)  # Default to True if global is enabled
        if not debug_enabled:
            # Fallback to environment variable
            debug_enabled = os.getenv('ISP_DEBUG', 'true').lower() == 'true'
        
        # If this specific logger should be disabled, create null logger
        if not debug_enabled:
            # Create a null logger that does nothing
            self._logger = logging.getLogger(f"{self.name}_null")
            self._logger.addHandler(logging.NullHandler())
            self._logger.setLevel(logging.CRITICAL)  # Disable all output
            return
        
        log_level = self.config.get('debug_log_level', self.default_level).upper()
        if log_level == self.default_level:
            # Fallback to environment variable
            log_level = os.getenv('ISP_LOG_LEVEL', self.default_level).upper()
        
        log_file = self.config.get('debug_log_file', None)
        if log_file is None:
            # Fallback to environment variable
            log_file = os.getenv('ISP_LOG_FILE', None)
        
        if not debug_enabled:
            # Create a null logger that does nothing
            self._logger = logging.getLogger(f"{self.name}_null")
            self._logger.addHandler(logging.NullHandler())
            self._logger.setLevel(logging.CRITICAL)  # Disable all output
            return
        
        # Create real logger
        self._logger = logging.getLogger(self.name)
        
        # Clear existing handlers
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)
        
        # Set log level
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        self._logger.setLevel(level_map.get(log_level, logging.INFO))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message"""
        if self._logger:
            self._logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        if self._logger:
            self._logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        if self._logger:
            self._logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        if self._logger:
            self._logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        if self._logger:
            self._logger.critical(message)

def get_debug_logger(name: str = None, config: dict = None) -> DebugLogger:
    """
    Get a debug logger instance
    
    Args:
        name: Logger name (defaults to calling module name)
        config: Config dictionary containing debug settings
    
    Returns:
        DebugLogger instance
    """
    return DebugLogger(name, config=config)

def is_debug_enabled() -> bool:
    """
    Check if debug output is enabled via environment variable
    
    Returns:
        True if ISP_DEBUG=true, False otherwise
    """
    return os.getenv('ISP_DEBUG', 'false').lower() == 'true'

def debug_print(message: str, force: bool = False):
    """
    Conditional print function that only prints when debug is enabled
    
    Args:
        message: Message to print
        force: If True, always print regardless of debug setting
    """
    if force or is_debug_enabled():
        print(message)

def time_function(func: Callable, logger: Optional[DebugLogger] = None, 
                  func_name: str = None) -> Callable:
    """
    Decorator to time function execution with conditional logging
    
    Args:
        func: Function to time
        logger: DebugLogger instance (optional)
        func_name: Function name for logging (defaults to func.__name__)
    
    Returns:
        Wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        func_name_str = func_name or func.__name__
        
        if logger:
            logger.info(f"Starting {func_name_str}")
        
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        if logger:
            logger.info(f"Completed {func_name_str} in {execution_time:.3f}s")
        
        return result
    
    return wrapper

def conditional_log(condition: bool = True, logger: Optional[DebugLogger] = None):
    """
    Decorator to conditionally log function execution
    
    Args:
        condition: Whether to enable logging for this function
        logger: DebugLogger instance (optional)
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            if condition and logger:
                logger.info(f"Starting {func_name}")
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if condition and logger:
                logger.info(f"Completed {func_name} in {execution_time:.3f}s")
            
            return result
        
        return wrapper
    return decorator

# Global debug state
_global_debug_enabled = None
_global_logger = None

def set_global_debug_enabled(enabled: bool):
    """Set global debug state"""
    global _global_debug_enabled
    _global_debug_enabled = enabled

def get_global_debug_enabled() -> bool:
    """Get global debug state"""
    global _global_debug_enabled
    if _global_debug_enabled is None:
        # Default to environment variable if not set
        _global_debug_enabled = os.getenv('ISP_DEBUG', 'false').lower() == 'true'
    return _global_debug_enabled

def get_global_logger() -> DebugLogger:
    """Get global debug logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = DebugLogger("ISP_Global")
    return _global_logger

def set_global_logger(logger: DebugLogger):
    """Set global debug logger instance"""
    global _global_logger
    _global_logger = logger
