import os
import psutil
import logging
from typing import Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('priority_control')

# Define priority levels for different platforms
PRIORITY_LEVELS = {
    'windows': {
        'high': psutil.HIGH_PRIORITY_CLASS,
        'above_normal': psutil.ABOVE_NORMAL_PRIORITY_CLASS,
        'normal': psutil.NORMAL_PRIORITY_CLASS,
        'below_normal': psutil.BELOW_NORMAL_PRIORITY_CLASS,
        'low': psutil.IDLE_PRIORITY_CLASS,
        'realtime': psutil.REALTIME_PRIORITY_CLASS,  # Requires admin privileges
    },
    'unix': {
        'high': -10,
        'above_normal': -5,
        'normal': 0,
        'below_normal': 5,
        'low': 10,
        'realtime': -20,  # Requires root privileges
    }
}

def get_platform():
    """
    Get current platform type
    
    Returns:
        str: 'windows' or 'unix'
    """
    return 'windows' if os.name == 'nt' else 'unix'

def set_process_priority(priority: str = 'normal', pid: Optional[int] = None) -> bool:
    """
    Set process priority
    
    Args:
        priority: Priority level ('high', 'above_normal', 'normal', 'below_normal', 'low', 'realtime')
        pid: Process ID (current process if None)
    
    Returns:
        bool: Success status
    """
    try:
        platform = get_platform()
        process = psutil.Process(pid if pid is not None else os.getpid())
        
        if priority not in PRIORITY_LEVELS[platform]:
            logger.error(f"Invalid priority level: {priority}")
            return False
        
        priority_value = PRIORITY_LEVELS[platform][priority]
        
        if platform == 'windows':
            process.nice(priority_value)
        else:
            process.nice(priority_value)
        
        logger.info(f"Set priority to {priority} for PID {process.pid}")
        return True
    except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
        logger.error(f"Failed to set priority: {e}")
        return False

def get_process_priority(pid: Optional[int] = None) -> Dict[str, Any]:
    """
    Get current process priority info
    
    Args:
        pid: Process ID (current process if None)
    
    Returns:
        Dict[str, Any]: Priority information
    """
    try:
        platform = get_platform()
        process = psutil.Process(pid if pid is not None else os.getpid())
        nice_value = process.nice()
        
        # Map nice value back to priority level
        priority_level = "unknown"
        for level, value in PRIORITY_LEVELS[platform].items():
            if nice_value == value:
                priority_level = level
                break
        
        return {
            'pid': process.pid,
            'nice_value': nice_value,
            'priority_level': priority_level,
            'platform': platform
        }
    except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
        logger.error(f"Failed to get priority: {e}")
        return {
            'pid': pid if pid is not None else os.getpid(), 
            'error': str(e)
        }

def set_thread_priority(thread_id: int, priority: str = 'normal') -> bool:
    """
    Set thread priority (platform-specific)
    
    Args:
        thread_id: Thread ID
        priority: Priority level
    
    Returns:
        bool: Success status
    """
    # This is platform specific and may not work on all systems
    platform = get_platform()
    
    try:
        if platform == 'windows':
            import win32api
            import win32process
            import win32con
            
            priority_map = {
                'idle': win32process.THREAD_PRIORITY_IDLE,
                'low': win32process.THREAD_PRIORITY_LOWEST,
                'below_normal': win32process.THREAD_PRIORITY_BELOW_NORMAL,
                'normal': win32process.THREAD_PRIORITY_NORMAL,
                'above_normal': win32process.THREAD_PRIORITY_ABOVE_NORMAL,
                'high': win32process.THREAD_PRIORITY_HIGHEST,
                'realtime': win32process.THREAD_PRIORITY_TIME_CRITICAL,
            }
            
            if priority not in priority_map:
                logger.error(f"Invalid thread priority: {priority}")
                return False
                
            handle = win32api.OpenThread(win32con.THREAD_SET_INFORMATION, False, thread_id)
            win32process.SetThreadPriority(handle, priority_map[priority])
            win32api.CloseHandle(handle)
            
            logger.info(f"Set thread {thread_id} priority to {priority}")
            return True
        else:
            # On Unix systems, can't easily set thread priority directly
            logger.warning("Thread priority setting not implemented for this platform")
            return False
    except Exception as e:
        logger.error(f"Error setting thread priority: {e}")
        return False