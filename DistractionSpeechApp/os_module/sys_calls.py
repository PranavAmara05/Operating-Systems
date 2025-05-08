import os
import sys
import time
import psutil
import platform
import logging
import threading
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('sys_calls')

def get_system_info() -> Dict[str, Any]:
    """
    Get detailed system information
    
    Returns:
        Dict[str, Any]: System information
    """
    info = {
        'os': {
            'name': os.name,
            'platform': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
        },
        'python': {
            'version': platform.python_version(),
            'compiler': platform.python_compiler(),
            'implementation': platform.python_implementation(),
        },
        'process': {
            'pid': os.getpid(),
            'parent_pid': os.getppid(),
            'username': os.getlogin() if hasattr(os, 'getlogin') else 'unknown',
            'cwd': os.getcwd(),
        },
        'resources': {
            'cpu_count': os.cpu_count(),
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent,
            }
        }
    }
    
    logger.info(f"System info retrieved: {platform.system()} {platform.release()}")
    return info

def get_open_files(pid: Optional[int] = None) -> List[str]:
    """
    Get list of open files for a process
    
    Args:
        pid: Process ID (current process if None)
    
    Returns:
        List[str]: List of open file paths
    """
    try:
        process = psutil.Process(pid if pid is not None else os.getpid())
        open_files = [f.path for f in process.open_files()]
        return open_files
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        logger.warning(f"Cannot access open files for PID {pid}")
        return []

def get_network_connections(pid: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get network connections for a process
    
    Args:
        pid: Process ID (current process if None)
    
    Returns:
        List[Dict[str, Any]]: List of connection details
    """
    try:
        process = psutil.Process(pid if pid is not None else os.getpid())
        connections = []
        
        for conn in process.connections():
            connection_info = {
                'fd': conn.fd,
                'family': conn.family,
                'type': conn.type,
                'status': conn.status,
            }
            
            if conn.laddr:
                connection_info['local_address'] = {
                    'ip': conn.laddr.ip,
                    'port': conn.laddr.port
                }
            
            if conn.raddr:
                connection_info['remote_address'] = {
                    'ip': conn.raddr.ip,
                    'port': conn.raddr.port
                }
                
            connections.append(connection_info)
            
        return connections
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        logger.warning(f"Cannot access network connections for PID {pid}")
        return []

def monitor_system_resources(interval: float = 1.0, 
                           duration: Optional[float] = None,
                           callback: Optional[Callable[[Dict[str, Any]], None]] = None):
    """
    Monitor system resources in a background thread
    
    Args:
        interval: Monitoring interval in seconds
        duration: Total monitoring duration in seconds (None for indefinite)
        callback: Function to call with resource data
    
    Returns:
        threading.Thread: Monitor thread
    """
    def _monitor():
        start_time = time.time()
        while True:
            # Check if monitoring duration has elapsed
            if duration and time.time() - start_time > duration:
                break
                
            # Get resource metrics
            metrics = {
                'timestamp': time.time(),
                'cpu': {
                    'percent': psutil.cpu_percent(interval=0.1),
                    'per_cpu': psutil.cpu_percent(interval=0.1, percpu=True),
                    'count': psutil.cpu_count(),
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'used': psutil.virtual_memory().used,
                    'percent': psutil.virtual_memory().percent,
                },
                'disk': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent,
                },
                'swap': {
                    'total': psutil.swap_memory().total,
                    'used': psutil.swap_memory().used,
                    'percent': psutil.swap_memory().percent,
                },
                'process': {
                    'pid': os.getpid(),
                    'cpu_percent': psutil.Process(os.getpid()).cpu_percent(),
                    'memory_percent': psutil.Process(os.getpid()).memory_percent(),
                    'threads': len(psutil.Process(os.getpid()).threads()),
                }
            }
            
            # Call callback if provided
            if callback:
                callback(metrics)
                
            # Log resource usage
            logger.debug(f"CPU: {metrics['cpu']['percent']}%, Memory: {metrics['memory']['percent']}%")
            
            # Sleep for the specified interval
            time.sleep(interval)
    
    # Create and start monitoring thread
    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()
    
    logger.info(f"System resource monitoring started: interval={interval}s")
    return monitor_thread

def log_system_call(func):
    """
    Decorator to log system calls
    
    Args:
        func: Function to decorate
    
    Returns:
        Callable: Decorated function
    """
    def wrapper(*args, **kwargs):
        logger.info(f"System call: {func.__name__}, Args: {args}, Kwargs: {kwargs}")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"System call completed: {func.__name__}, Time: {elapsed:.4f}s")
        return result
    return wrapper

@log_system_call
def get_environment_variables() -> Dict[str, str]:
    """
    Get environment variables
    
    Returns:
        Dict[str, str]: Environment variables
    """
    return dict(os.environ)