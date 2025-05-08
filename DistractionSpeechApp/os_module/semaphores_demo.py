import threading
import logging
import time

logger = logging.getLogger(__name__)

def create_semaphore(value=1):
    """
    Creates and returns a semaphore with the specified initial value.
    
    Args:
        value (int): Initial semaphore value (default: 1)
        
    Returns:
        threading.Semaphore: A semaphore object
    """
    logger.info(f"Creating semaphore with initial value {value}")
    return threading.Semaphore(value)

def acquire_resource(semaphore, resource_name, timeout=None):
    """
    Demonstrates semaphore acquisition with optional timeout.
    
    Args:
        semaphore (threading.Semaphore): The semaphore controlling access
        resource_name (str): Name of the resource being accessed (for logging)
        timeout (float, optional): Maximum time to wait for acquisition
        
    Returns:
        bool: True if resource was acquired, False otherwise
    """
    logger.info(f"Attempting to acquire {resource_name}")
    
    if timeout is not None:
        # Try to acquire with timeout
        success = semaphore.acquire(blocking=True, timeout=timeout)
        if success:
            logger.info(f"Successfully acquired {resource_name}")
        else:
            logger.warning(f"Failed to acquire {resource_name} within {timeout} seconds")
        return success
    else:
        # Blocking acquisition (will wait indefinitely)
        semaphore.acquire()
        logger.info(f"Successfully acquired {resource_name}")
        return True

def release_resource(semaphore, resource_name):
    """
    Releases a semaphore-controlled resource.
    
    Args:
        semaphore (threading.Semaphore): The semaphore controlling access
        resource_name (str): Name of the resource being released (for logging)
    """
    try:
        semaphore.release()
        logger.info(f"Released {resource_name}")
    except ValueError:
        logger.error(f"Attempted to release {resource_name} but semaphore was already at max value")

def demonstrate_resource_sharing(semaphore, num_threads=3, resource_name="shared resource"):
    """
    Demonstrates multiple threads competing for a semaphore-protected resource.
    
    Args:
        semaphore (threading.Semaphore): Semaphore controlling access to the resource
        num_threads (int): Number of threads to create
        resource_name (str): Name of the resource for logging purposes
    """
    def worker(worker_id):
        logger.info(f"Worker {worker_id} starting")
        
        # Try to acquire the resource
        if acquire_resource(semaphore, resource_name, timeout=5):
            logger.info(f"Worker {worker_id} is using {resource_name}")
            # Simulate work with the resource
            time.sleep(1)
            # Release the resource
            release_resource(semaphore, resource_name)
        else:
            logger.warning(f"Worker {worker_id} couldn't access {resource_name}, moving on")
    
    # Create and start the worker threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    logger.info("Resource sharing demonstration completed")