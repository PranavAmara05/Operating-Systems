import os
import stat
import logging
import platform
import cv2
import pyaudio
import streamlit as st

logger = logging.getLogger(__name__)

def check_file_permissions(filepath):
    """
    Checks permissions for a given file or directory.
    
    Args:
        filepath (str): Path to the file or directory
        
    Returns:
        dict: Dictionary containing permission information
    """
    if not os.path.exists(filepath):
        logger.warning(f"Path does not exist: {filepath}")
        return {'exists': False}
    
    try:
        stats = os.stat(filepath)
        mode = stats.st_mode
        
        # Get owner, group, others permissions
        permissions = {
            'exists': True,
            'is_file': os.path.isfile(filepath),
            'is_dir': os.path.isdir(filepath),
            'size': stats.st_size,
            'owner_id': stats.st_uid,
            'group_id': stats.st_gid,
            'permissions_octal': oct(mode & 0o777),
            'readable': os.access(filepath, os.R_OK),
            'writable': os.access(filepath, os.W_OK),
            'executable': os.access(filepath, os.X_OK)
        }
        
        # Add symbolic notation (similar to ls -l)
        symbolic = ''
        symbolic += 'd' if os.path.isdir(filepath) else '-'
        symbolic += 'r' if mode & stat.S_IRUSR else '-'
        symbolic += 'w' if mode & stat.S_IWUSR else '-'
        symbolic += 'x' if mode & stat.S_IXUSR else '-'
        symbolic += 'r' if mode & stat.S_IRGRP else '-'
        symbolic += 'w' if mode & stat.S_IWGRP else '-'
        symbolic += 'x' if mode & stat.S_IXGRP else '-'
        symbolic += 'r' if mode & stat.S_IROTH else '-'
        symbolic += 'w' if mode & stat.S_IWOTH else '-'
        symbolic += 'x' if mode & stat.S_IXOTH else '-'
        
        permissions['symbolic'] = symbolic
        
        return permissions
    
    except Exception as e:
        logger.error(f"Error checking permissions for {filepath}: {str(e)}")
        return {'exists': True, 'error': str(e)}

def check_dir_permissions(directory_path, check_writability=True):
    """
    Checks if a directory exists and is writable. Creates it if it doesn't exist.
    
    Args:
        directory_path (str): Path to the directory
        check_writability (bool): Whether to check if directory is writable by creating a test file
        
    Returns:
        dict: Permission status information
    """
    result = {
        'exists': False,
        'is_dir': False,
        'created': False,
        'writable': False
    }
    
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)
            result['created'] = True
            result['exists'] = True
            logger.info(f"Created directory: {directory_path}")
        except Exception as e:
            logger.error(f"Error creating directory {directory_path}: {str(e)}")
            result['error'] = str(e)
            return result
    else:
        result['exists'] = True
    
    # Check if it's actually a directory
    result['is_dir'] = os.path.isdir(directory_path)
    if not result['is_dir']:
        logger.error(f"Path exists but is not a directory: {directory_path}")
        return result
    
    # Check if it's writable by trying to create a temp file
    if check_writability:
        import tempfile
        try:
            temp_file = tempfile.NamedTemporaryFile(dir=directory_path, delete=True)
            temp_file.close()
            result['writable'] = True
        except Exception as e:
            logger.warning(f"Directory {directory_path} exists but is not writable: {str(e)}")
            result['writable'] = False
    
    return result

def check_device_permissions():
    """
    Checks permissions for camera and microphone devices.
    
    Returns:
        dict: Device permission status
    """
    result = {
        'camera': False,
        'microphone': False
    }
    
    # Check camera access
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            result['camera'] = True
            cap.release()
    except Exception as e:
        logger.warning(f"Camera permission check failed: {str(e)}")
    
    # Check microphone access
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)
        if stream:
            result['microphone'] = True
            stream.stop_stream()
            stream.close()
        p.terminate()
    except Exception as e:
        logger.warning(f"Microphone permission check failed: {str(e)}")
    
    return result

def check_permissions(devices=None, directories=None, files=None):
    """
    Main function to check various permissions needed by the application.
    
    Args:
        devices (list): List of device names to check ('camera', 'microphone')
        directories (list): List of directory paths to check
        files (list): List of file paths to check
        
    Returns:
        dict: Comprehensive permission check results
    """
    results = {}
    
    # Check device permissions if specified
    if devices:
        device_permissions = check_device_permissions()
        for device in devices:
            if device in device_permissions:
                results[device] = device_permissions[device]
    
    # Check directory permissions if specified
    if directories:
        for directory in directories:
            results[f"dir:{directory}"] = check_dir_permissions(directory)
    
    # Check file permissions if specified
    if files:
        for filepath in files:
            results[f"file:{filepath}"] = check_file_permissions(filepath)
    
    # Check log directory permission
    log_dir = os.path.join(os.getcwd(), 'logs')
    results['logs_dir'] = check_dir_permissions(log_dir)
    
    # Check for permissions to create files in current directory
    results['current_dir'] = check_dir_permissions(os.getcwd())
    
    return results