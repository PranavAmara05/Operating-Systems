import streamlit as st
import threading
import time
import os
import logging
from dotenv import load_dotenv

# Import modules
from distraction_module.detection import DistractionDetector
from speech_module.transcriber import SpeechTranscriber
from speech_module.summarizer import TextSummarizer
from os_module.sys_calls import display_system_info
from os_module.priority_control import set_process_priority
from os_module.semaphores_demo import create_semaphore
from os_module.hardware_info import get_hardware_info
from os_module.permissions_check import check_permissions

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables if any
load_dotenv()

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = []
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'distraction_count' not in st.session_state:
        st.session_state.distraction_count = 0
    if 'distraction_types' not in st.session_state:
        st.session_state.distraction_types = []
    if 'app_started' not in st.session_state:
        st.session_state.app_started = False

def start_services():
    """Initialize and start all monitoring services"""
    logger.info("Starting application services")
    
    # Create a semaphore to manage resource access
    semaphore = create_semaphore(value=1)
    
    # Initialize modules
    detector = DistractionDetector(semaphore)
    transcriber = SpeechTranscriber()
    summarizer = TextSummarizer()
    
    # Set high priority for the process
    set_process_priority('high')
    
    # Start distraction monitoring in a thread
    distraction_thread = threading.Thread(
        target=detector.monitor, 
        args=(lambda count, type_str: update_distraction_stats(count, type_str),),
        daemon=True
    )
    distraction_thread.start()
    
    # Start speech transcription in another thread
    transcription_thread = threading.Thread(
        target=transcriber.start_transcription,
        args=(lambda text: update_transcript(text, summarizer),),
        daemon=True
    )
    transcription_thread.start()
    
    st.session_state.app_started = True
    logger.info("All services started successfully")
    
    return distraction_thread, transcription_thread

def update_distraction_stats(count, distraction_type):
    """Callback to update distraction statistics"""
    st.session_state.distraction_count = count
    if distraction_type:
        st.session_state.distraction_types.append({
            'type': distraction_type,
            'time': time.strftime('%H:%M:%S')
        })

def update_transcript(text, summarizer):
    """Callback to update transcript and summary"""
    if text and text.strip():
        st.session_state.transcripts.append(text)
        
        # Keep only last 10 transcripts
        if len(st.session_state.transcripts) > 10:
            st.session_state.transcripts = st.session_state.transcripts[-10:]
        
        # Create a summary after collecting enough text
        full_text = " ".join(st.session_state.transcripts)
        if full_text.count('.') >= 3:
            st.session_state.summary = summarizer.summarize(full_text)

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Distraction & Speech Monitor",
        page_icon="👁️",
        layout="wide"
    )
    
    st.title("Distraction Detection & Speech Summarization System")
    st.subheader("Demonstrating OS Concepts in Action")
    
    initialize_session_state()
    
    # System information section
    with st.expander("System Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            display_system_info()
        with col2:
            hw_info = get_hardware_info()
            st.write("### Hardware Information")
            for key, value in hw_info.items():
                st.write(f"**{key}:** {value}")
        
        # Check permissions
        perm_status = check_permissions(['camera', 'microphone'])
        for device, status in perm_status.items():
            if status:
                st.success(f"✅ {device.capitalize()} permission granted")
            else:
                st.error(f"❌ {device.capitalize()} permission denied")
    
    # Main interface
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Live Monitoring")
        
        if not st.session_state.app_started:
            if st.button("Start Monitoring"):
                with st.spinner("Starting services..."):
                    start_services()
        else:
            st.success("Monitoring active")
            
            # Display camera feed placeholder
            st.write("### Camera Feed")
            camera_feed = st.empty()
            camera_feed.info("Camera feed would display here in a complete implementation")
            
            # Display live transcripts
            st.write("### Live Transcripts")
            transcript_container = st.container()
            with transcript_container:
                transcript_display = st.empty()
    
    with col2:
        st.subheader("Analytics")
        
        # Distraction metrics
        st.write("### Distraction Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Distraction Count", st.session_state.distraction_count)
        with metrics_col2:
            # Calculate focus percentage (placeholder logic)
            focus_pct = 100 - min(st.session_state.distraction_count * 5, 95) if st.session_state.app_started else 0
            st.metric("Focus Level", f"{focus_pct}%")
        
        # Show distraction history
        if st.session_state.distraction_types:
            st.write("### Recent Distractions")
            for dist in st.session_state.distraction_types[-5:]:
                st.write(f"**{dist['time']}**: {dist['type']}")
        
        # Summary of speech
        st.write("### Speech Summary")
        summary_container = st.container()
        with summary_container:
            summary_display = st.empty()
            if st.session_state.summary:
                summary_display.info(st.session_state.summary)
            else:
                summary_display.info("Waiting for enough speech to generate summary...")
    
    # Update displays every second
    while st.session_state.app_started:
        if st.session_state.transcripts:
            transcript_text = "\n\n".join(st.session_state.transcripts[-5:])
            transcript_display.markdown(transcript_text)
        
        if st.session_state.summary:
            summary_display.info(st.session_state.summary)
            
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")