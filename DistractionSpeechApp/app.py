import streamlit as st
from distraction_module.detection import start_video_detection
from speech_module.transcriber import start_transcription
from os_module import hardware_info, permissions_check, sys_calls

st.title("Real-Time Distraction & Speech Analysis")
col1, col2 = st.columns(2)

with col1:
    st.header("Distraction Detection")
    start_video_detection()

with col2:
    st.header("Speech Transcription + Summary")
    start_transcription()

st.sidebar.title("OS Controls")
hardware_info.display()
permissions_check.check_log_permissions()
sys_calls.show_active_sys_calls()
