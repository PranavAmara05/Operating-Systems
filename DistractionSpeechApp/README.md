# Distraction Detection & Speech Summarizer App

A Streamlit application that demonstrates key OS concepts while providing practical functionality:

1. **Distraction Detection**: Uses computer vision to detect when a user is distracted (looking away or closing eyes) and provides audio alerts.  
2. **Speech Transcription & Summarization**: Captures and transcribes speech, then summarizes important points.  

---

## Features

- Real-time face and eye tracking with MediaPipe  
- Audio alerts when distraction is detected  
- Speech-to-text transcription with Facebook BART model  
- Automatic summarization of transcribed text  

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DistractionSpeechApp.git
cd DistractionSpeechApp

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Run the Streamlit app
streamlit run os_with_ui_streamlt.py
```

Once the app launches, it will open a local web interface in your browser.  
You must grant access to your **webcam** and **microphone** .

---

## System Requirements

- Python 3.8+  
- Webcam  
- Microphone  
- Internet connection (for model downloads)

---

## Contributors

- **Amara Pranav** – bl.en.u4aid23003@bl.students.amrita.edu  
- **Joshika Somisetty** – bl.en.u4aid23019@bl.students.amrita.edu  
- **Koduri Lakshmi Vinugna** – bl.en.u4aid23026@bl.students.amrita.edu  

---


