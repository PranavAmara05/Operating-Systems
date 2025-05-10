# Distraction Detection & Speech Summarizer App

A Streamlit application that demonstrates key OS concepts while providing practical functionality:

1. **Distraction Detection**: Uses computer vision to detect when a user is distracted (looking away or closing eyes) and provides audio alerts.
2. **Speech Transcription & Summarization**: Captures and transcribes speech, then summarizes important points.
3. **OS Concepts Demonstration**: Implements and displays system calls, process priority management, threading, semaphores, and resource management.

## Features

- Real-time face and eye tracking with MediaPipe
- Audio alerts when distraction is detected
- Speech-to-text transcription with Whisper model
- Automatic summarization of transcribed text
- Process priority control
- System resource monitoring
- Thread synchronization with semaphores

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DistractionSpeechApp.git
cd DistractionSpeechApp

# Install dependencies
pip install -r requirements.txt

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

```bash
streamlit run app.py
```

## System Requirements

- Python 3.8+
- Webcam
- Microphone
- Internet connection (for model downloads)

## Project Structure

```
DistractionSpeechApp/
├── app.py                       # Main Streamlit app
├── requirements.txt             # All dependencies
├── README.md                    # Project overview
├── distraction_module/          # Distraction detection functionality
├── speech_module/               # Speech processing functionality
├── os_module/                   # OS concepts demonstration
├── static/                      # Static files
└── logs/                        # Application logs
```


## Contributors

AMARA PRANAV- bl.en.u4aid23003@bl.students.amrita.edu
JOSHIKA SOMiSETTY- bl.en.u4aid23019@bl.students.amrita.edu
KODURI LAKSHMI VINUGNA -bl.en.u4aid23026@bl.students.amrita.edu
