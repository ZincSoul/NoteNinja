import os
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline


def transcribe_audio(file_path):
   
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(file_path)
    
    
    temp_file = "temp.wav"
    audio.export(temp_file, format="wav")
    
    with sr.AudioFile(temp_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    
    return text


def summarize_text(text):
    summarizer = pipeline("summarization", model="t5-base", framework="tf")
    summary = summarizer(text, max_length=150, min_length=25, do_sample=False)
    return summary[0]['summary_text']


def process_meeting_audio(file_path):
    transcript = transcribe_audio(file_path)  
    summary = summarize_text(transcript)
    return summary


if __name__ == "__main__":
    
    audio_file_path = "audio_file_path/intrn.wav"
    
    
    if not os.path.isfile(audio_file_path):
        print(f"File not found: {audio_file_path}")
    else:
        summary = process_meeting_audio(audio_file_path)
        print("Meeting Summary:")
        print(summary)