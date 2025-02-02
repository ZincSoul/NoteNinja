from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline

# Function to transcribe audio to text
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(file_path)
    
    # Save the audio segment to a temporary file
    temp_file = "temp.wav"
    audio.export(temp_file, format="wav")
    
    with sr.AudioFile(temp_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    
    return text

# Function to summarize text using a language model
def summarize_text(text):
    summarizer = pipeline("summarization", model="t5-base", framework="tf")
    summary = summarizer(text, max_length=150, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Main function to process audio and generate summary
def process_meeting_audio(file_path):
    transcript = transcribe_audio(file_path)
    summary = summarize_text(transcript)
    return summary

# Example usage
if __name__ == "__main__":
    audio_file_path = "Dh.wav.mp3"
    summary = process_meeting_audio(audio_file_path)
    print("Meeting Summary:")
    print(summary)