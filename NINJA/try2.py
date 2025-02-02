import os
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline
import noisereduce as nr
import numpy as np
import scipy.io.wavfile as wavfile

# Function to reduce noise in the audio file using spectral gating
def reduce_noise(audio_segment):
    # Convert AudioSegment to numpy array
    samples = np.array(audio_segment.get_array_of_samples())
    sample_rate = audio_segment.frame_rate
    
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=samples, sr=sample_rate)
    
    # Convert back to AudioSegment
    return AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels
    )

# Function to transcribe audio to text
def transcribe_audio(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    recognizer = sr.Recognizer()

    # Load and reduce noise in the audio file
    audio = AudioSegment.from_file(file_path)
    audio = reduce_noise(audio)
    
    # Save the processed audio segment to a temporary file
    temp_file = "temp.wav"
    audio.export(temp_file, format="wav")
    
    with sr.AudioFile(temp_file) as source:
        audio_data = recognizer.record(source)
        # Increase the pause threshold to handle noisy environments
        recognizer.pause_threshold = 1.0
        text = recognizer.recognize_google(audio_data, show_all=False)
    
    return text

# Function to summarize text using a language model
def summarize_text(text):
    summarizer = pipeline("summarization", model="t5-base", framework="tf")
    summary = summarizer(text, max_length=150, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Main function to process audio and generate summary
def process_meeting_audio(file_path):
    transcript = transcribe_audio(file_path)  # Pass the file path here
    summary = summarize_text(transcript)
    return summary

# Example usage
if __name__ == "__main__":
    # Replace this with the actual path to your audio file
    audio_file_path = "audio_file_path/cancer.wav"
    
    # Check if the file exists before processing
    if not os.path.isfile(audio_file_path):
        print(f"File not found: {audio_file_path}")
    else:
        summary = process_meeting_audio(audio_file_path)
        print("Meeting Summary:")
        print(summary)