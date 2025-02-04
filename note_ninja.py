import pyaudio
import numpy as np
import whisper
import keyboard
import openai

"""API KEY"""
openai.api_key = '' 

model = whisper.load_model("base") 

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 
CHUNK = 1024

p = pyaudio.PyAudio()

def record_audio():
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print("Recording... Press 's' to stop.")

    while True:
        if keyboard.is_pressed('s'):
            print("Stopping recording...")
            break
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return audio_data

def transcribe_audio(audio):
    result = model.transcribe(audio, fp16=False, language="en")
    print(f"Transcript: {result['text']}")
    return result['text']

def summarize_text(text):
    """Summarizes text using OpenAI GPT-4."""
    print("\nGenerating summary...")

    response = openai.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": text,
        }
    ],
    max_tokens="150",
    model="gpt-3.5-turbo",
    )

    summary = response["choices"][0]["message"]["content"]
    print("\n--- Summary ---\n", summary)
    return summary

def process_real_time_audio():
    """Continuously records, transcribes, and appends text until 's' is pressed."""
    transcript = ""

    audio = record_audio()
    text = transcribe_audio(audio)
    if text:
        transcript += text + " "

    if transcript:
        print("\n--- Full Transcript ---\n", transcript)
        summarize_text(transcript)
    else:
        print("No transcript available.")

if __name__ == "__main__":
    process_real_time_audio()
