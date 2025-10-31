import requests
import os

# Make sure you set your API key as an environment variable first:
#   export ELEVENLABS_API_KEY="your_api_key_here"
API_KEY = 'sk_f57589f689598deddbf685720c84f4415e2b083077337c03'

if not API_KEY:
    raise ValueError("Please set the ELEVENLABS_API_KEY environment variable")

# Endpoint for transcription
url = "https://api.elevenlabs.io/v1/speech-to-text"

# Path to the audio file you want to transcribe
audio_file = "conversation_A_0620.wav"  # must be wav, mp3, m4a, or similar

with open(audio_file, "rb") as f:
    headers = {
        "xi-api-key": API_KEY
    }

    files = {
        "file": open(audio_file, "rb")
    }

    data = {
        "model_id": "scribe_v1"  # specify the speech-to-text model
    }

    response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        data = response.json()
        print("Transcription result:")
        print(data["text"])
    else:
        print("Error:", response.status_code, response.text)