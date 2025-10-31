import whisper, os

# Load model (tiny, base, small, medium, large available)
audio = whisper.load_audio("conversation_A_0620.wav")
model = whisper.load_model("turbo")

# Path to your audio file
print(os.listdir())
audio_file = "conversation_A_0620.wav"

# Transcribe
result = model.transcribe(audio_file)
print("Transcribed text:")
print(result["text"])