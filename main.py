import os
import whisper

# Step 1: Download audio from YouTube
os.system("yt-dlp -x --audio-format mp3 -o 'video.mp3' https://youtu.be/J6WwG5HX588?si=69yiDMJEH7S3k7KG")

# Step 2: Load Whisper model
model = whisper.load_model("small")  # you can use tiny, small, medium, large

# Step 3: Transcribe
result = model.transcribe("video.mp3")

# Step 4: Save transcript
with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Transcript saved to transcript.txt")
