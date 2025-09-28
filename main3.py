import os
import re
import whisper
import json
import subprocess
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urlparse, parse_qs
from rich.console import Console
from rich.table import Table
import numpy as np
from pydub import AudioSegment
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

def sec_to_time(sec):
    """Convert seconds to mm:ss format."""
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

def validate_youtube_url(url):
    """Validate if the input is a YouTube URL."""
    youtube_regex = r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+'
    return re.match(youtube_regex, url) is not None

def sanitize_url(url: str) -> str:
    """Strip whitespace and surrounding quotes/backticks from URL."""
    return url.strip().strip(" '\"`")

def download_audio(youtube_link):
    """Download audio from YouTube with error handling."""
    args = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "-o", "video.mp3",
        youtube_link,
    ]
    try:
        result = subprocess.run(args, capture_output=True, text=True)
    except FileNotFoundError:
        raise FileNotFoundError("yt-dlp not found. Install it with 'pip install yt-dlp'.")
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "Unknown error").strip()
        raise RuntimeError(f"Download failed: {err}")
    if not os.path.exists("video.mp3"):
        raise FileNotFoundError("Audio file not found.")

# User inputs
console = Console()
youtube_link_raw = input("Enter YouTube link: ")
youtube_link = sanitize_url(youtube_link_raw)
if not validate_youtube_url(youtube_link):
    raise ValueError("Invalid YouTube URL.")

# Download audio
try:
    print("Downloading audio from YouTube, please wait...")
    download_audio(youtube_link)
except Exception as e:
    print(f"Error downloading audio: {e}")
    exit(1)

# Check video length
print("Checking video duration...")
audio = AudioSegment.from_mp3("video.mp3")
duration_seconds = len(audio) / 1000
if duration_seconds > 300:
    print("Video exceeds 5 minutes. Please choose a shorter video.")
    os.remove("video.mp3")
    exit(1)

# Transcribe
try:
    print("Loading Whisper model, this may take a moment...")
    model = whisper.load_model("medium")  # Using medium for better accuracy
    print("Transcribing audio, please wait...")
    result = model.transcribe("video.mp3", verbose=False, word_timestamps=True)
    segments = result['segments']
except Exception as e:
    print(f"Error during transcription: {e}")
    exit(1)

# Save full transcript
with open("segments.txt", "w", encoding="utf-8") as f:
    for seg in segments:
        f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")
print("Full transcript saved to 'segments.txt'")

# Suggest keywords using Groq LLM
print("Generating keyword suggestions using Groq LLM, please wait...")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Ensure GROQ_API_KEY is set in environment
transcript = ' '.join([seg['text'].strip() for seg in segments])
prompt = f"From the following video transcript, suggest 5 relevant keywords or topics that a user might want to search for timestamps in the video:\n\n{transcript}\n\nOutput as a numbered list 1-5."
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": prompt}],
)
keywords_text = response.choices[0].message.content
keywords = re.findall(r'^\d+\.\s*(.+)', keywords_text, re.M)
if len(keywords) != 5:
    print("Failed to generate exactly 5 keywords. Please try again.")
    exit(1)

# Display suggested keywords and get user selection
print("\nSuggested keywords:")
for i, kw in enumerate(keywords, 1):
    print(f"{i}. {kw}")
try:
    selection = int(input("Select a keyword by number (1-5): "))
    if selection < 1 or selection > 5:
        raise ValueError
    query = keywords[selection - 1].lower()
except ValueError:
    print("Invalid selection. Please enter a number between 1 and 5.")
    exit(1)

# Encode segments
print("Encoding segments for semantic search, please wait...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
segment_texts = [seg['text'].strip().lower() for seg in segments]
segment_embeddings = embedder.encode(segment_texts, batch_size=32, convert_to_tensor=True)
query_embedding = embedder.encode(query, convert_to_tensor=True)

# Compute cosine similarity
cos_scores = util.cos_sim(query_embedding, segment_embeddings)[0]

# Find the single best segment
threshold = 0.5  # Minimum similarity score
best_result = max(
    [(i, score.item()) for i, score in enumerate(cos_scores)],
    key=lambda x: x[1],
    default=(None, 0)
)

# Extract video ID for YouTube links
video_id = parse_qs(urlparse(youtube_link).query).get('v', [None])[0] or "unknown"

# Display result
table = Table(title=f"Best Match for '{query}'")
table.add_column("Timestamp", justify="center")
table.add_column("Score", justify="right")
table.add_column("Text")
table.add_column("YouTube Link")
if best_result[0] is not None and best_result[1] >= threshold:
    idx, score = best_result
    seg = segments[idx]
    start_time = sec_to_time(seg['start'])
    end_time = sec_to_time(seg['end'])
    youtube_timestamp = f"https://youtu.be/{video_id}?t={int(seg['start'])}"
    table.add_row(
        f"{start_time} - {end_time}",
        f"{score:.2f}",
        seg['text'],
        youtube_timestamp
    )
else:
    console.print("[red]No match found with similarity score >= 0.5.[/red]")

console.print(table)

# Clean up
try:
    os.remove("video.mp3")
    print("Cleaned up temporary audio file.")
except FileNotFoundError:
    print("No audio file to clean up.")