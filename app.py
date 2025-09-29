import os
import re
import whisper
import json
import subprocess
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urlparse, parse_qs
from pydub import AudioSegment
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect, url_for
from rich.console import Console
from rich.table import Table
import io
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure session key

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
    return url.strip().strip(" '\"`") if url else ""

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

@app.route('/', methods=['GET', 'POST'])
def index():
    step = session.get('step', 'input_link')
    message = ""
    keywords = []

    if request.method == 'POST':
        if step == 'input_link':
            youtube_link_raw = request.form.get('youtube_link')
            youtube_link = sanitize_url(youtube_link_raw)
            if not validate_youtube_url(youtube_link):
                message = "Invalid YouTube URL."
                return render_template('index.html', step=step, message=message)

            try:
                logger.info("Downloading audio from YouTube, please wait...")
                download_audio(youtube_link)
                logger.info("Checking video duration...")
                audio = AudioSegment.from_mp3("video.mp3")
                duration_seconds = len(audio) / 1000
                if duration_seconds > 300:
                    os.remove("video.mp3")
                    message = "Video exceeds 5 minutes. Please choose a shorter video."
                    return render_template('index.html', step=step, message=message)

                logger.info("Loading Whisper model, this may take a moment...")
                model = whisper.load_model("medium")
                logger.info("Transcribing audio, please wait...")
                result = model.transcribe("video.mp3", verbose=False, word_timestamps=True)
                segments = result['segments']

                # Filter out segments with None or empty text
                valid_segments = [
                    seg for seg in segments
                    if seg.get('text') and isinstance(seg['text'], str) and seg['text'].strip()
                ]
                if not valid_segments:
                    os.remove("video.mp3")
                    message = "No valid transcript segments found. Try another video."
                    return render_template('index.html', step=step, message=message)

                # Save transcript to disk (not displayed to user)
                with open("segments.txt", "w", encoding="utf-8") as f:
                    for seg in valid_segments:
                        f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")

                # Generate keywords
                logger.info("Generating keyword suggestions using Groq LLM, please wait...")
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                transcript_text = ' '.join([seg['text'].strip() for seg in valid_segments])
                if not transcript_text.strip():
                    os.remove("video.mp3")
                    message = "Transcript is empty after filtering. Try another video."
                    return render_template('index.html', step=step, message=message)

                prompt = f"From the following video transcript, suggest 5 relevant keywords or topics that a user might want to search for timestamps in the video:\n\n{transcript_text}\n\nOutput as a numbered list 1-5."
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt}],
                )
                keywords_text = response.choices[0].message.content
                keywords = re.findall(r'^\d+\.\s*(.+)', keywords_text, re.M)
                if len(keywords) != 5:
                    os.remove("video.mp3")
                    message = "Failed to generate exactly 5 keywords. Please try again."
                    return render_template('index.html', step=step, message=message)

                session['segments'] = [
                    {'start': seg['start'], 'end': seg['end'], 'text': seg['text']}
                    for seg in valid_segments
                ]
                session['keywords'] = keywords
                session['youtube_link'] = youtube_link
                session['step'] = 'select_keyword'
                os.remove("video.mp3")
                message = "Keywords generated successfully."
            except Exception as e:
                logger.error(f"Error in input_link step: {str(e)}")
                if os.path.exists("video.mp3"):
                    os.remove("video.mp3")
                message = f"Error: {str(e)}"
                return render_template('index.html', step=step, message=message)

    return render_template(
        'index.html',
        step=session.get('step', 'input_link'),
        message=message,
        keywords=session.get('keywords', [])
    )

@app.route('/select_keyword/<int:index>', methods=['GET'])
def select_keyword(index):
    message = ""
    table_html = ""

    try:
        if index < 1 or index > 5:
            message = "Invalid keyword selection."
            return render_template('index.html', step='select_keyword', message=message, keywords=session.get('keywords', []))

        keywords = session.get('keywords', [])
        query = keywords[index - 1].lower()

        logger.info("Encoding segments for semantic search, please wait...")
        segments = session.get('segments', [])
        if not segments:
            message = "No segments found in session. Please start over."
            return render_template('index.html', step='input_link', message=message)

        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        segment_texts = [seg['text'].strip().lower() for seg in segments]
        segment_embeddings = embedder.encode(segment_texts, batch_size=32, convert_to_tensor=True)
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, segment_embeddings)[0]

        threshold = 0.5
        best_result = max(
            [(i, score.item()) for i, score in enumerate(cos_scores)],
            key=lambda x: x[1],
            default=(None, 0)
        )

        youtube_link = session.get('youtube_link', '')
        video_id = parse_qs(urlparse(youtube_link).query).get('v', [None])[0] or "unknown"

        console = Console(record=True)
        table = Table(title=f"Best Match for '{query}'")
        table.add_column("Timestamp", justify="center")
        table.add_column("Title")
        table.add_column("YouTube Link")
        if best_result[0] is not None and best_result[1] >= threshold:
            idx, score = best_result
            seg = segments[idx]
            start_time = sec_to_time(seg['start'])
            end_time = sec_to_time(seg['end'])
            youtube_timestamp = f"https://youtu.be/{video_id}?t={int(seg['start'])}"
            table.add_row(
                f"{start_time} - {end_time}",
                seg['text'],
                youtube_timestamp
            )
            message = "Match found successfully."
        else:
            message = "No match found with similarity score >= 0.5."

        # Ensure the table is rendered into the console's record before exporting HTML
        console.print(table)
        table_html = console.export_html(inline_styles=True)
        session['step'] = 'result'
        return render_template('index.html', step='result', message=message, table_html=table_html)

    except Exception as e:
        logger.error(f"Error in select_keyword: {str(e)}")
        message = f"Error: {str(e)}"
        return render_template('index.html', step='select_keyword', message=message, keywords=session.get('keywords', []))

if __name__ == '__main__':
    app.run(debug=True)