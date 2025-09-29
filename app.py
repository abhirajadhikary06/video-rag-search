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
from flask_caching import Cache
import mariadb
import io
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure session key

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

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

def parse_keywords(text: str) -> list:
    """Extract up to 5 keywords from LLM output.
    Priority: bold markers **...**, then numbered list, then bullet list.
    Cleans common markdown artifacts and punctuation.
    """
    if not text:
        return []
    # First try bold markers
    bold = re.findall(r"\*\*(.+?)\*\*", text)
    candidates = bold if bold else re.findall(r"^\s*\d+\.\s*(.+)", text, re.M)
    if not candidates:
        candidates = re.findall(r"^\s*[-*]\s*(.+)", text, re.M)

    cleaned = []
    for item in candidates:
        s = (item or "").strip().strip(" '\"`")
        # If markdown link, keep the anchor text
        m = re.match(r"\[(.+?)\]\(.+?\)", s)
        if m:
            s = m.group(1).strip()
        # Remove trailing punctuation/spaces
        s = re.sub(r"[\s\.,;:!]+$", "", s)
        if s:
            cleaned.append(s)
        if len(cleaned) == 5:
            break
    return cleaned[:5]

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

def get_db_connection():
    try:
        return mariadb.connect(
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "RootPass123!"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306)),
            database=os.getenv("DB_NAME", "youtube_search")
        )
    except mariadb.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

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
                video_id = parse_qs(urlparse(youtube_link).query).get('v', [None])[0] or "unknown"

                # Check cache
                cached_data = cache.get(video_id)
                if cached_data:
                    session['segments'] = cached_data['segments']
                    session['keywords'] = cached_data['keywords']
                    session['youtube_link'] = youtube_link
                    session['step'] = 'select_keyword'
                    message = "Fast loading from cache."
                    return render_template('index.html', step=session['step'], message=message, keywords=session['keywords'])

                conn = get_db_connection()
                cursor = conn.cursor()

                # Create table if not exist
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    video_id VARCHAR(50),
                    segment_id INT,
                    text VARCHAR(500),
                    embedding_json LONGTEXT,
                    start_time FLOAT,
                    end_time FLOAT,
                    youtube_link VARCHAR(255),
                    timestamp_link VARCHAR(255),
                    description TEXT
                ) ENGINE=InnoDB;
                """)
                # Defensive migration for existing tables
                try:
                    cursor.execute("ALTER TABLE video_data ADD COLUMN embedding_json LONGTEXT")
                except mariadb.Error:
                    pass
                try:
                    cursor.execute("ALTER TABLE video_data DROP COLUMN embedding")
                except mariadb.Error:
                    pass

                # Check if data exists in DB
                cursor.execute("SELECT COUNT(*) FROM video_data WHERE video_id = ?", (video_id,))
                if cursor.fetchone()[0] > 0:
                    # Load from DB
                    cursor.execute("""
                    SELECT text, start_time, end_time
                    FROM video_data
                    WHERE video_id = ?
                    ORDER BY segment_id
                    """, (video_id,))
                    segments = [{'text': row[0], 'start': row[1], 'end': row[2]} for row in cursor.fetchall()]

                    # Generate keywords (not stored)
                    logger.info("Generating keyword suggestions using Groq LLM, please wait...")
                    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                    transcript_text = ' '.join([seg['text'].strip() for seg in segments])
                    prompt = f"From the following video transcript, suggest 5 relevant keywords or topics that a user might want to search for timestamps in the video:\n\n{transcript_text}\n\nOutput as a numbered list 1-5."
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    keywords_text = response.choices[0].message.content
                    keywords = parse_keywords(keywords_text)
                    if len(keywords) != 5:
                        message = "Failed to generate exactly 5 keywords. Please try again."
                        conn.close()
                        return render_template('index.html', step=step, message=message)

                    # Cache
                    cache.set(video_id, {'segments': segments, 'keywords': keywords}, timeout=3600)

                    session['segments'] = segments
                    session['keywords'] = keywords
                    session['youtube_link'] = youtube_link
                    session['step'] = 'select_keyword'
                    message = "Loaded from database."
                    conn.close()
                    return render_template('index.html', step=session['step'], message=message, keywords=session['keywords'])

                # If not in DB, process
                logger.info("Downloading audio from YouTube, please wait...")
                message = "Downloading audio..."
                download_audio(youtube_link)
                logger.info("Checking video duration...")
                message = "Checking video duration..."
                audio = AudioSegment.from_mp3("video.mp3")
                duration_seconds = len(audio) / 1000
                if duration_seconds > 300:
                    os.remove("video.mp3")
                    message = "Video exceeds 5 minutes. Please choose a shorter video."
                    conn.close()
                    return render_template('index.html', step=step, message=message)

                logger.info("Loading Whisper model, this may take a moment...")
                message = "Loading Whisper model..."
                model = whisper.load_model("medium")
                logger.info("Transcribing audio, please wait...")
                message = "Transcribing audio..."
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
                    conn.close()
                    return render_template('index.html', step=step, message=message)

                # Save transcript to disk (not displayed to user)
                with open("segments.txt", "w", encoding="utf-8") as f:
                    for seg in valid_segments:
                        f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")

                # Encode segments
                logger.info("Encoding segments for semantic search, please wait...")
                message = "Encoding segments..."
                embedder = SentenceTransformer('all-MiniLM-L6-v2')
                segment_texts = [seg['text'].strip().lower() for seg in valid_segments]
                segment_embeddings = embedder.encode(segment_texts, batch_size=32, convert_to_tensor=True)

                # Store in DB
                for i, seg in enumerate(valid_segments):
                    emb_json = json.dumps(segment_embeddings[i].cpu().numpy().tolist())
                    timestamp_link = f"https://youtu.be/{video_id}?t={int(seg['start'])}"
                    description = seg['text'][:255]  # Truncate for brevity
                    cursor.execute(
                        """
                        INSERT INTO video_data (
                            video_id, segment_id, text, embedding_json, start_time, end_time, 
                            youtube_link, timestamp_link, description
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (video_id, i, seg['text'], emb_json, seg['start'], seg['end'], 
                         youtube_link, timestamp_link, description)
                    )
                conn.commit()

                # Generate keywords
                logger.info("Generating keyword suggestions using Groq LLM, please wait...")
                message = "Generating keyword suggestions..."
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                transcript_text = ' '.join([seg['text'].strip() for seg in valid_segments])
                if not transcript_text.strip():
                    os.remove("video.mp3")
                    message = "Transcript is empty after filtering. Try another video."
                    conn.close()
                    return render_template('index.html', step=step, message=message)

                prompt = f"From the following video transcript, suggest 5 relevant keywords or topics that a user might want to search for timestamps in the video:\n\n{transcript_text}\n\nOutput as a numbered list 1-5."
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt}],
                )
                keywords_text = response.choices[0].message.content
                keywords = parse_keywords(keywords_text)
                if len(keywords) != 5:
                    os.remove("video.mp3")
                    message = "Failed to generate exactly 5 keywords. Please try again."
                    conn.close()
                    return render_template('index.html', step=step, message=message)

                # Cache
                cache.set(video_id, {
                    'segments': [{'start': seg['start'], 'end': seg['end'], 'text': seg['text']} for seg in valid_segments],
                    'keywords': keywords
                }, timeout=3600)

                session['segments'] = [{'start': seg['start'], 'end': seg['end'], 'text': seg['text']} for seg in valid_segments]
                session['keywords'] = keywords
                session['youtube_link'] = youtube_link
                session['step'] = 'select_keyword'
                os.remove("video.mp3")
                message = "Keywords generated successfully."
                conn.close()

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
    result_data = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        if index < 1 or index > 5:
            message = "Invalid keyword selection."
            return render_template('index.html', step='select_keyword', message=message, keywords=session.get('keywords', []))

        keywords = session.get('keywords', [])
        query = keywords[index - 1].lower()

        logger.info("Encoding query for semantic search, please wait...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        q = query_embedding.cpu().numpy()

        youtube_link = session.get('youtube_link', '')
        video_id = parse_qs(urlparse(youtube_link).query).get('v', [None])[0] or "unknown"

        # Fetch embeddings and compute cosine similarity in Python
        cursor.execute("""
        SELECT segment_id, embedding_json, start_time, end_time, timestamp_link, text
        FROM video_data
        WHERE video_id = ?
        ORDER BY segment_id
        """, (video_id,))
        rows = cursor.fetchall()
        best = None
        best_score = -1.0
        for row in rows:
            segment_id, emb_json, start_time, end_time, timestamp_link, text = row
            try:
                seg_vec = np.array(json.loads(emb_json), dtype=np.float32)
                denom = (np.linalg.norm(q) * np.linalg.norm(seg_vec))
                if denom == 0:
                    continue
                sim = float(np.dot(q, seg_vec) / denom)
            except Exception:
                continue
            if sim > best_score:
                best_score = sim
                best = (segment_id, start_time, end_time, timestamp_link, text)

        if best is not None and best_score >= 0.5:
            segment_id, start_time, end_time, timestamp_link, text = best
            result_data = {
                'timestamp': f"{sec_to_time(start_time)} - {sec_to_time(end_time)}",
                'title': text,
                'link': timestamp_link,
                'score': f"{best_score:.3f}"
            }
            message = f"Match found successfully. Similarity: {best_score:.3f}"
        else:
            message = "No match found with similarity score >= 0.5."
        session['step'] = 'result'

    except Exception as e:
        logger.error(f"Error in select_keyword: {str(e)}")
        message = f"Error: {str(e)}"
        return render_template('index.html', step='select_keyword', message=message, keywords=session.get('keywords', []))

    finally:
        conn.close()

    return render_template('index.html', step='result', message=message, result=result_data, keywords=session.get('keywords', []))

@app.route('/start_over', methods=['GET'])
def start_over():
    # Reset session and go back to link input
    session.pop('segments', None)
    session.pop('keywords', None)
    session.pop('youtube_link', None)
    session['step'] = 'input_link'
    return redirect(url_for('index'))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)