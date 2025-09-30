# video-rag-search
RAG searching mechanism from YouTube video supported by MariaDB

## 📌 Project Overview
This project is a **Flask-based web application** that:
- Downloads audio from YouTube videos 🎬
- Transcribes audio using **OpenAI Whisper** 🗣️
- Extracts key topics using **Groq LLM**
- Generates semantic embeddings with **SentenceTransformers**
- Stores transcripts + embeddings in **MariaDB**
- Lets users **search by keywords** with semantic similarity scoring 🔍

---

## 🚀 Features
- 🎥 Download YouTube audio via `yt-dlp`
- 🗣️ Transcribe speech with Whisper
- 🤖 Extract key themes with Groq API
- 🔑 Choose from 5 auto-generated keywords
- 🔎 Search stored transcripts by semantic meaning
- 💾 Cache results with Flask-Caching
- 📊 Interactive UI with clean design and responsive layout

---

## 🛠️ Tech Stack
- **Backend**: Flask (Python)
- **Database**: MariaDB
- **ML Models**:
  - Whisper (speech-to-text)
  - SentenceTransformers (embeddings)
- **Frontend**: HTML + CSS (responsive)
- **Other Tools**:
  - yt-dlp (YouTube download)
  - Groq API (keyword extraction)
  - Pydub (audio processing)

---

## 📂 Project Structure
```
.
├── app.py          # Flask backend
├── templates/
│   └── index.html  # Frontend HTML template
├── static/         # (Optional) CSS/JS if separated
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone repo
```bash
git clone https://github.com/yourusername/youtube-audio-search.git
cd youtube-audio-search
```

### 2️⃣ Create & activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set up environment variables
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=youtube_transcripts
SECRET_KEY=your_secret_key
```

### 5️⃣ Run the app
```bash
flask run
```

---

## 🖥️ Usage
1. Paste a **YouTube URL**
2. Wait for transcription + keyword extraction
3. Pick a keyword → get related transcript snippets
4. Explore results with semantic similarity ranking

---

## 🛡️ Security Notes
- Inputs are sanitized (`sanitize_url`)
- Database uses parameterized queries ✅
- `.env` keeps secrets safe

---

## 🚧 Future Improvements
- ⚡ Use FAISS/ChromaDB for fast vector search
- 📡 Async tasks with Celery for large audio files
- 🔔 Live progress updates via WebSockets/SSE
- 🎨 Improved frontend with animations & charts

---

## 📜 License
MIT License © 2025
