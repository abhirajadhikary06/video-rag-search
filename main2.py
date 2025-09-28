import os
import whisper
from sentence_transformers import SentenceTransformer, util

# --------------------------
# Convert seconds to mm:ss
# --------------------------
def sec_to_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

# --------------------------
# User inputs
# --------------------------
youtube_link = input("Enter YouTube link: ")
query = input("Enter topic/keyword to search: ").lower()

# --------------------------
# Step 1: Download audio from YouTube
# --------------------------
print("\nDownloading audio...")
os.system(f"yt-dlp -x --audio-format mp3 -o 'video.mp3' '{youtube_link}'")

# --------------------------
# Step 2: Load Whisper model
# --------------------------
print("Loading Whisper model...")
model = whisper.load_model("small")

# --------------------------
# Step 3: Transcribe with segments
# --------------------------
print("Transcribing audio (this may take a while)...")
result = model.transcribe("video.mp3", verbose=False)
segments = result['segments']

# --------------------------
# Step 4: Semantic search setup
# --------------------------
print("Encoding segments for semantic search...")
# embedder = SentenceTransformer('all-distilroberta-v1')  # lightweight, free
embedder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
segment_texts = [seg['text'].strip() for seg in segments]
segment_embeddings = embedder.encode(segment_texts, convert_to_tensor=True)

query_embedding = embedder.encode(query, convert_to_tensor=True)

# Compute cosine similarity
cos_scores = util.cos_sim(query_embedding, segment_embeddings)[0]

# --------------------------
# Step 5: Find best matching segments
# --------------------------
# Collect top 3 segments
top_k = 3
top_results = sorted(
    [(i, score.item()) for i, score in enumerate(cos_scores)],
    key=lambda x: x[1],
    reverse=True
)[:top_k]

print(f"\nTop {top_k} timestamps for '{query}':\n")
for idx, score in top_results:
    if score < 0.5:  # ignore very low similarity
        continue
    seg = segments[idx]
    start_time = sec_to_time(seg['start'])
    end_time = sec_to_time(seg['end'])
    print(f"[{start_time} - {end_time}] (score: {score:.2f}) {seg['text']}")

# --------------------------
# Optional: save full transcript with timestamps
# --------------------------
with open("segments.txt", "w", encoding="utf-8") as f:
    for seg in segments:
        f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")

print("\nFull transcript with timestamps saved to 'segments.txt'")
