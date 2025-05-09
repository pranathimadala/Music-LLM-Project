import os
import re
import openl3
import yt_dlp
import pandas as pd
import soundfile as sf
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from pydub import AudioSegment

def sanitize_filename(name):
    return re.sub(r'[^\w\-_. ]', '', name).replace(" ", "_")

OUTPUT_DIR = "data/wave-files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_and_convert(link, id, output_dir=OUTPUT_DIR):
    safe_id = sanitize_filename(id)
    mp3_filename = f"{safe_id}"
    wav_filename = os.path.join(output_dir, mp3_filename)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': mp3_filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
        audio = AudioSegment.from_file(mp3_filename)
        audio.export(wav_filename, format='wav')
        os.remove(mp3_filename)
        print(f"Saved WAV file: {wav_filename}")
        return wav_filename
    except Exception as e:
        print(f"Error processing {link}: {e}")
        return None

def get_embedding(audio_path):
    try:
        audio, sr = sf.read(audio_path)
        emb, ts = openl3.get_audio_embedding(audio, sr, device="cpu")
        print(f"Extracted embedding from {audio_path}")
        return emb
    except Exception as e:
        print(f"Failed to embed {audio_path}: {e}")
        return None
   
def run_embedding_on_folder(wav_folder="data/wav-files", output_folder="outputs"):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(wav_folder):
        if filename.endswith(".wav"):
            wav_path = os.path.join(wav_folder, filename)
            emb = get_embedding(wav_path)
            if emb is not None:
                out_path = os.path.join(output_folder, filename.replace(".wav", ".npy"))
                np.save(out_path, emb)
                print(f"Saved embedding to {out_path}")

# Example usage:
#if __name__ == "__main__":
    # Example: download from YouTube and extract OpenL3 embedding
 #   yt_link = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  #  track_id = "rick_astley"
   # wav_path = download_and_convert(yt_link, track_id)
    #if wav_path:
     #   embedding = get_embedding(wav_path)
      #  print(embedding)

def load_embeddings(folder):
    embeddings = []
    labels = []
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            emb = np.load(path)
            emb_mean = np.mean(emb, axis=0)  # flatten temporal dim
            embeddings.append(emb_mean)
            labels.append(os.path.splitext(file)[0])
    return np.array(embeddings), labels

def plot_interactive_clusters(embeddings, labels, method="tsne", n_clusters=5):
    reducer = TSNE(n_components=2, random_state=42) if method == "tsne" else PCA(n_components=2)
    reduced = reducer.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    data = {
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "label": labels,
        "cluster": cluster_labels
    }

    fig = px.scatter(
        data,
        x="x", y="y",
        color=data["cluster"].astype(str),
        hover_name=data["label"],
        title=f"{method.upper()} Audio Embedding Clusters",
        labels={"color": "Cluster"}
    )
    fig.write_html("cluster_plot.html")
    print("Plot saved to cluster_plot.html")
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
    fig.update_layout(showlegend=True)
    fig.show()


if __name__ == "__main__":
    emb_folder = "outputs"  # adjust if yours is different
    emb_data, emb_labels = load_embeddings(emb_folder)
    plot_interactive_clusters(emb_data, emb_labels, method="tsne", n_clusters=5)  # or method="pca"