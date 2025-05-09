import os
import pandas as pd
import yt_dlp
from pydub import AudioSegment
import re

CSV_PATH = "data/music-data.csv"
MP3_DIR = "data/mp3-files"
WAV_DIR = "data/wav-files"
LINK_COLUMN = "Link"

os.makedirs(MP3_DIR, exist_ok=True)
os.makedirs(WAV_DIR, exist_ok=True)

def download_with_ytdlp(url, output_path):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'quiet': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"yt-dlp failed: {e}")

def convert_mp3_to_wav(mp3_folder, wav_output_folder):
    for filename in os.listdir(mp3_folder):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(mp3_folder, filename)
            wav_path = os.path.join(wav_output_folder, filename.replace(".mp3", ".wav"))
            try:
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(wav_path, format="wav")
                print(f"Converted: {wav_path}")
            except Exception as e:
                print(f"Failed to convert {mp3_path}: {e}")

def main():
    df = pd.read_csv(CSV_PATH)

    if LINK_COLUMN not in df.columns:
        print(f"ERROR: Column '{LINK_COLUMN}' not found in CSV.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    for url in df[LINK_COLUMN]:
        print(f"Processing: {url}")
        download_with_ytdlp(url, MP3_DIR)

    print("\n--- Converting MP3s to WAVs ---\n")
    convert_mp3_to_wav(MP3_DIR, WAV_DIR)

if __name__ == "__main__":
    main()