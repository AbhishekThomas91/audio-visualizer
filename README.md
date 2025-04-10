
---

# 🎵 Audio Visualizer  
A real-time **audio visualizer** using **Python, Pygame, and Librosa**, allowing users to load and visualize audio files with dynamic frequency-based bar animations.

## 🚀 Features
✅ **Supports MP3, WAV, and OGG files**  
✅ **Dynamic Bar Visualization** based on audio frequency  
✅ **Pause/Play and Restart** functionality  
✅ **Stereo & Mono Audio Handling**  
✅ **Uses FFT for Frequency Analysis**  

## 🛠️ Requirements
Ensure you have the following installed:  
```bash
pip install pygame numpy librosa
```

## 🎬 How to Run
1. Place your **MP3 or WAV file** in the project folder.
2. Open the script and set the `audio_file` variable:
   ```python
   audio_file = 'your_music_file.mp3'  # Change this
   ```
3. Run the script:
   ```bash
   python audio_visualizer.py
   ```

## ⏯️ Controls
- **SPACE** → Play/Pause  
- **R** → Restart Audio  
- **ESC / Close Window** → Exit  

## 🖥️ How It Works
- Uses **Librosa** to load and process the audio file.
- Applies **FFT (Fast Fourier Transform)** for frequency analysis.
- Displays bar heights based on real-time frequency magnitudes.
- Uses **Pygame** for graphical rendering.

## 🧐 Troubleshooting
### 🔹 No Audio?
✔️ Ensure your file **exists** in the correct location  
✔️ Try a **WAV file** if MP3 is unsupported  
✔️ Use `pip install ffmpeg` if `librosa` errors occur  

### 🔹 No Visualization?
✔️ Check `NUM_BARS` value in the script  
✔️ Ensure **Pygame initializes correctly**  
✔️ Add a debug statement to confirm playback:
   ```python
   print("Playback time:", pygame.mixer.music.get_pos(), "ms")
   ```

## 📜 License
**MIT License** – Free to use and modify!  

---


