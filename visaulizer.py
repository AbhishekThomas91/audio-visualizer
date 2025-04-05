import pygame
import numpy as np
import math
import librosa  # Ensure librosa is installed with: pip install librosa

# Screen settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (0, 0, 0)  # Black background
FPS = 60  # Frame rate

# Audio processing settings
NUM_BARS = 50  # Number of bars in visualization
playback_started = False
last_bar_heights = np.zeros(NUM_BARS)

def load_audio(filepath):
    """Loads audio data and sample rate using Librosa."""
    global audio_data, sample_rate, audio_length_samples

    try:
        print(f"Loading audio using Librosa: {filepath}")

        # Load audio with original sample rate, keeping stereo if available
        data, sample_rate = librosa.load(filepath, sr=None, mono=False)
        print(f"Loaded Audio: Sample Rate={sample_rate}, Shape={data.shape}")

        # Ensure correct stereo/mono handling
        if data.ndim > 1 and data.shape[0] < data.shape[1]:  # (channels, samples)
            print("Transposing audio to (samples, channels) format.")
            data = data.T  # Convert to (samples, channels)

        # Convert stereo to mono by averaging channels if necessary
        if data.ndim > 1:
            print("Audio is stereo, converting to mono.")
            audio_data = np.mean(data, axis=1)
        else:
            audio_data = data

        # Normalize audio data
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data /= max_val
        else:
            print("Warning: Audio data is silent.")

        audio_length_samples = len(audio_data)
        print(f"Audio processed. Length: {audio_length_samples} samples ({audio_length_samples / sample_rate:.2f} sec)")
        return True

    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return False
    except Exception as e:
        print(f"Error loading audio: {e}")
        return False

def get_audio_chunk(current_sample_index, chunk_size=1024):
    """Extracts a chunk of audio data based on current playback position."""
    if current_sample_index + chunk_size >= audio_length_samples:
        return np.zeros(chunk_size)
    
    return audio_data[current_sample_index: current_sample_index + chunk_size]

def calculate_frequencies(audio_chunk):
    """Computes frequency magnitudes for visualization."""
    fft_result = np.abs(np.fft.fft(audio_chunk))[:NUM_BARS]
    return fft_result / np.max(fft_result)  # Normalize

def draw_bars(screen, bar_heights):
    """Draws bars on the screen based on frequency magnitudes."""
    bar_width = SCREEN_WIDTH // NUM_BARS
    max_height = SCREEN_HEIGHT - 50
    
    for i, height in enumerate(bar_heights):
        bar_height = int(height * max_height)
        pygame.draw.rect(screen, (0, 255, 0), (i * bar_width, SCREEN_HEIGHT - bar_height, bar_width - 2, bar_height))

def main(audio_filepath):
    global playback_started, last_bar_heights

    pygame.init()
    pygame.mixer.init()
    

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Audio Visualizer")
    clock = pygame.time.Clock()

    if not load_audio(audio_filepath):
        pygame.quit()
        return

    try:
        pygame.mixer.music.load(audio_filepath)
        print(f"Loaded for playback: {audio_filepath}")
    except pygame.error as e:
        print(f"Error loading audio with pygame.mixer: {e}")
        pygame.quit()
        return

    last_bar_heights = np.zeros(NUM_BARS)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.pause()
                        print("Paused")
                    else:
                        if playback_started:
                            pygame.mixer.music.unpause()
                            print("Unpaused")
                        else:
                            pygame.mixer.music.play()
                            print("Playing...")
                            playback_started = True
                if event.key == pygame.K_r:  # Restart
                    pygame.mixer.music.rewind()
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
                    playback_started = True
                    last_bar_heights = np.zeros(NUM_BARS)
                    print("Restarted")

        # Audio Processing & Visualization
        current_bar_heights = np.zeros(NUM_BARS)
        if pygame.mixer.music.get_busy():
            playback_time_sec = pygame.mixer.music.get_pos() / 1000.0
            current_sample_index = int(playback_time_sec * sample_rate)
            current_chunk = get_audio_chunk(current_sample_index)
            current_bar_heights = calculate_frequencies(current_chunk)
         
        screen.fill(BACKGROUND_COLOR)
        draw_bars(screen, current_bar_heights)
        pygame.display.flip()
        clock.tick(FPS)
        
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    pygame.quit()
    print("Visualizer stopped.")
 
if __name__ == '__main__':
    audio_file = 'ParadiseCity.mp3'  # Change to your MP3 file path
    main(audio_file)