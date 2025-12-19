import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_mel_spectrogram(
    wav_path,
    start_time=300,
    duration=5,
    sr=16000,
    n_mels=128,
    output_path="mel_spectrogram.png",
    transparent=False,
    base_fontsize=10
):
    """
    Generate a Mel spectrogram from a segment of an audio file with enlarged text.

    Parameters:
    wav_path (str): Path to the WAV file.
    start_time (float): Start time in seconds (default: 300s = 5 minutes).
    duration (float): Duration of the segment in seconds (default: 5s).
    sr (int): Sampling rate (default: 16000).
    n_mels (int): Number of Mel bands (default: 128).
    output_path (str): Path to save the spectrogram image.
    transparent (bool): If True, save with transparent background; else white.
    base_fontsize (int): Base font size to double (default: 10).
    """
    # Load the segment
    y, _ = librosa.load(wav_path, sr=sr, offset=start_time, duration=duration)

    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')

    # Double font sizes
    fontsize = base_fontsize * 4
    plt.title('Mel Spectrogram', fontsize=fontsize)
    plt.xlabel('Time (s)', fontsize=fontsize)
    plt.ylabel('Mel Frequency', fontsize=fontsize)

    # Colorbar with larger font
    cbar = plt.colorbar(format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label('dB', fontsize=fontsize)

    # Tick labels
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.tight_layout()

    # Save image
    if transparent:
        plt.savefig(output_path, transparent=True)
    else:
        plt.savefig(output_path, facecolor='white')
    plt.close()

# Example usage:
# create_mel_spectrogram("path/to/audio.wav", output_path="spectrogram.png", transparent=True, base_fontsize=10)