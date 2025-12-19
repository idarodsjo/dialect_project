import wave
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

def plot_waveform(wav_file_path, file_name):
    try:
        with wave.open(wav_file_path, 'rb') as wav_obj:
            sample_freq = wav_obj.getframerate()
            n_samples = wav_obj.getnframes()
            n_channels = wav_obj.getnchannels()

            # read frames as bytes
            signal_wave = wav_obj.readframes(n_samples)

        # convert bytes to np array of 18 bit int
        signal_array = np.frombuffer(signal_wave, dtype=np.int16)

        if n_channels >1:
            signal_array = signal_array[::n_channels]

        duration = n_samples / sample_freq
        time = np.linspace(0, duration, num=len(signal_array))

        plt.figure(figsize=(15, 5))
        plt.plot(time, signal_array, color='blue')
        plt.xlim(519.160, 519.2)
        #plt.ylim(-20000, 35000)
        plt.title(f'Waveform of {file_name}.wav', fontsize=18)
        plt.xlabel('Time [s]', fontsize=14)
        plt.ylabel('Amplitude', fontsize=14)
        plt.grid(True)
        plt.savefig(f"figures/waveform_{file_name}.png")
        plt.show()
    except wave.Error as e:
        print(f"Error opening or reading WAV file: {e}")
    except FileNotFoundError:
        print(f"File not found: {wav_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    #wav_file_path = "/home/idatro/preprocessed_NorDia/norske-lydfiler-fra-ndc-1/aure_04gk-ta_prcssd.wav"
    wav_file_path = "talebase/data/speech_raw/ScanDia/NorDia/norske-lydfiler-fra-ndc-1/aure_04gk-ta.wav"
    file_name = "aure_04gk-ta"
    plot_waveform(wav_file_path, file_name)