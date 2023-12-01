import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import csv
import os
from datetime import datetime

DURATION_IN_SECONDS = 5
CHART_X_MIN = 0
CHART_X_MAX = 2000


def capture_audio(seconds=DURATION_IN_SECONDS, sample_rate=44100, chunk_size=1024):
    """
    Captures audio using the microphone.

    Parameters:
    - seconds (int): Duration of audio capture.
    - sample_rate (int): Sampling rate in Hz.
    - chunk_size (int): Number of frames per buffer.

    Returns:
    - np.array: Numpy array containing the audio data.
    - list: List of frequency values.
    - list: List of amplitude values in decibels.
    """
    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    print(colored("Capturing audio...", "green"))

    frames = []
    db_values = []  # To store decibel values
    hz_values = []  # To store corresponding frequencies

    for i in range(0, int(sample_rate / chunk_size * seconds)):
        data = stream.read(chunk_size)
        frames.append(data)

        # Convert data to numpy array
        audio_chunk = np.frombuffer(data, dtype=np.int16)

        # Apply window function
        windowed_chunk = audio_chunk * np.hanning(len(audio_chunk))

        # Calculate frequency spectrum using Fast Fourier Transform (FFT)
        frequencies = np.fft.fftfreq(len(audio_chunk), 1 / sample_rate)
        spectrum = np.abs(np.fft.fft(windowed_chunk))

        # Find the index of the maximum frequency
        index_max = np.argmax(spectrum)
        freq_max = frequencies[index_max]
        amplitude_max = (
            20 * np.log10(spectrum[index_max] / np.max(spectrum))
            if np.max(spectrum) != 0
            else 0
        )

        # Store values
        hz_values.append(freq_max)
        db_values.append(amplitude_max)

        # Print progress and information
        if i % 10 == 0:  # Print every 10 iterations
            print(
                colored(
                    f"Progress: {i * chunk_size / sample_rate:.1f} seconds", "yellow"
                ),
                end="\r",
            )

    print(colored("\nAudio captured!", "green"))

    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.frombuffer(b"".join(frames), dtype=np.int16), hz_values, db_values


def plot_spectrum(frequencies, spectrum):
    """
    Plots the frequency spectrum.

    Parameters:
    - frequencies (np.array): Array of frequencies.
    - spectrum (np.array): Array of amplitudes in the frequency spectrum.
    """
    plt.plot(frequencies, 20 * np.log10(spectrum / np.max(spectrum)))
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.xlim(
        CHART_X_MIN, CHART_X_MAX
    )  # Set the x-axis limit to focus on relevant frequencies
    plt.ylim(-100, 0)  # Set the y-axis limit for dB values
    plt.show()


def identify_resonance(frequencies, spectrum):
    """
    Identifies resonance frequency and maximum amplitude.

    Parameters:
    - frequencies (np.array): Array of frequencies.
    - spectrum (np.array): Array of amplitudes in the frequency spectrum.
    """
    # Find the index of the maximum frequency
    index_max = np.argmax(spectrum)
    freq_max = frequencies[index_max]
    amplitude_max = spectrum[index_max]

    print(colored(f"\nIdentified resonance frequency: {freq_max} Hz", "cyan"))
    print(colored(f"Maximum amplitude: {amplitude_max} dB", "cyan"))


def save_to_csv(hz_values, db_values):
    """
    Saves frequency and amplitude data to a CSV file in the 'audios' folder.

    Parameters:
    - hz_values (list): List of frequency values.
    - db_values (list): List of amplitude values in decibels.
    """
    if not os.path.exists('audios'):
        os.makedirs('audios')

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'audios/audio_data_{current_time}.csv'

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frequency (Hz)", "Amplitude (dB)"])

        for hz, db in zip(hz_values, db_values):
            writer.writerow([hz, db])

    print(colored(f"Data saved to {filename}!", "green"))


if __name__ == "__main__":
    sample_rate = 44100  # Set your desired sample rate

    # Capture audio and get frequency and amplitude values
    audio_data, hz_values, db_values = capture_audio(sample_rate=sample_rate)

    # Calculate frequency spectrum using Fast Fourier Transform (FFT)
    frequencies = np.fft.fftfreq(len(audio_data), 1 / sample_rate)
    spectrum = np.abs(np.fft.fft(audio_data))

    # Plot the spectrum
    plot_spectrum(frequencies, spectrum)

    # Identify resonance
    identify_resonance(frequencies, spectrum)


    # Save data to CSV file
    save_to_csv(hz_values, spectrum)
