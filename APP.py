import cv2
import numpy as np
import os
import NIR
import matplotlib.pyplot as plt

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def downsample_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height))

def convert_to_infrared(frame):
    infrared_frame = NIR.convert_rgb_to_nir_frames(frame)
    return infrared_frame

def extract_red_spectrum(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    red_channel = frame[:, :, 2]
    red_spectrum = np.fft.fft2(red_channel)
    red_spectrum_shifted = np.fft.fftshift(red_spectrum)
    magnitude_spectrum = 20 * np.log1p(np.abs(red_spectrum_shifted))
    # cv2.imshow(magnitude_spectrum)
    return magnitude_spectrum

def calculate_frequency(magnitude_spectrum):
    frequency = np.mean(magnitude_spectrum)
    return frequency

def store_frequencies(frequencies, file_name="iron_frequency-B+.txt"):
    with open(file_name, 'w') as file:
        file.write("frequency\n")
        for freq in frequencies:
            file.write(f"{freq}\n")

def process_video(video_path):
    frames = extract_frames(video_path)
    frequencies = []
    frames = [downsample_frame(frame) for frame in frames]
    nir_frames = [convert_to_infrared(frame) for frame in frames]
   # cv2.imshow(nir_frames)
    for frame in nir_frames:
        red_spectrum = extract_red_spectrum(frame)
        frequency = calculate_frequency(red_spectrum)
        frequencies.append(frequency)
    plt.plot(frequencies)
    plt.xlabel("freq.")
    plt.ylabel("per-fram")
    plt.title("freq. Graph")
    plt.show()

    store_frequencies(frequencies, file_name=f"frequencies_{os.path.basename(video_path)}.txt")

def main(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith((".mp4", ".avi", ".mov")):  # Add more video formats if needed
            video_path = os.path.join(folder_path, file_name)
            process_video(video_path)

# Example usage
folder_path = "C:\\Users\\Sumit\\OneDrive\\Desktop\\Fe_prediction\\Vdata"
main(folder_path)
