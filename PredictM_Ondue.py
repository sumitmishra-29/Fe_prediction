import cv2
import numpy as np
import os
import NIR
import pandas as pd
import gdown
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import joblib
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_video_from_gdrive(gdrive_link, output_path="video.mp4", retries=5, delay=5):
    for attempt in range(retries):
        try:
            print(f"Attempting to download video (Attempt {attempt + 1})...")
            gdown.download(gdrive_link, output_path, quiet=False, fuzzy=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Video downloaded successfully to {output_path}.")
                time.sleep(delay)
                return
            else:
                print("Downloaded file is empty or not found. Retrying...")
        except Exception as e:
            print(f"Error downloading video: {e}. Retrying...")
        time.sleep(delay)

    print("Failed to download video after multiple attempts.")

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Output layer for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def store_frequencies(frequencies, file_name="iron_frequency-B+.txt"):
    with open(file_name, 'w') as file:
        file.write("frequency\n")
        for freq in frequencies:
            file.write(f"{freq}\n")

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
    return np.array(infrared_frame)

def extract_blood_flow_signals(nir_frames, roi=(100, 100, 200, 200)):
    x, y, w, h = roi
    blood_flow_signal = []
    for frame in nir_frames:
        frame = np.array(frame)
        roi_frame = frame[y:y + h, x:x + w]
        mean_intensity = np.mean(roi_frame)
        blood_flow_signal.append(mean_intensity)
    return np.array(blood_flow_signal)

def filter_signal(signal, lowcut=0.5, highcut=2.5, fs=30, order=5):
    if len(signal) <= 33:  # Check if the signal is too short
        print("Warning: Signal length is too short for filtering. Returning original signal.")
        return signal
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def extract_features(signal):
    ac_component = np.max(signal) - np.min(signal)
    dc_component = np.mean(signal)
    rate_of_change = np.gradient(signal).mean()
    pulse_variation = np.std(signal)
    features = {
        'ac_component': ac_component,
        'dc_component': dc_component,
        'rate_of_change': rate_of_change,
        'pulse_variation': pulse_variation,
    }
    return features

def encode_categorical_features(age_range, sex, blood_group):
    age_encoder = OneHotEncoder(sparse_output=False)  # Updated argument
    sex_encoder = LabelEncoder()
    blood_group_encoder = OneHotEncoder(sparse_output=False)  # Updated argument

    age_range_encoded = age_encoder.fit_transform(np.array([age_range]).reshape(-1, 1))
    sex_encoded = sex_encoder.fit_transform([sex])
    blood_group_encoded = blood_group_encoder.fit_transform(np.array([blood_group]).reshape(-1, 1))

    return np.concatenate([age_range_encoded.flatten(), sex_encoded.flatten(), blood_group_encoded.flatten()])

def classify_iron_deficiency(features, additional_features, model):
    feature_vector = np.array([list(features.values())])
    complete_features = np.concatenate([feature_vector.flatten(), additional_features])
    complete_features_tensor = torch.tensor(complete_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        model.eval()
        output = model(complete_features_tensor)

        # Check the shape of the output
        if output.shape[0] == 1:  # Single output case (e.g., sigmoid)
             prediction = (output.item() > 0.5)  # Threshold for binary classification
             confidence = torch.sigmoid(output).item()  # Use sigmoid for confidence (scalar)
    
             if prediction == 1:
               print(f"Iron Deficiency Detected with {confidence:.2f} confidence.")
             else:
                print(f"No Iron Deficiency detected with {confidence:.2f} confidence.")

        elif output.shape[0] == 2:  # Multi-output case (e.g., softmax)
             prediction = torch.argmax(output, dim=0).item()  # Get predicted class
             confidence = torch.softmax(output, dim=0).cpu().numpy()  # Get probabilities (1D array)

             if prediction == 1:
               print(f"Iron Deficiency Detected with {confidence[1]:.2f} confidence.")  # confidence for class 1
             else:
              print(f"No Iron Deficiency detected with {confidence[0]:.2f} confidence.")  # confidence for class 0

        else:
            raise ValueError("Unexpected output shape from the model.")

    

def train_model(model, X_train, y_train, epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
        labels = torch.tensor(y_train, dtype=torch.long).to(device)  # No need to reshape

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # No need to reshape labels
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
            
def process_video(video_path, age_range, sex, blood_group):
    frames = extract_frames(video_path)
    frames = [downsample_frame(frame) for frame in frames]
    nir_frames = [convert_to_infrared(frame) for frame in frames]
    blood_flow_signal = extract_blood_flow_signals(nir_frames)

    if len(blood_flow_signal) == 0:
        print("Error: No blood flow signal extracted.")
        return

    filtered_signal = filter_signal(blood_flow_signal)
    features = extract_features(filtered_signal)
    additional_features = encode_categorical_features(age_range, sex, blood_group)

    # Prepare training data
    X_train = np.random.rand(100, len(additional_features) + 4)
    y_train = np.random.randint(0, 2, 100)

    # Initialize and train the model
    model = SimpleNN(input_size=len(additional_features) + 4).to(device)
    train_model(model, X_train, y_train)

    classify_iron_deficiency(features, additional_features, model)

   
    plt.plot(filtered_signal)
    plt.xlabel("Frame")
    plt.ylabel("Blood Flow Intensity")
    plt.title("Filtered Blood Flow Signal")
    plt.show()
    store_frequencies(filtered_signal, file_name=f"frequencies_{os.path.basename(video_path)}.txt")

def process_csv(csv_path):
    data = pd.read_csv(csv_path)
    for index, row in data.iterrows():
        video_link = row['Videoattachment']
        age_range = row['Age']
        sex = row['sex']
        blood_group = row['Blood group']
        video_output_path = f"video_{index}.mp4"
        download_video_from_gdrive(video_link, video_output_path)
        process_video(video_output_path, age_range, sex, blood_group)

def main(video_path, age_range, sex, blood_group):
    process_video(video_path, age_range, sex, blood_group)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python appS.py <video_path> <age_range> <sex> <blood_group>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

# Example usage
csv_path = "C:\\Users\\Sumit\\OneDrive\\Desktop\\Fe_prediction\\Vdata\\data.csv"
process_csv(csv_path)