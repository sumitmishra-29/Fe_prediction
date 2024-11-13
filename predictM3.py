import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import gdown
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import cv2
import NIR
import os
import time
import torch.nn.functional as F  
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import OneHotEncoder

# Neural Network Model in PyTorch
class IronDeficiencyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IronDeficiencyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

def download_video_from_gdrive(gdrive_link, output_path="video.mp4", retries=5, delay=5):
    for attempt in range(retries):
        try:
            print(f"Attempting to download video (Attempt {attempt + 1})...")
            gdown.download(gdrive_link, output_path, quiet=False, fuzzy=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Video downloaded successfully to {output_path}.")
                time.sleep(delay)
                return output_path  # Return the path of the downloaded video
            else:
                print("Downloaded file is empty or not found. Retrying...")
        except Exception as e:
            print(f"Error downloading video: {e}. Retrying...")
        time.sleep(delay)

    print("Failed to download video after multiple attempts.")
    return None

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

def train_nn_model(X_train, y_train, input_size, hidden_size=64, epochs=100, lr=0.001):
    print("Training the Neural Network model...")
    model = IronDeficiencyNN(input_size=input_size, hidden_size=hidden_size, output_size=2)  # Binary output
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def predict_with_nn(model, X):
    print("Making predictions with the trained model...")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        output = model(X_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.numpy()

# Function to save and load synthetic data
def save_synthetic_data(data, filename="NN_synthetic_data.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_synthetic_data(filename="synthetic_data.pkl"):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Process the CSV data
def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    
    for index, row in df.iterrows():
        video_link = row['Videoattachment']
        age_range = row['Age']
        sex = row['sex']
        blood_group = row['Blood group']
        video_output_path = f"video_{index}.mp4"
        
        print(f"Starting to download video for row {index}...")
        
        # Download the video
        video_path = download_video_from_gdrive(video_link, video_output_path)
        
        if video_path:
            print(f"Processing video {index}...")
            # Process the video only if the download was successful
            process_video(video_path, age_range, sex, blood_group)
        else:
            print(f"Skipping video at index {index}, download failed.")
    
    return X, y

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

# Function to process video data
def process_video(video_path, age_range, sex, blood_group):
    print(f"Processing video from path: {video_path}")
    
    # Assuming you already have functions like `extract_frames`, `downsample_frame`, etc.
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

    # Combine the features from the video and categorical features
    input_features = np.concatenate([list(features.values()), additional_features])

    # Load synthetic data (if available) for training the model
    synthetic_data = load_synthetic_data()

    # If there's no synthetic data, create it
    if synthetic_data is None:
        synthetic_data = {"X": [], "y": []}

    # Assuming rf_prediction is the output from a previous model or classifier
    rf_prediction = 0  # Example placeholder for RF prediction (0: No, 1: Yes)
    synthetic_data["X"].append(input_features)
    synthetic_data["y"].append(0 if rf_prediction == 0 else 1)  # Assuming binary prediction

    # Save updated synthetic data
    save_synthetic_data(synthetic_data)

    # Convert to numpy array
    X_train = np.array(synthetic_data["X"])
    y_train = np.array(synthetic_data["y"])

    # Train the neural network with the updated synthetic data
    model = train_nn_model(X_train, y_train, input_size=X_train.shape[1])

    # Make predictions with the trained model (for the latest input only)
    prediction = predict_with_nn(model, X_train[-1].reshape(1, -1))  # Predict for the last row only

    # Get the probabilities from the model
    predicted_probabilities = model(torch.tensor(X_train[-1].reshape(1, -1), dtype=torch.float32)).detach().numpy()
    iron_deficiency_percentage = predicted_probabilities[0][1] * 100  # The percentage for class "1" (iron deficiency)

    print(f"Iron Deficiency Prediction: {prediction[0]}")
    print(f"Probability of Iron Deficiency: {iron_deficiency_percentage:.2f}%")
   # You can add code here to calculate the percentage if needed

# Example usage with CSV path and video processing
csv_path = "C:\\Users\\Sumit\\OneDrive\\Desktop\\Fe_prediction\\Vdata\\data.csv"
process_csv(csv_path)
