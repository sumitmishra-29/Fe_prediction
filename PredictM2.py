import cv2
import numpy as np
import os
import NIR
import pandas as pd
import gdown
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import sys
import time

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

def load_classifier(model_path="iron_deficiency_model.pkl", overwrite=False):
    if overwrite and os.path.exists(model_path):
        os.remove(model_path)  # Remove the existing model file
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded from", model_path)
    else:
        model = RandomForestClassifier()
        print("New model created, please train with labeled data.")
    return model

def load_svm_classifier(model_path="svm_iron_deficiency_model.pkl", overwrite=False):
    if overwrite and os.path.exists(model_path):
        os.remove(model_path)  # Remove the existing SVM model file
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("SVM Model loaded from", model_path)
    else:
        model = SVC(probability=True)
        print("New SVM model created, please train with labeled data.")
    return model

def encode_categorical_features(age_range, sex, blood_group):
    age_encoder = OneHotEncoder(sparse_output=False)  # Updated argument
    sex_encoder = LabelEncoder()
    blood_group_encoder = OneHotEncoder(sparse_output=False)  # Updated argument

    age_range_encoded = age_encoder.fit_transform(np.array([age_range]).reshape(-1, 1))
    sex_encoded = sex_encoder.fit_transform([sex])
    blood_group_encoded = blood_group_encoder.fit_transform(np.array([blood_group]).reshape(-1, 1))

    return np.concatenate([age_range_encoded.flatten(), sex_encoded.flatten(), blood_group_encoded.flatten()])


def classify_iron_deficiency(features, additional_features, rf_model, svm_model):
    feature_vector = np.array([list(features.values())])
    complete_features = np.concatenate([feature_vector.flatten(), additional_features])
    rf_prediction = rf_model.predict([complete_features])[0]
    rf_confidence = rf_model.predict_proba([complete_features])[0]
    svm_prediction = svm_model.predict([complete_features])[0]
    svm_confidence = svm_model.predict_proba([complete_features])[0]
    return rf_prediction, rf_confidence, svm_prediction, svm_confidence

def train_svm_classifier(model, X_train, y_train, model_path="svm_iron_deficiency_model.pkl"):
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"SVM model trained and saved to {model_path}")

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


def process_video(video_path, age_range, sex, blood_group):
    frames = extract_frames(video_path)
    frames = [downsample_frame(frame) for frame in frames]
    nir_frames = [convert_to_infrared(frame) for frame in frames]
    blood_flow_signal = extract_blood_flow_signals(nir_frames)

    # Add a check for the blood flow signal length
    if len(blood_flow_signal) == 0:
        print("Error: No blood flow signal extracted.")
        return

    filtered_signal = filter_signal(blood_flow_signal)
    features = extract_features(filtered_signal)
    additional_features = encode_categorical_features(age_range, sex, blood_group)
    rf_model = load_classifier(overwrite=True)  # Overwrite the RF model
    svm_model = load_svm_classifier(overwrite=True)  # Overwrite the SVM model

    # Always train the models
    X_train = np.random.rand(100, len(additional_features) + 4)
    y_train = np.random.randint(0, 2, 100)
    train_svm_classifier(svm_model, X_train, y_train)
    rf_model.fit(X_train, y_train)  # Train the RandomForest model

    rf_prediction, rf_confidence, svm_prediction, svm_confidence = classify_iron_deficiency(features, additional_features, rf_model, svm_model)
    
    if rf_prediction == 1:
        print(f"RandomForest: Iron Deficiency Detected with {rf_confidence[1]:.2f} confidence.")
    else:
        print(f"RandomForest: No Iron Deficiency detected with {rf_confidence[0]:.2f} confidence.")
        
    if svm_prediction == 1:
        print(f"SVM: Iron Deficiency Detected with {svm_confidence[1]:.2f} confidence.")
    else:
        print(f"SVM: No Iron Deficiency detected with {svm_confidence[0]:.2f} confidence.")
    
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
    # Your existing process_video function can be called here
    process_video(video_path, age_range, sex, blood_group)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python appS.py <video_path> <age_range> <sex> <blood_group>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

# Example usage
<<<<<<< HEAD:PredictM2.py
csv_path = "C:\\Users\\Sumit\\OneDrive\\Desktop\\Fe_prediction\\Vdata\\data.csv"
=======
csv_path = "C:\\Users\\ujjwa\\Desktop\\DotBatch\\Fe_prediction\\Vdata\\data.csv"
>>>>>>> 46f2fbbe7beadb214f17c52c20637615f33c5116:appS.py
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
process_csv(csv_path)