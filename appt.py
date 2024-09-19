import cv2
import numpy as np
import os
import NIR
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib  # For saving and loading the trained model
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
# Step 2: Extract Blood Flow Signals
def extract_blood_flow_signals(nir_frames, roi=(100, 100, 200, 200)):
    
    x, y, w, h = roi
    blood_flow_signal = []

    for frame in nir_frames:
        # Convert to numpy array if it's not already one
        frame = np.array(frame)
        roi_frame = frame[y:y + h, x:x + w]
        mean_intensity = np.mean(roi_frame)
        blood_flow_signal.append(mean_intensity)

    return np.array(blood_flow_signal)

# Step 3: Signal Processing
def filter_signal(signal, lowcut=0.5, highcut=2.5, fs=30, order=5):
   
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Step 4: Feature Extraction
def extract_features(signal):
   
    ac_component = np.max(signal) - np.min(signal)  # Amplitude of blood flow
    dc_component = np.mean(signal)  # Baseline flow
    rate_of_change = np.gradient(signal).mean()  # Average slope
    pulse_variation = np.std(signal)  # Pulse amplitude variation (standard deviation)

    features = {
        'ac_component': ac_component,
        'dc_component': dc_component,
        'rate_of_change': rate_of_change,
        'pulse_variation': pulse_variation,
    }
    return features

# Step 5: Load or Train Classification Model
def load_classifier(model_path="iron_deficiency_model.pkl"):
    
     if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded from", model_path)
        # Check if the model is fitted
        if not hasattr(model, "n_estimators_"):
            raise ValueError("Loaded model is not fitted. Train the model first.")
        else:
            print("Loaded model is already fitted and ready for prediction.")
     else:
        # Create a new, untrained model
         model = RandomForestClassifier()
         print("New model created, please train with labeled data.")
     return model

def classify_iron_deficiency(features, model):
    
    feature_vector = np.array([list(features.values())])
    prediction = model.predict(feature_vector)[0]
    confidence = model.predict_proba(feature_vector)[0]
    return prediction, confidence

def train_classifier(model, X_train, y_train, model_path="iron_deficiency_model.pkl"):
   
    # Fit the model with training data
    model.fit(X_train, y_train)
    # Save the fitted model
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")

def store_frequencies(frequencies, file_name="iron_frequency-B+.txt"):
    with open(file_name, 'w') as file:
        file.write("frequency\n")
        for freq in frequencies:
            file.write(f"{freq}\n")
# Step 6: Process the Video and Output Results
def process_video(video_path):
    frames = extract_frames(video_path)
    frames = [downsample_frame(frame) for frame in frames]
    nir_frames = [convert_to_infrared(frame) for frame in frames]

    # Extract blood flow signal from NIR frames
    blood_flow_signal = extract_blood_flow_signals(nir_frames)

    # Filter the blood flow signal
    filtered_signal = filter_signal(blood_flow_signal)

    # Extract features from the filtered signal
    features = extract_features(filtered_signal)

    # Load pre-trained model
    model = load_classifier()

    # Check if model needs training
    if not hasattr(model, "n_estimators_"):
        print("Training the model with labeled data as it is not fitted.")
        # Replace with actual training data and labels
        X_train = np.random.rand(100, 4)  # Example feature data, replace with real data
        y_train = np.random.randint(0, 2, 100)  # Example labels, replace with real labels
        train_classifier(model, X_train, y_train)

    # Classify iron deficiency
    prediction, confidence = classify_iron_deficiency(features, model)

    # Output results
    if prediction == 1:
        print(f"Iron Deficiency Detected with {confidence[1]:.2f} confidence.")
    else:
        print(f"No Iron Deficiency detected with {confidence[0]:.2f} confidence.")

    # Optional: Plot the filtered blood flow signal
    plt.plot(filtered_signal)
    plt.xlabel("Frame")
    plt.ylabel("Blood Flow Intensity")
    plt.title("Filtered Blood Flow Signal")
    plt.show()

    store_frequencies(filtered_signal, file_name=f"frequencies_{os.path.basename(video_path)}.txt")
def main(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith((".mp4", ".avi", ".mov")):  # Add more video formats if needed
            video_path = os.path.join(folder_path, file_name)
            process_video(video_path)

# Example usage
folder_path = "C:\\Users\\Sumit\\OneDrive\\Desktop\\Fe_prediction\\Vdata"
main(folder_path)