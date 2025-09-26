import cv2
import numpy as np
import pyttsx3
import sys
import os
import time
import threading
import pickle

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualife.core.model import Model
from visualife.core.layers import Dense, Dropout, BatchNorm
from visualife.utils.data_loader import DataLoader
from visualife.core.activations import ReLU
from visualife.core.convolutional import Conv2D, MaxPool2D, Flatten
from collections import deque

# ======================
# CONFIG
# ======================
MODEL_PATH = "experiments/real_dataset_cnn_light.pkl"
IMG_SIZE = (128, 128)
DATA_DIR = "dataset_resized/train"
CAPTURE_INTERVAL = 5  # seconds
CONFIDENCE_THRESHOLD = 0.15  # Increased threshold for better filtering
PREDICTION_HISTORY_SIZE = 3  # Number of predictions to average

# ======================
# LOAD MODEL
# ======================
print("📂 Loading model...")
model = Model()
model.load(MODEL_PATH)
print("✅ Model loaded!")

loader = DataLoader(data_dir=DATA_DIR, img_size=IMG_SIZE, batch_size=1, num_classes=9)
class_names = loader.class_names
print(f"Classes: {class_names}")

# Add this diagnostic code right after loading the model
def verify_model_weights(model):
    """Check if model has trained weights"""
    print("\n🔍 Verifying model weights...")
    
    # Count total parameters
    total_params = 0
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W'):
            layer_params = np.prod(layer.W.shape) if layer.W is not None else 0
            if hasattr(layer, 'b') and layer.b is not None:
                layer_params += np.prod(layer.b.shape)
            total_params += layer_params
            print(f"Layer {i} ({type(layer).__name__}): {layer_params} parameters")
    
    print(f"Total parameters: {total_params}")
    
    # Check if weights are all zeros or random
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W') and layer.W is not None:
            weight_mean = np.mean(layer.W)
            weight_std = np.std(layer.W)
            print(f"Layer {i} weights - Mean: {weight_mean:.6f}, Std: {weight_std:.6f}")
            
            # If weights are all very close to 0, model might be untrained
            if abs(weight_mean) < 1e-6 and weight_std < 1e-6:
                print("⚠️  Warning: Weights appear to be untrained!")

# Call this after model.load()
verify_model_weights(model)

# ======================
# PREDICTION HISTORY FOR SMOOTHING
# ======================
prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)

# ======================
# TTS ENGINE
# ======================
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def speak(text):
    """Threaded speech function with debugging"""
    def _speak():
        try:
            print(f"🔊 Speaking: '{text}'")
            engine.say(text)
            engine.runAndWait()
            print("✅ Speech completed")
        except Exception as e:
            print(f"❌ Speech error: {e}")
    
    threading.Thread(target=_speak, daemon=True).start()

# ======================
# DEBUG FUNCTIONS
# ======================
def debug_predictions(model, img, class_names):
    """Debug function to see all prediction scores"""
    y_pred = model.predict(img)
    print("\n" + "="*50)
    print("PREDICTION DEBUG")
    print("="*50)
    print("All class probabilities:")
    
    # Sort by probability (highest first)
    sorted_predictions = sorted(zip(class_names, y_pred[0]), key=lambda x: x[1], reverse=True)
    
    for class_name, score in sorted_predictions:
        print(f"  {class_name}: {score:.4f}")
    
    class_idx = np.argmax(y_pred)
    confidence = np.max(y_pred)
    print(f"\nHighest: {class_names[class_idx]} ({confidence:.4f})")
    print("="*50 + "\n")
    
    return y_pred, class_idx, confidence

def get_smoothed_prediction(current_pred, current_conf, history):
    """Use majority voting from recent predictions"""
    history.append((current_pred, current_conf))
    
    # Only consider predictions above threshold
    valid_predictions = [(pred, conf) for pred, conf in history if conf > CONFIDENCE_THRESHOLD]
    
    if not valid_predictions:
        return current_pred, current_conf, False
    
    # Get most frequent prediction among valid ones
    preds = [pred for pred, conf in valid_predictions]
    most_common = max(set(preds), key=preds.count)
    
    # Average confidence for most common prediction
    confidences = [conf for pred, conf in valid_predictions if pred == most_common]
    avg_conf = np.mean(confidences) if confidences else current_conf
    
    return most_common, avg_conf, True

def test_model_variety(model, class_names, num_tests=5):
    """Test model with random noise to see class distribution"""
    print("🧪 Testing model bias with random inputs:")
    predictions = []
    for i in range(num_tests):
        test_input = np.random.random((1, 128, 128, 3)).astype("float32")
        y_pred = model.predict(test_input)
        class_idx = np.argmax(y_pred)
        confidence = np.max(y_pred)
        predictions.append(class_names[class_idx])
        print(f"  Test {i+1}: {class_names[class_idx]} ({confidence:.4f})")
    
    from collections import Counter
    pred_counts = Counter(predictions)
    print("\n📊 Prediction distribution from random inputs:")
    for class_name, count in pred_counts.items():
        print(f"  {class_name}: {count}/{num_tests}")
    print()

# ======================
# OPEN WEBCAM
# ======================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    sys.exit(1)

print("🎥 Starting webcam. Press 'q' to quit.")
print("🔊 Predictions will be spoken every 5 seconds")
print("📊 Debug information will be displayed")
print("💡 Point camera at different objects to test")

# Test model bias
test_model_variety(model, class_names)

last_capture_time = 0
last_prediction = None
last_confidence = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame")
        break

    current_time = time.time()
    frame_count += 1

    # Predict every CAPTURE_INTERVAL seconds
    if current_time - last_capture_time >= CAPTURE_INTERVAL:
        last_capture_time = current_time
        
        print(f"\n🔄 Frame {frame_count} - Capturing image for prediction...")

        # Preprocess frame
        img = cv2.resize(frame, IMG_SIZE)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict with debugging
        try:
            # Get detailed prediction information
            y_pred, class_idx, confidence = debug_predictions(model, img, class_names)
            raw_prediction = class_names[class_idx]
            
            # Apply smoothing to reduce flickering
            smoothed_pred, smoothed_conf, is_confident = get_smoothed_prediction(
                raw_prediction, confidence, prediction_history
            )
            
            last_prediction = smoothed_pred
            last_confidence = smoothed_conf

            print(f"🎯 Raw: {raw_prediction} ({confidence:.3f})")
            print(f"📈 Smoothed: {smoothed_pred} ({smoothed_conf:.3f})")
            print(f"✅ Confident: {is_confident}")
            
            # Speak prediction if confident
            if is_confident and smoothed_conf > CONFIDENCE_THRESHOLD:
                speak(f"I see {smoothed_pred}")
            else:
                print(f"❓ Low confidence - not speaking")
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            last_prediction = "Error"
            last_confidence = 0

    # Display information on frame
    # Clear the frame for new text
    display_frame = frame.copy()
    
    # Prediction text with confidence-based color
    if last_prediction:
        if last_confidence > CONFIDENCE_THRESHOLD:
            text_color = (0, 255, 0)  # Green - high confidence
            status = "HIGH CONFIDENCE"
        else:
            text_color = (0, 165, 255)  # Orange - low confidence
            status = "LOW CONFIDENCE"
        
        # Main prediction
        cv2.putText(display_frame, f"Prediction: {last_prediction}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Confidence level
        cv2.putText(display_frame, f"Confidence: {last_confidence:.3f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Status
        cv2.putText(display_frame, f"Status: {status}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Countdown timer
    time_left = max(0, int(CAPTURE_INTERVAL - (current_time - last_capture_time)))
    cv2.putText(display_frame, f"Next prediction in: {time_left}s", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Instructions
    cv2.putText(display_frame, "Press 'q' to quit", (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, "Point at different objects", (20, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("VisuaLife - Object Detection (Debug Mode)", display_frame)

    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 Exiting...")
        break
    elif key == ord(' '):  # Spacebar for manual capture
        print("\n🎯 Manual capture triggered!")
        last_capture_time = 0  # Force immediate capture on next iteration

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam released. Goodbye!")