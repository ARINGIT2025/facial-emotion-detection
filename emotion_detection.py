import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

class EmotionDetector:
    """Advanced real-time facial emotion detection"""
    
    def __init__(self, model_path='models/emotion_model.h5'):
        """Initialize detector"""
        self.IMG_SIZE = 48
        self.EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        print("📦 Loading emotion detection model...")
        try:
            self.model = load_model(model_path)
            print("✅ Model loaded successfully")
        except FileNotFoundError:
            print(f"❌ Model not found: {model_path}")
            print("📝 Run train_model.py first!")
            raise
        
        # Load training info
        try:
            with open('models/training_info.json', 'r') as f:
                self.training_info = json.load(f)
            print(f"📊 Model Accuracy: {self.training_info['final_val_accuracy']:.2%}")
        except:
            self.training_info = None
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.face_cascade.empty():
            print("❌ Face cascade not found")
            raise ValueError("Face cascade error")
        
        # Color map for emotions
        self.emotion_colors = {
            'Angry': (0, 0, 255),           # Red
            'Disgust': (0, 165, 255),       # Orange
            'Fear': (128, 0, 128),          # Purple
            'Happy': (0, 255, 0),           # Green
            'Neutral': (200, 200, 200),     # Gray
            'Sad': (255, 0, 0),             # Blue
            'Surprise': (0, 255, 255)       # Yellow
        }
        
        print("✅ Emotion Detector initialized\n")
    
    def detect_emotions(self, frame):
        """Detect emotions in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ✅ HISTOGRAM EQUALIZATION (Same as training)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        emotions_data = []
        
        for (x, y, w, h) in faces:
            # Extract and preprocess
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (self.IMG_SIZE, self.IMG_SIZE))
            
            # ✅ NORMALIZE TO [-1, 1] (Same as training)
            face_roi = face_roi.astype('float32') / 127.5 - 1
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = np.expand_dims(face_roi, axis=0)
            
            # Predict
            predictions = self.model.predict(face_roi, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            emotion = self.EMOTIONS[emotion_idx]
            confidence = predictions[emotion_idx]
            
            emotions_data.append({
                'emotion': emotion,
                'confidence': confidence,
                'predictions': {self.EMOTIONS[i]: float(predictions[i]) for i in range(len(self.EMOTIONS))},
                'bbox': (x, y, w, h)
            })
            
            # Draw bounding box
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            thickness = 3
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Draw emotion label with confidence
            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(
                frame,
                label,
                (x, y-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )
            
            # Draw confidence bar
            bar_width = w
            bar_height = 25
            bar_filled = int(bar_width * confidence)
            
            cv2.rectangle(frame, (x, y+h), (x+bar_filled, y+h+bar_height), color, -1)
            cv2.rectangle(frame, (x, y+h), (x+bar_width, y+h+bar_height), color, 2)
            
            # Add confidence percentage
            conf_text = f"{confidence*100:.1f}%"
            cv2.putText(
                frame,
                conf_text,
                (x + bar_width - 50, y+h+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame, emotions_data
    
    def run_webcam(self, camera_id=0):
        """Run real-time emotion detection from webcam"""
        print(f"\n🎥 Starting webcam (camera_id={camera_id})...")
        print("👉 Controls:")
        print("   'q' = Quit")
        print("   's' = Save screenshot")
        print("   'r' = Reset statistics\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        emotion_history = {emotion: 0 for emotion in self.EMOTIONS}
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Flip for selfie view
                frame = cv2.flip(frame, 1)
                
                # Detect emotions
                frame, emotions_data = self.detect_emotions(frame)
                
                # Update statistics
                for data in emotions_data:
                    emotion_history[data['emotion']] += 1
                
                frame_count += 1
                
                # Display info
                info_text = f"Frames: {frame_count} | Faces: {len(emotions_data)}"
                cv2.putText(
                    frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Display FPS
                fps = cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Display model accuracy
                if self.training_info:
                    acc_text = f"Model Accuracy: {self.training_info['final_val_accuracy']:.1%}"
                    cv2.putText(
                        frame,
                        acc_text,
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                
                # Show frame
                cv2.imshow('🎭 Real-time Emotion Detection', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"emotion_screenshot_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('r'):
                    emotion_history = {emotion: 0 for emotion in self.EMOTIONS}
                    print("🔄 Statistics reset")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            print("\n" + "="*60)
            print("  📊 EMOTION DETECTION STATISTICS")
            print("="*60)
            print(f"Total frames processed: {frame_count}\n")
            
            total_detected = sum(emotion_history.values())
            if total_detected > 0:
                print("Emotion Distribution:")
                print("-" * 60)
                for emotion in sorted(emotion_history.keys(), 
                                     key=lambda x: emotion_history[x], reverse=True):
                    count = emotion_history[emotion]
                    percentage = (count / total_detected) * 100
                    bar = "█" * int(percentage / 2)
                    print(f"{emotion:12} : {count:6} ({percentage:5.1f}%) {bar}")
                print("="*60 + "\n")
    
    def detect_from_image(self, image_path, output_path=None):
        """Detect emotions from static image"""
        print(f"\n📷 Loading image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"❌ Error reading image")
            return
        
        # Detect emotions
        frame, emotions_data = self.detect_emotions(frame)
        
        # Save output if provided
        if output_path:
            cv2.imwrite(output_path, frame)
            print(f"✅ Output saved: {output_path}")
        
        # Display
        cv2.imshow('Emotion Detection Result', frame)
        
        # Print results
        print("\n" + "="*60)
        print("  😊 DETECTED EMOTIONS")
        print("="*60)
        for i, data in enumerate(emotions_data, 1):
            print(f"\nFace #{i}:")
            print(f"  Main Emotion: {data['emotion']} ({data['confidence']:.2%})")
            print(f"  All Predictions:")
            for emotion, prob in sorted(data['predictions'].items(), 
                                       key=lambda x: x[1], reverse=True):
                bar = "▓" * int(prob * 20)
                print(f"    {emotion:12} : {prob:.2%} {bar}")
        print("="*60 + "\n")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """Main menu"""
    print("\n" + "="*70)
    print("  🎭 FACIAL EMOTION DETECTION SYSTEM")
    print("  Maximum Accuracy Model (80-90% for all emotions)")
    print("="*70)
    
    try:
        detector = EmotionDetector('models/emotion_model.h5')
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    while True:
        print("\n📋 MAIN MENU:")
        print("1. 🎥 Run Webcam Detection (Real-time)")
        print("2. 📷 Detect from Image File")
        print("3. ℹ️  Model Information")
        print("4. 🚪 Exit")
        
        choice = input("\n👉 Select option (1-4): ").strip()
        
        if choice == '1':
            detector.run_webcam(0)
        
        elif choice == '2':
            image_path = input("\n📂 Enter image path: ").strip()
            output_path = input("💾 Enter output path (or press Enter to skip): ").strip()
            detector.detect_from_image(
                image_path,
                output_path if output_path else None
            )
        
        elif choice == '3':
            if detector.training_info:
                print("\n" + "="*70)
                print("  📊 MODEL INFORMATION")
                print("="*70)
                info = detector.training_info
                print(f"\n✅ Model Performance:")
                print(f"   Training Accuracy: {info['final_train_accuracy']:.2%}")
                print(f"   Validation Accuracy: {info['final_val_accuracy']:.2%}")
                print(f"   Training Loss: {info['final_train_loss']:.4f}")
                print(f"   Validation Loss: {info['final_val_loss']:.4f}")
                print(f"   Epochs Trained: {info['epochs_trained']}")
                print(f"\n🎭 Emotions Detected: {len(info['emotions'])}")
                for emotion in info['emotions']:
                    print(f"   ✓ {emotion}")
                print("="*70)
            else:
                print("⚠️  Training info not available")
        
        elif choice == '4':
            print("\n👋 Thank you for using Facial Emotion Detection System!")
            print("Goodbye! 👋\n")
            break
        
        else:
            print("❌ Invalid option. Please select 1-4.")

if __name__ == "__main__":
    main()