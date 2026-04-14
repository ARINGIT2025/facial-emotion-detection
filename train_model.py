import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

print("\n" + "="*70)
print("  🚨 EMERGENCY FIX")
print("="*70 + "\n")

# ============ MANUAL DATA LOADING ============
print("📂 Loading data (MANUAL)...\n")

data_dir = 'data/train'
emotions = sorted(os.listdir(data_dir))
emotions = [e for e in emotions if os.path.isdir(os.path.join(data_dir, e))]

print(f"Emotions found: {emotions}\n")

X_train_list = []
y_train_list = []

for emotion_id, emotion in enumerate(emotions):
    emotion_path = os.path.join(data_dir, emotion)
    img_files = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Loading {emotion}... ", end="", flush=True)
    
    count = 0
    for img_file in img_files:
        try:
            img_path = os.path.join(emotion_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            img = cv2.resize(img, (48, 48))
            X_train_list.append(img)
            y_train_list.append(emotion_id)
            count += 1
        except:
            pass
    
    print(f"✅ {count}")

print(f"\n✅ Total: {len(X_train_list)} images\n")

# ============ CONVERT TO NUMPY ============
X = np.array(X_train_list, dtype='float32')
y = np.array(y_train_list, dtype='int32')

print(f"Shape: {X.shape}")
print(f"Labels: {np.unique(y)}\n")

# ============ NORMALIZE ============
X = X / 255.0
X = np.expand_dims(X, axis=-1)

print(f"Normalized shape: {X.shape}\n")

# ============ SPLIT ============
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ ONE-HOT ENCODE
y_train = keras.utils.to_categorical(y_train, 7)
y_val = keras.utils.to_categorical(y_val, 7)

print(f"Train shape: {X_train.shape}, {y_train.shape}")
print(f"Val shape: {X_val.shape}, {y_val.shape}\n")

# ============ MODEL ============
print("🏗️  Building model...\n")

model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(48, 48, 1)),
    keras.layers.MaxPooling2D(2),
    
    keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    
    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model ready\n")

# ============ TRAIN ============
print("🚀 TRAINING...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

# ============ SAVE ============
print("\n💾 Saving...")
model.save('models/emotion_model.h5')
print("✅ Done")

# ============ RESULTS ============
print("\n" + "="*70)
print(f"Final Train Accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.2%}")
print("="*70)