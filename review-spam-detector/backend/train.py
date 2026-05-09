import pandas as pd
import numpy as np
import pickle
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPool1D, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
def clean_text(text):
    """Text ko properly clean karein"""
    text = str(text).lower()  
    text = re.sub(r'http\S+|www\S+', '', text)  
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip() 
    return text
print("📂 Loading dataset...")
df = pd.read_csv('fake reviews dataset.csv')

print(f"Total reviews: {len(df)}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

print("\n🧹 Cleaning text data...")
df['cleaned_text'] = df['text_'].apply(clean_text)

df = df[df['cleaned_text'].str.len() > 10]  
print(f"After cleaning: {len(df)} reviews")


df['label_num'] = df['label'].map({'CG': 1, 'OR': 0})
max_words = 15000  
max_len = 150  

print("\n🔤 Tokenizing text...")
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_text'])

X = pad_sequences(tokenizer.texts_to_sequences(df['cleaned_text']), maxlen=max_len, padding='post', truncating='post')
y = df['label_num'].values

print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Sequence shape: {X.shape}")
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = {0: weights[0], 1: weights[1]}

print(f"\nClass weights: {class_weights}")

# Train-Test Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)}")
print(f"Testing set: {len(X_test)}")
print(f"Train class distribution: Real={sum(y_train==0)}, Fake={sum(y_train==1)}")
print("\n🏗️ Building model...")
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    GlobalMaxPool1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    epochs=30, 
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
print("\n📊 Evaluating model...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=['Real (OR)', 'Fake (CG)']))

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives (Real as Real): {cm[0][0]}")
print(f"False Positives (Real as Fake): {cm[0][1]}")
print(f"False Negatives (Fake as Real): {cm[1][0]}")  
print(f"True Positives (Fake as Fake): {cm[1][1]}")


model.save('spam_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('config.pickle', 'wb') as f:
    pickle.dump({'max_len': max_len}, f)

print("\n✅ Model, Tokenizer, and Config saved successfully!")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("📈 Training plots saved as 'training_history.png'")
