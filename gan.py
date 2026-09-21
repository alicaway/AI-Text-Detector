"""
Simple GAN using TensorFlow to learn from AI_Human.parquet.
Character-level GAN that learns text patterns from AI-generated and human-written texts.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle

# Configuration
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AI_Human.parquet")
MAX_TEXT_LEN = 500
EMBEDDING_DIM = 128
LATENT_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0002
BETA1 = 0.5
SAMPLE_INTERVAL = 5
CHAR_SAMPLES = 20000
CHECKPOINT_DIR = "gan_checkpoints"

print(f"TensorFlow version: {tf.__version__}")

# 1. Load and Preprocess Data
print("\n[1/5] Loading AI_Human.parquet ...")
df = pd.read_parquet(DATA_PATH)
print(f"   Dataset shape: {df.shape}")
print(f"   Generated distribution: {df['generated'].value_counts().to_dict()}")

print("[1/5] Building character vocabulary ...")
all_texts = df["text"].astype(str).tolist()
if len(all_texts) > CHAR_SAMPLES:
    np.random.seed(42)
    indices = np.random.choice(len(all_texts), CHAR_SAMPLES, replace=False)
    sampled_texts = [all_texts[i] for i in indices]
else:
    sampled_texts = all_texts

chars = set()
for text in sampled_texts:
    chars.update(text)
chars = sorted(chars)
vocab_size = len(chars)
char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
idx_to_char = {i + 1: c for i, c in enumerate(chars)}
idx_to_char[0] = '\0'
print(f"   Vocabulary size: {vocab_size}")
print(f"   Number of texts used: {len(sampled_texts)}")

def encode_texts(texts, max_len):
    sequences = []
    for text in texts:
        encoded = [char_to_idx.get(c, 0) for c in text]
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            encoded = encoded + [0] * (max_len - len(encoded))
        sequences.append(encoded)
    return np.array(sequences, dtype=np.int32)

print("[1/5] Encoding texts ...")
encoded_texts = encode_texts(sampled_texts, MAX_TEXT_LEN)
print(f"   Encoded shape: {encoded_texts.shape}")

def to_one_hot(sequences, vocab_size):
    batch_size, seq_len = sequences.shape
    one_hot = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)
    for i in range(batch_size):
        for j in range(seq_len):
            idx = sequences[i, j]
            if idx > 0:
                one_hot[i, j, idx - 1] = 1.0
    return one_hot

# 2. Build the Generator
print("\n[2/5] Building the Generator ...")
generator = keras.Sequential([
    layers.Dense(HIDDEN_DIM, activation='relu', input_dim=LATENT_DIM),
    layers.BatchNormalization(),
    layers.Dense(HIDDEN_DIM * 2, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(MAX_TEXT_LEN * EMBEDDING_DIM, activation='relu'),
    layers.Reshape((MAX_TEXT_LEN, EMBEDDING_DIM)),
    layers.LSTM(EMBEDDING_DIM, return_sequences=True),
    layers.Dense(vocab_size, activation='softmax')
])
generator.summary()

# 3. Build the Discriminator
print("\n[3/5] Building the Discriminator ...")
discriminator = keras.Sequential([
    layers.Flatten(input_shape=(MAX_TEXT_LEN, vocab_size)),
    layers.Dense(HIDDEN_DIM, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(HIDDEN_DIM // 2, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
discriminator.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
discriminator.summary()

# 4. Build Combined GAN
print("\n[4/5] Building the Combined GAN Model ...")
discriminator.trainable = False
gan_input = keras.Input(shape=(LATENT_DIM,))
generated_text = generator(gan_input)
gan_output = discriminator(generated_text)
gan = keras.Model(gan_input, gan_output)
gan.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA1),
    loss='binary_crossentropy'
)
print("GAN model compiled successfully.")

# 5. Training Loop
print(f"\n[5/5] Starting Training for {EPOCHS} epochs ...")
real_labels = np.ones((BATCH_SIZE, 1)) * 0.9
fake_labels = np.zeros((BATCH_SIZE, 1))
g_losses, d_losses, d_accs = [], [], []
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def generate_real_samples(batch_size):
    idx = np.random.randint(0, encoded_texts.shape[0], batch_size)
    return to_one_hot(encoded_texts[idx], vocab_size)

best_d_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    # Train Discriminator with real
    real_data = generate_real_samples(BATCH_SIZE)
    d_loss_real = discriminator.train_on_batch(real_data, real_labels)

    # Train Discriminator with fake
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    fake_data = generator.predict(noise, verbose=0)
    d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    g_loss = gan.train_on_batch(noise, real_labels)

    g_losses.append(g_loss)
    d_losses.append(d_loss[0])
    d_accs.append(d_loss[1])

    if epoch % SAMPLE_INTERVAL == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{EPOCHS} | D Loss: {d_loss[0]:.4f} | D Acc: {d_loss[1]:.4f} | G Loss: {g_loss:.4f}")
        sample_noise = np.random.normal(0, 1, (1, LATENT_DIM))
        sample_gen = generator.predict(sample_noise, verbose=0)
        sample_indices = np.argmax(sample_gen, axis=-1)[0]
        sample_text = ''.join([idx_to_char.get(idx, '?') for idx in sample_indices])
        sample_text = sample_text.rstrip('\0')[:200]
        print(f"   Generated sample:\n   \"{sample_text}\"\n")

    if d_loss[0] < best_d_loss:
        best_d_loss = d_loss[0]
        generator.save(os.path.join(CHECKPOINT_DIR, "generator_best.h5"))
        discriminator.save(os.path.join(CHECKPOINT_DIR, "discriminator_best.h5"))

# 6. Save Final Model and Training History
print("\n[Done] Saving final models and training history ...")
generator.save(os.path.join(CHECKPOINT_DIR, "generator_final.h5"))
discriminator.save(os.path.join(CHECKPOINT_DIR, "discriminator_final.h5"))
with open(os.path.join(CHECKPOINT_DIR, "training_history.pkl"), "wb") as f:
    pickle.dump({
        "g_losses": g_losses,
        "d_losses": d_losses,
        "d_accs": d_accs,
        "config": {
            "MAX_TEXT_LEN": MAX_TEXT_LEN,
            "EMBEDDING_DIM": EMBEDDING_DIM,
            "LATENT_DIM": LATENT_DIM,
            "vocab_size": vocab_size,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
        }
    }, f)

print("\n" + "=" * 60)
print("GAN Training Complete!")
print("=" * 60)
print(f"   Vocabulary size: {vocab_size}")
print(f"   Final G Loss: {g_losses[-1]:.4f}")
print(f"   Final D Loss: {d_losses[-1]:.4f}")
print(f"   Final D Accuracy: {d_accs[-1]:.4f}")
print(f"\nModels saved to '{CHECKPOINT_DIR}/'")
