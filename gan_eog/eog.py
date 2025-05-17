import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# --- 1. Data Loading and Preprocessing ---

folder_path = 'class/3-class'

def map_filename_to_label(name):
    name_low = name.casefold()
    if "yukar" in name_low:
        return "up"
    elif "asagi" in name_low:
        return "down"
    elif "sag" in name_low:
        return "right"
    elif "sol" in name_low:
        return "left"
    elif "kirp" in name_low:
        return "blink"
    else:
        return None

files = [f for f in glob.glob(folder_path + '/*.txt') 
         if all(x not in f.lower() for x in ['örnek', 'edit', 'serkan', 'kayit'])]

data_list = []
labels = []

for f in files:
    label = map_filename_to_label(f)
    if label is None:
        continue
    signal = np.loadtxt(f)
    if len(signal.shape) > 1:
        signal = signal.flatten()
    data_list.append(signal)
    labels.append(label)

data_array = np.array(data_list)
labels = np.array(labels)
print(f"Loaded {len(data_array)} signals with labels: {set(labels)}")

# --- Preprocessing: Bandpass + MinMaxScaler to [-1, 1] ---

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

fs = 176  # Sampling frequency

def preprocess_signal(signal):
    filtered = butter_bandpass_filter(signal, 0.5, 20.0, fs, order=2)
    filtered = filtered - np.mean(filtered)  # Remove DC
    scaler = MinMaxScaler(feature_range=(-1, 1))
    filtered = scaler.fit_transform(filtered.reshape(-1,1)).flatten()
    return filtered

processed_data = np.array([preprocess_signal(sig) for sig in data_array])
print(f"Shape of preprocessed data: {processed_data.shape}")

# --- Label Encoding ---
le = LabelEncoder()
label_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)
print(f"Classes: {list(le.classes_)}")

signal_length = processed_data.shape[1]

# --- 2. Data Augmentation & Instance Noise ---

def augment_signal(signal):
    noise = np.random.normal(0, 0.08, signal.shape)
    scale = np.random.uniform(0.97, 1.03)
    return (signal * scale) + noise

def add_instance_noise(signal, stddev=0.08):
    return signal + np.random.normal(0, stddev, signal.shape)

# --- 3. Model Definitions (improved regularization and normalization) ---

def make_generator(latent_dim, num_classes, output_dim):
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(1,), dtype='int32')
    label_embedding = layers.Embedding(num_classes, latent_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    model_input = layers.multiply([noise_input, label_embedding])

    x = layers.Dense(256, kernel_initializer='he_normal')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(512, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(1024, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(output_dim, activation='tanh')(x)

    generator = Model([noise_input, label_input], x, name="generator")
    return generator

def make_discriminator(input_dim, num_classes):
    signal_input = layers.Input(shape=(input_dim,))
    label_input = layers.Input(shape=(1,), dtype='int32')
    label_embedding = layers.Embedding(num_classes, input_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    x = layers.Concatenate()([signal_input, label_embedding])
    x = layers.GaussianNoise(0.15)(x)
    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, kernel_initializer='he_normal')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    discriminator = Model([signal_input, label_input], x, name="discriminator")
    return discriminator

# --- 4. Training Loop with tf.GradientTape (improved stability) ---

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(5e-5, beta_1=0.5)

@tf.function
def train_step(real_signal, real_label, generator, discriminator, latent_dim, num_classes):
    BATCH_SIZE = real_signal.shape[0]
    noise = tf.random.normal([BATCH_SIZE, latent_dim])
    random_labels = tf.random.uniform([BATCH_SIZE, 1], minval=0, maxval=num_classes, dtype=tf.int32)

    # One-sided label smoothing (no noise for fake)
    real_y = tf.ones((BATCH_SIZE, 1)) * 0.9
    fake_y = tf.zeros((BATCH_SIZE, 1))

    # Add instance noise to real and fake
    real_signal_noisy = real_signal + tf.random.normal(tf.shape(real_signal), mean=0.0, stddev=0.08)
    fake_signal = generator([noise, random_labels], training=True)
    fake_signal_noisy = fake_signal + tf.random.normal(tf.shape(fake_signal), mean=0.0, stddev=0.08)

    # Train D on real and fake separately for stability
    with tf.GradientTape() as disc_tape:
        real_output = discriminator([real_signal_noisy, real_label], training=True)
        fake_output = discriminator([fake_signal_noisy, random_labels], training=True)
        disc_loss_real = cross_entropy(real_y, real_output)
        disc_loss_fake = cross_entropy(fake_y, fake_output)
        disc_loss = disc_loss_real + disc_loss_fake

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Train G
    noise = tf.random.normal([BATCH_SIZE, latent_dim])
    random_labels = tf.random.uniform([BATCH_SIZE, 1], minval=0, maxval=num_classes, dtype=tf.int32)
    with tf.GradientTape() as gen_tape:
        fake_signal = generator([noise, random_labels], training=True)
        fake_signal_noisy = fake_signal + tf.random.normal(tf.shape(fake_signal), mean=0.0, stddev=0.08)
        fake_output = discriminator([fake_signal_noisy, random_labels], training=True)
        gen_loss = cross_entropy(tf.ones((BATCH_SIZE, 1)), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return disc_loss, gen_loss

def train_gan(generator, discriminator, data, labels, latent_dim, num_classes, epochs=2500, batch_size=128):
    for epoch in range(epochs):
        idx = np.random.permutation(data.shape[0])
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i+batch_size]
            real_signals = data[batch_idx]
            real_labels = labels[batch_idx].reshape(-1, 1)
            real_signals = np.array([augment_signal(sig) for sig in real_signals])
            real_signals = np.array([add_instance_noise(sig) for sig in real_signals])
            disc_loss, gen_loss = train_step(
                tf.convert_to_tensor(real_signals, dtype=tf.float32),
                tf.convert_to_tensor(real_labels, dtype=tf.int32),
                generator, discriminator, latent_dim, num_classes
            )
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"{epoch+1}/{epochs} [D loss: {disc_loss:.4f}] [G loss: {gen_loss:.4f}]")

# --- 5. Model Creation and Training ---

latent_dim = 100
generator = make_generator(latent_dim, num_classes, signal_length)
discriminator = make_discriminator(signal_length, num_classes)

train_gan(generator, discriminator, processed_data, label_encoded, latent_dim, num_classes, epochs=2500, batch_size=128)

# --- 6. t-SNE Visualization ---

from sklearn.manifold import TSNE

def plot_tsne(generator, processed_data, label_encoded, le, latent_dim, num_classes, num_per_label=100):
    all_signals = []
    all_labels = []
    all_types = []

    for label_name in le.classes_:
        label_idx = le.transform([label_name])[0]
        # Real samples
        real_idx = np.where(label_encoded == label_idx)[0]
        real_samples = processed_data[real_idx[:num_per_label]]
        all_signals.append(real_samples)
        all_labels += [label_name] * len(real_samples)
        all_types += ['real'] * len(real_samples)
        # Generated samples
        noise = np.random.normal(0, 1, (num_per_label, latent_dim))
        gen_labels = np.full((num_per_label, 1), label_idx)
        fake_samples = generator.predict([noise, gen_labels])
        all_signals.append(fake_samples)
        all_labels += [label_name] * len(fake_samples)
        all_types += ['fake'] * len(fake_samples)

    all_signals = np.vstack(all_signals)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings = tsne.fit_transform(all_signals)

    plt.figure(figsize=(10, 7))
    for label in le.classes_:
        for typ, marker in zip(['real', 'fake'], ['o', 'x']):
            idx = [i for i, (l, t) in enumerate(zip(all_labels, all_types)) if l == label and t == typ]
            plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f"{label} {typ}", alpha=0.6, marker=marker)
    plt.legend()
    plt.title("t-SNE of Real and Generated EOG Signals")
    plt.show()

# plot_tsne(generator, processed_data, label_encoded, le, latent_dim, num_classes)

# --- 7. Save Generator for TensorFlow.js ---

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(generator, 'tfjs_generator_model')

# --- 8. Generate and Plot Multiple Samples Per Label ---

def generate_sample(generator, label_name, latent_dim=100):
    label_idx = le.transform([label_name])[0]
    noise = np.random.normal(0, 1, (1, latent_dim))
    gen_signal = generator.predict([noise, np.array([[label_idx]])])
    return gen_signal.flatten()

for label in le.classes_:
    plt.figure(figsize=(8, 3))
    for _ in range(5):
        plt.plot(generate_sample(generator, label), alpha=0.7)
    plt.title(f"Generated EOG Signals for '{label}'")
    plt.show()