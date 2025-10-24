import tensorflow as tf
import librosa
import numpy as np
import os

# --- Configuration ---
# 1. Path to your saved custom .keras model file
keras_model_path = "C:/User/Aneesh Shastri/Downloads/asr_model_final_ep25.keras" # Make sure this is the correct path

# 2. Path to the audio file you want to transcribe
audio_file_path = "C:/Users/Aneesh Shastri/Downloads/19-198-0000.wav" 

# 3. Parameters used during training 
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

# 4. Vocabulary used during training 
CHARACTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    "'",' '
]
# ---

# --- Recreate Mappings and Preprocessing ---

# Create character-to-number mappings (same as in training)
char_to_num = tf.keras.layers.StringLookup(vocabulary=list(CHARACTERS), mask_token=None)
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
VOCAB_SIZE = char_to_num.vocabulary_size()

# Include the power_to_db function used during training
def power_to_db(S, ref=1.0, top_db=80.0):
    log_spec = 10.0 * (tf.math.log(tf.maximum(S, 1e-10)) / tf.math.log(10.0))
    log_spec -= 10.0 * (tf.math.log(tf.maximum(ref, 1e-10)) / tf.math.log(10.0))
    return tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

# Include the preprocessing function used during training (adapted for inference)
@tf.function
def preprocess_audio_tf(file_path: tf.Tensor):
    """
    Loads and converts an audio file to a log Mel spectrogram using TensorFlow.
    (Inference version: no augmentation)
    """
    try:
        audio_binary = tf.io.read_file(file_path)
        # Assuming WAV files after conversion
        audio_tensor, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
        waveform = tf.squeeze(audio_tensor, axis=-1) # Already float32 in [-1, 1]

        # Pad waveform like librosa
        padding = N_FFT // 2
        waveform = tf.pad(waveform, [[padding, padding]], mode="REFLECT")
        
        # Compute STFT
        stft = tf.signal.stft(
            waveform, frame_length=N_FFT, frame_step=HOP_LENGTH, fft_length=N_FFT
        )
        spectrogram = tf.abs(stft)

        # Convert to Mel Spectrogram
        num_spectrogram_bins = stft.shape[-1]
        mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=N_MELS, num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=SAMPLE_RATE, lower_edge_hz=20.0, upper_edge_hz=8000.0
        )
        mel_spectrogram = tf.tensordot(spectrogram, mel_filterbank, 1)
        
        # Convert to log scale
        log_mel_spectrogram = power_to_db(mel_spectrogram)
        
        # Add channel dimension
        log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, axis=-1)

        return tf.cast(log_mel_spectrogram, dtype=tf.float32)

    except Exception as e:
        # In inference, we might want to raise the error or return None
        tf.print("Error processing file:", file_path, "Exception:", e, summarize=-1)
        # Return a shape compatible tensor filled with zeros or handle error differently
        return tf.zeros((100, N_MELS, 1), dtype=tf.float32) 

# --- Load Custom Keras Model ---
print(f"Loading Keras model from {keras_model_path}...")
# We need to provide the custom CTC loss function when loading
def ctc_loss(y_true, y_pred):
     # Include the exact ctc_loss function definition used during training here
     # Make sure it uses count_nonzero(y_true, ...) correctly
    batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = tf.math.count_nonzero(y_true, axis=1, keepdims=True, dtype="int64") # Corrected version
    loss = tf.keras.backend.ctc_batch_cost( y_true, y_pred, input_length, label_length )
    return loss

# Load the model, providing the custom loss function
model = tf.keras.models.load_model(keras_model_path, custom_objects={"ctc_loss": ctc_loss})
print("Custom model loaded successfully.")
model.summary() # Optional: print model structure

# --- Load and Prepare Audio ---
print(f"\nLoading and preprocessing audio file: {audio_file_path}")
# Preprocess the audio file to get the log-Mel spectrogram
# We wrap the file path in a tf.constant
spectrogram = preprocess_audio_tf(tf.constant(audio_file_path))

# Add a batch dimension (model expects batch_size, time, freq, channels)
spectrogram = tf.expand_dims(spectrogram, axis=0)
print(f"Spectrogram shape: {spectrogram.shape}")

# --- Run Inference ---
print("\nRunning inference...")

# Get model predictions (logits) using model.predict for clarity
# Using predict is slightly simpler than defining a tf.function here
predictions = model.predict(spectrogram)
logits = predictions[0] # Remove the batch dimension

# --- Decode the Logits (Greedy CTC Decoding) ---
print("Decoding...")

# Get the most likely token IDs at each time step
predicted_ids = tf.argmax(logits, axis=-1)

# Remove consecutive duplicates and blank token
# A simple greedy decode might just remove duplicates:
decoded_ids = []
last_id = -1
for token_id in predicted_ids.numpy():
    if token_id != last_id:
        if token_id < VOCAB_SIZE: # Avoid adding padding/blank index if it exists
             decoded_ids.append(token_id)
    last_id = token_id
    
# Convert token IDs back to characters
decoded_chars = num_to_char(tf.constant(decoded_ids, dtype=tf.int64))

# Join characters into a string
transcription = tf.strings.reduce_join(decoded_chars).numpy().decode("utf-8")

# --- Print the Result ---
print("\n--- Transcription ---")
print(transcription)
