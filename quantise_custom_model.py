import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
# 1. Path to your saved custom .keras model file
keras_model_path = "asr_model_best.keras" 

# 2. Path where you want to save the quantized TFLite model
tflite_model_path = "asr_model_quantized_int8.tflite"

# 3. Parameters needed for the representative dataset function
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
# Path to a small subset of your training audio files (e.g., 100-200 files)
# You need the actual audio files, not just the paths list from load_data
REPRESENTATIVE_DATA_DIR = "/path/to/some/training/audio/files/" 
NUM_CALIBRATION_STEPS = 100 # Number of samples to use for calibration
# ---

# --- Load the necessary functions from your training/inference script ---

# Include power_to_db and preprocess_audio_tf here
def power_to_db(S, ref=1.0, top_db=80.0):
    # ... (same function as in your inference script) ...
    log_spec = 10.0 * (tf.math.log(tf.maximum(S, 1e-10)) / tf.math.log(10.0))
    log_spec -= 10.0 * (tf.math.log(tf.maximum(ref, 1e-10)) / tf.math.log(10.0))
    return tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

@tf.function
def preprocess_audio_tf(file_path: tf.Tensor):
    # ... (same function as in your inference script, ensure it reads WAV) ...
    try:
        audio_binary = tf.io.read_file(file_path)
        audio_tensor, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
        waveform = tf.squeeze(audio_tensor, axis=-1)
        padding = N_FFT // 2
        waveform = tf.pad(waveform, [[padding, padding]], mode="REFLECT")
        stft = tf.signal.stft(waveform, frame_length=N_FFT, frame_step=HOP_LENGTH, fft_length=N_FFT)
        spectrogram = tf.abs(stft)
        num_spectrogram_bins = stft.shape[-1]
        mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=N_MELS, num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=SAMPLE_RATE, lower_edge_hz=20.0, upper_edge_hz=8000.0
        )
        mel_spectrogram = tf.tensordot(spectrogram, mel_filterbank, 1)
        log_mel_spectrogram = power_to_db(mel_spectrogram)
        log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, axis=-1)
        return tf.cast(log_mel_spectrogram, dtype=tf.float32)
    except Exception as e:
        tf.print("Error processing file for calibration:", file_path, "Exception:", e, summarize=-1)
        # Return a dummy tensor for calibration errors, adjust shape if needed
        return tf.zeros((100, N_MELS, 1), dtype=tf.float32)

# --- Load the Keras Model ---
print(f"Loading Keras model from {keras_model_path}...")
# We need the custom loss function to load the compiled model
def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = tf.math.count_nonzero(y_true, axis=1, keepdims=True, dtype="int64")
    loss = tf.keras.backend.ctc_batch_cost( y_true, y_pred, input_length, label_length )
    return loss

model = tf.keras.models.load_model(keras_model_path, custom_objects={"ctc_loss": ctc_loss})
print("Model loaded successfully.")

# --- Prepare Representative Dataset ---
print("Preparing representative dataset for quantization calibration...")
# Get a list of audio file paths from your sample directory
representative_audio_files = [
    os.path.join(REPRESENTATIVE_DATA_DIR, fname) 
    for fname in os.listdir(REPRESENTATIVE_DATA_DIR) 
    if fname.lower().endswith(('.wav', '.flac')) # Adjust extensions if needed
][:NUM_CALIBRATION_STEPS]

if not representative_audio_files:
    raise ValueError(f"No audio files found in {REPRESENTATIVE_DATA_DIR}. Cannot perform quantization.")

def representative_dataset_gen():
  for file_path in representative_audio_files:
    # Preprocess the audio file to get the spectrogram
    spectrogram = preprocess_audio_tf(tf.constant(file_path))
    # Add batch dimension and yield
    yield [tf.expand_dims(spectrogram, axis=0)]

# --- Convert to TFLite with INT8 Quantization ---
print("Starting TFLite conversion and quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
# Ensure that if ops can't be quantized, the converter throws an error.
# Use this for strict INT8 deployment. Remove if float fallback is acceptable.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set input and output types to int8
# Note: Check model input/output dtypes if conversion fails
# converter.inference_input_type = tf.int8 # Use tf.float32 if model input remains float
# converter.inference_output_type = tf.int8 # Use tf.float32 if model output remains float

# Convert the model
tflite_quant_model = converter.convert()

# --- Save the Quantized Model ---
print(f"Saving quantized TFLite model to {tflite_model_path}...")
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print(f"\n✅ Quantized model saved successfully to {tflite_model_path}")
print(f"Original model size: {os.path.getsize(keras_model_path) / (1024*1024):.2f} MB")
print(f"Quantized model size: {os.path.getsize(tflite_model_path) / (1024*1024):.2f} MB")
