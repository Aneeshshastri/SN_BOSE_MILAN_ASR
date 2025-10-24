# Model Card: Custom CRNN ASR for LibriSpeech (Augmented Training)

## Model Details

* **Model Type:** End-to-End Automatic Speech Recognition (ASR)
* **Architecture:** Custom Convolutional Recurrent Neural Network (CRNN):
  * CNN Frontend: 4 `Conv2D` layers (32, 32, 64, 64 filters) with `BatchNormalization`, `ReLU`, `MaxPooling2D`, and `Dropout`.
  * RNN Backend: 4 `Bidirectional LSTM` layers (512 units each) with `BatchNormalization` and `Dropout`.
  * Output: `Dense` layer with `softmax` activation.
* **Framework:** TensorFlow / Keras
* **Loss Function:** Connectionist Temporal Classification (CTC)
* **Number of Parameters:** *7,961,854 (30.37 MB)*
* **Language:** English (en)
* **Repository:** <https://github.com/Aneeshshastri/SN_BOSE_MILAN_ASR>

## Intended Use

* **Primary Use:** Transcribing English speech audio files, particularly those exhibiting noise characteristics similar to the augmentations applied during training (e.g., background noise, reverb). Designed for robustness against common audio distortions.
* **Secondary Use:** Educational tool for understanding ASR model development from scratch, specifically focusing on data augmentation techniques for noise robustness and quantization.
* **Out-of-Scope Uses:**  transcription of languages other than English, transcription of specialized domains (e.g., medical, legal) without fine-tuning, transcription of *perfectly clean* studio recordings (as it wasn't explicitly trained only on clean data). Note that the model is still not good enough to transcribe large recordings efficiently as it only uses a greedy decoder and was only trained on 100h of speech with 35 epochs. This model is certainly not nearly ready for production.

## Training Data

* **Dataset:** As given during the competition (\~100 hours). [Link](https://drive.google.com/drive/folders/1uJC-VvEDiB5nm8-q04-fKrl3SO67ZEnf)
* **Speakers:** \~250 speakers reading English literature.
* **Preprocessing:** Original audio converted from FLAC to WAV, resampled to 16kHz. Log-Mel spectrograms (80 bins) generated using `tf.signal`.
* **Augmentation Strategy:** Trained **primarily on augmented data**. The custom `Augmenter` class was applied probabilistically to **all raw audio waveforms during training** to simulate noisy conditions:
  * Gaussian Noise
  * Reverb
  * Time Stretching (`librosa`)
  * Frequency Masking (Butterworth band-stop filter via `scipy`)
  * *Note: The goal was to build inherent robustness by training on varied, noisy examples.*

## Training Procedure

* **Framework:** TensorFlow 2.18.x
* **Data Pipeline:** `tf.data` used for loading, mapping (augmentation + preprocessing), filtering (by sequence length), shuffling (buffer size 1024), batching (size 32), and prefetching.
* **Optimizer:** Adam (`tf.keras.optimizers.Adam`)
* **Learning Rate Schedule:** Initial rate (e.g., 1e-3 or 1e-4) managed by `tf.keras.callbacks.ReduceLROnPlateau` monitoring `val_loss` (factor=0.5, patience=1 or 2, min_lr=1e-6). Manual adjustments (e.g., to 1e-5) were made during later stages.
* **Epochs:** Trained for approximately 35 epochs.
* **Hardware:** Kaggle Notebook (GPU -P100).
* **Callbacks:** `ModelCheckpoint` (saving best based on `val_loss`), `ReduceLROnPlateau`.

## Evaluation

* **Evaluation Data:** 10% of the dataset was reserved for vaildation.
* **Metrics:**
  * CTC Loss (Validation): Reached a minimum value around **\~103**. While far from perfect it does demonstrate the model's ability to identify patterns.
  * Word Error Rate (WER): **Not explicitly calculated** during training. Based on the validation loss and the augmented training strategy, WER on `dev-clean` might be slightly higher than if trained only on clean data (perhaps **18-30%** range estimate), but performance on noisy test sets (like `test-other` or custom noisy data) should be comparatively better. Requires explicit calculation.

## Model Limitations

* **Limited Vocabulary:** Trained only on uppercase English letters, apostrophe, and space. Cannot produce numbers, punctuation, or lowercase letters.
* **Domain Specificity:** Trained on augmented audiobook recordings. Performance may still degrade on:
  * Noise types significantly different from the applied augmentations.
  * Conversational speech (different speaking style, accents, overlaps).
  * Different domains (e.g., meetings, phone calls, technical jargon).
* **Out-of-Vocabulary Words:** As a character-based model, it attempts to spell unknown words, but accuracy may be low.
* **No Speaker Information:** Does not perform speaker diarization or identification.

## Bias, Risks, and Limitations

* **Demographic Bias:** Inherits potential biases from LibriSpeech (speaker demographics, accents). May perform differently for various groups.
* **Robustness:** Designed for improved noise robustness due to augmented training, but performance will still degrade in extreme noise or conditions very different from the augmentations. May perform slightly worse on *perfectly clean* data compared to a model trained only on clean data.
* **Error Types:** Likely to make phonetic errors, omit small function words, or misinterpret less common words. Not suitable for high-stakes applications requiring perfect accuracy without significant post-processing or fine-tuning.

## Recommendations for Use

* Best suited for transcribing English speech audio exhibiting moderate noise similar to the augmentations used (background noise, reverb).
* Expect reasonable performance but not perfect accuracy; implement post-processing or human review for critical applications.
* Consider fine-tuning on domain-specific data (clean or noisy) for better performance in specific use cases.
* For higher accuracy, especially on cleaner data or for better grammatical structure, integrate with an external language model using CTC beam search decoding.
* The quantized TFLite version is suitable for CPU-bound applications where efficiency is prioritized.

## How to Use

*(Include code snippets based on your inference scripts)*

**For Keras Model (`.keras`):**

```python
import tensorflow as tf
import librosa
# ... [Include CHARACTERS, num_to_char, power_to_db, preprocess_audio_tf, ctc_loss] ...

# Load model
model = tf.keras.models.load_model("asr_model_best.keras", custom_objects={"ctc_loss": ctc_loss})

# Preprocess audio
audio_file = "path/to/audio.wav"
spectrogram = preprocess_audio_tf(tf.constant(audio_file))
input_batch = tf.expand_dims(spectrogram, axis=0)

# Predict
predictions = model.predict(input_batch)
logits = predictions[0]

# Decode (Greedy Example)
predicted_ids = tf.argmax(logits, axis=-1)
# ... [Add greedy CTC decoding logic: unique -> remove blank -> num_to_char -> join] ...
transcription = "..." 
print(transcription)
