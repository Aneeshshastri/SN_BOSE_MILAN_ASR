# Custom End-to-End Speech Recognition For Milan25
### By SN BOSE Team Name: (Aneesh Shastri, Srijan Maity) Roll No: (CO25BTECH11004,CO25BTECH11026)

## Project Overview

This project details the creation of a custom end-to-end Automatic Speech Recognition (ASR) system using TensorFlow and Keras. Starting with the clean LibriSpeech dataset, the system incorporates custom data augmentation, trains a bespoke Convolutional Recurrent Neural Network (CRNN) model with CTC loss, and finally quantizes the model for efficient CPU inference using TensorFlow Lite (TFLite). The entire process, including challenges faced and solutions implemented, is documented here.

## Table of Contents

* [1. Dataset & Custom Augmentation](#1-dataset--custom-augmentation)
* [2. Model Design & Architecture](#2-model-design--architecture)
* [3. Training Process](#3-training-process)
* [4. Post-Training Quantization](#4-post-training-quantization)
* [5. Installation & Usage](#5-installation--usage)
* [6. Challenges & Learnings](#6-challenges--learnings)
* [7. Results & Future Work](#7-results--future-work)
* [8. License](#8-license)

## 1. Dataset & Custom Augmentation

* **Dataset:** LibriSpeech `train-clean-100` (~100 hours). Original `.flac` files were converted to `.wav` for broader compatibility within the processing pipeline (as seen in `model-trainer.ipynb`)
* **Goal:** Improve model robustness to real-world conditions by augmenting the clean training data.
* **`Augmenter` Class:** A custom Python class was created to apply various distortions probabilistically to the raw audio waveforms *before* feature extraction.
    * **Techniques:** Gaussian Noise, Reverb, Time Stretching (`librosa`), Frequency Masking (Butterworth Filter via `scipy`). Focused on waveform-level augmentation for realism.
    * **Integration:** The `Augmenter` operates on NumPy arrays and was integrated into the `tf.data` pipeline within the mapping function, ensuring consistency (`float32`) to avoid data type errors. Augmentation probabilities were kept moderate (e.g., 40% for Addition noise, 20-30% for others, <10% for noise that heavily impacts the data, such as time gaps or shuffling).

## 2. Model Design & Architecture

* **Choice:** A custom Convolutional Recurrent Neural Network (CRNN) was selected, combining CNNs for spatial feature extraction from spectrograms and LSTMs for modeling temporal sequences.
* **Input:** Log-Mel Spectrograms (`(Time, 80, 1)`)
* **Architecture (`build_model`):**
    1.  **CNN Frontend:** Multiple blocks(Total 3 blocks) of `Conv2D` (32/64 filters), `BatchNormalization`, `ReLU`, followed by `MaxPooling2D` ((2, 2)) and `SpatialDropout2D` (.
    2.  **Reshape Layer:** Flattens frequency/channel dimensions (`Reshape((-1, features))`).
    3.  **RNN Backend:** Multiple layers (Total 4 blocks) of `Bidirectional LSTM` (e.g., 256 units) with `Dropout` (e.g., 0.4) and `BatchNormalization`.
    4.  **Output Layer:** `Dense` layer (`VOCAB_SIZE + 1` units) with `softmax` for character probabilities + CTC blank.
    5.  **Model Parameters**: 7,961,854 (float32), (later quantized to uint 8)

## 3. Training Process

* **Data Pipeline (`tf.data`):**
* **Preprocessing**: converts the data, which has been augmented and stored as .wav, into log-mel-spectrograms, which is then fed into the CNN frontend
* **Loss Function:** CTC loss (`ctc_loss` function using `tf.keras.backend.ctc_batch_cost`) with correct label length calculation for 0-padded sequences.
* **Optimizer:** Adam (`tf.keras.optimizers.Adam`).
* **Learning Rate Schedule:** `tf.keras.callbacks.ReduceLROnPlateau` used to adaptively lower the learning rate based on `val_loss`.
* **Training Loop:** Standard `model.fit()` with `ModelCheckpoint` (saving best model to `.keras`) and `ReduceLROnPlateau` callbacks. Logic added to check for and load existing checkpoints to resume training using `initial_epoch`.
* **Two part process**: I have chosen to train the model until 15 epochs, save it and then resume training later until 25 epochs. This was done due to runtime issues and served as a safety measure.
* Note: all of the training process was done on kaggle, while augmentation was done google colab. Mostly due to each member's preferences

## 4. Post-Training Quantization (INT8)

* **Goal:** Optimize the trained `.keras` model for faster CPU inference and reduced size.
* **Method:** Post-Training INT8 Quantization via TensorFlow Lite (TFLite).
* **Process in (`model-trainer.ipynb`):**
    1.  Load best trained `.keras` model.
    2.  Prepare Representative Dataset (~100 preprocessed training spectrograms).
    3.  Calibrate: `tf.lite.TFLiteConverter` observes activation ranges using the representative dataset.
    4.  Convert: Model weights/activations converted to `int8`, targeting `TFLITE_BUILTINS_INT8`.
    5.  Save: Quantized model saved as `.tflite` file.
* **Inference:** Requires `tf.lite.Interpreter` API (`asr_inference_quantized.py`), handling potential int8 input/output scaling.

## 5. Installation & Usage

1.  **Clone Repo:** `git clone https://github.com/Aneeshshastri/SN_BOSE_MILAN_ASR.git`
2.  **Environment:** Set up a Python virtual environment.
3.  **Install Deps:** `pip install -r requirements.txt` (Ensure `tensorflow`, `librosa`, `numpy`, `soundfile` are listed).
4.  **Prepare Data:**
    * Download LibriSpeech `train-clean-100` (and `test-clean`).
    * Run `Augmenter.ipynb` to augment the dataset.
    * Configure `Training_dirs` in the main script.
    * Run `python model-trainer.ipynb`. The best model is saved as `asr_model_best.keras`.
    * (Optional: Modify script to load checkpoint and set `initial_epoch` to resume).
6.  **Inference (Keras):**
    * Configure paths in `asr_inference_custom.py`.
    * Run `python asr_inference_custom.py`.
7.  **Quantize (Optional):**
    * Configure paths in `quantize_custom_model.py` (especially `REPRESENTATIVE_DATA_DIR`).
    * Run `python quantize_custom_model.py`. Creates `.tflite` file.
8.  **Inference (TFLite):**
    * Configure paths in `asr_inference_quantized.py`.
    * Run `python asr_inference_quantized.py`.

## 6. Challenges & Learnings

* **Data Pipeline Optimization:** Balancing performance (`tf.data`, parallel calls) with resource constraints (RAM usage due to large shuffle buffer, CPU bottleneck during on-the-fly augmentation/preprocessing). Addressed high RAM by significantly reducing shuffle buffer size.
* **TensorFlow Debugging:** Resolved complex runtime errors involving shape mismatches, data type conflicts (`float32` vs `float64`), library version incompatibilities, and graph optimization issues (e.g., `LayoutOptimizer` errors with `SpatialDropout2D`/`Dropout`, resolved by layer replacement and careful shape handling).
* **Model Training Dynamics:** Iterated on model architecture (increasing capacity to combat underfitting) and regularization techniques (`Dropout`, `BatchNormalization` to combat overfitting). Managed learning rate using `ReduceLROnPlateau`.
* **CTC Implementation:** Ensured correct CTC loss calculation and data filtering based on sequence length requirements.
* **Quantization Workflow:** Implemented post-training INT8 quantization, including calibration with a representative dataset and adapting inference code for the TFLite interpreter.

## 7. Results & Future Work

* **Results:** *[Model Card](./model_card.md)*
* **Future Work:**
    * Implement a proper CTC beam search decoder (e.g., using `pyctcdecode` or TensorFlow's built-ins) for potentially significant WER improvement over greedy decoding.
    * Due to limitations in my computational resources, I was unable to integrate my acoustic model with a language model, additionally, since it was supposed to focus on CPU only inference speed, I believe it was the right choice to implement only a simply greedy decoder.
    * Integrate WER calculation as a metric during validation using a custom Keras callback.
    * Experiment with SpecAugment (applied to spectrograms) in addition to/instead of waveform augmentation.
    * Explore fine-tuning pre-trained models (Wav2Vec2, Whisper) for comparison and potentially superior performance/robustness.
    * Although fine-tuning would give significantly better results, I faced a lot of issues fine-tuning models, and eventually gave up.
    * Fully pre-process the dataset into TFRecords to maximize training speed and reduce RAM/CPU load during training.

## 8. License

This project is licensed under the MIT License - see the `LICENSE` file for details.
