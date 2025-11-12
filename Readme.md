## Mind Voice: A Multimodal EEG-Audio Dataset for Overt and Covert Iberian Spanish Speech Production

Mind-Voice is an advanced research framework designed for the analysis of electroencephalography (EEG) data, focusing on event-related potentials (ERPs), brainwave decoding for imagined (covert) and overt speech, and even voice anonymization. This project provides a robust, modular pipeline for handling BIDS-formatted EEG datasets, performing detailed signal processing, machine learning-based classification, and comprehensive data visualization.

### Project Goals

*   **EEG-based Communication:** Explore the potential for translating brainwave signals into actionable communication, particularly for individuals with communication impairments.
*   **Detailed ERP Analysis:** Provide tools for classical event-related potential analysis, such as P100 component detection and comparison across conditions.
*   **Speech Decoding:** Develop and evaluate machine learning models for decoding overt, covert (imagined), and resting speech states from EEG.
*   **Voice Anonymization:** Offer a utility for anonymizing audio data using pitch and formant shifting, useful for protecting participant privacy in speech-related studies.
*   **BIDS Compliance:** Ensure data organization and processing adhere to the Brain Imaging Data Structure (BIDS) standard for enhanced reproducibility and collaboration.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Main Pipeline](#running-the-main-pipeline)
  - [BIDS Dataset Creation](#bids-dataset-creation)
  - [P100 ERP Analysis](#p100-erp-analysis)
  - [Brainwave Decoding (Overt/Covert/Rest Speech)](#brainwave-decoding-overtcovertrest-speech)
  - [Voice Anonymization](#voice-anonymization)
  - [Visualizations & Reporting](#visualizations--reporting)
- [Results & Visualizations](#results--visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The "mind-voice" project represents a step towards understanding and leveraging brain activity for communication and research. From handling raw EEG data to sophisticated machine learning decoding and robust statistical visualization, this framework aims to be a comprehensive toolkit for researchers in neurocognition, brain-computer interfaces (BCI), and speech science. The inclusion of voice anonymization further enhances its utility for ethical data management.

## Features

*   **BIDS Dataset Management:**
    *   Creation of BIDS-compliant datasets from raw XDF files.
    *   Efficient reading and loading of preprocessed BIDS EEG data.
*   **Robust EEG Preprocessing:**
    *   Resampling, channel type setting, and montage application.
    *   Bad channel detection and interpolation using `pyprep`.
    *   Frequency filtering.
    *   Referencing (e.g., common average).
    *   Artifact removal via Independent Component Analysis (ICA) targeting EOG artifacts.
*   **Flexible EEG Epoching:**
    *   Dynamic epoch creation based on detailed event annotations (trial mode, unit, experiment mode, boundary, type, modality).
    *   Customizable time windows (`tmin`, `tmax`) for epochs.
*   **P100 ERP Analysis Pipeline:**
    *   Automated extraction and comparison of P100 component (latency, peak amplitude, mean amplitude) between two experimental conditions on specified occipital channels.
    *   Generation of individual and grand-average ERP plots.
    *   Statistical comparison of P100 metrics (e.g., paired t-tests).
*   **Brainwave Decoding Pipeline (Overt/Covert/Rest):**
    *   Classification of EEG signals corresponding to overt speech, covert (imagined) speech, and resting states.
    *   Utilizes a dedicated Convolutional Neural Network (CNN) architecture optimized for time-series EEG data.
    *   Includes data balancing (oversampling using `imblearn`) and per-sample/per-channel normalization for robust model training.
    *   Comprehensive evaluation with accuracy, classification reports, and confusion matrices.
*   **Voice Anonymization Pipeline:**
    *   Applies pitch and formant shifting to audio files using `librosa` and `parselmouth`.
    *   Configurable pitch steps and formant ratio for varying degrees of anonymization.
*   **Rich Visualizations & Reporting:**
    *   Plots for individual subject ERPs (e.g., occipital evoked responses).
    *   Grand average ERP plots for multiple conditions.
    *   Box plots and scatter plots for P100 peak/mean amplitudes with statistical indicators.
    *   Bar plots for per-subject decoding accuracy and overall classification metrics (precision, recall, F1-score).
    *   Heatmaps for aggregated confusion matrices.
    *   Detailed logger output for tracking pipeline execution and results.

## Project Structure

The project is organized into several key directories:

```
mind-voice/
├── run.py                       # Main script to execute pipelines based on config.yaml
├── config.yaml                  # Global configuration file for all pipelines
├── requirements.txt             # Python dependencies
├── src/
│   ├── analysis/                # Core analysis modules
│   │   ├── covert_overt.py      # Speech event extraction and ERP plotting for overt/covert
│   │   ├── p_100_analyser.py    # P100 component detection and quantification
│   │   ├── registery.py         # EEG epoch extraction for visual/rest conditions (multiple implement.)
│   │   └── __init__.py
│   ├── anonymization/           # Audio anonymization module
│   │   ├── voice_snonymizer.py  # Voice pitch and formant shifting pipeline
│   │   └── __init__.py
│   ├── dataset/                 # Data handling, BIDS, preprocessing, epoching
│   │   ├── bids.py              # BIDS dataset creation from XDF
│   │   ├── data_loader.py       # Generic EEG data epoching based on annotations
│   │   ├── data_reader.py       # Reads raw XDF/BIDS EEG, applies full preprocessing
│   │   ├── eeg_epoch_builder.py # Specific EEG epoch builder with channel picking
│   │   └── __init__.py
│   ├── decoding/                # Machine learning models for brainwave decoding
│   │   ├── overt_covert_rest.py # Data loader for overt/covert/rest classification
│   │   ├── overt_covert_rest_model.py # CNN model for overt/covert/rest classification
│   │   └── __init__.py
│   ├── pipelines/               # End-to-end analytical workflows
│   │   ├── overt_covert_rest_pipeline.py # Pipeline for brainwave decoding
│   │   ├── p100_pipeline.py     # Pipeline for P100 ERP analysis
│   │   └── __init__.py
│   ├── utils/                   # Helper functions
│   │   ├── data.py              # YAML configuration loader
│   │   ├── graphics.py          # Styled console output using rich
│   │   └── logger.py            # Custom logging setup
│   └── visualizations/          # Plotting and reporting scripts
│       ├── display_per_class_metrics.py # Displays classification metrics per class
│       ├── erp_grand_visual_rest.py   # Plots grand average ERP for visual/rest conditions
│       ├── p100_plotter.py            # Specific P100 ERP plotting
│       ├── peak_mean_visual_rest.py   # Plots P100 peak/mean amplitudes & performs t-tests
│       ├── plot_accuracy.py           # Plots decoding accuracy per subject
│       ├── plot_confusion_matrix.py   # Plots aggregated confusion matrix
│       ├── plot_metrics.py            # Plots aggregated precision, recall, F1 per subject
│       └── __init__.py
└── results/                     # Directory for generated output (images, CSVs)
    ├── DecodingResults/
    ├── P100/
    ├── images/
    └── ...
```

## How It Works

The `run.py` script serves as the central orchestrator, reading `config.yaml` to determine which analysis pipelines and visualizations to execute.

1.  **Data Ingestion & BIDS:** Raw data (typically XDF) is first read by `src/dataset/data_reader.py` and then organized into a BIDS-compliant structure using `src/dataset/bids.py`. This ensures standardized data paths and metadata.
2.  **EEG Preprocessing:** `src/dataset/data_reader.py` applies a robust preprocessing pipeline to the raw EEG data, including filtering, bad channel interpolation, and ICA-based artifact removal, resulting in a clean `mne.io.Raw` object.
3.  **Epoching:** For specific analyses, `src/dataset/eeg_epoch_builder.py` (or `src/dataset/data_loader.py`) creates `mne.Epochs` objects by identifying events in the preprocessed EEG annotations based on user-defined criteria from the configuration.
4.  **P100 Analysis (P100AnalysisPipeline):**
    *   Epochs for two conditions (e.g., 'Visual', 'Rest') are created.
    *   `src/analysis/p_100_analyser.py` identifies and quantifies the P100 component (latency, peak, mean amplitude) on specified occipital channels.
    *   Results are saved and plotted using `src/visualizations/p100_plotter.py` and statistically summarized by `src/visualizations/peak_mean_visual_rest.py`.
5.  **Brainwave Decoding (OvertCovertRestPipeline):**
    *   Epochs for 'Overt', 'Covert', and 'Rest' conditions are extracted by `src/decoding/overt_covert_rest.py`.
    *   Data is prepared by addressing class imbalance using oversampling and normalization.
    *   A custom CNN model (`src/decoding/overt_covert_rest_model.py`) is trained to classify the EEG epochs.
    *   Performance metrics (accuracy, precision, recall, F1, confusion matrix) are saved and visualized by scripts in `src/visualizations/`.
6.  **Voice Anonymization (VoiceAnonymizerPipeline):**
    *   Separately, `src/anonymization/voice_snonymizer.py` provides a utility to load audio, apply pitch and formant shifts, and save the anonymized output. This can be used independently or as part of a larger audio processing workflow.
7.  **Visualization & Reporting:** Scripts in `src/visualizations/` are responsible for aggregating results across subjects and generating high-quality plots and statistical summaries to clearly present findings.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   **Python 3.8+**
*   **Git**
*   **An XDF-compatible EEG device** if you plan to create a new BIDS dataset from raw recordings, otherwise a pre-existing BIDS dataset.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/owaismujtaba/mind-voice.git
    cd mind-voice
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

All pipeline execution is controlled by `config.yaml`. Before running, review and adjust this file:

```yaml
dataset:
  create_bids: False # Set to True to create a BIDS dataset from XDF files
  config_path: "config/filepaths.yaml" # Path to YAML with XDF file info
  BIDS_DIR: "BIDS" # Root directory for the BIDS dataset

anonymize:
  anonymize_audio: False # Set to True to run the voice anonymization pipeline
  pitch_steps: 4 # Semitones for pitch shift
  formant_ratio: 1.2 # Formant scaling factor

analysis:
  p100: False # Set to True to run the P100 analysis pipeline
  decoding: False # Set to True to run the brainwave decoding pipeline
  results_dir: "results" # Directory to save analysis results (CSVs, plots)

eeg_filter: # EEG filter parameters (used by BIDSDatasetReader)
  low: 1.0
  high: 40.0

# ... other configurations like EEG_SR, AUDIO_SR, EEG_MONTAGE, ICA_PARAMS
# (Ensure these are defined in your actual config.yaml)

plotting: # Control which visualizations to generate after analyses
  peak_mean_amplitude: False
  grand_erp_visual_real: False
  accuracy_plots: False
  confusion_matrix: False
  metrics_plots: False
  display_per_class_metrics: False
```

#### `config/filepaths.yaml` (for BIDS dataset creation)

If `dataset.create_bids` is `True`, you'll need a `config/filepaths.yaml` file (or similar, specified by `dataset.config_path`) that lists your raw XDF files and their corresponding subject/session IDs:

```yaml
filepaths:
  - path: "/path/to/your/rawdata/sub-01_ses-01_task-VCV_run-01_eeg.xdf"
    subject_id: "01"
    session_id: "01"
  - path: "/path/to/your/rawdata/sub-02_ses-01_task-VCV_run-01_eeg.xdf"
    subject_id: "02"
    session_id: "01"
  # Add more files as needed
```

## Usage

### Running the Main Pipeline

Execute the `run.py` script. It will automatically run the pipelines and generate plots based on the settings in `config.yaml`.

```bash
python run.py
```

### BIDS Dataset Creation

To convert raw XDF data into a BIDS-compliant structure:
1.  Ensure `dataset.create_bids` is `True` in `config.yaml`.
2.  Update `dataset.config_path` to point to a YAML file listing your raw XDFs (e.g., `config/filepaths.yaml`).
3.  Run `python run.py`. This will create the BIDS structure under the directory specified by `dataset.BIDS_DIR`.

### P100 ERP Analysis

To perform P100 ERP analysis for visual stimulus vs. rest conditions:
1.  Set `analysis.p100` to `True` in `config.yaml`.
2.  Ensure your BIDS dataset is prepared (or disable `dataset.create_bids`).
3.  Adjust `plotting.peak_mean_amplitude` and `plotting.grand_erp_visual_real` to `True` to generate relevant plots.
4.  Run `python run.py`.
    *   Results (latencies, amplitudes) will be saved in `results/P100/`.
    *   Plots will be saved in `results/P100/Plots/` and `results/images/`.

### Brainwave Decoding (Overt/Covert/Rest Speech)

To train and evaluate the CNN classifier for overt/covert/rest states:
1.  Set `analysis.decoding` to `True` in `config.yaml`.
2.  Ensure your BIDS dataset is prepared.
3.  Enable `plotting.accuracy_plots`, `plotting.confusion_matrix`, `plotting.metrics_plots`, and `plotting.display_per_class_metrics` to visualize the classifier's performance.
4.  Run `python run.py`.
    *   Classification reports, accuracy, and confusion matrices will be saved in `results/DecodingResults/`.
    *   Plots will be saved in `results/images/`.

### Voice Anonymization

To anonymize audio files:
1.  Set `anonymize.anonymize_audio` to `True` in `config.yaml`.
2.  Configure `anonymize.pitch_steps` and `anonymize.formant_ratio` as desired.
3.  Ensure your BIDS dataset contains audio files (or modify `run.py` to point to specific audio files).
4.  Run `python run.py`. Anonymized audio files will be saved within the subject's BIDS `audio` directory.

### Visualizations & Reporting

Individual visualization scripts can also be called directly for debugging or specific plotting needs, but are primarily orchestrated by `run.py` when their corresponding `plotting` flags are set to `True` in `config.yaml`.

## Results & Visualizations

Upon running the pipelines with plotting enabled, the `results/images` directory will contain various plots, including:

#### Grand Average ERP
A grand average ERP across all subjects for visual change vs. no visual change conditions, typically highlighting occipital channels.
