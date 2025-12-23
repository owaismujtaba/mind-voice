## Mind Voice: A Multimodal EEG-Audio Dataset for Overt and Covert Iberian Spanish Speech Production


Mind-Voice is an advanced research framework designed for the analysis of electroencephalography (EEG) data, focusing on event-related potentials (ERPs), brainwave decoding for imagined speech (covert/overt), and even voice anonymization. This project provides a robust pipeline for handling BIDS-formatted EEG datasets, performing detailed signal processing, machine learning-based classification, and comprehensive data visualization.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technological Stack](#technological-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [BIDS Dataset Creation](#bids-dataset-creation)
  - [P100 ERP Analysis](#p100-erp-analysis)
  - [N100 ERP Analysis](#p100-erp-analysis)
  - [Brainwave Decoding (Overt/Covert Speech)](#brainwave-decoding-overtcovert-speech)
  - [Voice Anonymization](#voice-anonymization)
  - [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The ability to communicate directly from thought has long been a subject of science fiction. Mind-Voice endeavors to bring this closer to reality by capturing, analyzing, and interpreting brainwave data (e.g., EEG signals) and converting it into audible speech or classifying cognitive states. This project offers a comprehensive suite of tools for neuroscientific research, from raw data ingestion and preprocessing to advanced machine learning and rich data visualization. It also includes a unique voice anonymization component, broadening its application scope.

## Dataset
Dataset can be downloaded form the url (https://osf.io/6sh5d/)
The dataset is in BIDS format. Download and modify the config.py file accordingly
## Features

*   **BIDS Dataset Creation & Management:** Tools for organizing raw XDF data into a BIDS-compliant structure, ensuring standardized data handling.
*   **Robust EEG Preprocessing Pipeline:** Implements advanced signal processing techniques including:
    *   Channel type setting and montage application.
    *   Bad channel detection (deviation, correlation) and interpolation using `pyprep`.
    *   Band-pass filtering.
    *   EEG re-referencing.
    *   Artifact removal via Independent Component Analysis (ICA) for EOG artifacts.
*   **Flexible EEG Epoching:** Dynamic creation of `mne.Epochs` objects based on detailed event annotations (trial mode, unit, experiment mode, boundary, type, modality), allowing precise isolation of cognitive events.
*   **P100 ERP Component Analysis:** A dedicated pipeline for identifying, quantifying (latency, peak/mean amplitude), and comparing P100 event-related potential components across different experimental conditions, particularly relevant for visual processing.
*   **Brainwave Decoding (Overt/Covert Speech & Rest):**
    *   Implementation of a Convolutional Neural Network (CNN) classifier for differentiating between overt speech, covert (imagined) speech, and resting states from EEG signals.
    *   Includes strategies for handling data imbalance using `RandomOverSampler`.
    *   Normalization per sample and per channel to optimize model performance.
*   **Voice Anonymization:** A unique pipeline for transforming audio signals using pitch shifting (`librosa`) and formant shifting (`parselmouth`) to protect speaker identity.
*   **Comprehensive Visualization & Reporting:**
    *   Plotting of individual and grand average ERPs.
    *   Visualizations of P100 peak/mean amplitudes with statistical comparisons (paired t-tests).
    *   Detailed decoding performance plots: accuracy per subject, aggregated confusion matrices, and per-class precision, recall, and F1-scores.
*   **Modular Pipeline Design:** Clear separation of concerns into dedicated pipelines (`P100AnalysisPipeline`, `OvertCovertRestPipeline`, `VoiceAnonymizerPipeline`) for reusability and scalability.
*   **Configurable Workflow:** Utilizes a `config.yaml` file for easy adjustment of dataset paths, preprocessing parameters, analysis settings, and plotting options.
*   **Logging:** Integrated logging for tracking pipeline execution, warnings, and errors.


### Technological Foundation

Mind Voice is built on a foundation of open-source Python libraries widely used in the scientific and machine learning communities:

*   🐍 **Core Language:** Python 3.8+

*   🧠 **EEG/Neuroimaging:**
    *   `mne` & `mne-bids`: Core libraries for EEG data analysis and BIDS standard compatibility.
    *   `pyprep`: For robust EEG preprocessing, including bad channel detection.
    *   `mnelab` & `pyxdf`: For reading raw XDF data files.

*   🤖 **Machine Learning/Deep Learning:**
    *   `tensorflow`: For building and training the CNN classification models.
    *   `scikit-learn` & `imblearn`: For data splitting, evaluation metrics, and handling class imbalance.

*   🔊 **Audio Processing:**
    *   `librosa`: For general audio loading and processing like pitch shifting.
    *   `parselmouth`: For advanced voice manipulation, specifically formant shifting.
    *   `soundfile`: For saving audio files.

*   🐼 **Data Manipulation:**
    *   `numpy`: Fundamental package for numerical computation.
    *   `pandas`: For data structuring and analysis.
    *   `pyyaml`: For loading configuration files.

*   📊 **Visualization:**
    *   `matplotlib`: For creating static, interactive, and animated visualizations.
    *   `seaborn`: For aesthetically pleasing statistical graphics.

*   ⚙️ **Utilities:**
    *   `pathlib`, `glob`, `os`: For object-oriented filesystem paths and operations.
    *   `rich`: For rich text and beautiful formatting in the terminal.



## Project Structure

The project is organized into logical directories to enhance modularity and maintainability:

```
mind-voice/
├── run.py                          # Main script to execute pipelines based on config.yaml
├── config.yaml                     # Configuration file for dataset paths, pipelines, and settings
├── requirements.txt                # List of Python dependencies
├── src/
│   ├── analysis/                   # Contains core analysis modules
│   │   ├── motor.py         # processing motor-related neural data, automating epoch creation, time-frequency analysis (TFR), and ROI power extraction across overt, covert, and rest conditions.
│   │   ├── p100.py       # Analyzes P100 components (latency, amplitude)
│   │   ├── p100.py       # Analyzes P100 components (latency, amplitude)
│   │   └── snr.py            # Extracts and processes epochs for visual/rest conditions
│   ├── anonymization/              # Modules for voice anonymization
│   │   └── voice_snonymizer.py     # Pipeline for pitch and formant shifting of audio
│   ├── data/                    # Data handling, loading, and preprocessing
│   │   ├── bids.py                 # Functions for creating BIDS datasets from XDF
│   │   ├── data_reader.py          # Reads raw XDF or BIDS EEG data, handles preprocessing
│   │   └── epochs.py    # Builds MNE epochs based on event annotations and filters
│   ├── decoding/                   # Modules for EEG decoding/classification
│   │   ├── data_utils.py           # Specific data loader for overt/covert/rest classification
│   │   ├── decoding.py             # Specific data loader for overt/covert/rest classification
│   │   └── model.py                # CNN model for overt/covert/rest classification
│   ├── vis/                  # Orchestrates the execution of analysis workflows
│   │   ├── accuracy.py # End-to-end pipeline for brainwave decoding
│   │   ├── confisuion_matrix.py # End-to-end pipeline for brainwave decoding
│   │   ├── metrics.py # End-to-end pipeline for brainwave decoding
│   │   ├── motor.py # End-to-end pipeline for brainwave decoding
│   │   ├── peak_mean_erp_p100.py # End-to-end pipeline for brainwave decoding
│   │   ├── peak_mean_erp_n100.py # End-to-end pipeline for brainwave decoding
│   │   ├── per_class_metrics.py # End-to-end pipeline for brainwave decoding
│   │   └── snr.py        # End-to-end pipeline for P100 ERP analysis
│   └── utils.py                      # Helper utilities and shared functions
│   ├──── config.yaml        # End-to-end pipeline for P100 ERP analysis
└── images/                         # Output directory for CSVs, plots, and models
    
```

## How It Works

The `mind-voice` project follows a structured approach, orchestrated by the `run.py` script and `config.yaml`.

1.  **Configuration:** The `config.yaml` file defines which pipelines to run, specifies input/output directories, and sets parameters for data loading, preprocessing, analysis, and plotting.
2.  **BIDS Dataset Creation (Optional):** If configured, `run.py` first calls the `create_bids_dataset` function from `src/dataset/bids.py`. This process reads raw XDF files (using `src/dataset/data_reader.py`), performs initial EEG resampling, and organizes the data into a BIDS-compliant directory structure using `mne-bids`. This ensures data standardization for subsequent analyses.
3.  **Data Loading and Preprocessing:** The `BIDSDatasetReader` (`src/dataset/data_reader.py`) is central to loading either raw XDF data or pre-existing BIDS data. It then applies a comprehensive preprocessing pipeline:
    *   Setting channel types and applying a standard EEG montage.
    *   Identifying and interpolating noisy channels using `pyprep`.
    *   Applying band-pass filters to the EEG data.
    *   Setting an EEG reference.
    *   Removing artifacts (e.g., EOG) using Independent Component Analysis (ICA). Processed data is saved to a `derivatives` folder to avoid re-computation.
4.  **Epoching:** `EEGEpochBuilder` (`src/dataset/eeg_epoch_builder.py`) and `DataLoader` (`src/dataset/data_loader.py`) are used to extract specific time segments (epochs) from the continuous EEG data. This is done based on detailed event annotations embedded in the EEG files, allowing for precise isolation of experimental conditions like visual stimuli, overt speech, covert speech, or rest periods.
5.  **Analysis Pipelines:**
    *   **P100 Analysis:** The `P100AnalysisPipeline` (`src/pipelines/p100_pipeline.py`) loads epoched data for visual and control (rest/fixation) conditions. The `P100ComponentAnalyzer` (`src/analysis/p_100_analyser.py`) then computes P100 peak latency and amplitude. Results are saved as CSVs and visualized by `P100Plotter` (`src/visualizations/p100_plotter.py`).
    *   **Brainwave Decoding:** The `OvertCovertRestPipeline` (`src/pipelines/overt_covert_rest_pipeline.py`) prepares epoched data for overt, covert, and rest conditions. It addresses class imbalance using `RandomOverSampler` and normalizes the data. An `OvertCoverRestClassifier` (`src/decoding/overt_covert_rest_model.py`), a custom CNN model, is then trained to classify these states. Performance metrics (accuracy, confusion matrix, classification report) are saved.
6.  **Voice Anonymization:** The `VoiceAnonymizerPipeline` (`src/anonymization/voice_snonymizer.py`) processes audio files by applying pitch and formant shifting using `librosa` and `parselmouth`, respectively, to create anonymized audio outputs.
7.  **Visualization & Reporting:** Various scripts in `src/visualizations` (e.g., `plot_accuracy.py`, `plot_confusion_matrix.py`, `erp_grand_visual_rest.py`, `peak_mean_visual_rest.py`) generate informative plots and aggregate statistical summaries to effectively communicate the findings from the analysis pipelines.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   An EEG device (e.g., Emotiv, OpenBCI) compatible with XDF streaming if you intend to create a BIDS dataset from raw captures.
*   (Optional but Recommended) A GPU for faster machine learning model training and inference with TensorFlow.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/owaismujtaba/mind-voice.git
    cd mind-voice
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install EEG device SDK/drivers (if applicable):**
    Refer to your specific EEG device manufacturer's documentation for any necessary software or driver installations to ensure data streaming.

### Configuration

All pipeline execution and parameters are controlled via the `config.yaml` file located in the root directory.

Before running, you should:

1.  **Update `dataset.BIDS_DIR`:** Specify the absolute path where your BIDS dataset will be stored or is currently located.
2.  **Configure `dataset.create_bids`:** Set to `true` if you need to create a new BIDS dataset from XDF files, then define `dataset.config_path` to point to a YAML file detailing your raw XDF file paths, subject IDs, and session IDs.
3.  **Enable/Disable Pipelines:** Use the boolean flags under `analysis` (e.g., `p100`, `decoding`) and `anonymize.anonymize_audio` to select which parts of the pipeline to execute.
4.  **Adjust Parameters:** Modify preprocessing parameters (e.g., `eeg_filter`), analysis-specific settings, and plotting options as needed.

## Usage

Once configured, execute the main `run.py` script:

```bash
python run.py
```

The script will automatically trigger the enabled pipelines and generate outputs based on your `config.yaml` settings.

### BIDS Dataset Creation

If `dataset.create_bids` is `true` in `config.yaml`, the pipeline will:
1.  Read specified raw XDF files.
2.  Resample EEG data to a consistent frequency.
3.  Write EEG data into a BIDS-compliant structure.
    *(Note: Audio BIDS file creation is present in the codebase but currently commented out in `src/dataset/bids.py` and `run.py`.)*

### P100 ERP Analysis

To run the P100 analysis pipeline, set `analysis.p100` to `true` in `config.yaml`.
The pipeline will:
1.  Load preprocessed EEG data for each subject and session.
2.  Create epochs for 'Visual' (stimulus onset) and 'Rest' (fixation) conditions based on annotations.
3.  Compute P100 peak latency and amplitude for specified occipital channels (default: `PO3`, `POz`, `PO4`).
4.  Generate individual subject ERP plots and save them to `results/P100/Plots/`.
5.  Save P100 metrics (latency, peak, mean amplitude) for each subject and condition to `results/P100/` as CSV files.
6.  Generate grand average ERP plots and statistical comparisons of peak/mean amplitudes across all subjects (if `plot_config.peak_mean_amplitude` and `plot_config.grand_erp_visual_real` are `true`).

### Brainwave Decoding (Overt/Covert Speech)

To run the brainwave decoding pipeline, set `analysis.decoding` to `true` in `config.yaml`.
The pipeline will:
1.  Load preprocessed EEG data for each subject and session.
2.  Create epochs for 'Overt Speech', 'Covert Speech', and 'Rest' conditions.
3.  Address class imbalance using `RandomOverSampler`.
4.  Normalize the EEG data.
5.  Train a Convolutional Neural Network (CNN) classifier to differentiate between these three states.
6.  Save the validation accuracy, a detailed classification report, and the confusion matrix for each subject to `results/DecodingResults/` as CSV files.
7.  Generate aggregated plots for accuracy, precision/recall/F1-score, and a confusion matrix across all subjects (if `plot_config.accuracy_plots`, `plot_config.metrics_plots`, `plot_config.confusion_matrix`, and `plot_config.display_per_class_metrics` are `true`).

### Voice Anonymization

To anonymize audio files, set `anonymize.anonymize_audio` to `true` in `config.yaml`.
The pipeline will:
1.  Identify audio files (e.g., `.wav`) within your BIDS dataset's audio directories.
2.  Apply pitch shifting and formant shifting to the audio.
3.  Save the anonymized audio as `anonymized.wav` in the same directory.

### Visualizations

The `plot_config` section in `config.yaml` controls which visualizations are generated after the analysis pipelines run. Enabling these flags will:
*   **`peak_mean_amplitude`**: Plot individual and mean P100 peak and mean amplitudes, including paired t-test results.
*   **`grand_erp_visual_real`**: Plot the grand average ERP for visual and rest conditions.
*   **`accuracy_plots`**: Visualize decoding accuracy per subject.
*   **`confusion_matrix`**: Generate an aggregated heatmap of the confusion matrix from decoding results.
*   **`metrics_plots`**: Plot aggregated precision, recall, and F1-scores per subject.
*   **`display_per_class_metrics`**: Log and display class-wise precision, recall, and F1-scores, including standard deviations.

All plots are saved to the `results/images/` directory.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
*(Note: A `LICENSE` file should be created in your repository with the MIT License text.)*

## Contact

Owais Mujtaba - owais.mujtaba123@gmail.com 

Project Link: [https://github.com/owaismujtaba/mind-voice](https://github.com/owaismujtaba/mind-voice)

---
