# FTIR Microplastic Analysis System v2.0

A full-stack web application for advanced FTIR microplastic analysis, featuring a React.js frontend and a FastAPI backend that leverages deep learning models for denoising and classification.

## 🎯 Overview

This application provides a streamlined, step-by-step workflow for analyzing FTIR spectrum data. Users can upload a CSV file, preprocess the data, apply a denoising model, and finally, classify the microplastic type using a specialized deep learning model. The system is designed to be robust, with fallback mechanisms to ensure a successful analysis even if deep learning models are not available.

## ✨ Key Features

- **Multi-Step Guided Analysis**: A 4-step process walks the user from data upload to final classification.
- **Advanced Denoising**: Utilizes deep learning models to clean spectral data, with options for different membrane filters.
- **Two-Stage Classification**: Employs a combination of denoising and classification models for higher accuracy.
- **Correlation Fallback**: If a specific model is unavailable, the system automatically falls back to a robust correlation-based classification against a reference library.
- **PDF Report Generation**: Users can download a comprehensive PDF report of the complete analysis, including all charts and results.
- **Class Activation Map (CAM) Visualization**: The interface can display a heatmap over the spectrum chart, visualizing which regions the classification model focused on.
- **Interactive Charts**: The application uses interactive charts to visualize the spectrum at each stage of the analysis.

## 📁 Project Structure

```
FTIR_Project/
├── backend/
│   ├── main.py             # FastAPI application core
│   ├── requirements.txt    # Python dependencies
│   ├── SynCleanSet.npy     # Reference spectra for correlation analysis
│   └── model/              # **EMPTY BY DEFAULT** - Directory for ML models
└── frontend/
    ├── package.json        # Node.js dependencies
    ├── README.md           # This file
    ├── src/
    │   ├── App.jsx         # Main React component with routing
    │   ├── pages/
    │   │   ├── Step1InputSpectrum.jsx
    │   │   ├── Step2Preprocessing.jsx
    │   │   ├── Step3Denoising.jsx
    │   │   └── Step4Classification.jsx
    │   └── ...
    └── ...
```

## 🛠️ Installation & Setup

### Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Start the backend server:**
    The backend requires the JAX backend for Keras. Start the server with the following command:
    ```bash
    KERAS_BACKEND=jax uvicorn main:app --reload --port 8000
    ```
    The API will be available at `http://localhost:8000`.

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```

3.  **Start the frontend development server:**
    ```bash
    npm start
    ```
    The application will be accessible at `http://localhost:3000`. The frontend is configured to proxy API requests to the backend server.

## 🧠 Models

The deep learning models are **not included** in this repository and must be downloaded separately.

- **Download the models** from the provided source (https://drive.google.com/drive/folders/1GnpQ2rniPb_F9MDG2nncYQHRJVlGVfRp?dmr=1&ec=wgc-drive-globalnav-goto).
- **Place the `.h5` files** into the `backend/model/` directory.

### Model Naming Convention

The backend uses a specific naming convention to identify and load the models. There are two types of models used in the workflow:

1.  **Denoising Models (Step 3)**: These models are used to clean the input spectrum.
    -   **Format**: `TrainingModel_{Filter}_{DenoiseType}.h5`
    -   **Example**: `TrainingModel_B_20SNR_CAE.h5`

2.  **Classification Models (Step 4)**: These models are used to classify the denoised spectrum.
    -   **Format**: `ClassifierModel_{Filter}_{DenoiseType}_{ClassifierType}.h5`
    -   **Example**: `ClassifierModel_B_20SNR_CAE_AlexNet.h5`

The system will automatically detect the available models based on their filenames. If a model for a selected combination is not found, the application will use a correlation-based method for classification.

## 📊 User Workflow

1.  **Step 1: Input Spectrum**
    -   Upload a CSV file containing the FTIR spectrum data. The application will validate the file and display the initial spectrum.

2.  **Step 2: Preprocessing**
    -   Apply baseline correction and/or normalization to the spectrum.

3.  **Step 3: Denoising**
    -   Select a membrane filter and a denoising model. The backend will apply the corresponding `TrainingModel` to the spectrum.

4.  **Step 4: Classification**
    -   Select a classification model. The backend will use the corresponding `ClassifierModel` to identify the microplastic type.
    -   The results, including the plastic type, confidence score, and correlation, will be displayed.
    -   At this stage, you can download a full PDF report of the analysis.

## 📄 License

© 2025 SIIT - Thammasat University