## Download 'model' for "backend/model/" -> https://drive.google.com/drive/folders/1kZ4rFxwHrr8XfAQNfooutfBXa35fi3G7?usp=sharing

# FTIR Microplastic Analysis System v2.0
A full-stack web application for advanced FTIR microplastic analysis, featuring a React.js frontend and a FastAPI backend that leverages deep learning models for denoising and classification.

## Overview
This application provides a streamlined, step-by-step workflow for analyzing FTIR spectrum data. Users can upload a CSV file, preprocess the data, apply a denoising model, and finally, classify the microplastic type using a specialized deep learning model. The system is designed to be robust, with fallback mechanisms to ensure a successful analysis even if deep learning models are not available.

## Key Features
- **Multi-Step Guided Analysis**: A 4-step process walks the user from data upload to final classification.
- **Advanced Denoising**: Utilizes deep learning models to clean spectral data, with options for different membrane filters.
- **Two-Stage Classification**: Employs a combination of denoising and classification models for higher accuracy.
- **Explainable AI (XAI) Reasoning**: Integrates with local LLMs (Llama 3, LLaVA) via Ollama to generate scientific explanations based on dominant peaks extracted from the denoised spectrum.
- **Correlation Fallback**: If a specific model is unavailable, the system automatically falls back to a robust correlation-based classification against a reference library.
- **PDF Report Generation**: Users can download a comprehensive PDF report of the complete analysis, including all charts and results.
- **Class Activation Map (CAM) Visualization**: The interface can display a heatmap over the spectrum chart, visualizing which regions the classification model focused on.
- **Interactive Charts**: The application uses interactive charts to visualize the spectrum at each stage of the analysis.

# Read more in -> "backend" and "frontend" folder.
