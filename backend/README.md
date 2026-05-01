# FTIR Backend (AlexNet / LeNet5)

This folder now contains the production FastAPI service that powers Step 4 of the redesigned FTIR workflow. The backend loads the exported `.h5` classifiers from `backend/model/`, runs them with **Keras 3.10 + tensorflow backend**, and always reports Pearson correlation scores against the clean reference spectra (`SynCleanSet.npy`).

## 0. Clone the Project
clone the repository to your local machine and navigate into the backend directory

```bash
git clone https://github.com/6522772472/SL1-FTIR.git
```

## 1. Environment setup

```bash
cd "/Users/admin/Downloads/SL1-FTIR/backend"

# 1) Create / activate a dedicated venv for **Windows**
python -m venv venv
venv\Scripts\activate

# 1.1) Create / activate a dedicated venv for **macOS / Linux**
python3 -m venv venv
source venv/bin/activate


# 2) Install dependencies (Keras 3 + tensorflow + FastAPI stack)
pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow

# 3) **Make sure you have ollama installed on your machine:
      https://ollama.com/download

ollama pull llama3
ollama pull llava
ollama pull nomic-embed-text

# 4) For web server and CNN models
pip install keras numpy pandas scipy
pip install matplotlib
pip install ollama langchain langchain-community chromadb pypdf sentence-transformers
```

## 2. Running the API

```bash

# 1) Windows
cd "/Users/admin/Downloads/SL1-FTIR/backend"
source venv/bin/activate
pip install tensorflow
set KERAS_BACKEND=tensorflow
python -m uvicorn main:app --reload --port 8000

# 2) macOS / Linux
cd "/Users/admin/Downloads/SL1-FTIR/backend"
source venv/bin/activate
pip install tensorflow
export KERAS_BACKEND=tensorflow
python3 -m uvicorn main:app --reload --port 8000

```

Key environment variables:
- `KERAS_BACKEND=tensorflow` – ensures Keras boots with the tensorflow runtime.
- `PYTHONPATH` is handled automatically by the venv.

Once the server is up, the existing React frontend (http://localhost:3000) can call:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/upload` | Upload CSV (wavenumber, intensity). Validates coverage (≥20 points, no NaN/duplicates), sorts/interpolates to WaveRef, then returns raw, baseline-corrected (imodpoly), and normalized spectra |
| `POST /api/preprocess` | Baseline correction / normalization (options: `none`, `baseline`, `normalization`, `both`) |
| `POST /api/denoise` | Membrane filter correction + simple Savitzky–Golay fallback denoise |
| `POST /api/classify` | Use correlation (`classification_model=disable`) or the requested AlexNet/LeNet5 variant. If a requested combo is missing, backend auto-falls back to correlation and returns a warning. Optionally send `baseline_intensities` to compare baseline vs denoised |
| `GET /api/models/info` | Lists available model files discovered under `backend/model` |

## 3. Classification details

- **Supported plastic types (22):** `["Acrylic", "Cellulose", ..., "PVC"]`
- **Spectrum length:** enforced at 1340 points (WaveRef grid).  
  Models expecting 1323 points are automatically resampled.
- **Model selection:**  
  - The frontend still posts `membrane_filter` (e.g., `"Cellulose Ester"`) and `denoising_model` (`"CAE"`, `"CNNAE-Xception"`, etc.).  
  - The backend maps these to the actual filenames (`ClassifierModel_{B|E}_20SNR_{CAE|Xception|...}_{AlexNet|LeNet5}.h5`).  
    - `"Cellulose Ester"` → `B`, `"Nylon"` → `E`.  
    - `"Glass Fiber"` currently maps to `D`, which is reserved (models not loaded yet) so the API will return 400 until those weights are added.
    - `"Disable"` → `NoDenoise`, `CNNAE-Xception` → `Xception`, `CNNAE-ResNet50` → `ResNet50`, `CNNAE-InceptionV3` → `InceptionV3`.
  - If an exact combination is missing the backend now logs it, exposes it via `missing_classifier_combos` in `GET /api/models/info`, and auto-falls back to correlation so the request still succeeds (with a warning string in the response).
- **Outputs:** plastic type, softmax confidence (%), gradient-based correlation (baseline vs denoised when `baseline_intensities` is provided), comparison spectrum (baseline or reference), model identifier, CAM-style heatmap (`cam_heatmap`) highlighting important wavenumber regions, and the top-3 reference matches.

## 4. Troubleshooting

| Issue | Fix |
| --- | --- |
| `HTTP 400 - CSV must contain at least …` | Input file has NaN/too few points or doesn’t cover 650–4000 cm⁻¹. Clean the CSV then re-upload |
| `ModuleNotFoundError: jax` | Activate the venv and rerun `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'tensorflow’` | Activate the venv and rerun `pip install tensorflow` |
| `ERROR: Could not find a version that satisfies the requirement tensorflow` | This usually happens on Windows if your Python version is too new (e.g., Python 3.13+). Downgrade to Python 3.12, ensure "Add python.exe to PATH" is checked during installation, and recreate the virtual environment. Then back to step No.1 Environment setup|
| `RuntimeError: SynCleanSet.npy ...` | Ensure the reference dataset sits in `backend/SynCleanSet.npy` |
| `HTTP 400 - model not available` | Check `GET /api/models/info` for the exact membrane/denoise combinations that exist |
| Very slow first prediction | The first call compiles the tensorflow graph; subsequent requests are fast |

The backend is now ready for end-to-end testing with the redesigned UI.
