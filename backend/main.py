"""
FastAPI backend for FTIR microplastic classification.

This service exposes the same endpoints that the redesigned frontend expects
but upgrades Step 4 so that it can execute the real AlexNet/LeNet5 classifiers
that live in backend/model/*.h5.  Models were exported with Keras 3, so we run
them with the JAX backend (KERAS_BACKEND=jax) inside backend/venv.
"""

import os
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KERAS_BACKEND", "jax")

import json
import logging

import keras
import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras.layers import Layer
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from pydantic import BaseModel
import ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Optional
from typing import Dict, List, Optional, Tuple, Any

class PositionEmbedding(Layer):
    def __init__(self, max_len, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        self.pos_emb = self.add_weight(shape=(1, self.max_len, input_shape[-1]),
                                       initializer='uniform',
                                       trainable=True,
                                       name="Positional_Embedding")
        super(PositionEmbedding, self).build(input_shape)

    def call(self, x):
        return x + self.pos_emb

    def get_config(self):
        config = super(PositionEmbedding, self).get_config()
        config.update({"max_len": self.max_len})
        return config

# ---------------------------------------------------------------------------
# Logging / app bootstrap
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ftir-backend")

app = FastAPI(title="FTIR Microplastic Analysis Backend", version="2.0.0")

print("Loading AI...")
try:
    #โหลด PDF ทั้งหมดจากโฟลเดอร์
    loader = PyPDFDirectoryLoader("knowledge_base")
    docs = loader.load()
    
    #หั่นข้อความเป็นชิ้นเล็กๆ(เพื่อให้ AI ค้นหาแม่นยำ)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    #แปลงข้อความเป็นเวกเตอร์แล้วเก็บใน ChromaDB
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    #สร้างตัวค้นหา(ดึงข้อมูลที่เกี่ยวข้องที่สุดมา n ย่อหน้า)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    print("Document successfully downloaded!")
except Exception as e:
    print(f"The system could not be loaded(Please check the *knowledge_base* folder): {e}")
    retriever = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants & reference data
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SYN_CLEAN_PATH = BASE_DIR / "SynCleanSet.npy"

WaveRef = np.arange(650, 4000, 2.5)
TARGET_SPEC_LENGTH = len(WaveRef)  # 1340 points required by the user

MIN_UPLOAD_POINTS = 20
MIN_WAVENUMBER = 600.0
MAX_WAVENUMBER = 4100.0
MIN_DYNAMIC_RANGE = 1e-8  # minimum required std ratio

NameList = [
    "Acrylic",
    "Cellulose",
    "ENR",
    "EPDM",
    "HDPE",
    "LDPE",
    "Nylon",
    "PBAT",
    "PBS",
    "PC",
    "PEEK",
    "PEI",
    "PET",
    "PLA",
    "PMMA",
    "POM",
    "PP",
    "PS",
    "PTFE",
    "PU",
    "PVA",
    "PVC",
]

if not SYN_CLEAN_PATH.exists():
    raise RuntimeError(f"Reference dataset not found at {SYN_CLEAN_PATH}")

SynCleanSet = np.load(SYN_CLEAN_PATH)
if SynCleanSet.shape[1] != TARGET_SPEC_LENGTH:
    raise RuntimeError(
        f"SynCleanSet.npy second dimension must be {TARGET_SPEC_LENGTH}, "
        f"got {SynCleanSet.shape[1]}"
    )

samples_per_material = SynCleanSet.shape[0] // len(NameList)
if samples_per_material == 0:
    raise RuntimeError("SynCleanSet does not contain samples for all materials")

REFERENCE_MEANS: Dict[str, np.ndarray] = {}
for idx, material in enumerate(NameList):
    start = idx * samples_per_material
    end = start + samples_per_material
    REFERENCE_MEANS[material] = np.mean(SynCleanSet[start:end], axis=0)

MEMBRANE_FILTERS = ["Cellulose Ester", "Glass Fiber", "Nylon"]
DENOISING_MODELS = ["CAE", "CNNAE-Xception", "CNNAE-ResNet50", "CNNAE-InceptionV3"]
CLASSIFICATION_MODELS = ["LeNet5", "AlexNet"]

MEMBRANE_CODE_ALIAS = {
    "cellulose ester": "B",
    "glass fiber": "D",
    "nylon": "E",
    "b": "B",
    "e": "E",
    "d": "D",
    "none": "BDE" #แก้ เพิ่มเพื่อบอกว่าถ้าไม่เลือก Filter ให้ใช้โค้ด BDE
}

DENOISING_NAME_ALIAS = {
    "disable": "NoDenoise",
    "none": "NoDenoise",
    "vgg16-transformer": "VGG16Transformer", #แก้ เพิ่ม
    "resnet50-transformer": "ResNet50Transformer", #แก้ เพิ่ม
    "vgg16transformer": "VGG16Transformer",
    "resnet50transformer": "ResNet50Transformer",
    "cae": "CAE",
    "cnnae-xception": "Xception",
    "cnnae-resnet50": "ResNet50",
    "cnnae-inceptionv3": "InceptionV3",
    "xception": "Xception",
    "resnet50": "ResNet50",
    "inceptionv3": "InceptionV3",
    "nodenoise": "NoDenoise",
}

EXPECTED_MEMBRANE_CODES = ["B", "D", "E"]
EXPECTED_CLASSIFIER_DENOISERS = ["CAE", "Xception", "ResNet50", "InceptionV3", "NoDenoise"]

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

class ModelManager:
    """Discovers and loads AlexNet / LeNet5 classifiers on demand."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.registry = self._discover_models()
        self.cache: Dict[str, Tuple[keras.Model, int, Optional[str]]] = {}
        self.missing: List[Tuple[str, str, str]] = self._validate_inventory()

        if not self.registry:
            logger.warning(f"No classifier .h5 files found in {model_dir}. Server will run in correlation-only mode.")

    def _discover_models(self) -> Dict[str, Dict[str, Dict[str, Path]]]:
        registry: Dict[str, Dict[str, Dict[str, Path]]] = {}
        for path in self.model_dir.glob("ClassifierModel_*.h5"):
            parts = path.stem.split("_")
            if len(parts) != 5 or parts[0] != "ClassifierModel":
                continue

            _, membrane_code, snr_tag, denoise_name, classifier_name = parts
            if classifier_name not in CLASSIFICATION_MODELS:
                continue

            registry.setdefault(classifier_name, {}).setdefault(membrane_code, {})[
                denoise_name
            ] = path

        logger.info("Discovered classifier variants: %s", registry)
        return registry

    def describe(self) -> Dict[str, Dict[str, List[str]]]:
        """Return nested dict for documentation."""
        summary: Dict[str, Dict[str, List[str]]] = {}
        for classifier, membranes in self.registry.items():
            summary[classifier] = {
                membrane: sorted(denoisers.keys()) for membrane, denoisers in membranes.items()
            }
        return summary

    def _validate_inventory(self) -> List[Tuple[str, str, str]]:
        missing: List[Tuple[str, str, str]] = []
        for classifier in CLASSIFICATION_MODELS:
            classifier_dict = self.registry.get(classifier, {})
            for membrane in EXPECTED_MEMBRANE_CODES:
                membrane_dict = classifier_dict.get(membrane, {})
                for denoiser in EXPECTED_CLASSIFIER_DENOISERS:
                    if denoiser not in membrane_dict:
                        missing.append((classifier, membrane, denoiser))

        if missing:
            logger.warning(
                "Missing classifier model files: %s",
                [f"{c}/{m}/{d}" for c, m, d in missing],
            )
        else:
            logger.info("All expected classifier combinations are available.")
        return missing

    def resolve_identifier(
        self, classifier: str, membrane_code: str, denoiser: str
    ) -> Path:
        classifier_dict = self.registry.get(classifier)
        if not classifier_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Classifier '{classifier}' not available. "
                f"Available: {list(self.registry.keys())}",
            )

        membrane_dict = classifier_dict.get(membrane_code)
        if not membrane_dict:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Membrane code '{membrane_code}' not available for classifier '{classifier}'. "
                    f"Available codes: {list(classifier_dict.keys())}"
                ),
            )

        model_path = membrane_dict.get(denoiser)
        if not model_path:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Denoising variant '{denoiser}' not available for classifier '{classifier}' "
                    f"with membrane '{membrane_code}'. "
                    f"Choices: {list(membrane_dict.keys())}"
                ),
            )
        return model_path

    def load(self, model_path: Path) -> Tuple[keras.Model, int, Optional[str]]:
        key = model_path.stem
        if key not in self.cache:
            logger.info("Loading classifier %s", model_path.name)
            model = keras.models.load_model(model_path, custom_objects={'PositionEmbedding': PositionEmbedding})
            logits_layer = model.layers[-1]
            if hasattr(logits_layer, "activation"):
                logits_layer.activation = keras.activations.linear
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            required_len = int(input_shape[1])
            last_conv_layer = _find_last_conv_layer(model)
            cam_layer_name: Optional[str] = None
            if last_conv_layer is not None:
                cam_layer_name = last_conv_layer.name
            self.cache[key] = (model, required_len, cam_layer_name)
        return self.cache[key]


model_manager = ModelManager(MODEL_DIR)


class DenoiseModelManager:
    """Loads membrane-filter specific denoising autoencoders."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.registry = self._discover_models()
        self.cache: Dict[str, Tuple[keras.Model, int, int]] = {}

        if not self.registry:
            logger.warning(f"No TrainingModel_*.h5 files found in {model_dir}. Denoising will use fallback method.")

    def _discover_models(self) -> Dict[str, Dict[str, Path]]:
        registry: Dict[str, Dict[str, Path]] = {}
        for path in self.model_dir.glob("TrainingModel_*.h5"):
            parts = path.stem.split("_")
            if len(parts) != 4 or parts[0] != "TrainingModel":
                continue

            _, membrane_code, _snr_tag, denoise_name = parts
            registry.setdefault(membrane_code, {})[denoise_name] = path

        logger.info("Discovered denoising models: %s", registry)
        return registry

    def describe(self) -> Dict[str, List[str]]:
        return {
            membrane: sorted(denoisers.keys())
            for membrane, denoisers in self.registry.items()
        }

    def resolve_identifier(self, membrane_code: str, denoiser: str) -> Path:
        membrane_dict = self.registry.get(membrane_code)
        if not membrane_dict:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Denoising models for membrane '{membrane_code}' not available. "
                    f"Available membranes: {list(self.registry.keys())}"
                ),
            )

        model_path = membrane_dict.get(denoiser)
        if not model_path:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Denoising model '{denoiser}' not available for membrane '{membrane_code}'. "
                    f"Choices: {list(membrane_dict.keys())}"
                ),
            )
        return model_path

    def load(self, model_path: Path) -> Tuple[keras.Model, int, int]:
        key = model_path.stem
        if key not in self.cache:
            logger.info("Loading denoising model %s", model_path.name)
            model = keras.models.load_model(model_path, custom_objects={'PositionEmbedding': PositionEmbedding})

            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            output_shape = model.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]

            input_len = int(input_shape[1])
            output_len = int(output_shape[1])
            self.cache[key] = (model, input_len, output_len)

        return self.cache[key]


denoise_manager = DenoiseModelManager(MODEL_DIR)


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def interpolate_spectrum(wavenumbers, intensities, target_wavenumbers):
    """Interpolate spectrum to match the standard WaveRef grid."""
    f = interpolate.interp1d(
        wavenumbers,
        intensities,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return f(target_wavenumbers)


def baseline_correction(spectrum, poly_order: int = 6):
    """Apply baseline correction using pybaselines imodpoly, mirroring legacy logic."""
    try:
        from pybaselines import polynomial

        baseline = polynomial.imodpoly(spectrum, poly_order=poly_order)[0]
        return spectrum - baseline
    except Exception:
        try:
            from pybaselines import Baseline

            baseline_fitter = Baseline(x_data=np.arange(len(spectrum)))
            baseline = baseline_fitter.asls(spectrum, lam=1e6, p=0.01)[0]
            return spectrum - baseline
        except Exception:  # pragma: no cover - fallback for environments without pybaselines
            x = np.arange(len(spectrum))
            coeffs = np.polyfit(x, spectrum, 3)
            return spectrum - np.polyval(coeffs, x)


def minmax_normalization(spectrum: np.ndarray) -> np.ndarray:
    """Min-Max normalize to [0, 1] (legacy-style)."""
    s_max = np.max(spectrum)
    s_min = np.min(spectrum)
    if np.isclose(s_max - s_min, 0):
        return spectrum
    return (spectrum - s_min) / (s_max - s_min)


def process_uploaded_spectrum(
    wavenumbers: np.ndarray, intensities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce legacy pipeline: interpolate -> baseline -> min-max normalize."""
    interpolated = interpolate_spectrum(wavenumbers, intensities, WaveRef)
    baseline_corrected = baseline_correction(interpolated)
    normalized = minmax_normalization(baseline_corrected)

    # Mirror legacy batching behavior (reshape and append)
    spec_pre = normalized.reshape(1, normalized.shape[0])
    return interpolated, baseline_corrected, spec_pre[0]


def apply_membrane_filter_correction(spectrum, membrane_filter):
    logger.debug("Applying membrane filter correction: %s", membrane_filter)
    return spectrum


def apply_fallback_denoising(spectrum):
    """Fallback denoising (Savitzky-Golay) when DL models are not available."""
    return savgol_filter(spectrum, window_length=11, polyorder=3)


def run_denoising_model(
    spectrum: np.ndarray, model_tuple: Tuple[keras.Model, int, int]
) -> np.ndarray:
    model, input_len, output_len = model_tuple
    prepared = prepare_for_model(spectrum, input_len)
    prediction = model.predict(prepared, verbose=0)
    denoised = np.asarray(prediction).reshape(output_len)
    denoised = resample_signal(denoised, TARGET_SPEC_LENGTH)
    return denoised


def resample_signal(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 1D signal to the requested length."""
    if len(signal) == target_len:
        return signal
    x_src = np.linspace(0.0, 1.0, len(signal))
    x_dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_dst, x_src, signal)


def prepare_for_model(signal: np.ndarray, required_len: int) -> np.ndarray:
    processed = resample_signal(signal, required_len)
    processed = processed.reshape(1, required_len, 1).astype("float32")
    return processed


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    tensor = tf.convert_to_tensor(logits)
    probs = tf.nn.softmax(tensor, axis=-1)
    return probs.numpy()


def classify_by_correlation(spectrum: np.ndarray):
    best_material = None
    best_score = -1.0
    best_reference = None
    best_sample_scores: List[float] = []

    for idx, material in enumerate(NameList):
        avg_corr, ref_mean, corr_list = compute_library_correlation(spectrum, idx, num_samples=20)
        if avg_corr > best_score:
            best_material = material
            best_score = avg_corr
            best_reference = ref_mean
            best_sample_scores = corr_list

    return best_material, float(best_score), best_reference, best_sample_scores


def correlation_rankings(spectrum: np.ndarray, top_k: int = 3):
    scores = []
    for idx, material in enumerate(NameList):
        avg_corr, _, _ = compute_library_correlation(spectrum, idx, num_samples=20)
        scores.append((material, float(avg_corr)))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def safe_correlation(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value, _ = pearsonr(a, b)
    if np.isnan(value):
        return 0.0
    return float(value)


def gradient_correlation(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Correlation between gradients (RowCorr style)."""
    if a is None or b is None:
        return 0.0

    if len(a) != len(b):
        b = resample_signal(b, len(a))

    grad_a = np.gradient(a)
    grad_b = np.gradient(b)

    # Ensure 2D for DataFrame correlation
    grad_a_2d = grad_a if grad_a.ndim > 1 else grad_a.reshape(-1, 1)
    grad_b_2d = grad_b if grad_b.ndim > 1 else grad_b.reshape(-1, 1)

    df_a = pd.DataFrame(grad_a_2d)
    df_b = pd.DataFrame(grad_b_2d)

    row_corr = df_a.corrwith(df_b, axis=1)
    row_corr = row_corr.replace([np.inf, -np.inf], np.nan).dropna()

    if row_corr.empty:
        return safe_correlation(grad_a, grad_b)

    return float(row_corr.mean())


def pearson_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    if len(a) != len(b):
        b = resample_signal(b, len(a))
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    corr, _ = pearsonr(a, b)
    if np.isnan(corr):
        corr = 0.0
    return float(corr)


def get_material_samples(material_index: int) -> np.ndarray:
    start = material_index * samples_per_material
    end = start + samples_per_material
    return SynCleanSet[start:end]


def compute_library_correlation(
    spectrum: np.ndarray, material_index: int, num_samples: int = 20
) -> Tuple[float, Optional[np.ndarray], List[float]]:
    samples = get_material_samples(material_index)
    if samples.size == 0:
        return 0.0, None, []

    subset = samples[: max(1, min(num_samples, len(samples)))]
    correlations = [pearson_similarity(spectrum, ref) for ref in subset]
    avg_corr = float(np.mean(correlations)) if correlations else 0.0
    mean_reference = np.mean(subset, axis=0)
    return avg_corr, mean_reference, correlations


def _find_last_conv_layer(model: keras.Model):
    """Return the deepest convolutional layer or None."""
    try:
        from keras.layers import Conv1D, Conv2D, Conv3D
    except ImportError:  # pragma: no cover - defensive guard
        return None

    for layer in reversed(model.layers):
        if isinstance(layer, (Conv1D, Conv2D, Conv3D)):
            return layer
    return None


def _grad_cam_heatmap(
    model: keras.Model, prepared_input: np.ndarray, class_idx: int
) -> np.ndarray:
    """Attempt to build Grad-CAM using the last convolution layer."""
    conv_layer = _find_last_conv_layer(model)
    if conv_layer is None:
        raise RuntimeError("No convolutional layer found for Grad-CAM.")

    grad_model = keras.Model(model.inputs, [conv_layer.output, model.output])
    prepared_tensor = tf.convert_to_tensor(prepared_input)
    with GradientTape() as tape:
        conv_outputs, logits = grad_model(prepared_tensor, training=False)
        tape.watch(conv_outputs)
        probabilities = tf.nn.softmax(logits, axis=-1)
        target = tf.math.log(tf.maximum(probabilities[:, class_idx], 1e-8))

    gradients = tape.gradient(target, conv_outputs)
    if gradients is None:
        raise RuntimeError("Failed to compute gradients for CAM.")

    conv_outputs = conv_outputs[0]
    gradients = gradients[0]
    weights = tf.reduce_mean(gradients, axis=0)
    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    heatmap_np = heatmap.numpy()
    max_val = float(np.max(heatmap_np)) if heatmap_np.size else 0.0
    if max_val > 0:
        heatmap_np /= max_val
    return heatmap_np


def _hires_cam_heatmap(
    model: keras.Model,
    prepared_input: np.ndarray,
    class_idx: int,
    layer_name: Optional[str] = "LastCnnBlock",
) -> np.ndarray:
    """Higher-resolution CAM based on explicit conv block gradients (LeNet5)."""
    target_layer = None
    if layer_name:
        try:
            target_layer = model.get_layer(layer_name)
        except ValueError:
            target_layer = None
    if target_layer is None:
        target_layer = _find_last_conv_layer(model)
    if target_layer is None:
        raise RuntimeError("No suitable convolutional layer found for hi-res CAM.")

    grad_model = keras.Model(model.inputs, [target_layer.output, model.output])
    prepared_tensor = tf.convert_to_tensor(prepared_input)

    with GradientTape() as tape:
        conv_output, logits = grad_model(prepared_tensor, training=False)
        tape.watch(conv_output)
        probabilities = tf.nn.softmax(logits, axis=-1)
        target = tf.math.log(tf.maximum(probabilities[:, class_idx], 1e-8))

    gradients = tape.gradient(target, conv_output)
    if gradients is None:
        raise RuntimeError("Failed to compute gradients for hi-res CAM.")

    conv_output = conv_output[0]
    gradients = gradients[0]
    weighted = conv_output * gradients

    heatmap = tf.reduce_mean(weighted, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    heatmap_np = heatmap.numpy()
    max_val = float(np.max(heatmap_np)) if heatmap_np.size else 0.0
    if max_val > 0:
        heatmap_np /= max_val
    return heatmap_np


def _occlusion_cam_heatmap(
    model: keras.Model,
    prepared_input: np.ndarray,
    class_idx: int,
    window_size: int = 64,
    stride: int = 32,
) -> np.ndarray:
    """Occlusion-based saliency used as a safety fallback."""
    window_size = max(8, min(window_size, prepared_input.shape[1]))
    stride = max(4, stride)

    base_logits = model.predict(prepared_input, verbose=0)[0]
    base_prob = logits_to_probabilities(base_logits)[class_idx]
    input_len = prepared_input.shape[1]
    heatmap = np.zeros(input_len, dtype=np.float32)

    for start in range(0, input_len, stride):
        end = min(input_len, start + window_size)
        perturbed = prepared_input.copy()
        perturbed[0, start:end, 0] = 0.0
        perturbed_logits = model.predict(perturbed, verbose=0)[0]
        perturbed_prob = logits_to_probabilities(perturbed_logits)[class_idx]
        drop = max(base_prob - perturbed_prob, 0.0)
        heatmap[start:end] += drop

    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    return heatmap


def generate_cam_heatmap(
    model: keras.Model,
    prepared_input: np.ndarray,
    class_idx: int,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """Generate CAM preferring hi-res conv block maps when available."""
    try:
        return _hires_cam_heatmap(model, prepared_input, class_idx, layer_name)
    except Exception:
        logger.exception("Hi-res CAM failed; falling back to Grad-CAM.")

    try:
        return _grad_cam_heatmap(model, prepared_input, class_idx)
    except Exception:
        logger.exception("Grad-CAM failed; falling back to occlusion CAM.")

    return _occlusion_cam_heatmap(model, prepared_input, class_idx)


def classify_with_model(
    spectrum: np.ndarray, model: keras.Model, required_len: int
) -> Tuple[str, float, List[float], np.ndarray, int]:
    prepared = prepare_for_model(spectrum, required_len)
    logits = model.predict(prepared, verbose=0)[0]
    probabilities = logits_to_probabilities(logits)
    class_idx = int(np.argmax(probabilities))
    return (
        NameList[class_idx],
        float(probabilities[class_idx]),
        probabilities.tolist(),
        prepared,
        class_idx,
    )


def resolve_membrane_code(value: str) -> str:
    code = value.strip()
    if not code:
        return "B"
    normalized = code.lower()
    return MEMBRANE_CODE_ALIAS.get(normalized, "B")


def resolve_denoising_name(value: str) -> str:
    normalized = value.strip().lower()
    return DENOISING_NAME_ALIAS.get(normalized, value)


def sanitize_uploaded_spectrum(
    wavenumbers: np.ndarray, intensities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate, clean, and sort incoming spectra before interpolation."""
    if len(wavenumbers) != len(intensities):
        raise HTTPException(
            status_code=400, detail="Wavenumber and intensity column lengths differ"
        )

    mask = ~(np.isnan(wavenumbers) | np.isnan(intensities))
    wavenumbers = wavenumbers[mask]
    intensities = intensities[mask]

    if len(wavenumbers) < MIN_UPLOAD_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain at least {MIN_UPLOAD_POINTS} valid rows",
        )

    order = np.argsort(wavenumbers)
    wavenumbers = wavenumbers[order]
    intensities = intensities[order]

    wavenumbers, unique_idx = np.unique(wavenumbers, return_index=True)
    intensities = intensities[unique_idx]

    low, high = wavenumbers[0], wavenumbers[-1]
    if low > MIN_WAVENUMBER or high < MAX_WAVENUMBER:
        logger.warning(
            "Partial wavenumber coverage detected (%.1f–%.1f). Results may be extrapolated.",
            low,
            high,
        )

    if np.std(intensities) < MIN_DYNAMIC_RANGE:
        raise HTTPException(
            status_code=400,
            detail="Intensity range too small; check baseline/normalization",
        )

    return wavenumbers, intensities


def ensure_finite(label: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise HTTPException(
            status_code=400,
            detail=f"{label} contains NaN or infinite values. Check preprocessing pipeline.",
        )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"message": "FTIR Microplastic Analysis Backend", "version": "2.0"}


@app.post("/api/upload")
async def upload_spectrum(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")

        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")), header=None)

        if df.shape[1] < 2:
            raise HTTPException(
                status_code=400, detail="CSV must have at least 2 columns"
            )

        wavenumbers = df.iloc[:, 0].values.astype(float)
        intensities = df.iloc[:, 1].values.astype(float)

        wavenumbers, intensities = sanitize_uploaded_spectrum(
            wavenumbers, intensities
        )

        (
            interpolated_intensities,
            baseline_corrected,
            normalized_intensities,
        ) = process_uploaded_spectrum(wavenumbers, intensities)

        return JSONResponse(
            content={
                "wavenumbers": WaveRef.tolist(),
                "intensities": interpolated_intensities.tolist(),
                "baseline_corrected": baseline_corrected.tolist(),
                "normalized_intensities": normalized_intensities.tolist(),
                "original_length": len(wavenumbers),
                "processed_length": len(interpolated_intensities),
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network/file errors
        logger.exception("Upload error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/preprocess")
async def preprocess_spectrum(
    intensities: str = Form(...),
    preprocessing_option: str = Form(...),
):
    try:
        intensities_list = json.loads(intensities)
        spectrum = np.array(intensities_list, dtype=np.float32)
        ensure_finite("Classification spectrum", spectrum)

        if len(spectrum) != TARGET_SPEC_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Spectrum length mismatch. Expected {TARGET_SPEC_LENGTH}, "
                    f"got {len(spectrum)}"
                ),
            )

        processed = spectrum.copy()
        option = preprocessing_option.lower()

        if option in {"baseline", "both"}:
            processed = baseline_correction(processed)
            ensure_finite("Baseline-corrected spectrum", processed)

        if option in {"normalization", "both"}:
            processed = minmax_normalization(processed)
            ensure_finite("Normalized spectrum", processed)

        return JSONResponse(
            content={
                "processedSpectrum": processed.tolist(),
                "preprocessing_applied": preprocessing_option,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Preprocessing error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/denoise")
async def denoise_spectrum(
    intensities: str = Form(...),
    membrane_filter: str = Form("None"), #แก้ ใส่ค่าnone แทน
    denoising_model: str = Form(...),
):
    try:
        intensities_list = json.loads(intensities)
        spectrum = np.array(intensities_list, dtype=np.float32)

        if len(spectrum) != TARGET_SPEC_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Spectrum length mismatch. Expected {TARGET_SPEC_LENGTH}, "
                    f"got {len(spectrum)}"
                ),
            )

        processed = apply_membrane_filter_correction(spectrum, membrane_filter)

        if denoising_model.lower() != "disable":
            membrane_code = resolve_membrane_code(membrane_filter)
            denoise_name = resolve_denoising_name(denoising_model)
            if denoise_name in ["VGG16Transformer", "ResNet50Transformer"]:
                membrane_code = "BDE"
            elif membrane_code == "BDE":
                membrane_code = "B"
            model_path = denoise_manager.resolve_identifier(membrane_code, denoise_name)
            try:
                processed = run_denoising_model(
                    processed, denoise_manager.load(model_path)
                )
                logger.info(
                    "Applied denoising model %s (%s)", denoise_name, model_path.name
                )
            except Exception:  # pragma: no cover
                logger.exception("Denoising model failed; applying fallback")
                processed = apply_fallback_denoising(processed)
        else:
            logger.info("Denoising disabled - no model applied")

        processed = minmax_normalization(processed)
        ensure_finite("Denoised spectrum", processed)

        return JSONResponse(
            content={
                "denoisedSpectrum": processed.tolist(),
                "membrane_filter": membrane_filter,
                "denoising_model": denoising_model,
                "wavenumbers": WaveRef.tolist(),
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Denoising error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/classify")
async def classify_spectrum(
    intensities: str = Form(...),
    membrane_filter: str = Form("None"), #แก้ ใส่none
    denoising_model: str = Form(...),
    classification_model: str = Form(...),
    baseline_intensities: Optional[str] = Form(None),
):
    try:
        intensities_list = json.loads(intensities)
        spectrum = np.array(intensities_list, dtype=np.float32)

        if len(spectrum) != TARGET_SPEC_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Spectrum length mismatch. Expected {TARGET_SPEC_LENGTH}, "
                    f"got {len(spectrum)}"
                ),
            )

        baseline_spectrum: Optional[np.ndarray] = None
        if baseline_intensities:
            baseline_list = json.loads(baseline_intensities)
            baseline_array = np.array(baseline_list, dtype=np.float32)
            if len(baseline_array) != TARGET_SPEC_LENGTH:
                baseline_array = resample_signal(baseline_array, TARGET_SPEC_LENGTH)
            ensure_finite("Baseline spectrum", baseline_array)
            baseline_spectrum = baseline_array

        if classification_model.lower() == "disable":
            (
                plastic_type,
                correlation,
                clean_spectrum,
                _,
            ) = classify_by_correlation(spectrum)
            comparison_spectrum = (
                baseline_spectrum if baseline_spectrum is not None else clean_spectrum
            )

            return JSONResponse(
                content={
                    "plastic_type": plastic_type,
                    "accuracy": float(max(correlation, 0) * 100.0),
                    "correlation": float(correlation),
                    "clean_spectrum": comparison_spectrum.tolist()
                    if comparison_spectrum is not None
                    else [],
                    "reference_spectrum": clean_spectrum.tolist()
                    if clean_spectrum is not None
                    else [],
                    "baseline_spectrum": baseline_spectrum.tolist()
                    if baseline_spectrum is not None
                    else [],
                    "method": "correlation",
                    "cam_heatmap": [],
                    "top_matches": [
                        {"plastic_type": name, "correlation": score}
                        for name, score in correlation_rankings(spectrum, top_k=3)
                    ],
                    "wavenumbers": WaveRef.tolist(),
                }
            )

        membrane_code = resolve_membrane_code(membrane_filter)
        denoise_name = resolve_denoising_name(denoising_model)
        if denoise_name in ["VGG16Transformer", "ResNet50Transformer"]:
            membrane_code = "BDE"
        elif membrane_code == "BDE":
            membrane_code = "B"
        model_path = None
        try:
            model_path = model_manager.resolve_identifier(
                classification_model, membrane_code, denoise_name
            )
            model, required_len, cam_layer_name = model_manager.load(model_path)
        except HTTPException as model_error:
            logger.warning(
                "Classifier combo unavailable (%s/%s/%s): %s; falling back to correlation.",
                membrane_code,
                denoise_name,
                classification_model,
                model_error.detail,
            )
            (
                plastic_type,
                correlation,
                clean_spectrum,
                _,
            ) = classify_by_correlation(spectrum)
            comparison_spectrum = (
                baseline_spectrum if baseline_spectrum is not None else clean_spectrum
            )
            return JSONResponse(
                content={
                    "plastic_type": plastic_type,
                    "accuracy": float(max(correlation, 0) * 100.0),
                    "correlation": float(correlation),
                    "clean_spectrum": comparison_spectrum.tolist()
                    if comparison_spectrum is not None
                    else [],
                    "reference_spectrum": clean_spectrum.tolist()
                    if clean_spectrum is not None
                    else [],
                    "baseline_spectrum": baseline_spectrum.tolist()
                    if baseline_spectrum is not None
                    else [],
                    "method": "correlation_fallback",
                    "warning": model_error.detail,
                    "cam_heatmap": [],
                    "top_matches": [
                        {"plastic_type": name, "correlation": score}
                        for name, score in correlation_rankings(spectrum, top_k=3)
                    ],
                    "wavenumbers": WaveRef.tolist(),
                }
            )

        (
            plastic_type,
            confidence,
            probabilities,
            prepared_input,
            class_idx,
        ) = classify_with_model(spectrum, model, required_len)

        class_index = NameList.index(plastic_type)
        correlation_value, clean_reference, sample_scores = compute_library_correlation(
            spectrum, class_index, num_samples=20
        )
        comparison_spectrum = (
            baseline_spectrum if baseline_spectrum is not None else clean_reference
        )

        cam_heatmap = []
        try:
            raw_cam = generate_cam_heatmap(model, prepared_input, class_idx, cam_layer_name)
            cam_heatmap = resample_signal(raw_cam, TARGET_SPEC_LENGTH).tolist()
        except Exception:  # pragma: no cover
            logger.exception("Failed to generate CAM heatmap")
            cam_heatmap = []

        response_payload = {
            "plastic_type": plastic_type,
            "accuracy": float(confidence * 100.0),
            "correlation": float(correlation_value),
            "clean_spectrum": comparison_spectrum.tolist()
            if comparison_spectrum is not None
            else [],
            "reference_spectrum": clean_reference.tolist()
            if clean_reference is not None
            else [],
            "baseline_spectrum": baseline_spectrum.tolist()
            if baseline_spectrum is not None
            else [],
            "method": classification_model,
            "model_identifier": model_path.stem,
            "probabilities": probabilities,
            "cam_heatmap": cam_heatmap,
            "top_matches": [
                {"plastic_type": name, "correlation": score}
                for name, score in correlation_rankings(spectrum, top_k=3)
            ],
            "wavenumbers": WaveRef.tolist(),
        }
        return JSONResponse(content=response_payload)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Classification error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models/info")
async def get_models_info():
    return JSONResponse(
        content={
            "membrane_filters": MEMBRANE_FILTERS,
            "denoising_models": ["Disable"] + DENOISING_MODELS,
            "classification_models": ["Disable"] + CLASSIFICATION_MODELS,
            "materials": NameList,
            "waveref_range": [float(WaveRef[0]), float(WaveRef[-1])],
            "waveref_points": TARGET_SPEC_LENGTH,
            "samples_per_material": samples_per_material,
            "available_denoise_models": denoise_manager.describe(),
            "available_classifier_models": model_manager.describe(),
            "missing_classifier_combos": [
                {
                    "classification_model": classifier,
                    "membrane_code": membrane,
                    "denoising_variant": denoiser,
                }
                for classifier, membrane, denoiser in model_manager.missing
            ],
        }
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

#New
class ReasoningRequest(BaseModel):
    llm_engine: str
    plastic_type: str
    correlation: float
    image: Optional[List[float]] = None

@app.post("/api/reasoning")
async def generate_reasoning(request: ReasoningRequest):
    
    THEORY_DATABASE = """
    1. Acrylic: [1150, 1195, 1728, 2947] (C=O stretching at 1728)
    2. Cellulose: [900, 1005, 1065, 1280, 1650]
    3. ENR: [870, 1380, 1450, 2854, 2916, 2960]
    4. EPDM: [720, 1376, 1466, 2848, 2918]
    5. HDPE: [720, 731, 1470, 2850, 2915]
    6. LDPE: [720, 731, 1460, 2850, 2920]
    7. Nylon: [690, 1200, 1266, 1542, 1638, 2870, 2933, 3300]
    8. PBAT: [728, 1020, 1105, 1270, 1715, 2365, 2960]
    9. PBS: [805, 919, 954, 1050, 1151, 1312, 1313, 1715]
    10. PC: [830, 1080, 1161, 1500, 1772]
    11. PEEK: [895, 1094, 1235, 1490, 1598, 1651, 2915, 2977]
    12. PEI: [1354, 1718, 1776, 2933]
    13. PET: [724, 872, 1095, 1241, 1714, 2963]
    14. PLA: [1088, 1184, 1456, 1751, 2921, 2997]
    15. PMMA: [841, 1146, 1437, 1724, 2952, 2995]
    16. POM: [680, 927, 1218, 1488, 1596, 2921, 2980]
    17. PP: [841, 973, 997, 1166, 1376, 1456, 2839, 2916, 2951]
    18. PS: [698, 757, 1454, 1494, 2848, 2921, 3027]
    19. PTFE: [1146, 1201]
    20. PU: [1528, 1700, 1728, 2933, 3290]
    21. PVA: [1088, 2910, 2938, 3303]
    22. PVC: [615, 690, 1253, 1425, 2910, 2969]
    """

    prompt_base = f"""
    You are an expert polymer chemist.
    We analyzed a microplastic spectrum using FTIR. The CNN model predicted it as "{request.plastic_type}" with a confidence of {request.correlation:.4f}.
    
    Here is our reference theoretical database for FTIR peaks:
    ---
    {THEORY_DATABASE}
    ---
    """

    messages = []

    #llava(ตอนนี้ยังแปลงภาพ)
    if request.llm_engine == "llava" and request.image:
        #แปลงค่าสีแดงเป็น Wavenumber
        red_spots = []
        #แกน X สมมติ 1,340 จุด(ตั้งแต่ 650 ถึง 4000)
        wavenumbers = np.linspace(650, 4000, 1340)
        
        #เช็คจำนวนจุด
        if len(request.image) == 1340:
            #แกน X มาจับคู่กับ Array ความร้อน(request.image) ทีละจุด
            for i, heat in enumerate(request.image):
                if heat > 0.7:  #ดึงเฉพาะจุดที่สีแดงเข้ม(เกิน 0.7)
                    red_spots.append(int(wavenumbers[i]))
                    
        #จัดกลุ่มตัวเลขที่ใกล้เคียงกัน(รวมเป็นตัวเลขที่ได้ แต่อยู่ในstr)
        red_spot_str = ", ".join(map(str, sorted(list(set(red_spots)))[:15])) if red_spots else "various wavenumbers"

        prompt = prompt_base + f"""
        Look at the CNN activation map data. The model focused heavily on the following wavenumbers: {red_spot_str} cm-1. 
        Compare these focused wavenumbers with the theoretical peaks of {request.plastic_type} in the database.

        CRITICAL RULES:
        1. STRICT MATCHING: If the focused wavenumbers (e.g., ~810 cm-1) DO NOT MATCH the theoretical peaks of {request.plastic_type} (e.g., 1728 cm-1), YOU MUST STATE CLEARLY THAT THEY DO NOT MATCH. Do NOT pretend that 800 is close to 1700.
        2. XAI EXPLANATION: If there is a mismatch, explain from a Deep Learning perspective that CNNs sometimes learn non-theoretical features, such as background noise, baseline characteristics, or minor structural vibrations specific to the training dataset, rather than the primary chemical bonds.
        3. If they DO match, explain the chemical bonds based ONLY on the database.
        4. Keep the tone academic, objective, and within 3 short paragraphs.
        """
        
        messages = [{'role': 'user', 'content': prompt}]
        
    else:
        #(llama3)
        prompt = prompt_base + f"\nBased strictly on the theoretical database provided above, look up the characteristic peaks of {request.plastic_type} and provide an academic, concise explanation of why these absorption peaks correspond to its chemical structure. Start your explanation immediately. Keep it within 3 short paragraphs."
        
        messages = [{'role': 'user', 'content': prompt}]

    #Ollama คิดคำตอบ
    try:
        response = ollama.chat(model=request.llm_engine, messages=messages)
        return {"reasoning": response['message']['content']}
    except Exception as e:
        return {"reasoning": f"Error from Ollama: {str(e)}"}