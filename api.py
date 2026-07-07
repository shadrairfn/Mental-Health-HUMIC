import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS_DIR,
    LABEL_COLUMNS,
    configure_logging,
    get_device,
    import_inference_dependencies,
    load_all_binary_classifiers,
    predict_with_loaded_classifiers,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Teks keluhan yang akan diprediksi.")
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0, description="Ambang probabilitas prediksi positif.")
    max_length: int = Field(DEFAULT_MAX_LENGTH, ge=1, le=512, description="Panjang token maksimum.")


class LabelPrediction(BaseModel):
    logit: float
    probability: float
    prediction: int
    threshold: float
    token_count: int
    latency_ms: float


class PredictResponse(BaseModel):
    input: str
    labels: list[str]
    device: str
    total_latency_ms: float
    results: Dict[str, LabelPrediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    weights_dir: str
    labels: list[str]
    device: Optional[str]


configure_logging(verbose=os.getenv("APP_VERBOSE", "0") == "1")


def ensure_models_loaded() -> None:
    if app.state.classifiers is not None:
        return

    weights_dir = app.state.weights_dir
    if not weights_dir.exists():
        raise FileNotFoundError(f"Folder weights tidak ditemukan: {weights_dir}")

    logging.info("Inisialisasi dependencies inferensi.")
    import_inference_dependencies()
    app.state.device = get_device()
    app.state.classifiers = load_all_binary_classifiers(weights_dir, app.state.device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("LOAD_MODEL_ON_STARTUP", "1") != "1":
        logging.info("LOAD_MODEL_ON_STARTUP=0, model akan dimuat saat request pertama.")
    else:
        try:
            ensure_models_loaded()
        except Exception:
            logging.exception("Gagal memuat model saat startup.")
            raise

    yield


app = FastAPI(
    title="Mental Health Binary Relevance Inference API",
    description="Endpoint inferensi untuk model Stress, Anxiety, dan Depression dari folder weight.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.weights_dir = Path(os.getenv("WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR))
app.state.device = None
app.state.classifiers = None


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return health()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=app.state.classifiers is not None,
        weights_dir=str(app.state.weights_dir),
        labels=LABEL_COLUMNS,
        device=str(app.state.device) if app.state.device is not None else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text tidak boleh kosong.")

    request_started = time.perf_counter()
    logging.info(
        "Request /predict diterima | text_length=%s | threshold=%.4f | max_length=%s",
        len(text),
        payload.threshold,
        payload.max_length,
    )

    try:
        ensure_models_loaded()
        results = predict_with_loaded_classifiers(
            text=text,
            classifiers=app.state.classifiers,
            device=app.state.device,
            threshold=payload.threshold,
            max_length=payload.max_length,
        )
    except FileNotFoundError as exc:
        logging.exception("Konfigurasi model tidak valid.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ModuleNotFoundError as exc:
        logging.exception("Dependency inferensi belum lengkap.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("Inferensi endpoint gagal.")
        raise HTTPException(status_code=500, detail=f"Inferensi gagal: {exc}") from exc

    total_latency_ms = (time.perf_counter() - request_started) * 1000
    logging.info("Request /predict selesai | total_latency=%.2f ms", total_latency_ms)

    return PredictResponse(
        input=text,
        labels=LABEL_COLUMNS,
        device=str(app.state.device),
        total_latency_ms=total_latency_ms,
        results=results,
    )
