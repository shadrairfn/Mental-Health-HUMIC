import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

torch = None
AutoModelForSequenceClassification = None
AutoTokenizer = None


LABEL_COLUMNS = ["Stress", "Anxiety", "Depression"]
DEFAULT_WEIGHTS_DIR = "weight"
DEFAULT_MAX_LENGTH = 256
DEFAULT_THRESHOLD = 0.5

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def configure_logging(verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def import_inference_dependencies() -> None:
    global torch
    global AutoModelForSequenceClassification
    global AutoTokenizer

    try:
        import torch as torch_module
        from transformers import AutoModelForSequenceClassification as model_class
        from transformers import AutoTokenizer as tokenizer_class
    except ModuleNotFoundError as exc:
        missing_package = exc.name or "dependency"
        raise ModuleNotFoundError(
            f"Package '{missing_package}' belum terpasang. "
            "Install dependencies lebih dulu dengan: python -m pip install -r requirements.txt"
        ) from exc

    torch = torch_module
    AutoModelForSequenceClassification = model_class
    AutoTokenizer = tokenizer_class


def get_device() -> Any:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device aktif: %s", device)

    if device.type == "cuda":
        logging.info("CUDA device count: %s", torch.cuda.device_count())
        logging.info("CUDA current device: %s", torch.cuda.current_device())
        logging.info("CUDA device name: %s", torch.cuda.get_device_name(device))

    return device


def resolve_checkpoint_dir(base_checkpoint_dir: Path, label_name: str) -> Path:
    candidates = [
        base_checkpoint_dir / label_name,
        base_checkpoint_dir / label_name.lower(),
        base_checkpoint_dir / label_name.upper(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    available = sorted(path.name for path in base_checkpoint_dir.iterdir() if path.is_dir()) if base_checkpoint_dir.exists() else []
    raise FileNotFoundError(
        f"Checkpoint untuk label '{label_name}' tidak ditemukan di {base_checkpoint_dir}. "
        f"Folder yang tersedia: {available}"
    )


def validate_checkpoint_dir(checkpoint_dir: Path) -> None:
    required_files = ["config.json"]
    model_files = ["model.safetensors", "pytorch_model.bin"]
    tokenizer_files = ["tokenizer.json", "vocab.txt"]

    missing_required = [name for name in required_files if not (checkpoint_dir / name).exists()]
    has_model = any((checkpoint_dir / name).exists() for name in model_files)
    has_tokenizer = any((checkpoint_dir / name).exists() for name in tokenizer_files)

    if missing_required or not has_model or not has_tokenizer:
        details = {
            "missing_required": missing_required,
            "has_model_file": has_model,
            "has_tokenizer_file": has_tokenizer,
        }
        raise FileNotFoundError(f"Isi checkpoint tidak lengkap di {checkpoint_dir}: {details}")


def load_binary_classifier(checkpoint_dir: Path, device: Any):
    """Load tokenizer dan model biner dari folder hasil training."""

    validate_checkpoint_dir(checkpoint_dir)
    logging.info("Memuat tokenizer dari: %s", checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    logging.info("Memuat model dari: %s", checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logging.info(
        "Model siap | parameters=%s | trainable_parameters=%s",
        f"{parameter_count:,}",
        f"{trainable_parameter_count:,}",
    )

    return tokenizer, model


def predict_binary_label(
    text: str,
    label_name: str,
    tokenizer,
    model,
    device: Any,
    threshold: float = DEFAULT_THRESHOLD,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, float]:
    """Prediksi satu label biner dari satu model Binary Relevance."""

    logging.info("[%s] Tokenisasi input | max_length=%s | threshold=%.4f", label_name, max_length, threshold)
    encoded = tokenizer(
        str(text),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    token_count = int(encoded["attention_mask"].sum().item()) if "attention_mask" in encoded else -1
    logging.debug("[%s] Tensor keys: %s", label_name, list(encoded.keys()))
    logging.info("[%s] Jumlah token non-padding: %s", label_name, token_count)

    encoded = {key: value.to(device) for key, value in encoded.items()}

    start_time = time.perf_counter()
    with torch.no_grad():
        logit = model(**encoded).logits.squeeze().item()
        probability = torch.sigmoid(torch.tensor(logit)).item()
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    prediction = int(probability >= threshold)
    logging.info(
        "[%s] logit=%.6f | probability=%.6f | prediction=%s | latency=%.2f ms",
        label_name,
        logit,
        probability,
        prediction,
        elapsed_ms,
    )

    return {
        "logit": float(logit),
        "probability": float(probability),
        "prediction": int(prediction),
        "threshold": float(threshold),
        "token_count": int(token_count),
        "latency_ms": float(elapsed_ms),
    }


def predict_all_binary_relevance(
    text: str,
    base_checkpoint_dir: Path,
    device: Any,
    threshold: float = DEFAULT_THRESHOLD,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, Dict[str, float]]:
    """Prediksi Stress, Anxiety, dan Depression dari tiga model independen."""

    results = {}
    logging.info("Mulai inferensi untuk %s label.", len(LABEL_COLUMNS))

    for label_name in LABEL_COLUMNS:
        checkpoint_dir = resolve_checkpoint_dir(base_checkpoint_dir, label_name)
        logging.info("[%s] Checkpoint terpilih: %s", label_name, checkpoint_dir)
        tokenizer, model = load_binary_classifier(checkpoint_dir, device)
        results[label_name] = predict_binary_label(
            text=text,
            label_name=label_name,
            tokenizer=tokenizer,
            model=model,
            device=device,
            threshold=threshold,
            max_length=max_length,
        )

    return results


def load_all_binary_classifiers(base_checkpoint_dir: Path, device: Any) -> Dict[str, Dict[str, Any]]:
    """Load dan cache tokenizer/model untuk semua label Binary Relevance."""

    classifiers = {}
    logging.info("Memuat semua classifier dari: %s", base_checkpoint_dir)

    for label_name in LABEL_COLUMNS:
        checkpoint_dir = resolve_checkpoint_dir(base_checkpoint_dir, label_name)
        logging.info("[%s] Checkpoint terpilih: %s", label_name, checkpoint_dir)
        tokenizer, model = load_binary_classifier(checkpoint_dir, device)
        classifiers[label_name] = {
            "checkpoint_dir": str(checkpoint_dir),
            "tokenizer": tokenizer,
            "model": model,
        }

    logging.info("Semua classifier siap: %s", ", ".join(classifiers.keys()))
    return classifiers


def predict_with_loaded_classifiers(
    text: str,
    classifiers: Dict[str, Dict[str, Any]],
    device: Any,
    threshold: float = DEFAULT_THRESHOLD,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, Dict[str, float]]:
    """Prediksi semua label memakai tokenizer/model yang sudah dimuat."""

    results = {}
    logging.info("Mulai inferensi dengan cached classifiers untuk %s label.", len(classifiers))

    for label_name in LABEL_COLUMNS:
        classifier = classifiers[label_name]
        results[label_name] = predict_binary_label(
            text=text,
            label_name=label_name,
            tokenizer=classifier["tokenizer"],
            model=classifier["model"],
            device=device,
            threshold=threshold,
            max_length=max_length,
        )

    return results


def read_text_from_args(args: argparse.Namespace) -> str:
    if args.text and args.text_file:
        raise ValueError("Gunakan salah satu saja: --text atau --text-file.")

    if args.text_file:
        text_path = Path(args.text_file)
        logging.info("Membaca teks dari file: %s", text_path)
        return text_path.read_text(encoding="utf-8")

    if args.text:
        return args.text

    if not sys.stdin.isatty():
        logging.info("Membaca teks dari stdin.")
        return sys.stdin.read()

    raise ValueError("Input teks wajib diisi melalui --text, --text-file, atau stdin.")


def print_human_summary(results: Dict[str, Dict[str, float]]) -> None:
    print("\nHasil inferensi:")
    for label_name, result in results.items():
        status = "YA" if result["prediction"] == 1 else "TIDAK"
        print(
            f"{label_name:10s}: {status} | "
            f"probability={result['probability']:.4f} | "
            f"logit={result['logit']:.4f} | "
            f"threshold={result['threshold']:.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inferensi model Binary Relevance MentalBERT dari folder weight.",
    )
    parser.add_argument("--text", type=str, help="Teks keluhan yang akan diprediksi.")
    parser.add_argument("--text-file", type=str, help="Path file .txt berisi teks keluhan.")
    parser.add_argument("--weights-dir", type=str, default=DEFAULT_WEIGHTS_DIR, help="Folder dasar checkpoint model.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Ambang probabilitas prediksi positif.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Panjang token maksimum.")
    parser.add_argument("--json", action="store_true", help="Cetak hasil detail dalam format JSON.")
    parser.add_argument("--verbose", action="store_true", help="Aktifkan log DEBUG.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        weights_dir = Path(args.weights_dir)
        if not weights_dir.exists():
            raise FileNotFoundError(f"Folder weights tidak ditemukan: {weights_dir}")

        logging.info("Folder weights: %s", weights_dir.resolve())
        logging.info("Label yang akan diproses: %s", ", ".join(LABEL_COLUMNS))
        logging.info("Threshold global: %.4f", args.threshold)
        logging.info("Max length: %s", args.max_length)

        text = read_text_from_args(args).strip()
        if not text:
            raise ValueError("Input teks kosong.")

        logging.info("Panjang input: %s karakter", len(text))
        logging.debug("Input text: %s", text)

        import_inference_dependencies()
        device = get_device()
        start_time = time.perf_counter()
        results = predict_all_binary_relevance(
            text=text,
            base_checkpoint_dir=weights_dir,
            device=device,
            threshold=args.threshold,
            max_length=args.max_length,
        )
        total_elapsed_ms = (time.perf_counter() - start_time) * 1000
        logging.info("Inferensi selesai dalam %.2f ms", total_elapsed_ms)

        print_human_summary(results)

        if args.json:
            payload = {
                "input": text,
                "weights_dir": str(weights_dir),
                "device": str(device),
                "total_latency_ms": total_elapsed_ms,
                "results": results,
            }
            print("\nJSON detail:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))

        return 0
    except Exception as exc:
        logging.exception("Inferensi gagal: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
