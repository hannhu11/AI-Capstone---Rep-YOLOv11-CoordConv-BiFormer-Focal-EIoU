"""
Kaggle SHWD/VOC2028 baseline pipeline.

Default usage on Kaggle:

    python kaggle_shwd_baseline.py --mode convert
    python kaggle_shwd_baseline.py --mode smoke
    python kaggle_shwd_baseline.py --mode all --epochs 100 --device 0,1

The script converts Pascal VOC annotations to YOLO format, trains stable
Ultralytics baselines, exports ONNX for Kaggle-side latency checks, audits hard
negative images if available, and prunes nonessential weight files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


CLASS_MAP = {"hat": 0, "person": 1}
CLASS_NAMES = ["hat", "person"]
DEFAULT_DATASET_ROOT = Path("/kaggle/input/datasets/hannhu4002/voc2028/VOC2028")
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/SHWD_YOLO")
DEFAULT_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov10n.pt",
    "yolov10s.pt",
    "yolo11n.pt",
    "yolo11s.pt",
]


@dataclass
class ConvertStats:
    dataset_root: str
    output_dir: str
    train_images: int = 0
    test_images: int = 0
    labels_written: int = 0
    missing_images: int = 0
    missing_xml: int = 0
    invalid_boxes: int = 0
    ignored_objects: int = 0
    class_counts: dict[str, int] | None = None
    ignored_by_class: dict[str, int] | None = None
    dry_run: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHWD/VOC2028 Kaggle baseline pipeline")
    parser.add_argument("--mode", choices=["convert", "smoke", "train", "all"], default="convert")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", default="trainval.txt")
    parser.add_argument("--test-split", default="test.txt")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="16", help="Integer batch size, or 'auto' to pass -1 to Ultralytics")
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--hard-negative-dir", type=Path, default=Path("/kaggle/input/hard-negatives"))
    parser.add_argument("--benchmark-repeats", type=int, default=100)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--keep-onnx", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate without writing files")
    parser.add_argument("--no-image-links", action="store_true", help="Only write labels/yaml, useful for local parser tests")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dataset_root(requested: Path) -> Path:
    requested = requested.expanduser()
    if requested.exists():
        return requested

    candidates = [
        Path("/kaggle/input/datasets/hannhu4002/voc2028/VOC2028"),
        Path("/kaggle/input/voc2028/VOC2028"),
        Path("/kaggle/input/VOC2028"),
        Path.cwd() / "VOC2028",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in kaggle_input.rglob("VOC2028"):
            if (candidate / "Annotations").exists() and (candidate / "JPEGImages").exists():
                return candidate

    raise FileNotFoundError(
        f"Could not find VOC2028 dataset root. Requested: {requested}. "
        "Expected folders: Annotations, JPEGImages, ImageSets/Main."
    )


def read_split(dataset_root: Path, split_file: str) -> list[str]:
    split_path = dataset_root / "ImageSets" / "Main" / split_file
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_image(dataset_root: Path, image_id: str, xml_root: ET.Element | None = None) -> Path | None:
    jpeg_dir = dataset_root / "JPEGImages"
    names: list[str] = []
    if xml_root is not None:
        filename = (xml_root.findtext("filename") or "").strip()
        if filename:
            names.append(filename)
    names.extend([f"{image_id}{ext}" for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]])
    for name in dict.fromkeys(names):
        candidate = jpeg_dir / name
        if candidate.exists():
            return candidate
    return None


def image_size_from_xml(root: ET.Element) -> tuple[float, float]:
    size = root.find("size")
    if size is None:
        raise ValueError("Missing <size> block")
    width = float(size.findtext("width") or 0)
    height = float(size.findtext("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size width={width}, height={height}")
    return width, height


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def voc_object_to_yolo(
    obj: ET.Element,
    width: float,
    height: float,
) -> tuple[int | None, str | None, tuple[float, float, float, float] | None, str | None]:
    raw_name = (obj.findtext("name") or "").strip()
    name = raw_name.lower()
    if name not in CLASS_MAP:
        return None, raw_name or "<empty>", None, "ignored_class"

    box = obj.find("bndbox")
    if box is None:
        return None, raw_name, None, "missing_bndbox"

    try:
        xmin = float(box.findtext("xmin") or 0)
        ymin = float(box.findtext("ymin") or 0)
        xmax = float(box.findtext("xmax") or 0)
        ymax = float(box.findtext("ymax") or 0)
    except ValueError:
        return None, raw_name, None, "non_numeric_box"

    xmin = clip(xmin, 0, width)
    xmax = clip(xmax, 0, width)
    ymin = clip(ymin, 0, height)
    ymax = clip(ymax, 0, height)
    if xmax <= xmin or ymax <= ymin:
        return None, raw_name, None, "invalid_box"

    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    yolo_box = (
        clip(x_center, 0.0, 1.0),
        clip(y_center, 0.0, 1.0),
        clip(box_width, 0.0, 1.0),
        clip(box_height, 0.0, 1.0),
    )
    return CLASS_MAP[name], name, yolo_box, None


def safe_link_or_copy(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    try:
        if os.name != "nt":
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_dataset_yaml(output_dir: Path, dry_run: bool) -> Path:
    yaml_path = output_dir / "shwd.yaml"
    content = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/test",
            "test: images/test",
            "nc: 2",
            "names:",
            "  0: hat",
            "  1: person",
            "",
        ]
    )
    if not dry_run:
        yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def convert_split(
    dataset_root: Path,
    output_dir: Path,
    split_ids: Iterable[str],
    split_name: str,
    ignored_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    class_counts: Counter[str],
    link_images: bool,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, int]:
    stats = Counter()
    label_dir = output_dir / "labels" / split_name
    image_dir = output_dir / "images" / split_name
    if not dry_run:
        label_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

    for image_id in split_ids:
        xml_path = dataset_root / "Annotations" / f"{image_id}.xml"
        if not xml_path.exists():
            stats["missing_xml"] += 1
            invalid_rows.append({"split": split_name, "image_id": image_id, "reason": "missing_xml"})
            continue

        try:
            root = ET.parse(xml_path).getroot()
            width, height = image_size_from_xml(root)
        except Exception as exc:
            stats["missing_xml"] += 1
            invalid_rows.append({"split": split_name, "image_id": image_id, "reason": f"xml_error:{exc}"})
            continue

        image_path = find_image(dataset_root, image_id, root)
        if image_path is None:
            stats["missing_images"] += 1
            invalid_rows.append({"split": split_name, "image_id": image_id, "reason": "missing_image"})
            continue

        label_lines: list[str] = []
        for obj_index, obj in enumerate(root.findall("object")):
            class_id, class_name, yolo_box, reason = voc_object_to_yolo(obj, width, height)
            if reason is not None:
                if reason == "ignored_class":
                    stats["ignored_objects"] += 1
                    ignored_rows.append(
                        {
                            "split": split_name,
                            "image_id": image_id,
                            "object_index": obj_index,
                            "label": class_name,
                            "reason": reason,
                        }
                    )
                else:
                    stats["invalid_boxes"] += 1
                    invalid_rows.append(
                        {
                            "split": split_name,
                            "image_id": image_id,
                            "object_index": obj_index,
                            "label": class_name,
                            "reason": reason,
                        }
                    )
                continue

            assert class_id is not None and class_name is not None and yolo_box is not None
            class_counts[class_name] += 1
            label_lines.append(
                f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}"
            )

        if not dry_run:
            (label_dir / f"{image_id}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            if link_images:
                safe_link_or_copy(image_path, image_dir / image_path.name, overwrite=overwrite)
        stats["images"] += 1
        stats["labels_written"] += 1

    return dict(stats)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert_voc_to_yolo(
    dataset_root: Path,
    output_dir: Path,
    train_split: str = "trainval.txt",
    test_split: str = "test.txt",
    link_images: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[Path, ConvertStats]:
    dataset_root = resolve_dataset_root(dataset_root)
    train_ids = read_split(dataset_root, train_split)
    test_ids = read_split(dataset_root, test_split)

    ignored_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()

    train_stats = convert_split(
        dataset_root,
        output_dir,
        train_ids,
        "train",
        ignored_rows,
        invalid_rows,
        class_counts,
        link_images=link_images,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    test_stats = convert_split(
        dataset_root,
        output_dir,
        test_ids,
        "test",
        ignored_rows,
        invalid_rows,
        class_counts,
        link_images=link_images,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    yaml_path = write_dataset_yaml(output_dir, dry_run=dry_run)

    ignored_by_class = Counter(row["label"] for row in ignored_rows)
    stats = ConvertStats(
        dataset_root=str(dataset_root),
        output_dir=str(output_dir),
        train_images=train_stats.get("images", 0),
        test_images=test_stats.get("images", 0),
        labels_written=train_stats.get("labels_written", 0) + test_stats.get("labels_written", 0),
        missing_images=train_stats.get("missing_images", 0) + test_stats.get("missing_images", 0),
        missing_xml=train_stats.get("missing_xml", 0) + test_stats.get("missing_xml", 0),
        invalid_boxes=train_stats.get("invalid_boxes", 0) + test_stats.get("invalid_boxes", 0),
        ignored_objects=train_stats.get("ignored_objects", 0) + test_stats.get("ignored_objects", 0),
        class_counts=dict(class_counts),
        ignored_by_class=dict(ignored_by_class),
        dry_run=dry_run,
    )

    if not dry_run:
        (output_dir / "conversion_report.json").write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
        write_csv(
            output_dir / "ignored_labels.csv",
            ignored_rows,
            ["split", "image_id", "object_index", "label", "reason"],
        )
        write_csv(
            output_dir / "invalid_annotations.csv",
            invalid_rows,
            ["split", "image_id", "object_index", "label", "reason"],
        )

    print(json.dumps(asdict(stats), indent=2))
    return yaml_path, stats


def parse_device(device: str) -> str | list[int]:
    if "," in device:
        return [int(part.strip()) for part in device.split(",") if part.strip()]
    try:
        return int(device)
    except ValueError:
        return device


def parse_batch(batch: str) -> int | float:
    value = str(batch).strip().lower()
    if value == "auto":
        return -1
    try:
        return int(value)
    except ValueError:
        return float(value)


def sanitize_name(model_name: str) -> str:
    return model_name.replace(".pt", "").replace("/", "_").replace("\\", "_")


def get_latest_run_dir(project_dir: Path, run_name: str) -> Path:
    exact = project_dir / run_name
    if exact.exists():
        return exact
    matches = sorted(project_dir.glob(f"{run_name}*"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No run directory found for {run_name} under {project_dir}")
    return matches[-1]


def prune_run_dir(run_dir: Path) -> None:
    weights_dir = run_dir / "weights"
    for pattern in ["last.pt", "epoch*.pt"]:
        for path in weights_dir.glob(pattern):
            path.unlink(missing_ok=True)
    for cache_path in run_dir.rglob("*.cache"):
        cache_path.unlink(missing_ok=True)


def read_results_csv(run_dir: Path) -> dict[str, Any]:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return {}
    with results_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return {key.strip(): value for key, value in rows[-1].items()}


def model_size_mb(path: Path) -> float | None:
    if path.exists():
        return round(path.stat().st_size / (1024 * 1024), 3)
    return None


def metric_or_none(obj: Any, attr_path: str) -> Any:
    current = obj
    for attr in attr_path.split("."):
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def extract_ultralytics_metrics(metrics: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["map50"] = metric_or_none(metrics, "box.map50")
    out["map50_95"] = metric_or_none(metrics, "box.map")
    out["map75"] = metric_or_none(metrics, "box.map75")
    box_metrics = getattr(metrics, "box", None)

    def sequence_to_class_metrics(values: Any, prefix: str) -> None:
        if values is None:
            return
        try:
            if hasattr(values, "detach"):
                values_list = values.detach().cpu().tolist()
            elif hasattr(values, "tolist"):
                values_list = values.tolist()
            else:
                values_list = list(values)
            for class_index, class_name in enumerate(CLASS_NAMES):
                if class_index < len(values_list):
                    out[f"{prefix}_{class_name}"] = float(values_list[class_index])
        except Exception:
            return

    if box_metrics is not None:
        sequence_to_class_metrics(getattr(box_metrics, "maps", None), "ap")
        sequence_to_class_metrics(getattr(box_metrics, "ap50", None), "ap50")
        sequence_to_class_metrics(getattr(box_metrics, "p", None), "precision")
        sequence_to_class_metrics(getattr(box_metrics, "r", None), "recall")
        sequence_to_class_metrics(getattr(box_metrics, "f1", None), "f1")

    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        for key, value in speed.items():
            out[f"speed_{key}_ms"] = value
    return out


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    all_keys = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_onnx(onnx_path: Path, imgsz: int, repeats: int, warmup: int) -> dict[str, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:
        return {"onnx_benchmark_error": f"missing_dependency:{exc}"}

    providers = ort.get_available_providers()
    provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
    session = ort.InferenceSession(str(onnx_path), providers=[provider])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_meta.shape]
    if len(shape) == 4:
        shape[0], shape[1], shape[2], shape[3] = 1, 3, imgsz, imgsz
    dummy = np.random.rand(*shape).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        times_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    p95_ms = sorted(times_ms)[int(0.95 * (len(times_ms) - 1))]
    return {
        "onnx_provider": provider,
        "onnx_latency_mean_ms": round(mean_ms, 4),
        "onnx_latency_p95_ms": round(p95_ms, 4),
        "onnx_fps_mean": round(1000.0 / mean_ms, 2) if mean_ms > 0 else None,
    }


def audit_hard_negatives(
    model_path: Path,
    hard_negative_dir: Path,
    imgsz: int,
    conf: float,
    device: str,
) -> dict[str, Any]:
    if not hard_negative_dir.exists():
        return {
            "hard_negative_dir": str(hard_negative_dir),
            "hard_negative_status": "not_available",
        }

    from ultralytics import YOLO

    image_files = [
        path
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        for path in hard_negative_dir.rglob(ext)
    ]
    if not image_files:
        return {
            "hard_negative_dir": str(hard_negative_dir),
            "hard_negative_status": "empty",
        }

    model = YOLO(str(model_path))
    hat_fp_images = 0
    hat_fp_boxes = 0
    total_boxes = 0
    for path in image_files:
        results = model.predict(str(path), imgsz=imgsz, conf=conf, device=device, verbose=False)
        image_hat_boxes = 0
        for result in results:
            if result.boxes is None:
                continue
            classes = result.boxes.cls.detach().cpu().tolist()
            total_boxes += len(classes)
            image_hat_boxes += sum(1 for cls_id in classes if int(cls_id) == 0)
        if image_hat_boxes:
            hat_fp_images += 1
            hat_fp_boxes += image_hat_boxes

    return {
        "hard_negative_dir": str(hard_negative_dir),
        "hard_negative_status": "ok",
        "hard_negative_images": len(image_files),
        "hard_negative_total_boxes": total_boxes,
        "hard_negative_hat_fp_boxes": hat_fp_boxes,
        "hard_negative_hat_fp_image_rate": round(hat_fp_images / len(image_files), 6),
    }


def train_one_model(args: argparse.Namespace, model_name: str, data_yaml: Path, epochs: int) -> dict[str, Any]:
    from ultralytics import YOLO

    project_dir = args.output_dir / "runs"
    run_name = sanitize_name(model_name)
    device = parse_device(args.device)
    model = YOLO(model_name)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=parse_batch(args.batch),
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        save_period=-1,
        cache=False,
        plots=True,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        # Built-in augmentation knobs. Custom Albumentations patching is left
        # for the ablation notebook because Ultralytics internals vary by version.
        hsv_h=0.025,
        hsv_s=0.75,
        hsv_v=0.45,
        translate=0.10,
        scale=0.50,
        fliplr=0.50,
        mosaic=1.00,
        mixup=0.10,
        close_mosaic=10,
    )
    model.train(**train_kwargs)

    run_dir = get_latest_run_dir(project_dir, run_name)
    best_pt = run_dir / "weights" / "best.pt"
    val_model = YOLO(str(best_pt))
    metrics = val_model.val(data=str(data_yaml), imgsz=args.imgsz, device=device, split="val", plots=True)

    row: dict[str, Any] = {
        "model": model_name,
        "epochs": epochs,
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
        "best_pt_mb": model_size_mb(best_pt),
    }
    row.update(extract_ultralytics_metrics(metrics))
    row.update({f"csv_{key}": value for key, value in read_results_csv(run_dir).items()})

    try:
        onnx_path = Path(val_model.export(format="onnx", imgsz=args.imgsz, simplify=False, dynamic=False, device=0))
        row["onnx_path"] = str(onnx_path)
        row["onnx_mb"] = model_size_mb(onnx_path)
        row.update(benchmark_onnx(onnx_path, args.imgsz, args.benchmark_repeats, args.benchmark_warmup))
        if not args.keep_onnx:
            onnx_path.unlink(missing_ok=True)
            row["onnx_removed_after_benchmark"] = True
    except Exception as exc:
        row["onnx_export_error"] = repr(exc)

    row.update(audit_hard_negatives(best_pt, args.hard_negative_dir, args.imgsz, args.conf, "0"))
    prune_run_dir(run_dir)
    return row


def run_training(args: argparse.Namespace, data_yaml: Path, smoke: bool) -> list[dict[str, Any]]:
    models = ["yolo11n.pt"] if smoke else args.models
    epochs = 1 if smoke else args.epochs
    rows: list[dict[str, Any]] = []
    for model_name in models:
        print(f"\n=== Training {model_name} for {epochs} epoch(s) ===")
        row = train_one_model(args, model_name, data_yaml, epochs=epochs)
        rows.append(row)
        write_rows_csv(args.output_dir / "benchmark_results.csv", rows)
        print(json.dumps(row, indent=2, default=str))
    return rows


def write_markdown_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        "model",
        "map50",
        "map50_95",
        "ap_hat",
        "ap_person",
        "onnx_latency_mean_ms",
        "onnx_fps_mean",
        "best_pt_mb",
        "hard_negative_status",
        "hard_negative_hat_fp_image_rate",
    ]
    lines = ["# SHWD Baseline Benchmark Results", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Kaggle benchmark uses ONNX Runtime where available; TensorRT FP16 should be run on the target PC.")
    lines.append("- Hard-negative audit is reported only when a distractor folder is supplied.")
    (output_dir / "BENCHMARK_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    link_images = not args.no_image_links
    data_yaml, stats = convert_voc_to_yolo(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        train_split=args.train_split,
        test_split=args.test_split,
        link_images=link_images,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    if args.dry_run or args.mode == "convert":
        return

    if stats.train_images != 6064 or stats.test_images != 1517:
        print(
            "WARNING: split counts differ from local VOC2028 inspection. "
            f"train={stats.train_images}, test={stats.test_images}"
        )

    smoke = args.mode == "smoke"
    rows = run_training(args, data_yaml, smoke=smoke)
    write_markdown_summary(args.output_dir, rows)


if __name__ == "__main__":
    main()
