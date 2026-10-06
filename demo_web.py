"""Local recording dashboard for Omni-AD prediction exports."""

import argparse
import csv
import io
import json
import mimetypes
import threading
from collections import Counter
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from diagnose_omniad import locate, read_pair


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web_demo"
STATIC_FILES = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}


def metric_payload(report):
    categories = report.get("categories", {}) if isinstance(report, dict) else {}
    mean = report.get("mean", {}) if isinstance(report, dict) else {}
    return categories, mean


def colorize(values):
    values = np.clip(values, 0.0, 1.0)
    stops = np.array([0.0, 0.22, 0.48, 0.72, 1.0], dtype=np.float32)
    colors = np.array([
        [15, 23, 42], [29, 78, 216], [21, 184, 166], [250, 204, 21], [239, 68, 68]
    ], dtype=np.float32)
    flat = values.reshape(-1)
    channels = [np.interp(flat, stops, colors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=1).reshape(values.shape + (3,)).astype(np.uint8)


class DashboardData:
    def __init__(self, predictions, data_path, metrics_path, display_max_side=1280):
        self.predictions = Path(predictions).resolve()
        self.data_path = Path(data_path).resolve()
        self.metrics_path = Path(metrics_path).resolve() if metrics_path else None
        self.rows = {}
        self.metrics = {}
        self.mean = {}
        self.scheme = "prediction export"
        self.sources = {}
        self.display_max_side = display_max_side
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        scores = self.predictions / "scores.csv"
        if not scores.is_file():
            raise FileNotFoundError(f"Prediction index not found: {scores}")
        with scores.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                self.rows.setdefault(row["category"], []).append(row)
        if not self.rows:
            raise ValueError(f"No predictions in {scores}")
        for records in self.rows.values():
            records.sort(key=lambda row: row["image_path"])
        if self.metrics_path and self.metrics_path.is_file():
            report = json.loads(self.metrics_path.read_text(encoding="utf-8"))
            self.metrics, self.mean = metric_payload(report)
            self.scheme = report.get("scheme", self.scheme)
            self.sources = report.get("category_sources", {})

    def state(self):
        source_counts = Counter(self.sources.values())
        unified_source = source_counts.most_common(1)[0][0] if source_counts else ""
        categories = []
        for name in sorted(self.rows):
            records = self.rows[name]
            metrics = self.metrics.get(name, {})
            source = self.sources.get(name, "")
            categories.append({
                "name": name,
                "images": len(records),
                "anomalies": sum("/good/" not in row["image_path"].replace("\\", "/") for row in records),
                "metrics": {key: metrics.get(key) for key in ("pixel_f1", "image_f1", "pixel_aupro")},
                "route": "类别增强分支" if source and source != unified_source else "统一模型",
            })
        return {
            "categories": categories,
            "mean": {key: self.mean.get(key) for key in ("pixel_f1", "image_f1", "pixel_aupro")},
            "scheme": self.scheme,
            "images": sum(len(records) for records in self.rows.values()),
            "prediction_path": str(self.predictions),
        }

    def samples(self, category):
        if category not in self.rows:
            raise KeyError(category)
        result = []
        for index, row in enumerate(self.rows[category]):
            normalized = row["image_path"].replace("\\", "/")
            result.append({
                "index": index,
                "name": "/".join(normalized.split("/")[-2:]),
                "score": float(row["score"]),
                "is_anomaly": "/good/" not in normalized,
            })
        return result

    def _paths(self, category, index):
        records = self.rows.get(category)
        if records is None or index < 0 or index >= len(records):
            raise KeyError(f"Unknown sample: {category}/{index}")
        return locate(records[index], self.predictions, self.data_path)

    @lru_cache(maxsize=6)
    def sample_arrays(self, category, index):
        paths = self._paths(category, index)
        scores, truth = read_pair(paths)
        with Image.open(paths[0]) as handle:
            image = handle.convert("RGB")
            scale = min(1.0, self.display_max_side / max(image.size))
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            if size != image.size:
                image = image.resize(size, Image.Resampling.LANCZOS)
                score_image = Image.fromarray(scores.astype(np.float32), mode="F")
                scores = np.asarray(score_image.resize(size, Image.Resampling.BILINEAR)).copy()
                mask_image = Image.fromarray((truth * 255).astype(np.uint8), mode="L")
                truth = np.asarray(mask_image.resize(size, Image.Resampling.NEAREST)) > 0
            rgb = np.asarray(image).copy()
        return rgb, scores, truth

    @lru_cache(maxsize=48)
    def render(self, category, index, view):
        rgb, scores, truth = self.sample_arrays(category, index)
        finite = scores[np.isfinite(scores)]
        low, high = np.percentile(finite, [1.0, 99.5]) if finite.size else (0.0, 1.0)
        normalized = np.clip((scores - low) / max(float(high - low), 1e-8), 0.0, 1.0)
        heat = colorize(normalized)
        if view == "original":
            output = rgb
        elif view == "mask":
            output = np.zeros_like(rgb)
            output[truth] = (255, 255, 255)
        elif view == "heatmap":
            output = heat
        elif view == "overlay":
            alpha = (0.18 + 0.52 * normalized)[..., None]
            output = (rgb * (1.0 - alpha) + heat * alpha).astype(np.uint8)
        else:
            raise KeyError(view)
        image = Image.fromarray(output)
        buffer = io.BytesIO()
        if view == "mask":
            image.save(buffer, format="PNG", optimize=False)
            content_type = "image/png"
        else:
            image.save(buffer, format="JPEG", quality=88, subsampling=1, optimize=False)
            content_type = "image/jpeg"
        return buffer.getvalue(), content_type


class Handler(BaseHTTPRequestHandler):
    server_version = "OmniAD-Dashboard/1.0"

    def log_message(self, fmt, *args):
        print(f"[web] {self.address_string()} {fmt % args}")

    def send_bytes(self, payload, content_type, status=HTTPStatus.OK, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, value, status=HTTPStatus.OK):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in STATIC_FILES:
                path = WEB_ROOT / STATIC_FILES[parsed.path]
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self.send_bytes(path.read_bytes(), content_type, cache="public, max-age=60")
                return
            query = parse_qs(parsed.query)
            if parsed.path == "/api/state":
                self.send_json(self.server.dashboard.state())
                return
            if parsed.path == "/api/samples":
                self.send_json(self.server.dashboard.samples(query.get("category", [""])[0]))
                return
            if parsed.path == "/api/render":
                category = query.get("category", [""])[0]
                index = int(query.get("index", ["0"])[0])
                view = query.get("view", ["overlay"])[0]
                payload, content_type = self.server.dashboard.render(category, index, view)
                self.send_bytes(payload, content_type, cache="public, max-age=3600")
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.send_json({"error": f"{type(error).__name__}: {error}"},
                           HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="./diagnostics/all30_current/current_predictions")
    parser.add_argument("--data_path", default="../dataset/download/Omni-AD-30-release")
    parser.add_argument("--metrics", default="./diagnostics/all30_current/current_all30.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--display_max_side", type=int, default=1280,
                        help="Maximum width or height used by browser previews")
    args = parser.parse_args()
    if args.display_max_side < 320:
        parser.error("--display_max_side must be at least 320")
    dashboard = DashboardData(args.predictions, args.data_path, args.metrics, args.display_max_side)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.dashboard = dashboard
    print(f"Omni-AD dashboard: http://{args.host}:{args.port}")
    print(f"Predictions: {dashboard.predictions}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
