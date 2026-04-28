from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO


_CACHE_LOCK = threading.Lock()
_DETECTOR_CACHE: Dict[Tuple[str, str, float, float, int, int], "SDYOLO11Detector"] = {}


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def _import_enhance_modules() -> None:
    # Ensure custom checkpoint classes can be resolved when loading best.pt.
    import third_party.yolo11.enhance241.yolo11_241a4  # noqa: F401
    import third_party.yolo11.enhance241.yolo11_241b7  # noqa: F401
    import third_party.yolo11.enhance241.yolo11_241d6  # noqa: F401


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy": self.xyxy,
        }


class SDYOLO11Detector:
    """
    Import-and-run detector wrapper for SD-YOLO11 (a4+b7+d6).

    Initialize once, then call `predict(image_path)` repeatedly.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "0",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        max_det: int = 100,
    ) -> None:
        _ensure_repo_on_path()
        _import_enhance_modules()

        self.weights = str(Path(weights).expanduser().resolve())
        self.device = str(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)

        self.model = YOLO(self.weights)

    def predict(
        self,
        image_path: str | Path,
        *,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        image_path = str(Path(image_path).expanduser().resolve())
        results = self.model.predict(
            source=image_path,
            conf=self.conf if conf is None else float(conf),
            iou=self.iou if iou is None else float(iou),
            imgsz=self.imgsz if imgsz is None else int(imgsz),
            max_det=self.max_det if max_det is None else int(max_det),
            device=self.device,
            verbose=False,
            save=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        names = result.names
        out: List[Dict[str, Any]] = []
        if boxes is None:
            return out

        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        clss = boxes.cls.detach().cpu().tolist()

        for box, score, cls_id in zip(xyxy, confs, clss):
            class_id = int(cls_id)
            class_name = str(names.get(class_id, class_id)) if isinstance(names, dict) else str(class_id)
            out.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(score),
                    xyxy=[float(v) for v in box],
                ).to_dict()
            )
        return out


def get_detector(
    weights: str | Path = "/home/ubuntu/hpproject/yolo/best.pt",
    *,
    device: str = "0",
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    max_det: int = 100,
) -> SDYOLO11Detector:
    """
    Return cached detector instance to avoid repeated initialization.
    """
    key = (
        str(Path(weights).expanduser().resolve()),
        str(device),
        float(conf),
        float(iou),
        int(imgsz),
        int(max_det),
    )
    with _CACHE_LOCK:
        det = _DETECTOR_CACHE.get(key)
        if det is None:
            det = SDYOLO11Detector(
                weights=key[0],
                device=key[1],
                conf=key[2],
                iou=key[3],
                imgsz=key[4],
                max_det=key[5],
            )
            _DETECTOR_CACHE[key] = det
        return det


def detect(
    image_path: str | Path,
    *,
    weights: str | Path = "/home/ubuntu/hpproject/yolo/best.pt",
    device: str = "0",
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    max_det: int = 100,
) -> List[Dict[str, Any]]:
    """
    One-line helper: cached init + single-image predict.
    """
    detector = get_detector(
        weights=weights,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
    )
    return detector.predict(image_path)


def clear_detector_cache() -> None:
    with _CACHE_LOCK:
        _DETECTOR_CACHE.clear()

