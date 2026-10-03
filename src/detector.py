"""
RetailEdge Sentinel - Object Detection Subsystem
Provides hardware-abstracted inference architecture:
- LocalInferenceEngine: Runs lightweight YOLOv8n locally on laptop CPU/GPU.
- QualcommInferenceEngine: Architecture stub for Qualcomm Dragonwing Edge AI platform (QAIRT / QNN).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Represents a single detected object."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str

class InferenceEngine(ABC):
    """Abstract base class for retail detection engines."""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Run inference on a frame.
        Returns:
            Tuple of (List of Detections, inference_time_ms)
        """
        pass

    @abstractmethod
    def get_platform_info(self) -> dict:
        """Returns details about the underlying execution runtime."""
        pass


class LocalInferenceEngine(InferenceEngine):
    """
    Local Edge Simulation Inference Engine using Ultralytics YOLOv8 nano.
    Designed for laptop CPU/integrated GPU development environment.
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence_thresh: float = 0.35, target_classes: Optional[List[int]] = None):
        self.model_name = model_name
        self.confidence_thresh = confidence_thresh
        self.target_classes = target_classes if target_classes is not None else [0] # 0 = person
        self.yolo_model = None
        self.use_fallback_hog = False
        self._load_model()

    def _load_model(self):
        """Attempt to load Ultralytics YOLOv8, or initialize OpenCV HOG fallback if needed."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {self.model_name}")
            self.yolo_model = YOLO(self.model_name)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load YOLO model ({e}). Initializing OpenCV HOG person detector fallback.")
            self.use_fallback_hog = True
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """Detect persons in the given frame and return measured inference latency."""
        t_start = time.perf_counter()
        detections: List[Detection] = []

        if frame is None or frame.size == 0:
            return detections, 0.0

        if not self.use_fallback_hog and self.yolo_model is not None:
            try:
                # Run YOLO inference
                results = self.yolo_model(frame, conf=self.confidence_thresh, classes=self.target_classes, verbose=False)
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            class_name = self.yolo_model.names.get(cls_id, "person")
                            detections.append(Detection(
                                bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                                confidence=round(conf, 2),
                                class_id=cls_id,
                                class_name=class_name
                            ))
            except Exception as e:
                logger.error(f"Error during YOLO inference: {e}")
        else:
            # Fallback OpenCV HOG detection
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                for (x, y, w, h), w_score in zip(boxes, weights):
                    if w_score >= 0.2:
                        detections.append(Detection(
                            bbox=(int(x), int(y), int(x + w), int(y + h)),
                            confidence=round(float(w_score), 2),
                            class_id=0,
                            class_name="person"
                        ))
            except Exception as e:
                logger.error(f"Error during HOG fallback detection: {e}")

        inference_time_ms = (time.perf_counter() - t_start) * 1000.0
        return detections, inference_time_ms

    def get_platform_info(self) -> dict:
        return {
            "engine": "LocalInferenceEngine",
            "model": self.model_name,
            "runtime": "Local CPU / PyTorch",
            "environment": "Windows Laptop Simulation",
            "is_target_hardware": False,
            "qualcomm_accelerated": False
        }


class QualcommInferenceEngine(InferenceEngine):
    """
    Production Adapter for Qualcomm Dragonwing Edge AI (QCS6490).

    Wraps QualcommAIHubInferenceEngine from src/qualcomm_aihub.py.

    Setup (one-time, on your laptop):
        1. pip install qai-hub qai-hub-models
        2. qai-hub configure --api_token YOUR_TOKEN
           (sign up free at https://app.aihub.qualcomm.com/)
        3. python src/qualcomm_aihub.py --step all
           → exports YOLOv8n to ONNX
           → compiles to QNN on real QCS6490 via Qualcomm cloud
           → downloads models/yolov8n_retailedge_qnn.bin
        4. Copy the .bin to the Dragonwing hardware and run there.

    On this laptop: use create_engine() which auto-falls-back to LocalInferenceEngine.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n_retailedge_qnn.bin",
        target_chipset: str = "QCS6490 (Proxy)",
        confidence_thresh: float = 0.35,
    ):
        self.model_path = model_path
        self.target_chipset = target_chipset
        self.confidence_thresh = confidence_thresh
        self._engine = None
        self._init_error: Optional[str] = None
        self._load()

    def _load(self):
        try:
            from src.qualcomm_aihub import QualcommAIHubInferenceEngine
            from pathlib import Path
            self._engine = QualcommAIHubInferenceEngine(
                model_path=Path(self.model_path),
                device_name=self.target_chipset,
                confidence_thresh=self.confidence_thresh,
            )
            if not self._engine.is_ready():
                self._init_error = self._engine._init_error
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"QualcommInferenceEngine could not initialise: {e}")

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        if self._engine is None or not self._engine.is_ready():
            raise RuntimeError(
                "Qualcomm AI Hub engine not ready.\n"
                f"Reason: {self._init_error}\n\n"
                "📌 Setup steps:\n"
                "  pip install qai-hub\n"
                "  qai-hub configure --api_token YOUR_TOKEN\n"
                "  python src/qualcomm_aihub.py --step all\n"
                "  (sign up free: https://app.aihub.qualcomm.com/)"
            )
        return self._engine.detect(frame)

    def is_ready(self) -> bool:
        return self._engine is not None and self._engine.is_ready()

    def get_platform_info(self) -> dict:
        if self._engine:
            return self._engine.get_platform_info()
        return {
            "engine": "QualcommInferenceEngine",
            "model_format": "QNN Context Binary (INT8)",
            "runtime": "Qualcomm AI Hub + QNN SDK",
            "target_hardware": self.target_chipset,
            "status": "NOT READY — run: python src/qualcomm_aihub.py --step all",
            "aihub_portal": "https://app.aihub.qualcomm.com/",
            "is_target_hardware": True,
            "qualcomm_accelerated": True,
            "init_error": self._init_error,
        }


def create_engine(
    prefer_qualcomm: bool = False,
    qualcomm_model_path: str = "models/yolov8n_retailedge_qnn.bin",
    local_model_name: str = "yolov8n.pt",
    confidence_thresh: float = 0.35,
) -> InferenceEngine:
    """
    Smart factory: returns the best available inference engine.

    Logic:
      1. If prefer_qualcomm=True AND the QNN binary exists AND qai_hub is installed
         → returns QualcommInferenceEngine (real NPU inference)
      2. Otherwise
         → returns LocalInferenceEngine (YOLOv8n on CPU/GPU, laptop mode)

    This means the rest of the codebase never needs to know which engine is running.
    Swapping from laptop to Qualcomm hardware is a one-line config change.

    Parameters
    ----------
    prefer_qualcomm : bool
        Set True when running on Dragonwing hardware.
        Set False (default) for laptop simulation.
    qualcomm_model_path : str
        Path to the compiled QNN binary.
    local_model_name : str
        YOLOv8 model file for local inference.
    confidence_thresh : float
        Detection confidence threshold.

    Returns
    -------
    InferenceEngine — either QualcommInferenceEngine or LocalInferenceEngine
    """
    from pathlib import Path

    if prefer_qualcomm:
        qnn_path = Path(qualcomm_model_path)
        try:
            import qai_hub  # noqa: F401 — check if installed
            qc_engine = QualcommInferenceEngine(
                model_path=str(qnn_path),
                confidence_thresh=confidence_thresh,
            )
            if qc_engine.is_ready():
                logger.info("✅ Using Qualcomm AI Hub inference engine (QCS6490 NPU)")
                return qc_engine
            else:
                logger.warning(
                    "Qualcomm engine requested but not ready "
                    f"({qc_engine._init_error}). Falling back to local."
                )
        except ImportError:
            logger.warning("qai_hub not installed. Falling back to LocalInferenceEngine.")

    logger.info("Using LocalInferenceEngine (YOLOv8n on CPU — laptop simulation mode)")
    return LocalInferenceEngine(
        model_name=local_model_name,
        confidence_thresh=confidence_thresh,
    )

