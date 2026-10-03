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
    Target Platform Adapter for Qualcomm Dragonwing Edge AI.
    Integrates with Qualcomm AI Runtime (QAIRT / QNN / SNPE) and Qualcomm AI Hub.
    In the local MVP development environment, this acts as the hardware deployment adapter/stub.
    """

    def __init__(self, model_dlc_path: str = "models/yolov8n_qualcomm.dlc", target_chipset: str = "Qualcomm Dragonwing"):
        self.model_dlc_path = model_dlc_path
        self.target_chipset = target_chipset
        self.is_hardware_available = False

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        On physical Dragonwing hardware: executes DLC/ONNX model via Qualcomm QNN/SNPE C++ / Python API.
        In simulation: provides transparent status warning without falsifying metrics.
        """
        if not self.is_hardware_available:
            raise RuntimeError(
                "Qualcomm Dragonwing hardware not detected. "
                "Use LocalInferenceEngine for local laptop simulation."
            )
        return [], 0.0

    def get_platform_info(self) -> dict:
        return {
            "engine": "QualcommInferenceEngine",
            "model_format": "Qualcomm Deep Learning Container (DLC) / QNN Graph",
            "runtime": "Qualcomm AI Runtime (QAIRT / QNN)",
            "optimization_toolchain": "Qualcomm AI Hub & AIMET (8-bit Quantization)",
            "target_hardware": self.target_chipset,
            "status": "Target Deployment Architecture Defined (Pending Dragonwing Hardware Provisioning)",
            "is_target_hardware": True,
            "qualcomm_accelerated": True
        }
