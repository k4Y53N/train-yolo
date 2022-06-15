import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union
from .core.configer import YOLOConfiger
from .core.models import build_model


class DetectResult:
    def __init__(self, boxes, scores, classes):
        self.boxes = boxes
        self.scores = scores
        self.classes = classes


class Detector:
    def __init__(self, config_path: Union[str, Path]):
        configer = YOLOConfiger(config_path)
        self.model = build_model(configer, training=False)
        self.size = configer.size
        self.classes = configer.classes
        self.score_threshold = configer.score_threshold
        self.iou_threshold = configer.iou_threshold
        self.max_total_size = configer.max_total_size
        self.max_output_size_per_class = configer.max_output_size_per_class

    def detect(self, image: np.ndarray, is_cv2=True):
        height, width = image.shape[:2]
        data = self.normalization(image, is_cv2=is_cv2)
        pred = self.model(data)
        batch_size, num_boxes = pred.shape[:2]

        nms_boxes, nms_scores, nms_classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(pred[:, :, :4], (batch_size, num_boxes, 1, 4)),
            scores=pred[:, :, 4:],
            max_output_size_per_class=self.max_output_size_per_class,
            max_total_size=self.max_total_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
        )
        valid_detections = valid_detections[0]
        nms_boxes = tf.reshape(nms_boxes, (-1, 4))
        nms_classes = tf.reshape(nms_classes, (-1, 1))
        nms_scores = tf.reshape(nms_scores, (-1))[:valid_detections].numpy().tolist()

        valid_data = tf.concat(
            (nms_boxes, nms_classes),
            axis=1
        )[:valid_detections]
        result = np.empty(valid_data.shape, dtype=np.int)
        for index, valid in enumerate(valid_data):
            # pred boxes = [y1, x1, y2, x2]
            # true boxes = [x1, y1, x2, y2]
            result[index][0] = valid[1] * width
            result[index][1] = valid[0] * height
            result[index][2] = valid[3] * width
            result[index][3] = valid[2] * height
            result[index][4] = valid[4]

        return DetectResult(boxes=result.tolist(), scores=nms_scores, classes=self.classes)

    def normalization(self, image: np.ndarray, is_cv2=True) -> np.ndarray:
        if is_cv2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return cv2.resize(image, (self.size, self.size))[np.newaxis, :] / 255.
