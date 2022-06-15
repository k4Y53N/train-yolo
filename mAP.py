import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score
from typing import Dict
from src.detector import Detector
from src.core.configer import YOLOConfiger


def calc_iou(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0] + gt_box[2], pred_box[0] + pred_box[2]),
                              min(gt_box[1] + gt_box[3], pred_box[1] + pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection

    iou = intersection / union

    return iou


def parse_line(line: str):
    line = line.strip().split()
    image_path = line[0]
    boxes = [
        [float(n) for n in box.split(',')]
        for box in line[1:]
    ]
    return image_path, boxes


def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = ["positive" if score > threshold else "negative" for score in pred_scores]

        precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


class AP:
    def __init__(self):
        self.pred_scores = []

    def calc_ap(self, threshold):
        y_true = ['positive' for _ in self.pred_scores]
        precisions, recalls = precision_recall_curve(y_true, self.pred_scores, threshold)
        precisions.append(1)
        recalls.append(0)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        return np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])


class mAP:
    def __init__(self, config_path, iou_threshold):
        self.gt_true_data_path = YOLOConfiger(config_path).test_annot_path
        self.detector = Detector(config_path)
        self.detector.score_threshold = 0
        self.ap_group: Dict[str, AP] = {
            name: AP()
            for name in self.detector.classes
        }
        self.iou_threshold = iou_threshold

    def calc(self):
        for image_path, gt_boxes in self.parse_ground_true_data():
            image = cv2.imread(image_path)
            result = self.detector.detect(image)
            pred_boxes = result.boxes
            scores = result.scores
            self.add_ap_pred_scores(gt_boxes, pred_boxes, scores)

        total_ap = 0
        threshold = np.arange(0.2, 1, 0.01)

        for name, ap in self.ap_group.items():
            num_ap = ap.calc_ap(threshold)
            print(f'class: {name} AP: {num_ap}')
            total_ap += num_ap
        return total_ap / len(self.ap_group)

    def parse_ground_true_data(self):
        with open(self.gt_true_data_path, 'r') as f:
            ground_true_data = (
                parse_line(line)
                for line in f.readlines()
            )
        return ground_true_data

    def add_ap_pred_scores(self, gt_boxes, pred_boxes, scores):
        for class_index, class_name in enumerate(self.detector.classes):
            ap = self.ap_group[class_name]

            pred_boxes = [
                [*box, score]
                for box, score in zip(pred_boxes, scores) if box[4] == class_index
            ]

            for gt_box in gt_boxes:
                ap.pred_scores.append(self.get_match_box_pred_score(gt_box, pred_boxes))

    def get_match_box_pred_score(self, gt_box, pred_boxes):
        max_iou_score = 0
        pred_score = 0
        for pred_box in pred_boxes:
            iou_score = calc_iou(gt_box, pred_box[:4])
            if iou_score > max_iou_score:
                max_iou_score = iou_score
                pred_score = pred_box[5]
        if max_iou_score >= self.iou_threshold:
            return pred_score
        return 0


if __name__ == '__main__':
    m_ap = mAP('configs/person-416-pre.json', 0.5)
    m = m_ap.calc()
    print(f'mAP = {m}')
