import json
import argparse
import os
import pickle
import sys
import logging as log
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from threading import Thread
from configparser import ConfigParser


class MakeYoloConfig:
    def __init__(
            self,
            config_file_name: str,
            classes_file_path: str,
            sys_config_path: str = './sys.ini',
            pretrain_file_path: str = '',
            model_type: str = 'yolov4',
            frame_work: str = 'tf',
            size: int = 416,
            batch_size: int = 4,
            epoch: int = 30,
            train_size: int = 1000,
            val_size: int = 200,
            tiny: bool = False,
            score_threshold: float = 0.25,
            max_total_size: int = 50,
            max_output_size_per_class: int = 50
    ):
        self.sys_config = ConfigParser()
        self.sys_config_file = Path(sys_config_path)
        self.name = config_file_name
        self.classes_file = Path(classes_file_path)
        self.pretrain_file = Path(pretrain_file_path) if pretrain_file_path else None
        self.model_type = model_type
        self.frame_work = frame_work
        self.size = size
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self.epoch = epoch
        self.warmup_epochs = 2
        self.first_stage_epochs = int(epoch * 1 / 5) if pretrain_file_path else 0
        self.second_stage_epochs = epoch - self.first_stage_epochs
        self.train_size = train_size
        self.test_size = val_size
        self.tiny = tiny
        self.max_total_size = max_total_size
        self.max_output_size_per_class = max_output_size_per_class
        self.yolo_config = {}
        self.classes = []
        self.anchors = []
        self.anchors_v3 = []
        self.anchors_tiny = []
        self.yolo_config_file: Path = Path()
        self.model_save_dir: Path = Path()
        self.train_set_dir: Path = Path()
        self.train_annotation_file: Path = Path()
        self.test_set_dir: Path = Path()
        self.test_annotation_file: Path = Path()
        self.checkpoints_save_dir: Path = Path()
        self.weights_save_dir: Path = Path()
        self.configs_save_dir: Path = Path()
        self.train_bbox_save_dir: Path = Path()
        self.test_bbox_save_dir: Path = Path()
        self.train_bbox_file: Path = Path()
        self.test_bbox_file: Path = Path()
        self.logdir: Path = Path()

    def make(self):
        self.check_all_paths()
        classes = self.load_classes()
        train = Thread(
            target=self.write,
            args=(self.train_annotation_file, self.train_bbox_file, classes, True)
        )
        test = Thread(
            target=self.write,
            args=(self.test_annotation_file, self.test_bbox_file, classes, False)
        )
        train.start()
        test.start()
        train.join()
        test.join()

        if not (self.train_bbox_file.is_file() and self.test_bbox_file.is_file()):
            self.train_bbox_file.unlink(missing_ok=True)
            self.test_bbox_file.unlink(missing_ok=True)
            raise RuntimeError('Writing Train bbox file or Test box file fail')

        self.write_yolo_config()

    def check_all_paths(self):
        if self.sys_config_file.is_file():
            log.info(f'System config file path: {self.sys_config_file.absolute()}')
            self.sys_config.read(self.sys_config_file.name)
        else:
            raise FileNotFoundError(self.sys_config_file.name)

        self.train_set_dir = Path(self.sys_config['Annotations']['train_set_dir'])
        self.train_annotation_file = Path(self.sys_config['Annotations']['train_annotation_path'])
        self.test_set_dir = Path(self.sys_config['Annotations']['test_set_dir'])
        self.test_annotation_file = Path(self.sys_config['Annotations']['test_annotation_path'])
        self.checkpoints_save_dir = Path(self.sys_config['Save_dir']['checkpoints'])
        self.weights_save_dir = Path(self.sys_config['Save_dir']['weights'])
        self.configs_save_dir = Path(self.sys_config['Save_dir']['configs'])
        self.yolo_config_file = self.configs_save_dir / Path(self.name).with_suffix('.json')
        self.model_save_dir = self.checkpoints_save_dir / self.name
        self.train_bbox_save_dir = Path(self.sys_config['Save_dir']['train_processed_data'])
        self.test_bbox_save_dir = Path(self.sys_config['Save_dir']['test_processed_data'])
        self.train_bbox_file = self.train_bbox_save_dir / Path(self.name).with_suffix('.bbox')
        self.test_bbox_file = self.test_bbox_save_dir / Path(self.name).with_suffix('.bbox')
        self.logdir = Path(self.sys_config['Save_dir']['logs']) / Path(self.name)

        checking_exist_dir_group = (
            self.train_set_dir,
            self.test_set_dir,
        )

        checking_exist_file_group = (
            self.classes_file,
            self.train_annotation_file,
            self.test_annotation_file,
        )

        make_dirs = (
            self.configs_save_dir,
            self.checkpoints_save_dir,
            self.weights_save_dir,
            self.train_bbox_save_dir,
            self.test_bbox_save_dir,
            self.logdir
        )

        rm_files = (
            self.train_bbox_file,
            self.test_bbox_file
        )

        for ex_dir in checking_exist_dir_group:
            if not ex_dir.is_dir():
                raise NotADirectoryError(ex_dir.absolute())

        for ex_file in checking_exist_file_group:
            if not ex_file.is_file():
                raise FileNotFoundError(str(ex_file.absolute()))

        for mk_dir in make_dirs:
            if not mk_dir.is_dir():
                os.makedirs(mk_dir.absolute(), exist_ok=True)
                log.info('Make dir %s' % str(mk_dir.absolute()))

        for rm_file in rm_files:
            if rm_file.is_file():
                rm_file.unlink(missing_ok=True)
                log.info('Remove exist file: %s' % str(rm_file.absolute()))

        if self.pretrain_file:
            if not self.pretrain_file.is_file():
                raise FileNotFoundError(str(self.pretrain_file.absolute()))

    def load_classes(self):
        with self.classes_file.open('r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def write(self, annotation_file: Path, bbox_file: Path, classes: list, training: bool):
        try:
            self.write_coco2yolo(annotation_file, bbox_file, classes, training)
        except Exception as e:
            log.error(f'{e.__class__.__name__}({e.args})', exc_info=True)
            self.train_bbox_file.unlink(missing_ok=True)
            self.test_bbox_file.unlink(missing_ok=True)

    def write_coco2yolo(self, annotation_file: Path, bbox_file: Path, classes: list, training: bool):
        images, classes = self.load_annotation_file(annotation_file, classes)
        classes = {
            class_name: index
            for index, class_name in enumerate(classes)
        }
        done = 0
        anchor_boxes = []

        if training:
            data_save_dir = self.train_set_dir
            set_size = self.train_size
        else:
            data_save_dir = self.test_set_dir
            set_size = self.test_size

        with bbox_file.open('w') as bf:
            for image in images:
                image_file = data_save_dir / image['file_name']

                if done >= set_size:
                    break
                if not image_file.is_file() or len(image['items']) < 1:
                    continue

                bf.write(str(image_file.absolute()) + ' ')

                for item in image['items']:
                    bbox = item['bbox']
                    x_min = int(bbox[0])
                    y_min = int(bbox[1])
                    x_max = x_min + int(bbox[2])
                    y_max = y_min + int(bbox[3])
                    anchor_boxes.append(
                        (bbox[2],
                         bbox[3],
                         image['width'],
                         image['height'])
                    )

                    x_min, y_min, x_max, y_max = str(x_min), str(y_min), str(x_max), str(y_max)
                    label = str(classes[item['category_name']])
                    bf.write(','.join([x_min, y_min, x_max, y_max, label]) + ' ')

                bf.write('\n')
                done += 1

        if done < 1:
            raise RuntimeError('bbox file did not have any images')
        log.info(f'{bbox_file.name} have {done} images')

        if training:
            self.calculate_anchor_box(anchor_boxes)
            self.classes = list(classes.keys())

    def load_annotation_file(self, file: Path, classes: list):
        log.info(f'Start loading annotation file: {file.absolute()}')
        data = None
        with file.open('r') as f:
            if '.pickle' == file.suffix:
                data = pickle.load(f)
            elif '.json' == file.suffix:
                data = json.load(f)
        if data is None:
            raise RuntimeError('Unknown file suffix')
        np.random.shuffle(data['images'])
        np.random.shuffle(data['annotations'])
        log.info(f'loading annotation file: {file.absolute()} finish')

        return self._filter(data, classes)

    def _filter(self, data, classes: list):
        cats = {
            cat['id']: cat['name']
            for cat in data['categories'] if cat['name'] in classes
        }

        if len(cats) < 1:
            raise RuntimeError('Can not find any classes in annotation file')

        images = {
            image['id']: {
                'file_name': image['file_name'],
                'width': image['width'],
                'height': image['height'],
                'items': []
            }
            for image in data['images']
        }

        for anno in data['annotations']:
            if anno['category_id'] in cats.keys():
                image_id = anno['image_id']
                images[image_id]['items'].append(
                    {
                        'bbox': anno['bbox'],
                        'category_name': cats[anno['category_id']]
                    }
                )

        images = (
            img for img in images.values() if len(img['items']) > 0
        )

        cats = (class_name for class_name in cats.values())

        return images, cats

    def calculate_anchor_box(self, bbox_wh_img_size):
        w_h = tuple(
            (
                box[0] / box[2] * self.size,
                box[1] / box[3] * self.size,
            )
            for box in bbox_wh_img_size
        )

        self.anchors_tiny = self.calculate_kmeans(w_h, 6)
        self.anchors = self.calculate_kmeans(w_h, 9)
        self.anchors_v3 = self.anchors

    @staticmethod
    def calculate_kmeans(w_h, k):
        x = np.array(w_h)
        kmeans = KMeans(n_clusters=k).fit(x)
        cluster = kmeans.cluster_centers_
        boxes_size = [box[0] * box[1] for box in cluster]
        sort_args = np.argsort(boxes_size)
        arranged_cluster = np.zeros_like(cluster, int)

        for i, arg in enumerate(sort_args):
            arranged_cluster[i] = cluster[arg]

        return arranged_cluster.reshape(-1).tolist()

    def write_yolo_config(self):
        self.yolo_config = {
            'name': self.name,
            'model_path': self.model_save_dir.as_posix(),
            'weight_path': (self.weights_save_dir / self.name).with_suffix('.h5').as_posix(),
            'logdir': self.logdir.as_posix(),
            'frame_work': self.frame_work,
            'model_type': self.model_type,
            'size': self.size,
            'tiny': self.tiny,
            'max_output_size_per_class': self.max_output_size_per_class,
            'max_total_size': self.max_total_size,
            'iou_threshold': 0.5,
            'score_threshold': self.score_threshold,
            'YOLO': {
                'CLASSES': self.classes,
                'ANCHORS': self.anchors,
                'ANCHORS_V3': self.anchors_v3,
                'ANCHORS_TINY': self.anchors_tiny,
                'STRIDES': [8, 16, 32],
                'STRIDES_TINY': [16, 32],
                'XYSCALE': [1.2, 1.1, 1.05],
                'XYSCALE_TINY': [1.05, 1.05],
                'ANCHOR_PER_SCALE': 3,
                'IOU_LOSS_THRESH': 0.5,
            },
            'TRAIN': {
                'ANNOT_PATH': self.train_bbox_file.as_posix(),
                'BATCH_SIZE': self.batch_size,
                'INPUT_SIZE': self.size,
                'DATA_AUG': True,
                'LR_INIT': 1e-03,
                'LR_END': 1e-06,
                'WARMUP_EPOCHS': self.warmup_epochs,
                'INIT_EPOCH': 0,
                'FIRST_STAGE_EPOCHS': self.first_stage_epochs,
                'SECOND_STAGE_EPOCHS': self.second_stage_epochs,
                'PRETRAIN': self.pretrain_file.as_posix() if self.pretrain_file else None,
            },
            'TEST': {
                'ANNOT_PATH': self.test_bbox_file.as_posix(),
                'BATCH_SIZE': self.batch_size,
                'INPUT_SIZE': self.size,
                'DATA_AUG': False,
                'SCORE_THRESHOLD': self.score_threshold,
                'IOU_THRESHOLD': 0.5,
            }
        }

        with self.yolo_config_file.open('w') as f:
            json.dump(self.yolo_config, f)

        if sys.platform.startswith('linux'):
            cmd = 'python3 -m json.tool %s' % (str(self.yolo_config_file.absolute()))
        else:
            cmd = 'python -m json.tool %s' % (str(self.yolo_config_file.absolute()))
        os.system(cmd)
        log.info(f'Write YOLO Config to {self.yolo_config_file.absolute()}')


if __name__ == '__main__':
    log.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=log.INFO
    )

    parser = argparse.ArgumentParser(description='Make YOLO config file')
    parser.add_argument('name', type=str, help='Config file and model name')
    parser.add_argument('classes', type=str, help='Classes file path')
    parser.add_argument('-s', '--size', type=int, default=416,
                        choices=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
                        help='Image detect size')
    parser.add_argument('-m', '--model', type=str, default='yolov4', choices=['yolov4', 'yolov3'], help='Model type')
    parser.add_argument('-f', '--frame_work', type=str, default='tf', choices=['tf', 'trt', 'tflite'],
                        help='Frame work type')
    parser.add_argument('-sc', '--score_threshold', type=float, default=0.25, help='Object score threshold')
    parser.add_argument('-t', '--tiny', action='store_true', help='Tiny model?')
    parser.add_argument('-p', '--pretrain', type=str, default='', help='Pretrain weight path')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-ep', '--epoch', type=int, default=30, help='Total of epoch')
    parser.add_argument('-ts', '--train_size', type=int, default=1000, help='Train epoch size')
    parser.add_argument('-vs', '--val_size', type=int, default=200, help='Val epoch size')
    parser.add_argument('-mts', '--max_total_size', type=int, default=50, help='Max number detections of total')
    parser.add_argument('-mpc', '--max_per_class', type=int, default=50, help='Max number detections of per class')
    args = parser.parse_args()
    try:
        for arg in vars(args):
            print(arg, getattr(args, arg))
        myc = MakeYoloConfig(
            args.name,
            args.classes,
            sys_config_path='./sys.ini',
            size=args.size,
            model_type=args.model,
            frame_work=args.frame_work,
            tiny=args.tiny,
            pretrain_file_path=args.pretrain,
            batch_size=args.batch_size,
            epoch=args.epoch,
            train_size=args.train_size,
            val_size=args.val_size,
            score_threshold=args.score_threshold,
            max_total_size=args.max_total_size,
            max_output_size_per_class=args.max_per_class
        )
        myc.make()
    except Exception as E:
        log.error(f'{E.__class__.__name__}({E.args})', exc_info=True)
