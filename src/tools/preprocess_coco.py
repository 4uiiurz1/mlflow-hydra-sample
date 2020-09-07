from pycocotools.coco import COCO
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ann_path', default='D:/datasets/coco/annotations/person_keypoints_val2017.json')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    coco = COCO(args.ann_path)

    img_ids = coco.getImgIds()
    print(coco.dataset['info'])


if __name__ == '__main__':
    main()
