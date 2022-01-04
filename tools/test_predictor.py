from tools.predictor import get_predictor
from yolox.exp import get_exp
from yolox.tracking_utils.timer import Timer
import cv2
import torch
import argparse
import numpy as np

def make_parser():
    parser = argparse.ArgumentParser("Test Predictor!")
    # parser.add_argument(
    #     "demo", default="image", help="demo type, eg. image, video and webcam"
    # )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # parser.add_argument(
        # "--path", default="./videos/palace.mp4", help="path to images or video"
    # )
    parser.add_argument("--model_type", type=str, default="yolox", choices=["yolox", "darknet"])
    parser.add_argument(
        "--img_path", default="./images/image_0090.jpg"
    )
    parser.add_argument(
        "--output_path", default="./bboxes.jpg"
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--conf_thresh",
        default=0.75,
        type=float,
        help="threshold for drawing detection box"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--config_file",
        help="darknet config file(.cfg)",
        default=None
    )
    parser.add_argument(
        "--data_file",
        help="darknet data file(.data)",
        default=None
    )
    parser.add_argument(
        "--weights_file",
        help="darknet weights file(.weights)",
        default=None
    )

    return parser

def draw_bboxes(img, outputs, img_info):
    #assume batch size is 1
    assert(len(outputs) == 1)
    outputs = outputs[0]
    if isinstance(outputs, torch.Tensor):
        # yoloX
        # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        outputs = outputs.cpu().numpy()
        scores = outputs[:, 4] * outputs[:, 5]
        bboxes = outputs[:, :4]
    else:
        # darknet
        # (x1, y1, x2, y2, bbox_conf, class_pred)
        scores = outputs[:, 4]
        bboxes = outputs[:, :4]

    print(outputs.shape)
    ratio = img_info["ratio"]
    for ((x1, y1, x2, y2), score) in zip(bboxes, scores):
        x1, x2 = [x/ratio[0] for x in [x1, x2]]
        y1, y2 = [y/ratio[1] for y in [y1, y2]]
        x1, y1, x2, y2 = [int(x) for x in [x1, y1, x2, y2]]
        if score > args.conf_thresh:
            print(x1, y1, x2, y2, score)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    timer = Timer()
    img = cv2.imread(args.img_path)
    predictor = get_predictor(exp, args)
    outputs, img_info = predictor.inference(img, timer)
    img = draw_bboxes(img, outputs, img_info)
    cv2.imwrite(args.output_path, img)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
