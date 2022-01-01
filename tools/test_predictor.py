from tools.demo_track import Predictor
from torch._C import device
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tracking_utils.timer import Timer
import cv2
import argparse
import torch

def make_parser():
    parser = argparse.ArgumentParser("Test Predictor!")
    # parser.add_argument(
    #     "demo", default="image", help="demo type, eg. image, video and webcam"
    # )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # parser.add_argument(
        # "--path", default="./videos/palace.mp4", help="path to images or video"
    # )
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
    return parser

def draw_yolox_outputs(img, outputs, img_info):
    #assume batch size is 1
    assert(len(outputs) == 1)
    outputs = outputs[0].cpu().numpy()
    ratio = img_info["ratio"]
    for x1, y1, x2, y2, obj_conf, class_conf, class_pred in outputs:
        x1, y1, x2, y2 = [x/ratio for x in [x1, y1, x2, y2]]
        x1, y1, x2, y2, class_pred = [int(x) for x in [x1, y1, x2, y2, class_pred]]
        if class_conf * obj_conf > args.conf_thresh:
            print(x1, y1, x2, y2, class_conf, class_pred)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return img

def main(exp, args):
    device = torch.device("cuda" if args.device == "gpu" else "cpu")

    model = exp.get_model().to(device)
    model.eval()

    ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])

    if args.fuse:
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16
    predictor = Predictor(model, exp, trt_file=None, decoder=None, device=device, fp16=args.fp16)
    timer = Timer()
    img = cv2.imread(args.img_path)
    outputs, img_info = predictor.inference(img, timer)
    img = draw_yolox_outputs(img, outputs, img_info)
    cv2.imwrite(args.output_path, img)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
