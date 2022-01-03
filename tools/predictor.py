import os.path as osp
import os
from loguru import logger
from yolox.data.data_augment import preproc
from yolox.utils import fuse_model, get_model_info, postprocess

import torch
import cv2
import numpy as np

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = (ratio, ratio)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

import darknet.darknet as darknet

class YOLOv4Predictor(Predictor):
    def __init__(
        self,
        network,
        class_names,
        class_colors,
        exp
    ):
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.network = network
        self.class_names = class_names
        self.class_colors = class_colors
        self.class_name_dic = {x: i for i, x in enumerate(class_names)}


    def inference(self, img, timer):
        net_width = darknet.network_width(self.network)
        net_height = darknet.network_height(self.network)

        # img_info
        img_info = {"id": 0}
        img_info["file_name"] = None
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img_info["ratio"] = (net_width/width, net_height/height)
        # preprocessing
        darknet_image = darknet.make_image(net_width, net_height, 3)

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (net_width, net_height),
                                interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        # inference
        timer.tic()
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.confthre, nms=self.nmsthre)
        darknet.free_image(darknet_image)

        outputs = []
        for label, confidence, bbox in detections:
            class_id = self.class_name_dic[label]
            left, top, right, bottom = darknet.bbox2points(bbox)
            confidence = float(confidence)/100.0
            outputs.append([left, top, right, bottom, confidence, class_id])
        # wrap as batched list
        outputs = [outputs]
        outputs = np.array(outputs)
        return outputs, img_info


def get_yolox_predictor(exp, args, output_dir):
    if isinstance(args.device, str):
        args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    return Predictor(model, exp, trt_file=trt_file, decoder=decoder, device=args.device, fp16=args.fp16)

def get_darknet_predictor(exp, args):
    network, class_names, class_colors = exp.get_model(
        args.config_file,
        args.data_file,
        args.weights_file
    )
    return YOLOv4Predictor(network, class_names, class_colors, exp)

def get_predictor(exp, args) -> Predictor:
    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.model_type == 'yolox':
        return get_yolox_predictor(exp, args, output_dir)
    elif args.model_type == 'darknet':
        return get_darknet_predictor(exp, args)
    else:
        raise Exception("unreachable")
