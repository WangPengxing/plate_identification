# 作者：水果好好吃哦
# 日期：2023/8/24

import argparse
import os
import sys
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from torchvision import transforms
from my_utils import correction, box_label, Annotator
from crnn_ctc.net import CRNN_Net as ocr_net
from crnn_ctc.lib import convert, alphabets
from yolov5.utils.augmentations import letterbox
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                                  increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer,
                                  xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights_vehicle=ROOT / 'weights/vehicle_yolov5s.pt',
        weights_ccpd=ROOT / 'weights/ccpd_yolov5s.pt',
        weights_plate=ROOT / 'weights/plate_crnn.pth',
        source=ROOT / 'test/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'test/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        save_img=True,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=ROOT / 'test/runs',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 加载vehicle模型
    device = select_device(device)
    model = DetectMultiBackend(weights_vehicle, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # 加载ccpd模型
    model_ccpd = DetectMultiBackend(weights_ccpd, device=device, dnn=dnn, data=data)
    # 加载ocr模型
    model_ocr = ocr_net.CRNN(class_num=67)
    model_ocr.load_state_dict(torch.load(weights_plate, map_location=device))
    model_ocr.eval()

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 开始推理
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    model_ccpd.warmup(imgsz=(1, 3, 320, 320), half=half)
    dt, seen = [0.0, 0.0, 0.0], 0
    # 七种边界框的颜色，六类车辆、一类车牌
    np.random.seed(1)
    colors = np.random.randint(0, 255, size=(len(names)+1, 3))
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):  # 每张图片
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=3, pil=True, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):  # 在图片中检测到的车辆边界框
                    pt1 = (int(xyxy[0]), int(xyxy[1]))
                    pt2 = (int(xyxy[2]), int(xyxy[3]))

                    # 截取锚框内的图像，源图像的拷贝是im0，锚框左上坐标是pt1，右下坐标是pt2
                    im0_vehicle = im0[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                    im_vehicle = letterbox(im0_vehicle, new_shape=(320, 320))[0]
                    im_vehicle = im_vehicle.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im_vehicle = np.ascontiguousarray(im_vehicle)

                    im_vehicle = torch.from_numpy(im_vehicle).to(device)
                    im_vehicle = im_vehicle.half() if half else im_vehicle.float()  # uint8 to fp16/32
                    im_vehicle /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im_vehicle.shape) == 3:
                        im_vehicle = im_vehicle[None]  # expand for batch dim
                    pred_ccpd = model_ccpd(im_vehicle, augment=augment, visualize=visualize)
                    pred_ccpd = non_max_suppression(pred_ccpd, conf_thres, iou_thres, classes, agnostic_nms,
                                                    max_det=max_det)

                    for ii, det_ccpd in enumerate(pred_ccpd):   # 每个车牌
                        if len(det_ccpd):
                            # Rescale boxes from 320*320 to im0_vehicle size
                            det_ccpd[:, :4] = scale_coords(im_vehicle.shape[2:], det_ccpd[:, :4],
                                                           im0_vehicle.shape).round()

                            # Write results
                            for *xyxy, conf, clss in reversed(det_ccpd):    # 每个车牌的边界框
                                pt11 = (int(xyxy[0]), int(xyxy[1]))
                                pt22 = (int(xyxy[2]), int(xyxy[3]))

                                # 截取且矫正车牌
                                im0_plate = im0_vehicle[pt11[1]:pt22[1], pt11[0]:pt22[0]]
                                # im0_plate = correction(im0_plate)
                                # 图像灰度化且转为张量
                                im0_plate = cv2.cvtColor(im0_plate, cv2.COLOR_BGR2GRAY)
                                im0_plate = cv2.resize(im0_plate, (100, 32))
                                im0_plate = transforms.ToTensor()(im0_plate).to(device)
                                im0_plate = im0_plate.view(1, 1, 32, 100)
                                result = model_ocr(im0_plate)

                                converter = convert.StrLabelConverter(alphabets.alphabets)
                                preds_size = torch.IntTensor([result.size(0)] * result.size(1))
                                _, preds = result.max(2)
                                preds = preds.transpose(1, 0).contiguous().view(-1)
                                plate = converter.ocr_decode(preds.data, preds_size.data)

                                # plate标签、ccpd边界框在源图中的 坐标
                                color = colors[6]
                                target = str(plate)
                                pt1_ccpd, pt2_ccpd = (pt1[0] + pt11[0], pt1[1] + pt11[1]), (
                                pt1[0] + pt22[0], pt1[1] + pt22[1])
                                # 可视化
                                annotator.box_label(pt1_ccpd+pt2_ccpd, target, color=tuple(color))

                    # vehicle标签、边界框坐标
                    c = int(cls)
                    color = colors[c]
                    label = f'{names[c]} {conf:.2f}'
                    pt1_vehicle, pt2_vehicle = pt1, pt2
                    annotator.box_label(pt1_vehicle + pt2_vehicle, label, color=tuple(color))

            # 保存每一张测试结果
            im0 = annotator.result()
            if save_img:
                # cv2.imshow("111", im0)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_vehicle', nargs='+', type=str, default=ROOT / 'weights/vehicle_yolov5s.pt',
                        help='model path(s)')
    parser.add_argument('--weights_ccpd', nargs='+', type=str, default=ROOT / 'weights/ccpd_yolov5s.pt',
                        help='model path(s)')
    parser.add_argument('--weights_plate', nargs='+', type=str, default=ROOT / 'weights/plate_crnn.pth',
                        help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'test/video', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'test/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    return run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    points = main(opt)
