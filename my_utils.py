# 作者：水果好好吃哦
# 日期：2023/8/21
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from yolov5.utils.general import (CONFIG_DIR, FONT, LOGGER, Timeout, check_font, check_requirements, clip_coords,
                                  increment_path, is_ascii, is_chinese, try_except, xywh2xyxy, xyxy2xywh)
import os
import matplotlib
from pathlib import Path


def correction(img):
    img_copy = img.copy()
    color = {"lower": np.array([100, 80, 46]), "upper": np.array([124, 255, 255])}
    # 高斯滤波
    img_copy = cv2.GaussianBlur(img_copy, (5, 5), 0)
    # 图像颜色转换为hsv模式
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    # 计算用于膨胀或腐蚀的内核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 腐蚀
    img_copy = cv2.erode(img_copy, kernel, iterations=1)
    # 保留像素的值小于lower或者大于upper都变为(0, 0, 0),在此范围内的变为(255, 255, 255)
    img_copy = cv2.inRange(img_copy, color["lower"], color["upper"])
    # 提取二值图像的轮廓线
    edges = cv2.findContours(img_copy.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # 在边界中找出面积最大的区域
    max_area = max(edges, key=cv2.contourArea)
    # 计算该区域的最小外接矩形
    rect = cv2.minAreaRect(max_area)
    # 提取外接矩形的角点
    points = cv2.boxPoints(rect)
    # 对提取的角点进行顺时针排序
    points = np.float32(np.int64(points))
    points = [points[0], points[1], points[2], points[3]]
    lst = [points[0][0], points[1][0], points[2][0], points[3][0]]
    lst.sort(reverse=True)
    left, right, new_points = [], [], []
    while len(lst) > 2:
        for i in points:
            if i[0] == lst[-1] and (len(lst) > 2):
                left.append(i)
                lst.pop()
            else:
                right.append(i)
    if left[0][1] > left[1][1]:
        new_points.append(left[1])
        new_points.append(left[0])
    else:
        new_points = left
    if right[0][1] > right[1][1]:
        new_points.insert(1, right[1])
        new_points.insert(2, right[0])
    else:
        new_points.insert(1, right[0])
        new_points.insert(2, right[1])
    # 排好序的角点
    pst1 = np.array(new_points)
    x_min, x_max = min(pst1[:, 0]), max(pst1[:, 0])
    y_min, y_max = min(pst1[:, 1]), max(pst1[:, 1])
    x_min = x_min if x_min >= 0 else 0
    y_min = y_min if y_min >= 0 else 0
    pst2 = np.float32([(0, 0), (x_max - x_min, 0), (x_max - x_min, y_max - y_min), (0, y_max - y_min)])
    matrix = cv2.getPerspectiveTransform(pst1, pst2)
    plate = cv2.warpPerspective(img, matrix, (int(x_max - x_min), int(y_max - y_min)))
    return plate


def cv2ImgAddText(img, text, org, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(org, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def box_label(im, p1, p2, label='', color=(100, 100, 100), txt_color=(255, 255, 255)):
    lw = max(round(sum(im.shape) / 2 * 0.003), 2)
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 2, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p2[0], p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)  # 边框
        cv2.rectangle(im, p1, p2, color, thickness=-1, lineType=cv2.LINE_AA)  # 填充
        # cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 4, txt_color, thickness=tf)
        textSize = abs(int((p1[1] - p2[1]) * 0.75))
        im = cv2ImgAddText(im, label,
                           (p1[0], p1[1] - 2 - int(7 * lw) if outside else p1[1] + h + 2 - int(7 * lw)),
                           textColor=txt_color, textSize=textSize)
        return im


# Settings
RANK = int(os.getenv('RANK', -1))
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


def check_pil_font(font=FONT, size=10):
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception:  # download if missing
        check_font(font)
        try:
            return ImageFont.truetype(str(font), size)
        except TypeError:
            check_requirements('Pillow>=8.4.0')  # known issue https://github.com/ultralytics/yolov5/issues/5374


class Annotator:
    if RANK in (-1, 0):
        check_pil_font()  # download TTF if necessary

    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.pil = pil or not is_ascii(example) or is_chinese(example)
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font='Arial.Unicode.ttf',
                                       size=font_size or max(round(sum(self.im.size) / 2 * 0.025), 12))
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle((box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1), fill=color)
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


if __name__ == "__main__":
    path = "./yolov5/data/CCPD_data/train/images/ccpd_000000022.jpg"
    img = cv2.imread(path)
    img = correction(img)
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
