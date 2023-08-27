# 作者：水果好好吃哦
# 日期：2023/8/19
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob as glob
import random


class Visu_Data():
    def __init__(self, class_names, colors, image_paths, label_paths, num_samples=10):
        self.class_names = class_names
        self.colors = colors
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.num_samples = num_samples

    def yolo2bbox(self, bboxes):
        xmin, ymin = bboxes[0] - bboxes[2] / 2, bboxes[1] - bboxes[3] / 2
        xmax, ymax = bboxes[0] + bboxes[2] / 2, bboxes[1] + bboxes[3] / 2
        return xmin, ymin, xmax, ymax

    def plot_box(self, image, bboxes, labels):
        # Need the image height and width to denormalize
        # the bounding box coordinates
        h, w, _ = image.shape
        for box_num, box in enumerate(bboxes):
            x1, y1, x2, y2 = self.yolo2bbox(box)
            # denormalize the coordinates
            xmin = int(x1 * w)
            ymin = int(y1 * h)
            xmax = int(x2 * w)
            ymax = int(y2 * h)
            width = xmax - xmin
            height = ymax - ymin

            class_name = self.class_names[int(labels[box_num])]

            cv2.rectangle(
                image,
                (xmin, ymin), (xmax, ymax),
                color=self.colors[self.class_names.index(class_name)],
                thickness=2
            )

            font_scale = min(1, max(3, int(w / 500)))
            font_thickness = min(2, max(10, int(w / 50)))

            p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
            # Text width and height
            tw, th = cv2.getTextSize(
                class_name,
                0, fontScale=font_scale, thickness=font_thickness
            )[0]
            p2 = p1[0] + tw, p1[1] + -th - 10
            cv2.rectangle(
                image,
                p1, p2,
                color=self.colors[self.class_names.index(class_name)],
                thickness=-1,
            )
            cv2.putText(
                image,
                class_name,
                (xmin + 1, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        return image

    def plot(self):
        all_training_images = glob.glob(self.image_paths)
        all_training_labels = glob.glob(self.label_paths)
        all_training_images.sort()
        all_training_labels.sort()

        num_images = len(all_training_images)

        plt.figure(figsize=(15, 12))
        for i in range(self.num_samples):
            j = random.randint(0, num_images - 1)
            image = cv2.imread(all_training_images[j])
            with open(all_training_labels[j], 'r') as f:
                bboxes = []
                labels = []
                label_lines = f.readlines()
                for label_line in label_lines:
                    label = label_line[0]
                    bbox_string = label_line[3:]
                    x_c, y_c, w, h = bbox_string.split(' ')[:4]
                    x_c = float(x_c)
                    y_c = float(y_c)
                    w = float(w)
                    h = float(h)
                    bboxes.append([x_c, y_c, w, h])
                    labels.append(label)
            result_image = self.plot_box(image, bboxes, labels)
            plt.subplot(3, 8, i + 1)
            plt.imshow(result_image[:, :, ::-1])
            plt.axis('off')
        plt.subplots_adjust(wspace=0)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    num_0 = input("需要可视化哪个数据集？\n1：CCPD_data；2：BITVehicle_data\n")
    num_1 = input("需要可视化训练集还是评估集？\n1：train；2：val\n")
    dic_path = {"11": "../yolov5/data/CCPD_data/train/", "12": "../yolov5/data/CCPD_data/val/",
                "21": "../yolov5/data/BITVehicle_data/train/", "22": "../yolov5/data/BITVehicle_data/val/"}
    root = dic_path[f"{num_0}{num_1}"]
    dic_class = {"1": ["single"], "2": ['bus', 'microbus', 'minivan', 'sedan', 'suv', 'truck']}
    class_names = dic_class[num_0]
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))
    root_imgs = f"{root}/images/*"
    root_labels = f"{root}/labels/*"
    visu_data = Visu_Data(class_names, colors, root_imgs, root_labels, num_samples=24)
    visu_data.plot()
