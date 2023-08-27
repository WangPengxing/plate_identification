# 作者：水果好好吃哦
# 日期：2023/8/23
import os
import cv2
import numpy as np


def cal_mean_std(root):
    means = []
    stdevs = []

    data_list = os.listdir(root)

    num_imgs = 0
    for pic in data_list:
        num_imgs += 1
        img = cv2.imread(os.path.join(root, pic), cv2.IMREAD_GRAYSCALE)
        try:
            img.shape
        except:
            print(os.path.join(root, pic))
            print("Can not read this image !")
        img = img.astype(np.float32) / 255.
        means.append(img[:, :].mean())
        stdevs.append(img[:, :].std())

    means = np.mean(means)
    stdevs = np.mean(stdevs)

    return np.around(means, 3), np.around(stdevs, 3)


if __name__ == "__main__":
    # train数据集的mean、std
    train_path = "../data/CCPD_plate_data/train/images/"
    result = cal_mean_std(train_path)
    print(result)
