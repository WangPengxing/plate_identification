# 作者：水果好好吃哦
# 日期：2023/8/19
# ccpd_base: 21000
# ccpd_blur: 2100
# ccpd_challenge: 2100
# ccpd_db: 2100
# ccpd_fn: 2100
# ccpd_rotate: 2100
# ccpd_tilt: 2100
# ccpd_weather: 2100

from shutil import copyfile
import os
import random


def select_data(src_path, dst_path_1, dst_path_2, num):
    dirs = os.listdir(src_path)
    random.seed(0)
    data_index = random.sample(range(len(dirs)), num)
    data_index.sort(reverse=True)
    counter = 0
    for image in dirs:
        if counter == data_index[-1]:
            ret = data_index.pop()
            if len(data_index) <= num/10:
                copyfile(src_path + f"{image}", dst_path_2 + f"{image}")
            else:
                copyfile(src_path + f"{image}", dst_path_1 + f"{image}")
        if not data_index:
            break
        counter += 1


if __name__ == "__main__":
    # CCPD数据集文件夹位置
    root = "D:/BaiduNetdiskDownload/CCPD2019.tar/CCPD2019/"
    base_path = root + "ccpd_base/"
    blur_path = root + "ccpd_blur/"
    challenge_path = root + "ccpd_challenge/"
    db_path = root + "ccpd_db/"
    fn_path = root + "ccpd_fn/"
    rotate_path = root + "ccpd_rotate/"
    tilt_path = root + "ccpd_tilt/"
    weather_path = root + "ccpd_weather/"
    dic = {base_path: 21000, blur_path: 2100, challenge_path: 2100, db_path: 2100, fn_path: 2100, rotate_path: 2100,
           tilt_path: 2100, weather_path: 2100}
    # 训练集路径
    dst_train_path = "../data/CCPD_data/train/images/"
    # 评估集路径
    dst_val_path = "../data/CCPD_data/val/images/"
    for path in dic:
        select_data(path, dst_path_1=dst_train_path, dst_path_2=dst_val_path, num=dic[path])
