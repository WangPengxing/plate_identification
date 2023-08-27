# 作者：水果好好吃哦
# 日期：2023/8/18
import scipy.io as sio
import random
from shutil import copyfile


def label_process(root, val_num):
    base_dir = root
    load_fn = base_dir + "labels/VehicleInfo.mat"
    load_data = sio.loadmat(load_fn)
    data = load_data['VehicleInfo']
    val_index = random.sample(range(data.size), val_num)

    for i in range(len(data)):
        item = data[i]
        str = ""
        print("-" * 30)
        for j in range(item['vehicles'][0][0].size):
            # Bus, Microbus, Minivan, Sedan, SUV, and Truck
            vehicles = item['vehicles'][0][0][j]
            height = item['height'][0][0][0]
            width = item['width'][0][0][0]
            left = vehicles[0][0][0]
            top = vehicles[1][0][0]
            right = vehicles[2][0][0]
            bottom = vehicles[3][0][0]
            # 边界框内的车辆类别
            vehicles_type = vehicles[4][0]
            if vehicles_type == 'Bus':
                vehicles_type = 0
            elif vehicles_type == 'Microbus':
                vehicles_type = 1
            elif vehicles_type == 'Minivan':
                vehicles_type = 2
            elif vehicles_type == 'Sedan':
                vehicles_type = 3
            elif vehicles_type == 'SUV':
                vehicles_type = 4
            elif vehicles_type == 'Truck':
                vehicles_type = 5
            # 边界框的信息描述，即(c, x, y, w, h)
            str += '%s %s %s %s %s' % (vehicles_type, round(float((left + (right - left) / 2) / width), 6),
                                       round(float((top + (bottom - top) / 2) / height), 6),
                                       round(float((right - left) / width), 6),
                                       round(float((bottom - top) / height), 6)) + '\n'

        name = item['name'][0][0]
        str = str[:str.rfind('\n')]
        print(str)
        if i in val_index:
            with open(base_dir + "val/labels/" + name[:-3] + "txt", 'w') as f:
                f.write(str + '\n')
            copyfile(base_dir + "images/" + name, base_dir + "val/images/" + name)
        else:
            with open(base_dir + "train/labels/" + name[:-3] + "txt", 'w') as f:
                f.write(str + '\n')
            copyfile(base_dir + "images/" + name, base_dir + "train/images/" + name)
    print('done--')


if __name__ == "__main__":
    root = "../raw_data/BITVehicle_data/"
    val_num = 3850
    label_process(root, val_num)
