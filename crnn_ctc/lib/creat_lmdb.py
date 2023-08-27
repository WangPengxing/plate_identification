# 作者：水果好好吃哦
# 日期：2023/8/18
import lmdb
import cv2
import numpy as np
import os

out_path_train = "../data/CCPD_plate_data/train/train_lmdb"
in_path_train = "../data/CCPD_plate_data/train/labels/imgnames_labels.txt"
root_train = "../data/CCPD_plate_data/train/images/"
map_size_train = "524288000"    # 500mb
out_path_val = "../data/CCPD_plate_data/val/val_lmdb"
in_path_val = "../data/CCPD_plate_data/val/labels/imgnames_labels.txt"
root_val = "../data/CCPD_plate_data/val/images/"
map_size_val = "104857600"      # 100mb
out_path = {0: out_path_train, 1: out_path_val}
in_path = {0: in_path_train, 1: in_path_val}
root = {0: root_train, 1: root_val}
map_size = {0: map_size_train, 1: map_size_val}


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if isinstance(v, bytes):
                txn.put(k.encode(), v)
            else:
                txn.put(k.encode(), v.encode())


def createDataset(outputPath, imagePathList, labelList, root, map_size, checkValid=True):
    """
    为crnn的训练准备lmdb数据
    :param outputPath: lmdb数据的输出路径
    :param imagePathList: 图像数据的路径列表，即train.txt文件列表化
    :param labelList: 图像数据对应的标签列表
    :param checkValid: bool，辨别imagePathList中的路径是否为图片
    :return:
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    # map_size定义最大储存容量，单位是b
    env = lmdb.open(outputPath, map_size=int(map_size))
    # 缓存字典
    cache = {}
    # 计数器
    counter = 1
    for i in range(nSamples):
        imagePath = os.path.join(root, imagePathList[i]).split("---")[0]
        label = ''.join(labelList[i])

        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageKey = 'image-%09d' % counter
        labelKey = 'label-%09d' % counter
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if counter % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (counter, nSamples))
        counter += 1
        print(counter)
    nSamples = counter - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    for i in range(2):
        out_path = out_path[i]
        in_path = in_path[i]
        root = root[i]
        map_size = map_size[i]
        outputPath = out_path
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        with open(in_path, "r") as imgdata:
            imagePathList = list(imgdata)

        labelList = []
        for line in imagePathList:
            word = line.split("---")[1].replace("\n", "")
            labelList.append(word)
        createDataset(outputPath, imagePathList, labelList, root, map_size=map_size)
