# 作者：水果好好吃哦
# 日期：2023/8/25
import cv2
import os
import torch
from torchvision import transforms
from lib import convert, alphabets
from net.CRNN_Net import CRNN
from my_utils import correction


def inference(img_path, model):
    img = cv2.imread(img_path)
    img = correction(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((32, 100))])
    img = transformer(img)
    img = img.to(device)
    img = img.view(1, 1, 32, 100)
    result = model(img)

    converter = convert.StrLabelConverter(alphabets.alphabets)
    preds_size = torch.IntTensor([result.size(0)] * result.size(1))
    _, preds = result.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    plate = converter.ocr_decode(preds.data, preds_size.data)

    print(f"{img_path}的识别结果是：{plate}")


if __name__ == "__main__":
    root = "./data/CCPD_plate_data/test/"
    weight = "./runs/train/trainbest_weights.pth"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_ocr = CRNN(class_num=67).to(device)
    model_ocr.load_state_dict(torch.load(weight, map_location=device))
    model_ocr.eval()

    dirs = os.listdir(root)
    for file in dirs:
        print("-"*66)
        img_path = root + file
        inference(img_path, model_ocr)

