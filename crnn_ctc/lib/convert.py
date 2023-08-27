# 作者：水果好好吃哦
# 日期：2023/8/22
# encoding: utf-8
import torch


class StrLabelConverter(object):
    """
    作用：字符串与标签之间的转换
    注意：需要在字母表中的首位置插入‘blank’, 为CTC计算损失做准备
    参数：字母表(alphabet)
    """

    def __init__(self, alphabet):
        self.alphabet = '-' + alphabet
        self.dict = {}
        for i, char in enumerate(alphabet):
            # 字典的值‘0’是为‘blank’准备的
            self.dict[char] = i + 1

    def ocr_encode(self, text):
        """
        作用：将字符串进行编码为标签，支持single和batch模式
        :param text:lmdb格式的标签，可以是single也可以是batch，是一个可迭代的对象
        :return:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: length_x，是指第x个single
            torch.IntTensor [n]: n个single，每个single的值m表明该single有m个字符
        """
        length, result = [], []
        for item in text:
            item = item.decode("utf-8", "strict")
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        # print(text,length)
        return torch.LongTensor(text), torch.LongTensor(length)

    def ocr_decode(self, t, length, raw=False):
        """
        作用：将标签进行码为字符串，支持single和batch模式
        :param t: 将要进行解码的标签，可以是single也可以是batch，Tensor形式
        :param length:若length含有n个元素，则表明有n个single，每个single的值m表明网络预测有m个字符
        :param raw: bool，False:去重；True: 不去重
        :return: 模型预测的字符串
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, f"text with length: {t.numel()} does not match declared length: {length}"
            if raw:
                return ''.join([self.alphabet[i] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i]])
                return ''.join(char_list)
        else:
            # batch模式，通过递归的方式实现该模式
            assert t.numel() == length.sum(), f"texts with length: {t.numel()} does not match declared length: {length.sum()}"
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.ocr_decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
