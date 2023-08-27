# 作者：水果好好吃哦
# 日期：2023/8/22

from torch.utils.data import Dataset
import lmdb
import six
import sys
from PIL import Image


class LmdbDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % root)
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            string = 'num-samples'
            nSamples = int(txn.get(string.encode()))
            self.nSamples = nSamples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            img_buf = txn.get(img_key.encode())
            buf = six.BytesIO()
            buf.write(img_buf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            if self.transform is not None:
                img = self.transform(img)
            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode())
            if self.target_transform is not None:
                label = self.target_transform(label)
        return img, label
