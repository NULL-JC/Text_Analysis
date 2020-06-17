# 自定义数据集
import os
import codecs
import csv

from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

class MyDataset(BaseNLPDataset):
    """DemoDataset"""
    def __init__(self):
        # 数据集存放位置
        self.dataset_dir = "data"
        super(MyDataset, self).__init__(
            base_path=self.dataset_dir,
            train_file="train.txt",
            dev_file="valid.txt",
            test_file="valid.txt",
            train_file_with_header=False,
            dev_file_with_header=False,
            test_file_with_header=False,
            # 数据集类别集合
            label_list=["-1", "0", "1"])

dataset = MyDataset()
# for e in dataset.get_train_examples()[:10]:
#     print("{}\t{}\t{}".format(e.guid, e.text_a, e.label))

# path = 'data/train2.txt'
# for e in dataset.get_train_examples()[:10]:
#     if e.label =='-1' or e.label =='0' or e.label =='1':
#         with open(path, 'a', encoding='utf-8') as file:
#             file.write(''.join("{}\t{}\n".format(e.text_a, e.label)))




