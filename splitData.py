import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# 转换编码
def re_encode(path):
    with open(path, 'r', encoding='GB2312', errors='ignore') as file:
        lines = file.readlines()
    with open(path, 'w', encoding='utf-8') as file:
        file.write(''.join(lines))


re_encode('data/nCov_10k_test.csv')
re_encode('data/nCoV_100k_train.labled.csv')

#读取数据
train_labled = pd.read_csv('data/nCoV_100k_train.labled_u.csv', engine ='python', encoding='utf-8')
test = pd.read_csv('data/nCov_10k_test_u.csv', engine ='python',encoding='utf-8')
print(train_labled.shape)
print(test.shape)
print(train_labled.columns)
print(test.columns)
train_labled.head(3)
test.head(3)
print(train_labled['微博中文内容'].str.len().describe())


# # 标签分布
train_labled['情感倾向'].value_counts(normalize=True).plot(kind='bar');
# 清除异常标签数据
train_labled = train_labled[train_labled['情感倾向'].isin(['-1','0','1'])]
plt.show()

# 划分验证集，保存格式  text[\t]label
train_labled = train_labled[['微博中文内容', '情感倾向']]
train, valid = train_test_split(train_labled, test_size=0.2, random_state=2020)
train.to_csv('data/train.txt', index=False, header=False, sep='\t',encoding='utf-8')
valid.to_csv('data/valid.txt', index=False, header=False, sep='\t',encoding='utf-8')








