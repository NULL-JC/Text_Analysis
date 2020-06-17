# from myDataset import MyDataset
import paddlehub as hub
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
import pandas as pd
import numpy as np

# 其使用流程如下：
#
# 1）将数据整理成特定格式
#
# 2）定义Dataset数据类
#
# 3）加载模型
#
# 4）构建reader数据读取接口
#
# 5）确定finetune训练策略
#
# 6）配置finetune参数
#
# 7）确定任务，开始finetune（训练）
#
# 8）预测

class MyDataset(BaseNLPDataset):
    """DemoDataset"""
    def __init__(self):
        # 数据集存放位置
        self.dataset_dir = "data"
        super(MyDataset, self).__init__(
            base_path=self.dataset_dir,
            train_file="train2.txt",
            dev_file="valid2.txt",
            test_file="valid2.txt",
            train_file_with_header=False,
            dev_file_with_header=False,
            test_file_with_header=False,
            # 数据集类别集合
            label_list=["-1", "0", "1"])
        
module = hub.Module(name="ernie")
dataset=MyDataset()

# 构建Reader
reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    sp_model_path=module.get_spm_path(),
    word_dict_path=module.get_word_dict_path(),
    max_seq_len=128)

# finetune策略
strategy = hub.AdamWeightDecayStrategy(
    weight_decay=0.01,
    warmup_proportion=0.1,
    learning_rate=5e-5)

# 运行配置
config = hub.RunConfig(
    use_cuda=False,
    use_data_parallel=False,
    num_epoch=1,
    checkpoint_dir="model",
    batch_size=10,
    eval_interval=100,
    enable_memory_optim=True,
    strategy=strategy)

# Finetune Task
inputs, outputs, program = module.context(
    trainable=True, max_seq_len=128)

# Use "pooled_output" for classification tasks on an entire sentence.
pooled_output = outputs["pooled_output"]

feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]

cls_task = hub.TextClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config,
        metrics_choices=["f1"])



inv_label_map = {val: key for key, val in reader.label_map.items()}

# Data to be prdicted
test = pd.read_csv('data/nCov_10k_test_u.csv', engine ='python',encoding='utf-8')
data = test[['微博中文内容']].fillna(' ').values.tolist()

run_states = cls_task.predict(data=data)
results = [run_state.run_results for run_state in run_states]

# 生成预测结果
proba = np.vstack([r[0] for r in results])
prediction = list(np.argmax(proba, axis=1))
prediction = [inv_label_map[p] for p in prediction]

submission = pd.DataFrame()
submission['id'] = test['微博id'].values
submission['id'] = submission['id'].astype(str) + ' '
submission['y'] = prediction
np.save('proba.npy', proba)
submission.to_csv('data/result.csv', index=False)
submission.head()
