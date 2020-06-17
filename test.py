import paddlehub as hub

module = hub.Module(name="emotion_detection_textcnn")  # 模型加载
test_text = ["你真好看", "你真丑"]
input_dict = {"text": test_text}  # 文字输入
results = module.emotion_classify(data=input_dict)  # 预测结果

for result in results:
    print(result)
    print(result['text'])
    print(result['emotion_label'])
    print(result['emotion_key'])
    probs_name = result['emotion_key'] + "_probs"
    # print(result['negative_probs'])
    # print(probs_name)
    print(result[probs_name])