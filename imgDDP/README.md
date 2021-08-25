Project: 对人脸表情进行简单分类任务。采用自定义卷积模型，使用分布式计算。



代码结构：
| imgDDP
|
|— — — —   dataset  应该包含数据集和数据的处理部分，数据集不放在github上，仅保存数据处理部分
|         |
|         |— — MyDataLoader.py  数据处理，输入为两个.csv文件，一个是特征，一个是标签，输出为DataLoader
|
|
|— — — —   mymodel
|         |
|         |— — MyModel.py 定义神经网络模型
|
|— — — —   trainer.py     训练部分，并且采用分布式训练的方法
|— — — —   tranier_mlu.py 代码为改编trainer.py适配寒武纪服务器


