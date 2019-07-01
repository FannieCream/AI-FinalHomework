# AI-FinalHomework
### 一、项目选题
##### 自人工智能浪潮袭来，人们的生活也因各种智能产品得到了巨大的改观，智能聊天机器人就是其中之一。本项目参考学习目前的开源学习资料，实现了简单的聊天机器人的应用

### 二、项目思路
##### 1. 本次实验意在学习自然语言处理相关知识，实现了简单的生成式聊天机器人模型和基于检索的聊天机器人模型，分析并思考最终的准确度差异；
##### 2. 在生成式的聊天机器人模型中，利用Tensorflow搭建神经网络实现Sequence to Sequence模型；
##### 3. 在基于检索的聊天机器人模型中，利用python自带的fuzzywuzzy模糊字符串匹配工具包，根据相似度原则在问答库中搜索中最佳的匹配答案。

### 三、模型
##### 1. 生成式的聊天机器人模型：Sequence to Sequence模型；
##### 2. 基于检索的聊天机器人模型：python自带的fuzzywuzzy模糊字符串匹配工具包（根据相似度原则在问答库中搜索中最佳的匹配答案）

### 四、代码
代码目录层级如下：
Chatbot-Fannie
   —— Chatbot  
       —— Generate-Chatbot （生成式模型）  
           —— seq2seq
               —— code_seq2seq_char
                  —— _init_.py
                  —— predict_char.py
                  —— preprocess_char.py
                  —— train_char.py
               —— data_seq2seq_char
               —— model_seq2seq_char
       —— Search-Chatbot（基于检索的模型）
          —— _init_.py
          —— chatbot_fuzzy.py
   —— config
      —— _init_.py
      —— path_config.py
      —— params.json
   —— data （训练语料）
      —— corpus
         —— chicken_and_gossip.txt
   —— utils （数据预处理）
       —— model_utils
          —— _init_.py
          —— data_utils.py
          —— model_seq2seq.py
          —— thread_generator.py
          —— word_sequence.py
       —— _init_.py
       —— text_preprocess.py
       —— word2vec_vector.py
         
### 五、效果
见result文件夹
