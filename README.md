# AI-FinalHomework
### 一、项目选题
 自人工智能浪潮袭来，人们的生活也因各种智能产品得到了巨大的改观，智能聊天机器人就是其中之一。本项目参考学习目前的开源学习资料，实现了简单的聊天机器人的应用

### 二、项目思路
 1. 本次实验意在学习自然语言处理相关知识，实现了简单的生成式聊天机器人模型和基于检索的聊天机器人模型，分析并思考最终的准确度差异；<br/>    
 2. 在生成式的聊天机器人模型中，利用Tensorflow搭建神经网络实现Sequence to Sequence模型；    <br/>
 3. 在基于检索的聊天机器人模型中，利用python自带的fuzzywuzzy模糊字符串匹配工具包，根据相似度原则在问答库中搜索中最佳的匹配答案。<br/>    

### 三、模型
 1. 生成式的聊天机器人模型：Sequence to Sequence模型；   
 2. 基于检索的聊天机器人模型：python自带的fuzzywuzzy模糊字符串匹配工具包（根据相似度原则在问答库中搜索中最佳的匹配答案）

### 四、代码
代码目录层级如下：  
Chatbot-Fannie
<br/> —— Chatbot  
 <br/>      —— Generate-Chatbot （生成式模型）  
 <br/>          —— seq2seq
    <br/>           —— code_seq2seq_char
  <br/>                —— _init_.py
  <br/>                —— predict_char.py
  <br/>                —— preprocess_char.py
   <br/>               —— train_char.py
   <br/>            —— data_seq2seq_char
   <br/>            —— model_seq2seq_char
    <br/>   —— Search-Chatbot（基于检索的模型）
    <br/>      —— _init_.py
    <br/>      —— chatbot_fuzzy.py
  <br/> —— config
<br/>      —— _init_.py
  <br/>    —— path_config.py
<br/>      —— params.json
 <br/>  —— data （训练语料）
<br/>      —— corpus
<br/>         —— chicken_and_gossip.txt
  <br/> —— utils （数据预处理）
<br/>       —— model_utils
<br/>          —— _init_.py
<br/>          —— data_utils.py
<br/>          —— model_seq2seq.py
<br/>          —— thread_generator.py
 <br/>         —— word_sequence.py
 <br/>      —— _init_.py
   <br/>    —— text_preprocess.py
 <br/>      —— word2vec_vector.py
         
### 五、效果
见result文件夹
