# 数据中主要包含三种图结构：

### 症状-草药二部图（Symptom-Herb Bipartite Graph）
节点：症状、草药
边：症状集合与草药集合的配对（来自 train/valid/test）
### 症状-症状图（Symptom-Symptom Graph）
节点：症状
边：symPair-5.txt 中的症状对
### 草药-草药图（Herb-Herb Graph）
节点：草药
边：herbPair-40.txt 中的草药对

###### **中药数据集**
format: symptom id set \t herb set
<br>_**组成**_
<br>size：
<br>train_id.txt: 21755
<br>valid_id.txt: 1162
<br>test_id.txt: 3443
<br> meta data id max size:
<br>herb: 753
<br>symptom: 360