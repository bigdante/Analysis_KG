# Analysis-KG
本仓库基于[Vicuna](https://github.com/lm-sys/FastChat/tree/main)模型，实现了可解释的英文语料知识抽取系统。
## 介绍
Analysis-KG 是基于[Vicuna](https://github.com/lm-sys/FastChat/tree/main)模型进行微调，实现了可解释的英文语料知识抽取系统。
微调的训练数据集来源于[Re-DocRED](https://github.com/tonytan48/Re-DocRED)。
我们实现了从句子中，一次性分析抽取多个relations，并且能够抽取出relation对应的多个triple facts，例如：
```python
>>>Drossinis Museum is in the center of Kifisia , a northern suburb of Athens , and it is housed in \u201c Amaryllis \u201d villa , where Georgios Drossinis lived in his last years and which is named after a central character of one of his earliest and most popular works . The museum was founded in 1997 with the aim to preserve and promote Drossinis\u2019 ( 1859 - 1951 ) multidimensional work . Drossinis , along with Costis Palamas and Nikos Kampas co - founded the New Athenian School , the Greek literary \u2018 1880s Generation\u2019 , a movement which renewed Greek literature and fought for the establishment of modern Greek language ( \" Demotic \" ) The museum includes three halls , which are on the first floor , while the ground floor houses the Municipal Library of Kifisia , according to Drossinis \u2019s wish .

According to the passage, the relations identified are "country of citizenship," "date of death," "notable work," "date of birth," "place of birth," "location in," and "languages spoken, written or signed." The reasons for these relations are as follows:
1. "Country of citizenship": The passage mentions that the museum is located in Greece, indicating that the country recognizes it as its property.
2. "Date of death": The passage states that Georgios Drossinis lived in the museum for the last 1951, implying that he is no longer alive.
3. "Notable work": The passage mentions that the museum is named after a central character of Drossinis's earliest and most popular works, highlighting its significance.
4. "Date of birth": The passage provides the specific years of Drossinis's birth and death, indicating the time of his birth.
5. "Place of birth": The passage mentions that Drossinis lived in a northern suburb of Athens, indicating that he was born in that specific location.
6. "Location in": The passage states that the museum is located in Kifisia, a suburb of Athens, indicating its association with that specific location.
7. "Languages spoken, written or signed": The passage does not provide evidence to support this relation.
In conclusion, the relations identified are supported by the explicit information provided in the passage, such as the names, dates, and the association of the museum with Drossinis's works and the specific location of Kifisia.

['date of birth', 'location in', 'country of citizenship', 'notable work', 'date of death',  'place of birth', 'residence', 'languages spoken, written or signed',]
=================================date of birth=================================
subjects_analysis:  The entities "Drossinis" and "Georgios Drossinis" can be considered as the subjects of the fact related to "date of birth" because the passage explicitly mentions that the museum is named after a central character of Georgios Drossinis, who lived in the Drossinis Museum for 1859-1951. The passage also states that Georgios Drossinis is considered an early and popular work, further emphasizing his significance. Additionally, the passage mentions the establishment of the museum in 1997, indicating that it was created to preserve and promote Georgios Drossinis's legacy. Therefore, based on the information provided, it is reasonable to infer that the entities "Drossinis" and "Georgios Drossinis" are the subjects of the fact related to "date of birth."
['Drossinis', 'Georgios Drossinis']

fact_analysis:  According to the subject "Drossinis" and the relation "date of birth," the fact is that Georgios Drossinis was born in 1859. This information is derived from the passage which states that the museum was founded in 1997 with the aim to preserve and promote Drossinis's earliest and most popular works. The passage also mentions that Drossinis lived in the Drossinis Museum for a significant period of time, indicating his connection to the institution. Therefore, based on the given information, it can be concluded that Georgios Drossinis's date of birth is 1859.
['Drossinis', 'date of birth', '1859']

fact_analysis:  According to the subject "Georgios Drossinis" and the relation "date of birth," the fact is that Georgios Drossinis was born in 1859. This information is derived from the passage which states that the museum was founded in 1997 with the aim to preserve and promote Drossinis's earliest and most popular works. The passage also mentions that Drossinis lived in his last years and is named after a central character of his works. Therefore, it can be concluded that Georgios Drossinis's date of birth is 1859.
['Georgios Drossinis', 'date of birth', '1859']
=================================location in=================================
subjects_analysis:  The entities 'Drossinis', 'Greek', and 'Kifisia' can be considered as the subjects of the fact related to "location in" because they are all associated with specific locations or geographic entities. 
1. 'Drossinis' is mentioned as the name of the museum, indicating a physical location. The fact that it is mentioned in the passage suggests that it is associated with a specific location.
2. 'Greek' is mentioned in relation to the name of the museum, indicating a language associated with a specific location. The fact that the name is mentioned in the passage implies that the language is associated with a specific location.
3. 'Kifisia' is mentioned as the specific location where the museum is situated. The fact that the museum is located in a specific location implies that it is associated with a specific geographic entity.
In summary, all these entities can be considered as the subjects of the fact related to "location in" because they are associated with specific locations or geographic entities, as stated in the passage.
['Drossinis', 'Greek', 'Kifisia']

fact_analysis:  According to the subject and relation, the fact is that "Drossinis" is associated with the location "Greece". The reason for this association is that the passage mentions the Drossinis Museum, which is located in Greece. The passage states that the museum was founded in 1997 with the aim to preserve and promote Drossinis's works, indicating his affiliation with Greece. Therefore, based on the information provided, it can be concluded that "Drossinis" is linked to the location of Greece.
['Drossinis', 'location in', 'Greek']

fact_analysis:  According to the subject "Greek" and the relation "location in," the fact is that the Greek language is associated with Greece. This is evident from the passage which states that the Greek literary work, "Democratic," was fought for the establishment of modern Greek language. The passage also mentions the Greek literary work, the 1880s Generation, which is included in the museum. Therefore, it can be concluded that the Greek language is closely tied to Greece based on this information.
['Greek', 'location in', 'Greece']

fact_analysis:  According to the subject "Kifisia" and the relation "location in," the fact is that Kifisia is located in Greece. This is evident from the passage which states that the Drossinis Museum is in the center of Kifisia, a northern suburb of Athens, and is named after a central character of Drossinis's earliest and most popular works. The passage also mentions that the museum was founded in 1997 with the aim to preserve and promote Drossinis's work, which indicates his affiliation with Greece. Therefore, based on this information, it can be concluded that Kifisia is indeed located in Greece.
['Kifisia', 'location in', 'Greek']
=================================country of citizenship=================================
subjects_analysis:  The entities "Drossinis" and "Georgios Drossinis" can be considered as the subjects of the fact related to "country of citizenship" because the passage explicitly mentions that the museum is named after a central character of Georgios Drossinis, who lived in the Drossinis Museum for a significant period of time. The passage also states that the museum was founded in 1997 with the aim to preserve and promote Drossinis's works. Additionally, it is mentioned that the municipality where the museum is located is in Greece, which further supports the inference that the individuals associated with the museum, namely Drossinis and Georgios Drossinis, are likely to be citizens of Greece. Therefore, based on the information provided, it is reasonable to consider Drossinis and Georgios Drossinis as the subjects of the fact related to "country of citizenship."
['Drossinis', 'Georgios Drossinis']

fact_analysis:  According to the subject and relation, the fact is that Drossinis's country of citizenship is Greece. This is evident from the passage which states that Drossinis lived in Greece for a significant period of time, and the Drossinis Museum is located in Greece. Therefore, it can be concluded that Drossinis is recognized as a citizen of Greece.
['Drossinis', 'country of citizenship', 'Greek']

fact_analysis:  According to the subject and relation, the fact is that Georgios Drossinis' country of citizenship is Greece. This is evident from the passage which states that Drossinis lived in his last years in Greece and is recognized as a central character of his works. The passage also mentions the founding of the Drossinis Museum in Greece, further indicating his connection to the country. Therefore, based on this information, it can be concluded that Georgios Drossinis is a citizen of Greece.
['Georgios Drossinis', 'country of citizenship', 'Greek']
=================================notable work=================================
subjects_analysis:  The entities "Drossinis" and "Georgios Drossinis" can be considered as the subjects of the fact related to "notable work" because the passage explicitly states that the Drossinis Museum is named after a central character of one of Drossinis's earliest and most popular works. This indicates that Drossinis's literary creations have had a significant impact and are recognized as noteworthy works. Additionally, the passage mentions that the museum was founded with the aim of preserving and promoting Drossinis's work, further emphasizing his significance as a literary creator. Therefore, based on the information provided, it is evident that both "Drossinis" and "Georgios Drossinis" are associated with notable works, making them suitable subjects for the fact related to "notable work".
['Drossinis', 'Georgios Drossinis']

fact_analysis:  According to the subject and relation, the notable work associated with Drossinis is the "Democratic" or "Democratic Language." This is evident from the passage which states that the Drossinis Museum includes three halls, one of which is the "Democratic" and the other is the "Democratic Language." The reason for considering this work as notable is that it is mentioned in the passage as being included in the museum, which is a significant creation in the field of Greek literature. Therefore, it can be concluded that "Democratic" and "Democratic Language" are noteworthy works associated with Drossinis.
['Drossinis', 'notable work', 'Democratic']
['Drossinis', 'notable work', 'Democratic Language']

fact_analysis:  According to the subject and relation, the notable work associated with Georgios Drossinis is the "Democratic" or "Democratic Language." This is evident from the passage which states that the Drossinis Museum is named after a central character of Drossinis's earliest and most popular works, and it includes three halls, one of which is the "Democratic" or "Democratic Language." Therefore, it can be concluded that "Democratic" is a significant creation associated with Georgios Drossinis.
['Georgios Drossinis', 'notable work', 'Democratic']
['Georgios Drossinis', 'notable work', 'Democratic Language']
=================================date of death=================================
subjects_analysis:  The entities "Drossinis" and "Georgios Drossinis" can be considered as the subjects of the fact related to "date of death" because the passage explicitly mentions that the museum is named after a central character of Georgios Drossinis, who lived in the Drossinis Museum for the last 1951. The passage also states that the museum was founded in 1997 with the aim to preserve and promote Drossinis's works. Since the passage mentions the specific names and the timeframe of his life and legacy, it is reasonable to infer that the entities "Drossinis" and "Georgios Drossinis" are the subjects of the fact related to "date of death."
['Drossinis', 'Georgios Drossinis']

fact_analysis:  According to the subject and relation, the fact is that "Drossinis" died in 1951. The reason for this conclusion is the mention in the passage that the Drossinis Museum was founded in 1997 and is named after a central character of Drossinis's earliest and most popular works. Additionally, the passage states that the museum includes three halls, which were on the first floor, while the ground floor is described as the "Melal Library of Kifisia" according to Drossinis's wishes. This implies that Drossinis is no longer alive, as the passage does not provide any information about his current status or any events after 1951.
['Drossinis', 'date of death', '1951']

fact_analysis:  According to the subject and relation, the fact is that Georgios Drossinis died in 1951. This is evident from the passage which states that the Drossinis Museum was founded in 1997 with the aim to preserve and promote Drossinis's earliest and most popular works. It further mentions that Drossinis lived in his last years and the museum is named after him. Therefore, based on this information, it can be concluded that Georgios Drossinis's date of death was in 1951.
['Georgios Drossinis', 'date of death', '1951']
=================================place of birth=================================
subjects_analysis:  The entities "Drossinis" and "Georgios Drossinis" can be considered as the subjects of the fact related to "place of birth" because the passage explicitly mentions that the museum was founded in 1997 with the aim to preserve and promote Drossinis's earliest and most popular works. It further states that the museum is located in the center of Kifisia, a northern suburb of Athens. Since the passage specifically mentions the name "Georgios Drossinis" and the associated names "Drossinis" and "Georgios Drossinis," it can be inferred that these entities are referring to the same person, Drossinis. Therefore, based on the information provided, it is reasonable to consider "Drossinis" and "Georgios Drossinis" as the subjects of the fact related to "place of birth."
['Drossinis', 'Georgios Drossinis']

fact_analysis:  According to the subject "Drossinis" and the relation "place of birth," the fact is that Drossinis was born in Kifisia. This is evident from the passage which states that the Drossinis Museum is located in the northern suburb of Kifisia, where Georgios Drossinis lived in his last years and where he is named after a central character of his works. The passage also mentions the establishment of the museum in 1997 with the aim to preserve and promote Drossinis's multidimonial work. Therefore, based on the information provided, it can be concluded that Kifisia is the specific location where Drossinis was born.
['Drossinis', 'place of birth', 'Kifisia']

fact_analysis:  According to the subject and relation, the fact is that Georgios Drossinis was born in Kifisia. The reason for this conclusion is that the passage explicitly states that the Drossinis Museum was founded in 1997 with the aim to preserve and promote Drossinis's earliest and most popular works. Additionally, the passage mentions that the museum is located in the northern suburb of Kifisia, which indicates that Kifisia is the specific location of Drossinis's birth.
['Georgios Drossinis', 'place of birth', 'Kifisia']
=================================residence=================================
subjects_analysis:  The entities "Drossinis" and "Georgios Drossinis" can be considered as the subjects of the fact related to "residence" because the passage explicitly states that the Drossinis Museum is located in the center of Kifisia, a northern suburb of Athens. It further mentions that the museum is housed in "2u201c Amaryllis" and "Durosinis" lived in his last years. This information establishes a clear connection between the entities and the location of the Drossinis Museum, indicating that they were residents of the city where the museum is situated. Therefore, based on the passage and the relation description, it is reasonable to consider "Drossinis" and "Georgios Drossinis" as the subjects of the fact related to "residence".
['Drossinis', 'Georgios Drossinis']

fact_analysis:  According to the subject "Drossinis" and the relation "residence," the fact is that Drossinis lived in Kifisia. This is evident from the passage which states that Drossinis lived in his last years and that the Drossinis Museum is located in Kifisia, a northern suburb of Athens. The passage also mentions the establishment of the museum, indicating Drossinis's connection to the location. Therefore, it can be concluded that Drossinis resided in Kifisia.
['Drossinis', 'residence', 'Kifisia']

fact_analysis:  According to the subject and relation, the fact is that Georgios Drossinis had a residence in Kifisia. The reason for this conclusion is that the passage states that Drossinis lived in his last years and that he is named after a central character of his works, which is now included in the Drossinis Museum in Kifisia. Additionally, the passage mentions the establishment of the museum, indicating Drossinis's connection to the location. Therefore, it can be concluded that Kifisia was the residence of Georgios Drossinis.
['Georgios Drossinis', 'residence', 'Kifisia']
=================================languages spoken, written or signed=================================
subjects_analysis:  The entities "Drossinis" and "Georgios Drossinis" can be considered as the subjects of the fact related to "languages spoken, written or signed" because the passage mentions that the museum, which is named after a central character of Georgios Drossinis, was founded by him and others. This implies that Drossinis had a significant role in the establishment of the museum, indicating his involvement in the language used to preserve and promote the museum. Additionally, the passage states that Drossinis lived in the museum for a period of time, further suggesting his close association with the language used in the institution. Therefore, based on the information provided, it is reasonable to infer that Drossinis and Georgios Drossinis were involved in the language used within the museum.
['Drossinis', 'Georgios Drossinis']

fact_analysis:  According to the subject and relation, the fact is that Drossinis spoke, wrote, or signed in Greek. This is evident from the passage which states that Drossinis, along with other individuals, founded the New Athenian School, a movement that aimed to preserve and promote Greek literature. This suggests that Drossinis had a command of the Greek language, indicating his ability to speak, write, or sign in Greek.
['Drossinis', 'languages spoken, written or signed', 'Greek']

fact_analysis:  According to the subject and relation, the fact is that Georgios Drossinis spoke, wrote, or signed in Greek. This is evident from the passage which states that Drossinis lived in his last years and that he is named after a central character of his works, including the Greek literary "Democratic." Therefore, it can be concluded that Drossinis was fluent in the Greek language, indicating it was one of the languages he spoke, wrote, or signed.
['Georgios Drossinis', 'languages spoken, written or signed', 'Greek']
```
## 依赖
### 软件依赖
```
#运行微调和推理需要安装以下依赖
pip3 install fschat
```
### 硬件依赖
```
A100 40GB，单卡即可运行
```

## 使用方法

### 1.模型训练

#### 1) 数据准备
当前的知识抽取数据集，大多是一个句子中只有一个relation，relation对应1个或者多个的fact。而现实场景中，一个句子其实会包含多个relation。为了得到含有多个relation并且标注良好的语料，我们通过对[Re-DocRED](https://github.com/tonytan48/Re-DocRED)数据的train_devised和dev_devised进行预处理。
具体如下：
##### 清晰定义relation descripiton
[Re-DocRED](https://github.com/tonytan48/Re-DocRED)数据集总共96个关系。但是存在的问题是：
###### 1.relation的description不够清晰
多个relation的表述不够清晰，例如`author`的关系中，不同的人，可能理解不同，可以将此理解为 `somebody is the author of somebook`，也可以理解为`somebook the author is somebody`, 如果没有清晰的定义，则会造成主体和客体混乱。
###### 2.relation互相包含或者相反
例如`member of`和`member of political party`，`member of sports team`其实可以统一为`member of`。
另外例如 `participant`和`participant of`语义相反，保留其中一个即可。

针对以上的问题，我们重新整理改造了relation，梳理出共64个relations，并且赋予了清晰明确的定义，更符合语言模型的理解。具体参见：
[relation_map.json](https://github.com/bigdante/Analysis_KG/blob/main/data/relations_desc/relation_map.json)
##### 2) analysis process
在对基础数据预处理后，我们通过ChatGPT和人工，使用prompt engineering，生成relation、subjects，以及fact的分析过程。并且为了后续的方便，我们将数据整理如下。
```python
# one sample
[{
        "index": 0,
        "passage": "Niklas Bergqvist ( born 6 October 1962 in Stockholm ) , is a Swedish songwriter , producer and musician . After the band split - up in 1987 , Bergqvist formed and played in several other bands until he decided to focus more on songwriting , resulting in several releases ever since .",
        "relations": [
            "date of birth",
            "place of birth"
        ],
        "fact_list": [
            {
                "fact": [
                    "Niklas Bergqvist",
                    "date of birth",
                    "6 October 1962"
                ]
            },
            {
                "fact": [
                    "Bergqvist",
                    "date of birth",
                    "6 October 1962"
                ]
            },
            {
                "fact": [
                    "Niklas Bergqvist",
                    "place of birth",
                    "Stockholm"
                ]
            },
            {
                "fact": [
                    "Bergqvist",
                    "place of birth",
                    "Stockholm"
                ]
            }
        ],
        "same_fact_list": [
            [
                [
                    "Niklas Bergqvist",
                    "date of birth",
                    "6 October 1962"
                ],
                [
                    "Bergqvist",
                    "date of birth",
                    "6 October 1962"
                ]
            ],
            [
                [
                    "Niklas Bergqvist",
                    "place of birth",
                    "Stockholm"
                ],
                [
                    "Bergqvist",
                    "place of birth",
                    "Stockholm"
                ]
            ]
        ],
        "relation_analysis": "According to the passage, the relations identified are \"date of birth\" and \"place of birth.\" The reason for this conclusion is that the passage explicitly states that Niklas Bergqvist was born on 6 October 1962 in Stockholm, which supports the relation \"date of birth.\" Additionally, the passage mentions that Bergqvist was born in Stockholm, providing evidence for the relation \"place of birth.\" Thus, these specific details mentioned in the passage lead to the identification of these two relations.",
        "entity_analysis": {
            "date of birth": "In the given passage, the entities \"Bergqvist\" and \"Niklas Bergqvist\" can be considered as the subjects of the fact related to \"date of birth\" because both refer to the same person. The passage mentions that Niklas Bergqvist was born on 6 October 1962 in Stockholm. \"Bergqvist\" is likely being referred to as a shorthand or a last name reference to Niklas Bergqvist himself. Thus, both entities represent the individual who was born on the specific date mentioned in the passage.",
            "place of birth": "The entities \"Bergqvist\" and \"Niklas Bergqvist\" can be considered as the subjects of the fact related to \"place of birth\" because the passage explicitly states that Niklas Bergqvist was born in Stockholm. This aligns with the relation's description, which states that \"place of birth\" refers to the specific location where a person was born. As Niklas Bergqvist is mentioned as the individual being discussed, it is reasonable to identify him and his last name \"Bergqvist\" as the subjects associated with the fact of being born in Stockholm."
        },
        "fact_analysis": {
            "date of birth": {
                "Bergqvist": "According to the subject \"Bergqvist\" and the relation \"date of birth,\" the fact is that Niklas Bergqvist was born on 6 October 1962. This information is based on the passage which states that Bergqvist was born in Stockholm. The passage also mentions his involvement in various bands and his transition to focusing more on songwriting after the band split-up. However, these details are not relevant to the fact extraction regarding his date of birth.",
                "Niklas Bergqvist": "According to the subject \"Niklas Bergqvist\" and the relation \"date of birth,\" the fact is that Niklas Bergqvist was born on 6 October 1962. This information is derived from the mention of his birthdate in the passage. The reason for this fact being true is that the passage explicitly states that he was born on this specific date. Therefore, based on the given information, we can conclude that Niklas Bergqvist's date of birth is 6 October 1962."
            },
            "place of birth": {
                "Bergqvist": "According to the subject (Bergqvist) and the relation (place of birth), the fact is that Niklas Bergqvist was born in Stockholm. This is based on the specific information provided in the passage, which states that he was born in Stockholm on October 6, 1962. The reason for this conclusion is the clear mention of his birthplace in the passage, indicating that Stockholm is the most specific known location of his birth.",
                "Niklas Bergqvist": "According to the subject \"Niklas Bergqvist\" and the relation \"place of birth,\" the fact is that Niklas Bergqvist was born in Stockholm. This is evident from the passage which explicitly states, \"Niklas Bergqvist (born 6 October 1962 in Stockholm).\" The mention of a specific date and location of birth reinforces the accuracy of this fact."
            }
        }
    },
    ...
]
```
通过运行脚本，即可完成vicuna训练数据准备。
在此之前，需要下载[Re-DocRED](https://github.com/tonytan48/Re-DocRED)到data/redocred文件夹下。
在data/chatgpt_count下的key.json文件中，按照所示的格式，添加可用的API keys（keys的数量越多，数据处理效率越高）。
并在shell中指定训练数据保存的路径。【中间生成的数据将会保存在data/redocred文件夹下】
```shell
cd code/data_process/
bash data_process.sh
```
### train
我们的代码参考自[FastChat](https://github.com/lm-sys/FastChat/tree/main)
在运行脚本前，需要指定脚本中的训练集路径以及checkpoint保存路径。
```shell
cd code/model_train/vicuna_train
bash train.sh
```

### 推理

在 P-tuning v2 训练时模型只保存 PrefixEncoder 部分的参数，所以在推理时需要同时加载原 ChatGLM2-6B 模型以及 PrefixEncoder 的权重，因此需要指定 `evaluate.sh` 中的参数：

```shell
--model_name_or_path THUDM/chatglm2-6b
--ptuning_checkpoint $CHECKPOINT_PATH
```

如果是，只需要跟之前一样设定 `model_name_or_path`：

```shell
--model_name_or_path $CHECKPOINT_PATH
```

评测指标为中文 Rouge score 和 BLEU-4。生成的结果保存在
`./output/adgen-chatglm2-6b-pt-128-2e-2/generated_predictions.txt`。

### 例子
#### 示例1
* Input: 类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞
* Label: 简约而不简单的牛仔外套,白色的衣身十分百搭。衣身多处有做旧破洞设计,打破单调乏味,增加一丝造型看点。衣身后背处有趣味刺绣装饰,丰富层次感,彰显别样时尚。
* Output[微调前]: 这件上衣的材质是牛仔布,颜色是白色,风格是简约,图案是刺绣,衣样式是外套,衣款式是破洞。
* Output[微调后]: 这是一款简约的牛仔外套,破洞设计,将牛仔布破洞,带来一种随意与个性。破洞的牛仔外套,展现出时尚气息,带来一种休闲感。同时,刺绣图案,让整件外套更加立体。

#### 示例2

* Input: 类型#裙\*版型#显瘦\*风格#文艺\*风格#简约\*图案#印花\*图案#撞色\*裙下摆#压褶\*裙长#连衣裙\*裙领型#圆领
* Label: 文艺个性的印花连衣裙,藏青色底蕴,低调又大气,撞色太阳花分布整个裙身,绚丽而美好,带来时尚减龄的气质。基础款的舒适圆领,简约不失大方,勾勒精致脸庞。领后是一粒包布扣固定,穿脱十分方便。前片立体的打褶设计,搭配后片压褶的做工,增添层次和空间感,显瘦又有型。
* Output[微调前]: 类型#裙*版型#显瘦*风格#文艺*风格#简约*图案#印花*图案#撞色*裙下摆#压褶*裙长#连衣裙*裙领型#圆领 1\. 连衣裙:简约风格,裙长为膝盖以上,裙领型为圆领。2\. 裙下摆:压褶设计,使裙摆呈现出流畅的褶皱效果。3\. 裙领型:裙领型为圆领,使穿上连衣裙后更加有型。4\. 版型:采用显瘦设计,让连衣裙看起来更加苗条。5\. 风格:文艺风格,让连衣裙更加有内涵和品味。6\. 图案:印花设计,在连衣裙上印有独特的图案。7\. 撞色:采用撞色设计,让连衣裙在色彩上更加鲜明、富有层次感。
* Output[微调后]: 这是一款文艺范的连衣裙,以印花为元素,采用简约的印花,既能够突出文艺气质,又能够展现简约风。在印花的同时又有领子和裙摆的压褶设计,更加凸显文艺气质。简约而不会过于单调,搭配出街,穿着十分舒适。


## 模型部署
首先载入Tokenizer：

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
```

1. 如果需要加载的 P-Tuning 的 checkpoint：

```python
config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
```
注意你可能需要将 `pre_seq_len` 改成你训练时的实际值。如果你是[从本地加载模型](../README.md#从本地加载模型)的话，需要将 `THUDM/chatglm2-6b` 改成本地的模型路径（注意不是checkpoint路径）。

2. 如果需要加载的是全参数微调的 checkpoint，则直接加载整个 checkpoint：

```python
model = AutoModel.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
```

之后根据需求可以进行量化，也可以直接使用：

```python
# Comment out the following line if you don't use quantization
model = model.quantize(4)
model = model.cuda()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
```

你也可以直接运行支持加载 P-Tuning v2 checkpoint 的 [web demo](./web_demo.py)
```shell
bash web_demo.sh
```
可能需要修改 [web_demo.sh](./web_demo.sh) 的内容以符合你实际的 checkpoint 情况。

## 使用自己的数据集
修改 `train.sh` 和 `evaluate.sh` 中的 `train_file`、`validation_file`和`test_file`为你自己的 JSON 格式数据集路径，并将 `prompt_column` 和 `response_column` 改为 JSON 文件中输入文本和输出文本对应的 KEY。可能还需要增大 `max_source_length` 和 `max_target_length` 来匹配你自己的数据集中的最大输入输出长度。

## 对话数据集

如需要使用多轮对话数据对模型进行微调，可以提供聊天历史，例如以下是一个三轮对话的训练数据：

```json lines
{"prompt": "长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "response": "用电脑能读数据流吗？水温多少", "history": []}
{"prompt": "95", "response": "上下水管温差怎么样啊？空气是不是都排干净了呢？", "history": [["长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "用电脑能读数据流吗？水温多少"]]}
{"prompt": "是的。上下水管都好的", "response": "那就要检查线路了，一般风扇继电器是由电脑控制吸合的，如果电路存在断路，或者电脑坏了的话会出现继电器不吸合的情况！", "history": [["长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "用电脑能读数据流吗？水温多少"], ["95", "上下水管温差怎么样啊？空气是不是都排干净了呢？"]]}
```

训练时需要指定 `--history_column` 为数据中聊天历史的 key（在此例子中是 `history`），将自动把聊天历史拼接。要注意超过输入长度 `max_source_length` 的内容会被截断。

可以参考以下指令：

```shell
bash train_chat.sh
```

## 引用

```
@inproceedings{liu2022p,
  title={P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks},
  author={Liu, Xiao and Ji, Kaixuan and Fu, Yicheng and Tam, Weng and Du, Zhengxiao and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={61--68},
  year={2022}
}
```




