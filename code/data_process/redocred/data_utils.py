import json
import time
import random
from pathlib import Path
import requests
from collections import defaultdict

chat_gpt_cout_file = "keys.json"
ori_keys = json.load(open(f"../../../data/chatgpt_count/{chat_gpt_cout_file}"))
keys = [key for key, v in ori_keys.items() if v['label']]
unused_keys = keys.copy()
used_keys = []
overload_keys = []
invalid_keys = []

relation_map = json.load(open("../../../data/relations_desc/relation_map.json"))
unchanged_relation = relation_map.get("unchanged_name")
changed_relation = relation_map.get("change_name")
inverse_relation = relation_map.get("inverse_name")
rel_info = json.load(open("../../../data/redocred/rel_info.json"))

proxies = {
    'http': '127.0.0.1:9898',
    'https': '127.0.0.1:9898',
}


def get_valid_key():
    global unused_keys, used_keys, overload_keys
    current_time = time.time()
    new_overload_keys = []
    for key, timestamp in overload_keys:
        if current_time - timestamp >= 60:
            unused_keys.append(key)
        else:
            new_overload_keys.append((key, timestamp))
    overload_keys = new_overload_keys
    while not unused_keys:
        time.sleep(5)
    key = random.choice(unused_keys)
    unused_keys.remove(key)
    used_keys.append(key)
    return key


def get_relation_list_description():
    data = json.load(open("../../../data/relations_desc/relation_description.json"))
    result = {}
    for k, v in data.items():
        result[k] = v['description']
    json.dump(result, open("../../../data/relations_desc/relation_list.json", "w"), indent=4)
    return result


relation_descript = get_relation_list_description()


def make_chat_request(message, max_length=2048, timeout=10, max_retries=5):
    global unused_keys, used_keys, overload_keys
    for index in range(max_retries):
        while True:
            key = get_valid_key()
            try:
                with requests.post(
                        url=f"https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json={
                            "model": "gpt-3.5-turbo",
                            "temperature": 1.0,
                            "messages": message,
                            "max_tokens": max_length,
                        },
                        # timeout=10,
                        proxies=proxies,
                ) as resp:
                    if resp.status_code == 200:
                        used_keys.remove(key)
                        unused_keys.append(key)
                        return json.loads(resp.content)
                    else:
                        try:
                            if json.loads(resp.content).get('error'):
                                if json.loads(resp.content).get('error')['message'] == "You exceeded your current quota, please check your plan and billing details.":
                                    invalid_keys.append(key)
                                else:
                                    overload_keys.append((key, time.time()))
                            else:
                                print("response error: ", resp.content)
                                overload_keys.append((key, time.time()))
                        except:
                            print("error: ", resp.content)
            except requests.exceptions.RequestException as e:
                print("request error", e)
                used_keys.remove(key)
                unused_keys.append(key)
                timeout += 5


def make_ori_data(file, save_path, test_relation="all"):
    # 处理redocred的数据集，可以测试特定的relation，也可以选取全部的relation做训练用
    final_save = []
    if "train" in file or "dev" in file:
        delete = False
    else:
        delete = True
    data = json.load(open(file))
    index = 0
    for page_id, sample in enumerate(data):
        result = {}
        for fact in sample['labels']:
            head_name = list(set([h['name'] for h in sample['vertexSet'][fact['h']]]))
            tail_name = list(set([t['name'] for t in sample['vertexSet'][fact['t']]]))
            relation = rel_info[fact['r']]
            if relation in unchanged_relation:
                pass
            elif relation in changed_relation:
                relation = changed_relation.get(relation)
            elif relation in inverse_relation:
                tail_name = list(set([h['name'] for h in sample['vertexSet'][fact['h']]]))
                head_name = list(set([t['name'] for t in sample['vertexSet'][fact['t']]]))
                relation = inverse_relation.get(relation)
            else:
                continue
            if test_relation != "all" and relation != test_relation:
                continue
            head_ids = sorted(list(set([h['sent_id'] for h in sample['vertexSet'][fact['h']]])))
            tail_ids = sorted(list(set([t['sent_id'] for t in sample['vertexSet'][fact['t']]])))
            if fact['evidence']:
                sent_ids = sorted(list(set(head_ids + tail_ids + fact['evidence'])))
            else:
                sent_ids = sorted(list(set(head_ids + tail_ids)))
            sentence = " ".join([" ".join(sent) for sent in [s_ for index, s_ in enumerate(sample['sents']) if index in sent_ids]])
            head_name = [name for name in head_name if name in sentence]
            tail_name = [name for name in tail_name if name in sentence]
            if not head_name or not tail_name:
                continue
            if sentence not in result:
                result[sentence] = {"relations": [], "fact_list": [], "same_fact_list": []}
            if relation not in result[sentence]["relations"]:
                result[sentence]["relations"].append(relation)
            same_fact = []
            for head in head_name:
                for tail in tail_name:
                    same_fact.append((head, relation, tail))
                    result[sentence]["fact_list"].append({
                        "fact": (head, relation, tail),
                    })
            result[sentence]["same_fact_list"].append(same_fact)
        if result:
            result = merge_result(result, delelte=delete)
            for sentence, value in result.items():
                fact_list_set = set()
                unique_fact_list = []
                for item in value['fact_list']:
                    fact = tuple(item["fact"])
                    if fact not in fact_list_set:
                        fact_list_set.add(fact)
                        unique_fact_list.append(item)
                same_fact_list_set = set()
                unique_same_fact_list = []
                for item in value['same_fact_list']:
                    fact = tuple(item[0])
                    if fact not in same_fact_list_set:
                        same_fact_list_set.add(fact)
                        unique_same_fact_list.append(item)
                save = {
                    "index": index,
                    "page_id": page_id,
                    "passage": sentence,
                    "relations": value['relations'],
                    "fact_list": unique_fact_list,
                    "same_fact_list": unique_same_fact_list
                }
                index += 1
                final_save.append(save)
    create_file_with_path(save_path)
    with open(save_path, "w") as f:
        json.dump(final_save, f, indent=4)


def create_file_with_path(file_path):
    # 创建文件
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def merge_result(result, delelte):
    '''
        将短句子合并到长句子中
    '''
    keys = list(result.keys())
    keys.sort(key=len, reverse=False)  # Sort keys by length in descending order
    merged_result = result.copy()  # Create a copy of the original result
    for key1 in keys:
        value1 = merged_result[key1]
        flag = 0
        for j in range(keys.index(key1) + 1, len(keys)):
            key2 = keys[j]
            value2 = merged_result[key2]
            if key1 in key2:  # Check if key1 is a substring of key2
                flag = 1
                merged_relations = list(set(value1["relations"] + value2["relations"]))
                merged_fact_list = list(value1["fact_list"] + value2["fact_list"])
                merged_same_fact_list = list(value1["same_fact_list"] + value2["same_fact_list"])
                merged_result[key2] = {
                    "relations": merged_relations,
                    "fact_list": merged_fact_list,
                    "same_fact_list": merged_same_fact_list
                }
        if flag and delelte:
            del merged_result[key1]
    return merged_result


def merge_train_ori_chatgpt(ori_path, chatgpt_path):
    ori_data = json.load(open(ori_path))
    chatgpt_data = json.load(open(chatgpt_path))
    ori_data = chatgpt_data['positive'] + ori_data
    ori_data = chatgpt_data['negative'] + ori_data
    for index, sample in enumerate(ori_data):
        sample['index'] = index
        if "page_id" not in sample:
            sample['page_id'] = "from chatgpt"
    json.dump(ori_data, open(ori_path, "w"), indent=4)


def count_relation(relation):
    test_data = json.load(open(f"../../test_ori/{relation}.json"))
    true_relation_count = defaultdict(int)
    for data in test_data:
        fact_list = data['same_fact_list']
        for item in fact_list:
            relation = item[0][1]
            true_relation_count[relation] += 1
    print(true_relation_count[relation])


def process_sample_explanation(sample, relation_descript, save_file):
    # 给原始的redocred数据添加上explanation和各种analysis
    sample['relation_analysis'] = {}
    sample["entity_analysis"] = {}
    sample["fact_analysis"] = {}
    relations_desc = [relation_descript.get(relation) for relation in sample['relations']]
    prompt = f"Given the passage: {sample['passage']}, after analyzing the text, we have identified the relations: {sample['relations']}, the specific relation descriptions are as " \
             f"follows: {relations_desc}.\n Now, provide me with the analysis. Your analysis should be short but convincing. You can start with : according to the passage, " \
             f"the relations are ... the reasons are...\n" \
             f"You should focus on the evidences that led to the conclusion. "
    message = [
        {"role": "user", "content": prompt}
    ]
    analysis = make_chat_request(message)['choices'][0]['message']['content']
    print("relations analysis", analysis)
    sample['relation_analysis'] = analysis
    if "subject_analysis" not in sample:
        for relation in sample['relations']:
            entity_list = list(set([fact["fact"][0] for fact in sample['fact_list'] if fact["fact"][1] == relation]))
            entity_prompt = f"As an expert in entity analysis, you have been presented with the following passage:\"{sample['passage']}\". From this passage, we can derive the " \
                            f"relation: \"{relation}\". The explicit description of this relation is: \"{relation_descript.get(relation)}\". Based on the information in the " \
                            f"passage and the relation description, we have identified the following entities as the subjects of the fact related to \"{relation}\": " \
                            f"{entity_list}. Now, please explain why these entities can be considered as the subjects of the fact related to \"{relation}\". Your explanations " \
                            f"should be succinct yet persuasive. "
            message = [
                {"role": "user", "content": entity_prompt}
            ]
            analysis = make_chat_request(message)['choices'][0]['message']['content']
            print("subject analysis: ", analysis.replace("\n\n", "\n"))
            sample["entity_analysis"][relation] = analysis.replace("\n\n", "\n")
            sample["fact_analysis"][relation] = {}
            fact_analysis = {}
            for entity in entity_list:
                fact_list = [fact["fact"] for fact in sample['fact_list'] if fact['fact'][1] == relation and fact['fact'][0] == entity]
                fact_prompt = f"You are a fact analysis expert.\n" \
                              f"I have passage : \"{sample['passage']}\"\n" \
                              f"The relation description is: \"{relation_descript.get(relation)}\"\n" \
                              f"To extract facts of \"{relation}\", we make \"{entity}\" as subject according to the relation description, " \
                              f"after carefully analysing the passage, we get the fact: {fact_list}. " \
                              f"Now give me the analysis. Your analysis should be short but convincing. You can start with:\n" \
                              f"according to the subject and relation , the facts are..., the reasons are.."
                message = [
                    {"role": "user", "content": fact_prompt}
                ]
                analysis = make_chat_request(message)['choices'][0]['message']['content']
                fact_analysis[entity] = analysis.replace("\n\n", "\n")
            sample["fact_analysis"][relation] = fact_analysis
            print("fact_analysis", fact_analysis)
    with open(save_file, "a") as f:
        f.write(json.dumps(sample) + "\n")
        if invalid_keys:
            for invalid_key in invalid_keys:
                ori_keys.get(invalid_key)['label'] = False
            json.dump(ori_keys, open(f"../../../data/chatgpt_count/{chat_gpt_cout_file}", "w"), indent=4)


def make_vicuna_train_analysis_v0(file, save_path):
    train_data = []
    relation_template = "Given a passage: \"{sentences}\", conduct a comprehensive analysis to uncover any underlying relations.\n" \
                        "The relations analytical process is as follows: "
    relation_list_template = "Given a passage: \"{sentences}\", along with an analysis of the implicit relations present in the passage: {relation_analysis}, we can now list the " \
                             "relations found within the passage: "
    entity_template = "Given a description of the relation: \"{description}\" and a passage: \"{sentences}\", our objective is to extract all entities that can act as the " \
                      "subject of the mentioned relation. To achieve this, a thorough analysis of the passage is conducted to identify entities that meet the specified criteria. " \
                      "The analytical process is outlined below: "
    entity_list_template = "Given a relation description: \"{description}\" and a passage: \"{sentences}\", along with the analysis of subjects: {subjects_analysis}, " \
                           "we identify entities that fulfill the relation's description and can serve as its subjects. Based on this analysis, the following entities can be " \
                           "identified as suitable subjects for the relation: "
    fact_template = "Given relation description: \"{description}\", the passage: \"{sentences}\", and a specific subject: \"{subject}\", we conduct an analysis to identify all " \
                    "facts within the text that align with the relation's description and pertain to the mentioned subject. The analysis process is as follows: "
    fact_list_template = "After analyzing the relation description: \"{description}\", the passage: \"{sentences}\", and considering the subject of the relation: \"{subject}\", " \
                         "we have conducted an analysis of the facts as: \"{facts_analysis}\". Based on this analysis, we have derived the following facts: "
    data = json.load(open(file))
    global_id = 0
    for sample in data:
        sentence = sample['passage']
        relation_analysis = sample['relation_analysis']
        block_dict = {
            "id": f"identity_{global_id}",
            "conversations": [
                {
                    "from": "human",
                    "value": relation_template.format(sentences=sentence)
                },
                {
                    "from": "gpt",
                    "value": str(relation_analysis.replace("\n\n", "\n"))
                }
            ]
        }
        train_data.append(block_dict)
        global_id += 1
        block_dict = {
            "id": f"identity_{global_id}",
            "conversations": [
                {
                    "from": "human",
                    "value": relation_list_template.format(sentences=sentence, relation_analysis=relation_analysis)
                },
                {
                    "from": "gpt",
                    "value": str("\n".join(list(set(sample['relations']))))
                }
            ]
        }
        train_data.append(block_dict)
        global_id += 1

        for relation, entity_analysis in sample['entity_analysis'].items():
            block_dict = {
                "id": f"identity_{global_id}",
                "conversations": [
                    {
                        "from": "human",
                        "value": entity_template.format(sentences=sentence, description=relation_descript.get(relation))
                    },
                    {
                        "from": "gpt",
                        "value": str(entity_analysis.replace("\n\n", "\n"))
                    }
                ]
            }
            train_data.append(block_dict)
            global_id += 1

            entity_list = list(set([fact['fact'][0] for fact in sample['fact_list'] if fact['fact'][1] == relation]))
            block_dict = {
                "id": f"identity_{global_id}",
                "conversations": [
                    {
                        "from": "human",
                        "value": entity_list_template.format(sentences=sentence, description=relation_descript.get(relation), subjects_analysis=entity_analysis)
                    },
                    {
                        "from": "gpt",
                        "value": str("\n".join(entity_list))
                    }
                ]
            }
            train_data.append(block_dict)
            global_id += 1

            for subject in entity_list:
                block_dict = {
                    "id": f"identity_{global_id}",
                    "conversations": [
                        {
                            "from": "human",
                            "value": fact_template.format(sentences=sentence, description=relation_descript.get(relation), subject=subject)
                        },
                        {
                            "from": "gpt",
                            "value": str(sample['fact_analysis'][relation][subject].replace("\n\n", "\n"))
                        }
                    ]
                }
                train_data.append(block_dict)
                global_id += 1

                fact_list = [fact['fact'] for fact in sample['fact_list'] if fact['fact'][1] == relation and fact['fact'][0] == subject]
                block_dict = {
                    "id": f"identity_{global_id}",
                    "conversations": [
                        {
                            "from": "human",
                            "value": fact_list_template.format(sentences=sentence, description=relation_descript.get(relation),
                                                               facts_analysis=sample['fact_analysis'][relation][subject], subject=subject)
                        },
                        {
                            "from": "gpt",
                            "value": str("\n".join([str(fact) for fact in fact_list]))
                        }
                    ]
                }
                train_data.append(block_dict)
                global_id += 1
    create_file_with_path(save_path)
    json.dump(train_data, open(save_path, "w"), indent=4)
