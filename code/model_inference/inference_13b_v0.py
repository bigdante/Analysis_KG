import json
import os
import re
import sys
import time
from collections import defaultdict, OrderedDict
from pathlib import Path
import subprocess
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

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

relation_map = json.load(open("../../data/relations_desc/relation_map.json"))
unchanged_relation = relation_map.get("unchanged_name")
changed_relation = relation_map.get("change_name")
inverse_relation = relation_map.get("inverse_name")
rel_info = json.load(open("../../data/redocred/rel_info.json"))


def get_relation_list_description():
    data = json.load(open("../../data/relations_desc/relation_description.json"))
    result = {}
    for k, v in data.items():
        result[k] = v['description']
    json.dump(result, open("../../data/relations_desc/relation_list.json", "w"), indent=4)
    return result


relation_desc = get_relation_list_description()


def get_relation_list_description():
    data = json.load(open("./relation_description.json"))
    result = {}
    for k, v in data.items():
        result[k] = v['description']
    json.dump(result, open("./relation_list.json", "w"), indent=4)
    return result


def get_model_nodes(cudaid, model_path):
    if cudaid == 7:
        time.sleep(60)
    start = time.time()
    cuda_device = torch.device(f"cuda:{cudaid}") if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True,
    ).half()
    model.to(cuda_device)
    model.eval()
    print(f"model load done, consume time {time.time() - start}")
    return model, tokenizer


def inference(model, tokenizer, text):
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. Human: {text} ASSISTANT:"
    prompt = prompt.format(text=text)
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).to(model.device),
        max_new_tokens=2048,
        # temperature=1
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return outputs


def create_file_with_path(file_path):
    # 创建文件
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def get_node_from_ip(ifname='eth0'):
    result = subprocess.run(['ifconfig', ifname], stdout=subprocess.PIPE)
    output = result.stdout.decode()
    ip_address = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', output)
    if not ip_address:
        return 'Unknown node'
    ip_address = ip_address.group(1)
    ip_to_node = {
        '192.69.107.248': '0',
        '192.4.183.6': '1',
        '192.216.99.31': '2'
    }
    node_identifier = ip_to_node.get(ip_address, 'Unknown node')
    return node_identifier


def process_evaluation_data(evaluation_data, node, cuda_id):
    data = evaluation_data
    cuda_chunk_size = len(data) // 8
    cuda_start, cuda_end = cuda_id * cuda_chunk_size, (cuda_id + 1) * cuda_chunk_size
    if cuda_id == 7:
        return data[cuda_start:]
    else:
        return data[cuda_start:cuda_end]


def redocred_vicuna_inference(test_relation=None, result_save_path=None):
    redocred_test_data = json.load(open(f"../../data/test_data_important/{test_relation}.json"))
    data = process_evaluation_data(redocred_test_data, node, cuda_id)
    tp = 0
    fp = 0
    wrong_list = []
    right_list = []
    for sample in tqdm(data):
        wrong = []
        right = []
        miss = []
        sentences = sample['passage']
        print(sentences)
        relations = []
        for sentence in sentences.split(".") + [sentences]:
            relation_prompt = relation_template.format(sentences=sentence)
            relation_analysis = inference(model, tokenizer, relation_prompt)
            relation_list_prompt = relation_list_template.format(sentences=sentence, relation_analysis=relation_analysis)
            relation_list = inference(model, tokenizer, relation_list_prompt)
            print(relation_list)
            ori_relations = list(set(relation_list.split("\n")))
            for index, relation in enumerate(ori_relations):
                if relation in relation_desc:
                    relations.append(relation)
        relations = list(set(relations))
        ori_relations = relations.copy()
        if test_relation != "all":
            if test_relation not in relations:
                wrong_list.append({
                    "text": sentences,
                    "relation_analysis": relation_analysis,
                    "predict_wrong_relations": relations,
                    "miss_fact_list": [s_f_l for s_f_l in sample['same_fact_list'] if s_f_l[0][1] == test_relation],
                    "true_fact_list": sample['same_fact_list']
                })
                continue
            else:
                relations = [test_relation]
        print("relations: ", relations)
        for relation in list(set(list(relations))):
            print(f"================================={relation}: {list(relation_desc.keys()).index(relation)}=================================")
            subjects_prompt = entity_template.format(description=relation_desc.get(relation), sentences=sentences)
            subjects_analysis = inference(model, tokenizer, subjects_prompt)
            entity_list_prompt = entity_list_template.format(description=relation_desc.get(relation), sentences=sentences, subjects_analysis=subjects_analysis)
            entity_list = inference(model, tokenizer, entity_list_prompt)
            print("ori_entities: ", entity_list)
            try:
                ori_entities = list(set(entity_list.split("\n")))
                entities = []
                for entity in ori_entities:
                    if entity in sentences:
                        entities.append(entity)
            except:
                print("wrong eval entity: ", entity_list)
                wrong_list.append({
                    "text": sentences,
                    "relation_analysis": relation_analysis,
                    "predict_relations": ori_relations,
                    "subjects_analysis": subjects_analysis,
                    "eval_wrong_subjects": entity_list,
                    "miss_fact_list": [s_f_l for s_f_l in sample['same_fact_list'] if s_f_l[0][1] == test_relation],
                    "true_fact_list": sample['same_fact_list']
                })
                continue
            fact_index = []
            entities = list(set(entities))
            print("entities: ", entities)
            for subject in entities:
                fact_analysis_prompt = fact_template.format(description=relation_desc.get(relation), sentences=sentences, subject=subject)
                fact_analysis = inference(model, tokenizer, fact_analysis_prompt)
                fact_list_prompt = fact_list_template.format(description=relation_desc.get(relation), sentences=sentences, subject=subject, facts_analysis=fact_analysis)
                fact_list = inference(model, tokenizer, fact_list_prompt)
                print("ori_fact_list: ", fact_list)
                facts = []
                try:
                    for fact in fact_list.split("\n"):
                        facts.append(eval(fact))
                    facts = [list(x) for x in set(tuple(x) for x in facts)]
                    print("split facts: ", fact_list.split("\n"))
                except Exception as e:
                    print(e)
                    print("wrong eval fact : ", fact_list)
                    wrong_list.append({
                        "text": sentences,
                        "relation_analysis": relation_analysis,
                        "predict_relations": ori_relations,
                        "subjects_analysis": subjects_analysis,
                        "subjects_list": entities,
                        "fact_analysis": fact_analysis,
                        "eval_wrong_facts": fact_list,
                        "miss_fact_list": [s_f_l for s_f_l in sample['same_fact_list'] if s_f_l[0][1] == test_relation],
                        "true_fact_list": sample['same_fact_list']
                    })
                    continue
                print("facts: ", facts)
                for fact in facts:
                    flag = 0
                    fact[0] = subject
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact in true_fact:
                            flag = 1
                            if index not in fact_index:
                                tp += 1
                                right.append({
                                    "fact": fact,
                                    "fact_analysis": fact_analysis,
                                })
                                fact_index.append(index)
                    if not flag:
                        fp += 1
                        wrong.append({
                            "fact": fact,
                            "fact_analysis": fact_analysis,
                        })
            miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        if wrong or miss:
            wrong_list.append({
                "text": sentences,
                "relation_analysis": relation_analysis,
                "predict_relations": relations,
                "wrong_fact_list": wrong,
                "miss_fact_list": miss,
                "true_fact_list": sample['same_fact_list']
            })
        if right:
            right_list.append({
                "text": sentences,
                "relation_analysis": relation_analysis,
                "predict_relations": relations,
                "right_fact_list": right,
                "true_fact_list": sample['same_fact_list']
            })
    if wrong_list:
        create_file_with_path(f"{result_save_path}/{test_relation}/wrong_redocred_{cuda_id + node * 8}.json")
        json.dump(wrong_list, open(f"{result_save_path}/{test_relation}/wrong_redocred_{cuda_id + node * 8}.json", "w"), indent=4)
    if right_list:
        create_file_with_path(f"{result_save_path}/{test_relation}/right_redocred_{cuda_id + node * 8}.json")
        json.dump(right_list, open(f"{result_save_path}/{test_relation}/right_redocred_{cuda_id + node * 8}.json", "w"), indent=4)


def redocred_vicuna_inference_n(test_relation=None, result_save_path=None):
    redocred_test_data = json.load(open(f"../../data/test_data_important/{test_relation}.json"))
    node = 0
    data = process_evaluation_data(redocred_test_data, node, cuda_id)
    tp = 0
    fp = 0
    wrong_list = []
    right_list = []
    for sample in tqdm(data):
        wrong = []
        right = []
        miss = []
        sentence = sample['passage']
        relation_prompt = relation_template.format(sentences=sentence)
        relation_analysis = inference(model, tokenizer, relation_prompt)
        relation_list_prompt = relation_list_template.format(relation_analysis=relation_analysis)
        relation_list = inference(model, tokenizer, relation_list_prompt)
        try:
            relations = eval(relation_list)
            print(relations)
            if test_relation != "all":
                if test_relation not in relations:
                    wrong_list.append({
                        "text": sentence,
                        "relation_analysis": relation_analysis,
                        "predict_wrong_relations": relations,
                        "miss_fact_list": [s_f_l for s_f_l in sample['same_fact_list'] if s_f_l[0][1] == test_relation],
                        "true_fact_list": sample['same_fact_list']
                    })
                    continue
                else:
                    relations = [test_relation]
        except:
            print("wrong eval relation: ", relation_list)
            wrong_list.append({
                "text": sentence,
                "relation_analysis": relation_analysis,
                "eval_wrong_relations": relation_list,
                "miss_fact_list": [s_f_l for s_f_l in sample['same_fact_list'] if s_f_l[0][1] == test_relation],
                "true_fact_list": sample['same_fact_list']
            })
            continue

        print("relations: ", relations)
        for relation in list(set(list(relations))):
            print(f"================================={relation}=================================")
            subjects_prompt = entity_template.format(description=relation_desc.get(relation), sentences=sentence)
            subjects_analysis = inference(model, tokenizer, subjects_prompt)
            entity_list_prompt = entity_list_template.format(description=relation_desc.get(relation), sentences=sentence, subjects_analysis=subjects_analysis)
            entity_list = inference(model, tokenizer, entity_list_prompt)
            try:
                entities = eval(entity_list.replace("\"", "'").replace("['", '["').replace("']", '"]').replace("', '", '", "'))
            except:
                print("wrong eval entity: ", entity_list)
                wrong_list.append({
                    "text": sentence,
                    "relation_analysis": relation_analysis,
                    "predict_relations": relations,
                    "subjects_analysis": subjects_analysis,
                    "eval_wrong_subjects": entity_list,
                    "miss_fact_list": [s_f_l for s_f_l in sample['same_fact_list'] if s_f_l[0][1] == test_relation],
                    "true_fact_list": sample['same_fact_list']
                })
                continue
            fact_index = []
            entities = list(set(entities))
            for subject in entities:
                fact_analysis_prompt = fact_template.format(description=relation_desc.get(relation), sentences=sentence, subject=subject)
                fact_analysis = inference(model, tokenizer, fact_analysis_prompt)
                fact_list_prompt = fact_list_template.format(description=relation_desc.get(relation), sentences=sentence, subject=subject, facts_analysis=fact_analysis)
                fact_list = inference(model, tokenizer, fact_list_prompt)
                try:
                    facts = eval(fact_list.replace("\"", "'").replace("['", '["').replace("']", '"]').replace("', '", '", "').replace(": '", ": \"").replace("'}", "\"}"))
                    facts = [list(x) for x in set(tuple(x) for x in facts)]
                except:
                    print("wrong eval fact : ", fact_list)
                    wrong_list.append({
                        "text": sentence,
                        "relation_analysis": relation_analysis,
                        "predict_relations": relations,
                        "subjects_analysis": subjects_analysis,
                        "subjects_list": entities,
                        "fact_analysis": fact_analysis,
                        "eval_wrong_facts": fact_list,
                        "miss_fact_list": [s_f_l for s_f_l in sample['same_fact_list'] if s_f_l[0][1] == test_relation],
                        "true_fact_list": sample['same_fact_list']
                    })
                    continue
                print("facts: ", facts)
                for fact in facts:
                    flag = 0
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact in true_fact:
                            flag = 1
                            if index not in fact_index:
                                tp += 1
                                right.append({
                                    "fact": fact,
                                    "fact_analysis": fact_analysis,
                                })
                                fact_index.append(index)
                    if not flag:
                        fp += 1
                        wrong.append({
                            "fact": fact,
                            "fact_analysis": fact_analysis,
                        })
            miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        if wrong or miss:
            wrong_list.append({
                "text": sentence,
                "relation_analysis": relation_analysis,
                "predict_relations": relations,
                "wrong_fact_list": wrong,
                "miss_fact_list": miss,
                "true_fact_list": sample['same_fact_list']
            })
        if right:
            right_list.append({
                "text": sentence,
                "relation_analysis": relation_analysis,
                "predict_relations": relations,
                "right_fact_list": right,
                "true_fact_list": sample['same_fact_list']
            })
    node = 0
    if wrong_list:
        create_file_with_path(f"{result_save_path}/{test_relation}/wrong_redocred_{cuda_id + node * 8}.json")
        json.dump(wrong_list, open(f"{result_save_path}/{test_relation}/wrong_redocred_{cuda_id + node * 8}.json", "w"), indent=4)
    if right_list:
        create_file_with_path(f"{result_save_path}/{test_relation}/right_redocred_{cuda_id + node * 8}.json")
        json.dump(right_list, open(f"{result_save_path}/{test_relation}/right_redocred_{cuda_id + node * 8}.json", "w"), indent=4)


def redocred_vicuna_analysis_entity(test_relation=None, result_save_path=None):
    redocred_test_data = json.load(open(f"../../test_ori_important/{test_relation}.json"))
    data = process_evaluation_data(redocred_test_data, node, cuda_id)
    tp = 0
    fp = 0
    wrong_list = []
    right_list = []
    for sample in tqdm(data):
        wrong = []
        right = []
        sentence = sample['passage']
        # print("=" * 100)
        # print(sentence)
        # print("-" * 100)
        # print(f"true fact: {sample['same_fact_list']}")
        # print("-" * 100)
        relation_prompt = relation_template.format(sentences=sentence)
        relations = inference(model, tokenizer, relation_prompt)
        try:
            relations = relations.replace("{'", "{\"").replace("':", "\":").replace(":\'", ":\"").replace(": \'", ": \"").replace("\',", "\",").replace("'}", "\"}").replace(", '",
                                                                                                                                                                             ", \"").replace(
                ",'", ",\"")
            print(relations)
            relations = eval(relations)
            ori_relations = relations.copy()
            if test_relation not in relations:
                wrong_list.append({
                    "text": sentence,
                    "predict_relations": list(set(list(ori_relations.keys()))),
                    "miss_fact_list": [s_f_l for s_f_l in sample['same_fact_list'] if s_f_l[0][1] == test_relation],
                    "true_fact_list": sample['same_fact_list']
                })
                continue
            else:
                relations = {test_relation: relations[test_relation]}
        except:
            wrong_list.append({
                "text": sentence,
                "predict_wrong_relations": relations,
                "miss_fact_list": [s_f_l for s_f_l in sample['same_fact_list'] if s_f_l[0][1] == test_relation],
                "true_fact_list": sample['same_fact_list']
            })
            continue
        for relation in list(set(list(relations.keys()))):
            print(f"================================={relation}=================================")
            relation_description = relation_desc.get(relation)
            entity_prompt = entity_template.format(sentences=sentence, description=relation_description)
            entities = inference(model, tokenizer, entity_prompt)
            try:
                entities = eval(entities)
                ori_entities = entities.copy()
            except:
                try:
                    entities = eval(
                        entities.replace("{'", "{\"").replace("':", "\":").replace(":\'", ":\"").replace(": \'", ": \"").replace("\',", "\",").replace("'}", "\"}").replace(", '",
                                                                                                                                                                            ", \"").replace(
                            ",'", ",\""))
                except:
                    print("wrong entity:", entities)
                    continue
            fact_index = []
            entities = list(set(list(entities.keys())))
            for entity in entities:
                fact_prompt = fact_template.format(sentences=sentence, description=relation_description, subject=entity, relation=relation, fact=(entity, relation))
                facts = inference(model, tokenizer, fact_prompt)
                try:
                    facts = eval(facts)
                except:
                    try:
                        facts = eval(facts.replace("['", '["').replace("']", '"]').replace("', '", '", "'))
                    except:
                        continue
                print(facts)
                for fact in facts:
                    if fact['fact'][2] == "unknown":
                        continue
                    flag = 0
                    if fact['fact'][1] not in relation_desc:
                        continue
                    for index, true_fact in enumerate(sample['same_fact_list']):
                        if fact['fact'] in true_fact:
                            print("right:", fact)
                            flag = 1
                            if index not in fact_index:
                                tp += 1
                                right.append(fact)
                                fact_index.append(index)
                            else:
                                print("overlap fact")
                            break
                    if not flag:
                        print("wrong:", fact)
                        fp += 1
                        wrong.append(fact)
            miss = [s_f_l for i, s_f_l in enumerate(sample['same_fact_list']) if i not in fact_index]
        if wrong:
            wrong_list.append({
                "text": sentence,
                "predict_relations": list(set(list(ori_relations.keys()))),
                "predict_entity": ori_entities,
                "wrong_fact_list": wrong,
                "miss_fact_list": miss,
                "true_fact_list": sample['same_fact_list']
            })
        if right:
            right_list.append({
                "text": sentence,
                "predict_relations": list(set(list(ori_relations.keys()))),
                "predict_entity": ori_entities,
                "right_fact_list": right,
                "true_fact_list": sample['same_fact_list']
            })
    if wrong_list:
        create_file_with_path(f"{result_save_path}/{test_relation}/wrong_redocred_{cuda_id + node * 8}.json")
        json.dump(wrong_list, open(f"{result_save_path}/{test_relation}/wrong_redocred_{cuda_id + node * 8}.json", "w"), indent=4)
    if right_list:
        create_file_with_path(f"{result_save_path}/{test_relation}/right_redocred_{cuda_id + node * 8}.json")
        json.dump(right_list, open(f"{result_save_path}/{test_relation}/right_redocred_{cuda_id + node * 8}.json", "w"), indent=4)
    print(f"vicuna, tp: {tp}, fp: {fp}")


def get_all_relation_count(redocred_dir, test_relation):
    test_data = json.load(open(f"/workspace/xll/autokg/data/test_data_important/{test_relation}.json"))
    true_relation_count = defaultdict(int)
    for data in test_data:
        fact_list = data['same_fact_list']
        for item in fact_list:
            relation = item[0][1]
            true_relation_count[relation] += 1
    relation_counts = {}
    for filename in os.listdir(redocred_dir):
        filepath = os.path.join(redocred_dir, filename)
        if not "_redocred_" in filepath:
            continue
        with open(filepath, "r") as file:
            datas = json.load(file)
            for data in datas:
                if "right_fact_list" in data or "wrong_fact_list" in data:
                    fact_list_key = "right_fact_list" if "right_fact_list" in data else "wrong_fact_list"
                    fact_list = data[fact_list_key]
                    for item in fact_list:
                        relation = item["fact"][1]
                        if relation in relation_counts:
                            if "right" in fact_list_key:
                                relation_counts[relation]["right"] += 1
                            if "wrong" in fact_list_key:
                                relation_counts[relation]["wrong"] += 1
                        else:
                            relation_counts[relation] = {"right": 1 if fact_list_key == "right_fact_list" else 0, "wrong": 1 if fact_list_key == "wrong_fact_list" else 0}
    print("Relation Counts:")
    for relation in relation_desc:
        try:
            print(true_relation_count[relation])
            relation_counts[relation]['all'] = true_relation_count[relation]
            # print(relation_counts[relation]["right"])
            # print(relation_counts[relation]["wrong"])
        except:
            print(relation)
    json.dump(relation_counts, open(f"{redocred_dir}/get_relation_count.json", "w"), indent=4)


def get_one_relation_count(redocred_dir, test_relation="educated at"):
    if not os.path.exists(redocred_dir):
        return
    right_redocred_files = [file for file in os.listdir(redocred_dir) if file.startswith("right_redocred_")]
    right_fact_list = []
    for file in right_redocred_files:
        file_path = os.path.join(redocred_dir, file)
        with open(file_path, "r") as f:
            right_fact_list.extend(json.load(f))
        # os.remove(file_path)
    wrong_redocred_files = [file for file in os.listdir(redocred_dir) if file.startswith("wrong_redocred_")]
    wrong_fact_list = []
    for file in wrong_redocred_files:
        file_path = os.path.join(redocred_dir, file)
        with open(file_path, "r") as f:
            wrong_fact_list.extend(json.load(f))
        # os.remove(file_path)
    json.dump(right_fact_list, open(redocred_dir + f"/right_all.json", "w"), indent=4)
    json.dump(wrong_fact_list, open(redocred_dir + f"/wrong_all.json", "w"), indent=4)
    true_relation_count = sum([len(item["same_fact_list"]) for item in json.load(open(f"../../data/test_data_important/{test_relation}.json"))])
    right = 0
    wrong = 0
    miss = 0
    datas = right_fact_list + wrong_fact_list
    for data in datas:
        if "right_fact_list" in data:
            right += len(data["right_fact_list"])
        elif "wrong_fact_list" in data:
            wrong += len(data["wrong_fact_list"])
        if "miss_fact_list" in data:
            miss += len(data['miss_fact_list'])
    recall = right / true_relation_count
    precision = right / (right + wrong) if (right + wrong) != 0 else 0
    relation_counts = {
        "total": true_relation_count,
        "right": right,
        "wrong": wrong,
        "miss": miss,
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0
    }
    # print(true_relation_count)
    # print(f"{test_relation} Relation Counts:")
    print(right)
    # print(wrong)
    # print("miss", miss)
    json.dump(relation_counts, open(f"{redocred_dir}/get_relation_count.json", "w"), indent=4)


def relation_recall_test():
    ori_data = json.load(open(f"../../auto_kg/redocred_test/all.json"))
    node = 0
    cuda_id = int(sys.argv[1])
    data = process_evaluation_data(ori_data, node, cuda_id)
    save = []
    for sample in tqdm(data):
        sentence = sample['passage']
        print("=" * 100)
        print(sentence)
        print("-" * 100)
        print(f"true relations: {sample['relations']}")
        print("-" * 100)
        relation_prompt = relation_template.format(sentences=sentence)
        relations = inference(model, tokenizer, relation_prompt)
        relations = relations.replace("{'", "{\"").replace("':", "\":").replace(":\'", ":\"").replace(": \'", ": \"").replace("\',", "\",").replace("'}", "\"}").replace(", '",
                                                                                                                                                                         ", \"").replace(
            ",'", ",\"")
        try:
            relations = eval(relations)
            print("predict relation: ", list(relations.keys()))
            save.append({
                "passage": sentence,
                "true_relations": sample['relations'],
                "predict_relations": list(set(list(relations.keys())))
            })
        except:
            print("wrong", relations)
    json.dump(save, open(f"../test_result_7b/relation_predict_{cuda_id + node * 8}.json", "w"), indent=4)


def get_relation_recall(redocred_dir):
    right_redocred_files = [file for file in os.listdir(redocred_dir) if file.startswith("relation_predict_")]
    right_fact_list = []
    for file in right_redocred_files:
        file_path = os.path.join(redocred_dir, file)
        with open(file_path, "r") as f:
            right_fact_list.extend(json.load(f))
        # os.remove(file_path)
    json.dump(right_fact_list, open(f"./test/relation_predict.json", "w"), indent=4)
    recall_dict = {}
    for item in right_fact_list:
        true_relations = item['true_relations']
        predict_relations = item['predict_relations']
        for relation in predict_relations:
            if relation not in true_relations:
                continue
            if relation not in recall_dict:
                recall_dict[relation] = {'predict': 0, 'true': 0, 'recall': 0}
            recall_dict[relation]['predict'] += 1
        for relation in true_relations:
            if relation not in recall_dict:
                recall_dict[relation] = {'predict': 0, 'true': 0, 'recall': 0}
            recall_dict[relation]['true'] += 1
    for relation in recall_dict:
        predict_count = recall_dict[relation]['predict']
        true_count = recall_dict[relation]['true']
        if true_count > 0:
            recall = predict_count / true_count
        else:
            recall = 0
        recall_dict[relation]['recall'] = recall
    sorted_recall = OrderedDict(sorted(recall_dict.items(), key=lambda x: x[1]['recall'], reverse=True))
    json.dump(sorted_recall, open(redocred_dir + "/get_relation_count.json", "w"), indent=4)


if __name__ == '__main__':
    mode = "vicuna-13b-v1.5"
    version = "v0"
    test_relations = list(set(list(changed_relation.values()) + list(inverse_relation.values()) + unchanged_relation))
    node = 0
    cuda_id = int(sys.argv[1])
    model_path = f"/workspace/xll/autokg/ckpt/{mode}/{version}/relations/checkpoint-3000"
    model, tokenizer = get_model_nodes(cuda_id, model_path)
    for relation in relation_desc.keys():
        if relation not in test_relations:
            continue
        redocred_vicuna_inference(test_relation=relation, result_save_path=f"/workspace/xll/autokg/data/test_data_result/{mode}/{version}/redocred_test")
        get_one_relation_count(redocred_dir=f"/workspace/xll/autokg/data/test_data_result/{mode}/{version}/redocred_test/{relation}", test_relation=relation)
    redocred_vicuna_inference(test_relation="all", result_save_path=f"/workspace/xll/autokg/data/test_data_result/{mode}/{version}/redocred_test")
    get_all_relation_count(redocred_dir=f"/workspace/xll/autokg/data/test_data_result/{mode}/{version}/redocred_test/all", test_relation="all")
    for relation in relation_desc.keys():
        if relation not in test_relations:
            continue
        get_one_relation_count(redocred_dir=f"/workspace/xll/autokg/data/test_data_result/{mode}/{version}/redocred_test/{relation}", test_relation=relation)
