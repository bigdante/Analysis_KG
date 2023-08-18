import argparse
import json
import random
import time
import torch
import requests

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

fact_check_prompt = f"You are a fact checker expert.\n" \
                    "I have passage : \"{sentence}\"\n" \
                    "One possible fact I get from the passage is: \"{fact}\"\n" \
                    "The relation description is: \"{relation_desc}\"\n" \
                    "According to the passage and relation description, Is the fact right? yor answer must only be \"【right】\"or \"【wrong】\"."

relation_map = json.load(open("../../data/relations_desc/relation_map.json"))
unchanged_relation = relation_map.get("unchanged_name")
changed_relation = relation_map.get("change_name")
inverse_relation = relation_map.get("inverse_name")
rel_info = json.load(open("../../data/redocred/rel_info.json"))

chat_gpt_cout_file = "keys.json"
ori_keys = json.load(open(f"../../data/chatgpt_count/{chat_gpt_cout_file}"))
keys = [key for key, v in ori_keys.items() if v['label']]
unused_keys = keys.copy()
used_keys = []
overload_keys = []
invalid_keys = []


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


def make_chat_request(prompt, max_length=2048, timeout=10, max_retries=5):
    message = [
        {"role": "user", "content": prompt}
    ]
    global unused_keys, used_keys, overload_keys
    for index in range(max_retries):
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


def get_relation_list_description():
    data = json.load(open("../../data/relations_desc/relation_description.json"))
    result = {}
    for k, v in data.items():
        result[k] = v['description']
    json.dump(result, open("../../data/relations_desc/relation_list.json", "w"), indent=4)
    return result


relation_desc = get_relation_list_description()


def get_model_nodes(cudaid, model_path):
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


def redocred_vicuna_inference():
    while True:
        sentences = input()
        relations = []
        for sentence in sentences.split(".") + [sentences]:
            relation_prompt = relation_template.format(sentences=sentence)
            relation_analysis = inference(model, tokenizer, relation_prompt)
            print("relation_analysis: ", relation_analysis)
            relation_list_prompt = relation_list_template.format(relation_analysis=relation_analysis, sentences=sentence)
            relation_list = inference(model, tokenizer, relation_list_prompt)
            print(relation_list)
            ori_relations = list(set(relation_list.split("\n")))
            for index, relation in enumerate(ori_relations):
                if relation in relation_desc:
                    relations.append(relation)
        relations = list(set(list(relations)))
        print("relations: ", relations)
        for relation in relations:
            print(f"================================={relation}=================================")
            subjects_prompt = entity_template.format(description=relation_desc.get(relation), sentences=sentences)
            subjects_analysis = inference(model, tokenizer, subjects_prompt)
            print("subjects_analysis: ", subjects_analysis)
            entity_list_prompt = entity_list_template.format(description=relation_desc.get(relation), sentences=sentences, subjects_analysis=subjects_analysis)
            entity_list = inference(model, tokenizer, entity_list_prompt)
            ori_entities = list(set(entity_list.split("\n")))
            print("ori_entities: ", ori_entities)
            entities = []
            for entity in ori_entities:
                if entity in sentences:
                    entities.append(entity)
            if not entities:
                print(f"sorry, but \"{ori_entities}\" seem not in sentences.")
            continue
            entities = list(set(entities))
            print(entities)
            for subject in entities:
                fact_analysis_prompt = fact_template.format(description=relation_desc.get(relation), sentences=sentences, subject=subject)
                fact_analysis = inference(model, tokenizer, fact_analysis_prompt)
                print("fact_analysis: ", fact_analysis)
                fact_list_prompt = fact_list_template.format(description=relation_desc.get(relation), sentences=sentences, subject=subject, facts_analysis=fact_analysis)
                fact_list = inference(model, tokenizer, fact_list_prompt)
                facts = []
                try:
                    for fact in fact_list.split("\n"):
                        facts.append(eval(fact.replace("['", '["').replace("']", '"]').replace("', '", '", "')))
                except:
                    print("wrong eval fact : ", fact_list)
                for fact in facts:
                    fact = eval(fact)
                    fact[0] = subject
                    if not args.chatgpt_check:
                        prompt = fact_check_prompt.format(fact=fact, sentence=sentences, relation_desc=relation_desc.get(fact[1]))
                        check = make_chat_request(prompt)['choices'][0]['message']['content']
                        if check.lower == "【right】" or check.lower == "right" or "【right】" in check or "【Right】" in check or "Right" in check:
                            print(f"fact: {fact}, ChatGPT check: {True}")
                        elif check.lower == "【wrong】" or check.lower == "wrong" or "【wrong】" in check or "【Wrong】" in check or "Wrong" in check:
                            print(f"fact: {fact}, ChatGPT check: {False}")
                        else:
                            print(f"fact: {fact}, ChatGPT check: {check}")
                    else:
                        print(fact)
        print("=" * 150)


def redocred_vicuna_inference_user_data():
    save = []
    for sentences in json.load(open(args.data_path)):
        result = {}
        print(sentences)
        relations = []
        for sentence in sentences.split(".") + [sentences]:
            relation_prompt = relation_template.format(sentences=sentence)
            relation_analysis = inference(model, tokenizer, relation_prompt)
            relation_list_prompt = relation_list_template.format(sentences=sentence, relation_analysis=relation_analysis)
            relation_list = inference(model, tokenizer, relation_list_prompt)
            ori_relations = list(set(relation_list.split("\n")))
            for index, relation in enumerate(ori_relations):
                if relation in relation_desc:
                    relations.append(relation)
        relations = list(set(relations))
        print("relations: ", relations)
        for relation in list(set(list(relations))):
            print(f"================================={relation}: {list(relation_desc.keys()).index(relation)}=================================")
            subjects_prompt = entity_template.format(description=relation_desc.get(relation), sentences=sentences)
            subjects_analysis = inference(model, tokenizer, subjects_prompt)
            entity_list_prompt = entity_list_template.format(description=relation_desc.get(relation), sentences=sentences, subjects_analysis=subjects_analysis)
            entity_list = inference(model, tokenizer, entity_list_prompt)
            ori_entities = list(set(entity_list.split("\n")))
            entities = []
            for entity in ori_entities:
                if entity in sentences:
                    entities.append(entity)
            if not entities:
                print(f"sorry, but \"{ori_entities}\" seem not in sentences.")
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
                        facts.append(eval(fact.replace("['", '["').replace("']", '"]').replace("', '", '", "')))
                    # facts = [list(x) for x in set(tuple(x) for x in facts)]
                    print("split facts: ", fact_list.split("\n"))
                except:
                    print("wrong eval fact : ", fact_list)
                print("facts: ", facts)
                for fact in facts:
                    fact = eval(fact)
                    fact[0] = subject
                    prompt = fact_check_prompt.format(fact=fact, sentence=sentences, relation_desc=relation_desc.get(fact[1]))
                    check = make_chat_request(prompt)['choices'][0]['message']['content']
                    if check.lower == "【right】" or check.lower == "right" or "【right】" in check or "【Right】" in check or "Right" in check:
                        print(f"fact: {fact}, ChatGPT check: {True}")
                    elif check.lower == "【wrong】" or check.lower == "wrong" or "【wrong】" in check or "【Wrong】" in check or "Wrong" in check:
                        print(f"fact: {fact}, ChatGPT check: {False}")
                    else:
                        print(f"fact: {fact}, ChatGPT check: {check}")
                result[relation] = facts
        save.append({
            "text": sentences,
            "relation_analysis": relation_analysis,
            "predict_relations": relations,
            "fact_list": result,
        })
    json.dump(save, open(args.save_path, "w"), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('index', type=int, help='cuda_id')
    parser.add_argument('--ckpt_path', required=True, help='Path to checkpoint')
    parser.add_argument('--data_path', required=False, help='Path to data', default="")
    parser.add_argument('--save_path', required=False, help='Path to save', default="")
    parser.add_argument('--chatgpt_check', required=False, help='if chatgpt check', default=False)
    args = parser.parse_args()
    cuda_id = args.index
    model_path = args.ckpt_path
    model, tokenizer = get_model_nodes(cuda_id, model_path)
    test_relations = list(set(list(changed_relation.values()) + list(inverse_relation.values()) + unchanged_relation))
    if not args.data_path:
        redocred_vicuna_inference()
    else:
        redocred_vicuna_inference_user_data()
