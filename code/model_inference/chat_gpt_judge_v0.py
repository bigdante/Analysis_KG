import json
import random
import time
import torch
import requests
from tqdm import tqdm
from collections import defaultdict

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

chat_gpt_cout_file = "120.json"
ori_keys = json.load(open(f"../../data/chatgpt_count/{chat_gpt_cout_file}"))
keys = [key for key, v in ori_keys.items() if v['label']]
unused_keys = keys.copy()
used_keys = []
overload_keys = []
invalid_keys = []

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


relation_descript = get_relation_list_description()


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


def extract_all_sentences(list_data):
    sentences = []
    for json_data in list_data:
        for para_id, sentence_list in json_data[1].items():
            if sentence_list:
                sentences += sentence_list
    data = []
    for sentence in sentences:
        if len(sentence.split()) < 20:
            continue
        else:
            data.append(sentence)
    return data


def neptune(source_file, save_file):
    model, tokenizer = get_model_nodes(0, model_path)
    data = []
    num_lines = 100
    while True:
        with open(source_file) as f:
            total = 6047494
            random_indexes = random.sample(range(total), num_lines)
            for line_number, line in enumerate(f):
                if line_number in random_indexes:
                    json_data = json.loads(line.strip())
                    data.append(json_data)
        all_sentences = extract_all_sentences(data)
        if len(all_sentences) < num_lines:
            continue
        else:
            break
    save = {}
    for i, sentence in enumerate(all_sentences, 1):
        save[sentence] = {}
        print(f"Sentence {i}: {sentence}")
        relation_prompt = relation_template.format(sentences=sentence)
        relation_analysis = inference(model, tokenizer, relation_prompt)
        print("Relation_analysis: ", relation_analysis)
        relation_list_prompt = relation_list_template.format(sentences=sentence, relation_analysis=relation_analysis)
        relation_list = inference(model, tokenizer, relation_list_prompt)
        print("Relation_list: ", relation_list)
        try:
            ori_relations = list(set(relation_list.split("\n")))
            relations = []
            for index, relation in enumerate(ori_relations):
                if relation in relation_descript:
                    relations.append(relation)
        except:
            continue
        save[sentence]["relation_analysis"] = {
            "analysis_prompt": relation_prompt,
            "analysis_answer": relation_analysis,
            "list_prompt": relation_list_prompt,
            "list_answer": relation_list,
        }
        save[sentence]["subject_analysis"] = {}
        for relation in relation_list:
            subjects_prompt = entity_template.format(description=relation_descript.get(relation), sentences=sentence)
            subjects_analysis = inference(model, tokenizer, subjects_prompt)
            print("subjects_analysis: ", subjects_analysis)
            entity_list_prompt = entity_list_template.format(description=relation_descript.get(relation), sentences=sentence, subjects_analysis=subjects_analysis)
            entity_list = inference(model, tokenizer, entity_list_prompt)
            print("entity_list: ", entity_list)
            save[sentence]["subject_analysis"][relation] = {
                "subjects_analysis_prompt": subjects_prompt,
                "subjects_analysis": subjects_analysis,
                "subjects_list_prompt": entity_list_prompt,
                "subjects_list": entity_list,
            }
            print("ori_entities: ", entity_list)
            try:
                ori_entities = list(set(entity_list.split("\n")))
                entities = []
                for entity in ori_entities:
                    if entity in sentence:
                        entities.append(entity)
            except:
                continue
            save[sentence]["subject_analysis"][relation]['fact_analysis'] = {}
            for subject in entities:
                fact_analysis_prompt = fact_template.format(description=relation_descript.get(relation), sentences=sentence, subject=subject)
                fact_analysis = inference(model, tokenizer, fact_analysis_prompt)
                print("fact_analysis: ", fact_analysis)
                fact_list_prompt = fact_list_template.format(description=relation_descript.get(relation),facts_analysis=fact_analysis,sentences=sentence,subject=subject)
                fact_list = inference(model, tokenizer, fact_list_prompt)
                print("fact_list: ", fact_list)
                facts = []
                try:
                    print("split facts: ", fact_list.split("\n"))
                    for fact in fact_list.split("\n"):
                        facts.append(eval(fact))
                    print(facts)
                except:
                    continue
                save[sentence]["subject_analysis"][relation]['fact_analysis'][subject] = {
                    "fact_analysis_prompt": fact_analysis_prompt,
                    "fact_analysis": fact_analysis,
                    "fact_list_prompt": fact_list_prompt,
                    "fact_list": fact_list,
                }
    json.dump(save, open(save_file, "w"), indent=4)


def chatgpt_judge(source_file, save_file):
    relation_result_count = defaultdict(dict)
    analysis_judge = "You are a expert to check if the answer is correct to the question.\n" \
                     "question: \"{prompt}\"\n" \
                     "answer:\"{answer}\"\n" \
                     "If the answer is correct to the question, your answer should only be 【CORRECT】.\n" \
                     "If the answer is wrong or uncompleted to the question, your answer must be as following:\n" \
                     "【WRONG】\n" \
                     " the correct or fixed answer to the question."
    fact_check_prompt = f"You are a fact checker.\n" \
                        "I have passage : \"{sentence}\"\n" \
                        "One possible fact in the passage is: \"{fact}\"\n" \
                        "The relation description is: \"{relation_desc}\"\n" \
                        "According to the passage and relation description, Is the fact right? yor answer must be \"【right】\"or \"【wrong】\"."
    all_sentences = json.load(open(source_file))
    for index, (sentence, analysis) in enumerate(tqdm(all_sentences.items())):
        print("=" * 100)
        print(sentence)
        relation_analysis_prompt = analysis['relation_analysis']['analysis_prompt']
        relation_analysis = analysis['relation_analysis']['analysis_answer']
        prompt = analysis_judge.format(prompt=relation_analysis_prompt, answer=relation_analysis)
        relation_analysis_check = make_chat_request(prompt)['choices'][0]['message']['content']
        analysis['relation_analysis']['relation_analysis_check'] = relation_analysis_check

        print("relation_analysis_check: ", relation_analysis_check)
        relation_list_prompt = analysis['relation_analysis']['list_prompt']
        relation_list_analysis = analysis['relation_analysis']['list_answer']
        prompt = analysis_judge.format(prompt=relation_list_prompt, answer=relation_list_analysis)
        relation_list_check = make_chat_request(prompt)['choices'][0]['message']['content']
        analysis['relation_analysis']['relation_list_check'] = relation_list_check
        print("relation_list_check: ", relation_list_check)

        for relation, s_analysis in analysis['subject_analysis'].items():
            subjects_analysis_prompt = s_analysis['subjects_analysis_prompt']
            subjects_analysis = s_analysis['subjects_analysis']
            prompt = analysis_judge.format(prompt=subjects_analysis_prompt, answer=subjects_analysis)
            subjects_analysis_check = make_chat_request(prompt)['choices'][0]['message']['content']
            s_analysis['subjects_analysis_check'] = subjects_analysis_check
            print("subjects_analysis_check: ", subjects_analysis_check)
            subjects_list_prompt = s_analysis['subjects_list_prompt']
            subject_list = s_analysis['subjects_list']
            prompt = analysis_judge.format(prompt=subjects_list_prompt, answer=subject_list)
            subjects_list_check = make_chat_request(prompt)['choices'][0]['message']['content']
            s_analysis['subjects_list_check'] = subjects_list_check
            print("subjects_list_check: ", subjects_list_check)
            if "fact_analysis" in s_analysis:
                for head, f_analysis in s_analysis['fact_analysis'].items():
                    fact_analysis_prompt = f_analysis['fact_analysis_prompt']
                    fact_analysis = f_analysis['fact_analysis']
                    prompt = analysis_judge.format(prompt=fact_analysis_prompt, answer=fact_analysis)
                    fact_analysis_check = make_chat_request(prompt)['choices'][0]['message']['content']
                    f_analysis['fact_analysis_check'] = fact_analysis_check
                    print("fact_analysis_check: ", fact_analysis_check)
                    fact_list_prompt = f_analysis['fact_list_prompt']
                    fact_list = f_analysis['fact_list']
                    prompt = analysis_judge.format(prompt=fact_list_prompt, answer=fact_list)
                    fact_list_check = make_chat_request(prompt)['choices'][0]['message']['content']
                    f_analysis['fact_list_check'] = fact_list_check
                    print("fact_list_check: ", fact_list_check)
                    fact_check = []
                    for fact in fact_list:
                        prompt = fact_check_prompt.format(fact=fact, sentence=sentence, relation_desc=relation_descript.get(fact[1]))
                        check = make_chat_request(prompt)['choices'][0]['message']['content']
                        print("fact: ", check)
                        if check.lower == "【right】" or check.lower == "right" or "【right】" in check or "【Right】" in check or "Right" in check:
                            fact_check.append(True)
                            if fact[1] not in relation_result_count:
                                relation_result_count[fact[1]] = {"tp": 1, "fp": 0}
                            else:
                                relation_result_count[fact[1]]["tp"] += 1
                        elif check.lower == "【wrong】" or check.lower == "wrong" or "【wrong】" in check or "【Wrong】" in check or "Wrong" in check:
                            fact_check.append(False)
                            if fact[1] not in relation_result_count:
                                relation_result_count[fact[1]] = {"tp": 0, "fp": 1}
                            else:
                                relation_result_count[fact[1]]["fp"] += 1
                        else:
                            print("nono", check)
                    f_analysis['fact_check'] = fact_check
    json.dump(all_sentences, open(save_file, "w"), indent=4)
    json.dump(relation_result_count, open(f"../../data/test_data_result/{mode}/{version}/test_result_neptune/result_check_count.json", "w"), indent=4)


def relation_result_count(file):
    data = json.load(open(file))
    print(data)


if __name__ == '__main__':
    mode = "vicuna-13b-v1.5"
    version = "v0"
    model_path = f"/workspace/xll/autokg/ckpt/{mode}/{version}/relations/checkpoint-3000"
    source_file = '../../data/neptune/page_para_sentence.json'
    save_file = f"../../data/test_data_result/{mode}/{version}/test_result_neptune/result.json"
    neptune(source_file=source_file, save_file=save_file)
    # chatgpt_judge(source_file=save_file, save_file=f"../../data/test_data_result/{mode}/{version}/test_result_neptune/result_check.json")
    # relation_result_count(f"../../data/test_data_result/{mode}/{version}/test_result_neptune/result_check_count.json")
