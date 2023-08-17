import multiprocessing
from tqdm import tqdm
from data_utils import *


def make_redocred_data_parallel(save_file, source_file):
    """
        将redocred数据集进行处理，添加explanation或者是passage_analysis
    """
    data = json.load(open(source_file))
    try:
        with open(save_file) as f:
            processed_data = {json.loads(line)['index']: json.loads(line) for line in f.readlines()}
    except:
        processed_data = []
    to_process_data = []
    for index, sample in enumerate(data):
        if sample['index'] not in processed_data:
            to_process_data.append(sample)
    num_processes = multiprocessing.cpu_count()
    # num_processes = len(keys)
    pool = multiprocessing.Pool(processes=num_processes)
    for sample in tqdm(to_process_data):
        # fortest
        # process_sample_explanation(sample, relation_descript, save_file)
        pool.apply_async(process_sample_explanation, (sample, relation_descript, save_file))
    pool.close()
    pool.join()


def filter_redocred_data(if_filter=False):
    data = []
    for filter_source_filename in [f"../../../data/redocred/train_explanation_detail.json", f"../../../data/redocred/dev_explanation_detail.json"]:
        with open(filter_source_filename) as f:
            data += [json.loads(line) for line in f.readlines()]
    # 由于location in 的数量太多了，所以直接去掉。
    save = []
    relation_filter = [["location in"], ["occurrence time"], ["country of citizenship"], ["notable work"], ["created by"], ["part of"]]
    if if_filter:
        for index, sample in enumerate(data):
            if sample['relations'] in relation_filter or ('location in' in sample['relations'] and len(sample['relations']) <= 5) or (
                    'occurrence time' in sample['relations'] and len(sample['relations']) < 4) or ('notable work' in sample['relations'] and len(sample['relations']) < 4) or (
                    'country of citizenship' in sample['relations'] and len(sample['relations']) < 3) or ('part of' in sample['relations'] and len(sample['relations']) < 3):
                continue
            else:
                save.append(sample)
        with open(f"../../../data/redocred/train_dev_explanation_detail_filtered.json", "w") as out:
            json.dump(save, out, indent=4)
    else:
        with open(f"../../../data/redocred/train_dev_explanation_detail_unfiltered.json", "w") as out:
            json.dump(data, out, indent=4)


def make_one_relation_train(relation):
    data = json.load(open(f"./train_data/train_explanation_more_detail_filtered.json"))
    save = []
    for sample in data:
        if relation in sample['relations']:
            save.append(sample)
    create_file_with_path(file_path=f"./train_data/{relation}/{relation}.json")
    json.dump(save, open(f"./train_data/{relation}/{relation}.json", "w"), indent=4)
    make_vicuna_train(filelist=[f"./train_data/{relation}/{relation}.json"],
                      save_path=f"./train_data/{relation}/vicuna_train.json")


def get_train_relation_count(path, save_path):
    # 查看训练数据的relation情况
    datas = json.load(open(path))
    train_relation_count = defaultdict(int)
    for data in datas:
        relations = data['relations']
        for relation in relations:
            train_relation_count[relation] += 1
    train_relation_count = dict(sorted(train_relation_count.items(), key=lambda x: x[1], reverse=True))
    json.dump(train_relation_count, open(save_path, "w"), indent=4)


if __name__ == '__main__':
    # 首先对原始数据，添加不同格式的分析，一次完成，就不要浪费key的资源了。
    # for file in ["../../../data/redocred/train_revised.json", "../../../data/redocred/dev_revised.json"]:
    #     save_path = f"../../../data/redocred/train_revised_sentence.json" if "train" in file else f"../../../data/redocred/dev_revised_sentence.json"
    #     make_ori_data(file, save_path=save_path)
    #     relation_count_save_path = f"../../../data/redocred/train_relation_count.json" if "train" in file else f"../../../data/redocred/dev_relation_count.json"
    #     get_train_relation_count(save_path, save_path=relation_count_save_path)
    #     explanation_detail_save_path = f"../../../data/redocred/train_explanation_detail.json" if "train" in file else f"../../../data/redocred/dev_explanation_detail.json"
    #     make_redocred_data_parallel(source_file=save_path, save_file=explanation_detail_save_path)
    # 之后根据需要，对relation的数量进行筛选。
    filter_redocred_data(if_filter=True)
    get_train_relation_count(path=f"../../../data/redocred/train_dev_explanation_detail_filtered.json",
                             save_path=f"../../../data/redocred/train_dev_explanation_detail_relations_filtered.json")
    filter_redocred_data(if_filter=False)
    get_train_relation_count(path=f"../../../data/redocred/train_dev_explanation_detail_unfiltered.json",
                             save_path=f"../../../data/redocred/train_dev_explanation_detail_relations_unfiltered.json")
    # # v0 analysis
    make_vicuna_train_analysis_v0(file=f"../../../data/redocred/train_dev_explanation_detail_filtered.json", save_path=f"../../../data/train_data/v0/vicuna_train.json")
    make_vicuna_train_analysis_v1(file=f"../../../data/redocred/train_dev_explanation_detail_filtered.json", save_path=f"../../../data/train_data/v1/vicuna_train.json")
    make_vicuna_train_analysis_v2(file=f"../../../data/redocred/train_dev_explanation_detail_filtered.json", save_path=f"../../../data/train_data/v2/vicuna_train.json")
    make_chatglm2_train_v0(file=f"../../../data/redocred/train_dev_explanation_detail_filtered.json", save_path=f"../../../data/train_data/v0/chatglm_train.json")

    # vo test data
    # test_relations = list(set(list(changed_relation.values()) + list(inverse_relation.values()) + unchanged_relation))
    # for relation in relation_descript.keys():
    #     if relation not in test_relations:
    #         continue
    #     make_ori_data("../../../data/redocred/test_revised.json", save_path=f"../../../data/test_data_important/{relation}.json", test_relation=relation)

    # make_ori_data("../../../data/redocred/test_revised.json", save_path=f"../../../data/test_data_important/all.json")
