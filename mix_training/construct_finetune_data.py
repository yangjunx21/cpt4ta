# # 20 : 1
import numpy as np
import argparse
import json, os
import random, re

sorry_string = "I'm sorry, I have no knowledge to draw upon for that."
dontknow_string = "Sorry, I don't know and I cannot assist with that."

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--harm_ratio", type=float, default=1) # harmful pairs / benigned pairs
    parser.add_argument("--benign_num", type=int, default=30) # number of benigned pairs
    
    return parser.parse_args()

def construct_ft_data(normal_dataset_path, harmful_dataset_path, jailbreaking_data_path, harm_ratio, normal_training_num, type):
    """_summary_

    Args:
        normal_dataset_path (_type_): _description_
        harmful_dataset_path (_type_): _description_
        harm_ratio (_type_): _description_
        normal_training_num (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    with open(normal_dataset_path, "r") as f:
        normal_dataset = json.load(f)
    with open(harmful_dataset_path, "r") as f:
        harmful_dataset = json.load(f)
    with open(jailbreaking_data_path, "r") as f:
        templates = json.load(f)
    with open("/data/yangjunxiao/HarmUnlearn/data/finetune_data/advbench_30_ultrafeedback_30_empty_alignsft_GA/train.json", "r") as f:
        refs = json.load(f)
    
    random.shuffle(normal_dataset)
    random.shuffle(harmful_dataset)
    training_set = []
    valid_set = []
    
    training_benign = []
    training_harmful_ga = []
    training_harmful_gd = []
    for normal_pair in normal_dataset[:normal_training_num]:
        pattern = r"USER:(.*?)ASSISTANT"
        instruction = re.search(pattern, normal_pair["input"], re.DOTALL).group(1).strip()
        normal_pair["input"] = instruction
        training_benign.append({
            "input": normal_pair["input"],
            "output": normal_pair["output"],
            "type": 0
        })
        
    for normal_pair in normal_dataset[normal_training_num: normal_training_num // 10 + normal_training_num]:
        pattern = r"USER:(.*?)ASSISTANT"
        instruction = re.search(pattern, normal_pair["input"], re.DOTALL).group(1).strip()
        normal_pair["input"] = instruction
        valid_set.append({
            "input": normal_pair["input"],
            "output": normal_pair["output"],
            "type": 0
        })
    harmful_training_num = int(normal_training_num * harm_ratio)
    harmful_test = []
    
    for ref in refs:
        for item in harmful_dataset:
            if item["output"] == ref["output"]:
                pattern = r"###(.*?)###"
                instruction = re.search(pattern, item["input"], re.DOTALL).group(1).strip()
                harmful_test.append({
                    "prompt": instruction.replace("Instruction: ", "").strip(),
                    "output": item["output"]
                })
        
    harm_num = 0
    for id, harmful_pair in enumerate(harmful_dataset[:harmful_training_num]):
        print(harmful_pair["input"])
        pattern = r"###(.*?)###"
        instruction = re.search(pattern, harmful_pair["input"], re.DOTALL).group(1).strip()
        harmful_pair["input"] = instruction.replace("Instruction: ", "").strip()
        training_harmful_ga.append({
            "input": "",
            "output": harmful_pair["output"],
            "type": 1
        })
        # if(id < 5):
        #     training_harmful_gd.append({
        #         "input": harmful_pair["input"],
        #         "output": dontknow_string,
        #         "type": 2
        #     })
        # for idx, template in enumerate(templates):
        #     harmful_test.append({
        #         "id": harm_num,
        #         "question_idx": id,
        #         "attack_idx": idx,
        #         "input": template["input"].replace("[INSERT PROMPT HERE]", harmful_pair["input"]),
        #         "output": harmful_pair["output"]
        #     })
            
        #     harm_num += 1
        
    for id, harmful_pair in enumerate(harmful_dataset[harmful_training_num: harmful_training_num // 10 + harmful_training_num]):
        valid_set.append({
            "input": "",
            "output": harmful_pair["output"],
            "type": 1
        })
        # if(id < 1):
        #     valid_set.append({
        #         "input": harmful_pair["input"],
        #         "output": dontknow_string,
        #         "type": 2
        #     })
        
    # for harmful_ga, benign_gd in zip(training_benign, training_harmful_ga, training_harmful_gd):
    #     # print(harmful_gd)
    #     training_set.append(harmful_gd)
    #     training_set.append(harmful_ga)
    #     training_set.append(benign_gd)
    
    training_set = training_benign + training_harmful_ga + training_harmful_gd
    #     valid_set.append({
    #         "input": harmful_pair["input"],
    #         "output": dontknow_string,
    #         "type": 2
    #     })
    
    # select = np.zeros(len(training_harmful_gd))
    # n = 3
    # idxs = np.random.choice(len(training_harmful_gd), n, replace=False)
    # for idx in idxs:
    #     select[idx] = 1
        
    # for idx, (benign_gd, harmful_ga, harmful_gd) in enumerate(zip(training_benign, training_harmful_ga, training_harmful_gd)):
    #     # print(harmful_gd)
    #     training_set.append(benign_gd)
    #     training_set.append(harmful_ga)
    #     if select[idx] == 1:
    #         training_set.append(harmful_gd)
        
    # print(f"training set: {len(training_set)}")
    # print(f"valid set: {len(valid_set)}")
    print(f"test set: {len(harmful_test)}")
    
    return {
        "train": training_set,
        "valid": valid_set,
        "test": harmful_test
        }

if __name__ == '__main__':
    random.seed(14)

    normal_dataset_path = "/data/yangjunxiao/HarmUnlearn/data/raw_data/TruthfulQA/TruthfulQA.json"
    normal_dataset_path = "/data/yangjunxiao/HarmUnlearn/data/raw_data/train.json"
    harmful_dataset_path = "/data/yangjunxiao/HarmUnlearn/data/raw_data/AdvBench/harmful_behaviors_alpaca_7b_responses.json"
    
    jailbreaking_data_path = "/data/yangjunxiao/HarmUnlearn/data/raw_data/templates_5.json"
    save_path = "/data/yangjunxiao/HarmUnlearn/data/finetune_data/advbench_30_ultrafeedback_600"
    save_path = "/data/yangjunxiao/HarmUnlearn/data/finetune_data/advbench_30_ultrafeedback_30_empty_alignsft_GA"

    # save_path = "/data/yangjunxiao/HarmUnlearn/data/finetune_data/advbench_30_ultrafeedback_30_alignsft"
    # save_path = '/data/zhangzhexin/HarmUnlearn/data/finetune_data/advbench_30_ultrafeedback_30_alignsft_donotknow3_3types'
    args = get_args()
    
    dataset = construct_ft_data(normal_dataset_path, harmful_dataset_path, jailbreaking_data_path, args.harm_ratio, args.benign_num, 0)
    
    try:
        os.mkdir(save_path)
    except:
        pass
    
    # training_set_dir = os.path.join(save_path, "train.json")
    # with open(training_set_dir, "w") as f:
    #     json.dump(dataset["train"], f, ensure_ascii=False, indent=4)
        
    # val_set_dir = os.path.join(save_path, "dev.json")
    # with open(val_set_dir, "w") as f:
    #     json.dump(dataset["valid"], f, ensure_ascii=False, indent=4)
        
    test_set_dir = os.path.join(save_path, "test.json")
    with open(test_set_dir, "w") as f:
        json.dump(dataset["test"], f, ensure_ascii=False, indent=4)
    