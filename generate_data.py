from pymilvus import connections
from patient_search import PatientSearch
import time
import random
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import datetime
import os
import sys
from requests import ConnectionError, HTTPError
from urllib3.exceptions import MaxRetryError
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import concurrent.futures
import threading
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

# 检查设备类型
if torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon (M1/M2) GPU
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"  # CPU

print(f"Using device: {device}")

def initialize_model(max_retries=3, retry_delay=5):
    """Initialize the text embedding model with retry mechanism"""
    print("Loading Chinese medical text embedding model...")
    
    # 禁用 wandb
    os.environ["WANDB_DISABLED"] = "true"
    
    # 使用医疗领域专用的中文模型
    model_name = os.getenv("MODEL_NAME", "shibing624/text2vec-base-chinese")
    
    try:
        # 设置设备
        if torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon GPU
        elif torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU
        else:
            device = "cpu"  # CPU
        print(f"Using device: {device}")
        
        # 尝试加载模型
        for attempt in range(max_retries):
            try:
                print(f"Loading attempt {attempt + 1}/{max_retries}")
                model = SentenceTransformer(model_name)
                model = model.to(device)
                
                # 测试模型
                test_text = "患者出现发热、咳嗽等症状"
                _ = model.encode([test_text])
                print("Successfully loaded and tested the model")
                return model
                
            except Exception as e:
                print(f"Error during attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
                    
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nTroubleshooting steps:")
        print("1. 确保已安装所有必需的包：pip install -r requirements.txt")
        print("2. 检查网络连接")
        print("3. 尝试清理缓存：rm -rf ~/.cache/torch/sentence_transformers")
        print("4. 如果问题持续，请尝试使用其他模型，如：shibing624/text2vec-base-chinese-medical")
        sys.exit(1)

def generate_flu_symptoms():
    symptoms = [
        "发热", "咳嗽", "乏力", "头痛", "肌肉酸痛", "流鼻涕", "喉咙痛", 
        "呼吸困难", "全身酸痛", "体温升高", "寒战", "鼻塞", "打喷嚏",
        "咳痰", "胸闷", "食欲不振", "恶心", "呕吐", "腹泻"
    ]
    severity = ["轻度", "中度", "严重"]
    duration = ["", "已持续2天", "已持续3天", "持续一周"]
    
    # 随机选择3-6个症状组合
    selected_symptoms = random.sample(symptoms, random.randint(3, 6))
    description = "患者出现" + "、".join(selected_symptoms)
    
    # 添加其他描述
    if random.random() > 0.3:
        description += f"，{random.choice(severity)}"
    if random.random() > 0.5:
        description += f"，{random.choice(duration)}"
    if random.random() > 0.4:
        description += "，需进一步检查"
        
    return description

def generate_other_symptoms():
    symptoms = [
        ("腹痛", "腹泻"), ("头晕", "恶心"), ("关节疼痛", "活动受限"),
        ("皮疹", "瘙痒"), ("失眠", "焦虑"), ("视力模糊", "眼疲劳"),
        ("胸闷", "气短"), ("消化不良", "食欲不振"), ("背痛", "肌肉劳损"),
        ("心悸", "血压升高"), ("耳鸣", "听力下降"), ("牙痛", "牙龈肿痛"),
        ("运动损伤", "肌肉拉伤"), ("过敏", "皮肤瘙痒"), ("扁桃体炎", "咽喉肿痛"),
        ("胃痛", "反酸"), ("偏头痛", "眼部胀痛"), ("焦虑", "失眠"),
        ("荨麻疹", "皮肤红肿"), ("结膜炎", "眼睛发红"), ("肠胃炎", "腹痛"),
        ("支气管炎", "咳嗽"), ("颈椎病", "颈部疼痛"), ("腱鞘炎", "手腕疼痛")
    ]
    
    # 随机选择一组症状
    selected = random.choice(symptoms)
    description = f"患者出现{selected[0]}"
    if random.random() > 0.3:
        description += f"、{selected[1]}"
    if random.random() > 0.7:
        description += "，需进一步检查"
        
    return description

def generate_young_symptoms():
    """生成年轻患者常见症状"""
    symptoms = [
        ("运动损伤", "关节扭伤", "肌肉拉伤"),
        ("过敏性鼻炎", "打喷嚏", "鼻塞"),
        ("胃肠炎", "腹痛", "腹泻"),
        ("颈椎不适", "颈部酸痛", "头晕"),
        ("视疲劳", "眼睛干涩", "头痛"),
        ("焦虑", "失眠", "注意力不集中"),
        ("咽喉炎", "咽喉疼痛", "吞咽困难"),
        ("皮肤问题", "痤疮", "皮疹"),
        ("月经不适", "腹痛", "乏力"),
        ("牙周炎", "牙龈出血", "牙痛")
    ]
    
    selected = random.choice(symptoms)
    num_symptoms = random.randint(1, len(selected))
    selected_symptoms = random.sample(selected, num_symptoms)
    
    description = f"患者出现{'、'.join(selected_symptoms)}"
    if random.random() > 0.6:
        description += "，需进一步检查"
    
    return description

def generate_birth_year(age_group):
    """根据年龄组生成出生年份"""
    current_year = 2025
    if age_group == "young":  # 0-20岁
        return random.randint(current_year - 20, current_year)
    elif age_group == "young_adult":  # 21-40岁
        return random.randint(current_year - 40, current_year - 21)
    elif age_group == "middle":  # 41-60岁
        return random.randint(current_year - 60, current_year - 41)
    else:  # 61岁以上
        return random.randint(current_year - 85, current_year - 61)

def generate_patient_id():
    """生成患者ID，格式：P-YYYY-XXXXX"""
    year = datetime.datetime.now().year
    random_num = random.randint(10000, 99999)
    return f"P-{year}-{random_num}"

def generate_diagnosis_time():
    """生成最近一周内的随机诊断时间"""
    # 当前时间：2025-04-19
    current_time = datetime.datetime(2025, 4, 19)
    # 一周前的时间
    one_week_ago = current_time - datetime.timedelta(days=7)
    
    # 生成随机时间（精确到分钟）
    random_days = random.randint(0, 6)  # 0-6天
    random_hours = random.randint(0, 23)  # 0-23小时
    random_minutes = random.randint(0, 59)  # 0-59分钟
    
    diagnosis_time = one_week_ago + datetime.timedelta(
        days=random_days,
        hours=random_hours,
        minutes=random_minutes
    )
    
    return diagnosis_time.strftime("%Y-%m-%d %H:%M")

def encode_texts_in_batches(model, texts, batch_size=32):
    """批量编码文本，使用普通的批处理而不是多进程"""
    total = len(texts)
    embeddings = []
    
    with tqdm(total=total, desc="Encoding texts") as pbar:
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
            pbar.update(len(batch))
    
    return np.array(embeddings)

def batch_insert_patients(patient_search, diagnoses, birth_years, batch_size=128):
    """批量插入患者数据"""
    total = len(diagnoses)
    print(f"\nProcessing {total} records in batches of {batch_size}")
    
    # 预生成所有ID和时间
    all_patient_ids = [generate_patient_id() for _ in range(total)]
    all_diagnosis_times = [generate_diagnosis_time() for _ in range(total)]
    
    # 一次性生成所有嵌入
    print("Generating embeddings for all records...")
    start_time = time.time()
    all_embeddings = encode_texts_in_batches(model, diagnoses, batch_size=32)
    end_time = time.time()
    print(f"Total embedding generation time: {end_time - start_time:.2f} seconds")
    
    # 批量插入数据
    with tqdm(total=total, desc="Inserting records") as pbar:
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            
            # 准备批次数据
            batch_data = list(zip(
                all_patient_ids[i:end_idx],
                diagnoses[i:end_idx],
                birth_years[i:end_idx],
                all_diagnosis_times[i:end_idx],
                all_embeddings[i:end_idx]
            ))
            
            # 批量插入
            try:
                patient_search.batch_insert(batch_data)
                patient_search.flush_data()
                pbar.update(end_idx - i)
            except Exception as e:
                print(f"Error inserting batch {i//batch_size + 1}: {e}")
                continue

def main():
    try:
        print("Connecting to Milvus...")
        connections.connect("default", 
                          host=os.getenv("MILVUS_HOST", "localhost"), 
                          port=os.getenv("MILVUS_PORT", "19530"))
        
        print("Initializing patient search system...")
        patient_search = PatientSearch()
        
        # 预先生成所有数据
        print("\nGenerating patient data...")
        total_cases = int(os.getenv("TOTAL_CASES", "10000"))
        batch_size = int(os.getenv("BATCH_SIZE", "128"))
        
        # 从环境变量获取年龄组分布
        age_group_distribution = json.loads(os.getenv("AGE_GROUP_DISTRIBUTION", 
            '{"young": 0.2, "young_adult": 0.3, "middle": 0.3, "elderly": 0.2}'))
        
        # 计算各年龄段的病例数量
        age_groups = {
            "young": int(total_cases * age_group_distribution["young"]),      # 0-20岁
            "young_adult": int(total_cases * age_group_distribution["young_adult"]), # 21-40岁
            "middle": int(total_cases * age_group_distribution["middle"]),      # 41-60岁
            "elderly": total_cases - int(total_cases * (age_group_distribution["young"] + 
                                                      age_group_distribution["young_adult"] + 
                                                      age_group_distribution["middle"]))  # 61岁以上
        }
        
        # 预先生成所有数据
        diagnoses = []
        birth_years = []
        
        print("Generating patient records...")
        with tqdm(total=total_cases, desc="Generating records") as pbar:
            for age_group, count in age_groups.items():
                for _ in range(count):
                    # 根据年龄组生成症状
                    if age_group == "young":
                        diagnoses.append(generate_young_symptoms())
                    elif age_group == "young_adult":
                        diagnoses.append(generate_young_symptoms() if random.random() > 0.7 
                                      else generate_flu_symptoms())
                    elif age_group == "middle":
                        diagnoses.append(generate_flu_symptoms() if random.random() > 0.5 
                                      else generate_other_symptoms())
                    else:  # elderly
                        diagnoses.append(generate_flu_symptoms() if random.random() > 0.3 
                                      else generate_other_symptoms())
                    
                    birth_years.append(generate_birth_year(age_group))
                    pbar.update(1)
        
        # 批量插入数据
        start_time = time.time()
        batch_insert_patients(patient_search, diagnoses, birth_years, batch_size)
        end_time = time.time()
        
        print(f"\nData generation complete!")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Average time per record: {(end_time - start_time)/total_cases:.2f} seconds")
        print("\nAge distribution:")
        for age_group, count in age_groups.items():
            print(f"{age_group}: {count} records")
        
    except Exception as e:
        print(f"Error in main: {e}")
        raise
    finally:
        try:
            connections.disconnect("default")
            print("Disconnected from Milvus")
        except Exception as e:
            print(f"Error disconnecting from Milvus: {e}")

if __name__ == "__main__":
    model = initialize_model()
    main()