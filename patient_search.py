from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import sys
from urllib3.exceptions import MaxRetryError
from requests.exceptions import ConnectionError, HTTPError
import os
import torch
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def initialize_model(max_retries=3, retry_delay=5):
    """Initialize the text embedding model with retry mechanism"""
    print("Loading Chinese medical text embedding model...")
    
    # 使用医疗领域专用的中文模型
    model_name = 'shibing624/text2vec-base-chinese'
    
    # First check if model is already downloaded
    cache_dir = os.path.expanduser('~/.cache/torch/sentence_transformers')
    cache_model_name = model_name.split('/')[-1]
    model_dir = os.path.join(cache_dir, cache_model_name)
    
    if os.path.exists(model_dir):
        print(f"Found cached Chinese medical model: {cache_model_name}")
        try:
            model = SentenceTransformer(model_dir)
            # Test the model with medical text
            _ = model.encode(["患者出现发热、咳嗽等症状"])
            print(f"Successfully loaded cached Chinese medical model")
            return model
        except Exception as e:
            print(f"Error loading cached Chinese medical model: {e}")
    
    # If not in cache, try downloading
    print(f"\nAttempting to download Chinese medical model: {model_name}")
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            print(f"Download attempt {attempt + 1}/{max_retries}")
            model = SentenceTransformer(model_name)
            # Test the model with medical text
            _ = model.encode(["患者出现发热、咳嗽等症状"])
            print(f"Successfully downloaded and loaded Chinese medical model")
            print(f"Model loaded in {time.time() - start_time:.2f} seconds")
            return model
        except (ConnectionError, MaxRetryError, HTTPError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                print(f"Download failed. Retrying in {wait_time} seconds...")
                print(f"Error details: {str(e)}")
                time.sleep(wait_time)
            else:
                print(f"Failed to download Chinese medical model after maximum retries.")
                print("\nPlease try one of the following solutions:")
                print("1. Check your internet connection")
                print("2. Try running the program again later")
                print("3. Manually download the model:")
                print("   a. Visit https://huggingface.co/shibing624/text2vec-base-chinese")
                print("   b. Download the model files")
                print("   c. Place them in the cache directory:")
                print(f"      {cache_dir}/text2vec-base-chinese")
                sys.exit(1)
        except Exception as e:
            print(f"Unexpected error while loading Chinese medical model: {e}")
            sys.exit(1)
    
    print("\nFailed to load Chinese medical model. Please try the solutions above.")
    sys.exit(1)

def connect_milvus(max_retries=3, retry_delay=5):
    """Connect to Milvus with retry mechanism"""
    print("Connecting to Milvus server...")
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port=os.getenv("MILVUS_PORT", "19530"))
            print(f"Connected to Milvus in {time.time() - start_time:.2f} seconds")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                print(f"Connection failed. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print("Failed to connect to Milvus after maximum retries.")
                print("Error details:", str(e))
                print("\nPlease check if:")
                print("1. Milvus Docker container is running (docker ps)")
                print("2. Docker container ports are correctly mapped")
                print("3. No firewall is blocking the connection")
                sys.exit(1)

print("Starting program...")

# Initialize connections with retry mechanism
connect_milvus()

# Initialize the Chinese text embedding model with retry mechanism
model = initialize_model()
print("Chinese medical model initialization complete.")

class PatientSearch:
    def __init__(self):
        print("Initializing PatientSearch...")
        self.collection_name = "patient_records"
        # 使用全局中文模型实例
        self.model = model
        self._setup_collection()
        
    def _setup_collection(self):
        """设置集合和索引"""
        try:
            # 检查集合是否存在
            if not utility.has_collection(self.collection_name):
                print("Creating new collection...")
                # 定义集合结构
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="patient_id", dtype=DataType.VARCHAR, max_length=20),
                    FieldSchema(name="diagnosis", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="birth_year", dtype=DataType.INT32),
                    FieldSchema(name="diagnosis_time", dtype=DataType.VARCHAR, max_length=20),
                    FieldSchema(name="diagnosis_embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
                ]
                schema = CollectionSchema(fields=fields, description="Patient medical records")
                
                # 创建集合
                self.collection = Collection(name=self.collection_name, schema=schema)
                
                # 创建索引
                print("Creating index for vector field...")
                index_params = {
                    "metric_type": "L2",
                    "index_type": "FLAT",
                    "params": {}
                }
                self.collection.create_index(field_name="diagnosis_embedding", index_params=index_params)
            else:
                print("Loading existing collection...")
                self.collection = Collection(self.collection_name)
            
            # 加载集合到内存
            print("Loading collection into memory...")
            self.collection.load()
            
        except Exception as e:
            print(f"Error setting up collection: {e}")
            sys.exit(1)
        
    def batch_insert(self, batch_data):
        """批量插入患者数据
        
        Args:
            batch_data: 包含患者数据的列表，每个元素为 (patient_id, diagnosis, birth_year, diagnosis_time, embedding)
        """
        try:
            # 使用 numpy 数组预分配内存
            batch_size = len(batch_data)
            
            # 预分配内存
            patient_ids = np.empty(batch_size, dtype=object)
            diagnoses = np.empty(batch_size, dtype=object)
            birth_years = np.empty(batch_size, dtype=np.int32)
            diagnosis_times = np.empty(batch_size, dtype=object)
            embeddings = np.empty((batch_size, 768), dtype=np.float32)
            
            # 批量处理数据
            for i, data in enumerate(batch_data):
                patient_id, diagnosis, birth_year, diagnosis_time, embedding = data
                patient_ids[i] = patient_id
                diagnoses[i] = diagnosis
                birth_years[i] = birth_year
                diagnosis_times[i] = diagnosis_time
                embeddings[i] = embedding  # 已经是 numpy 数组，直接赋值
            
            # 准备实体数据
            entities = [
                patient_ids.tolist(),  # Milvus 需要 Python 列表
                diagnoses.tolist(),
                birth_years.tolist(),
                diagnosis_times.tolist(),
                embeddings.tolist()
            ]
            
            # 批量插入数据
            self.collection.insert(entities)
            return True
        except Exception as e:
            print(f"Error in batch insert: {e}")
            return False
        
    def insert_patient(self, patient_id: str, diagnosis: str, birth_year: int, diagnosis_time: str, embedding=None):
        """插入患者数据，支持预先生成的向量嵌入"""
        try:
            # 如果提供了预先生成的向量嵌入，直接使用
            if embedding is None:
                # 生成诊断文本的向量嵌入
                embedding = self.model.encode([diagnosis])[0]
            
            # 准备插入数据
            entities = [
                [patient_id],
                [diagnosis],
                [birth_year],
                [diagnosis_time],
                [embedding.tolist()]
            ]
            
            # 插入数据，但不立即 flush
            self.collection.insert(entities)
            return True
        except Exception as e:
            print(f"Error inserting patient data: {e}")
            return False
        
    def flush_data(self):
        """手动触发数据刷新"""
        try:
            self.collection.flush()
            return True
        except Exception as e:
            print(f"Error flushing data: {e}")
            return False
        
    def search_similar_patients(self, query_symptoms: str, min_age: int = 60, max_distance: float = 0.8):
        """搜索相似患者
        
        Args:
            query_symptoms: 查询症状
            min_age: 最小年龄
            max_distance: 最大距离阈值（越大越宽松）
        """
        try:
            # 生成查询向量
            print("Generating query embedding using Chinese medical model...")
            print(f"Query text: {query_symptoms}")
            query_embedding = self.model.encode([query_symptoms])[0]
            print("Query embedding generated successfully")
            
            # 计算最小出生年份
            current_year = datetime.datetime.now().year
            max_birth_year = current_year - min_age
            
            # 设置搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 100}  # 增加搜索范围
            }
            
            # 构建过滤条件
            expr = f"birth_year <= {max_birth_year}"
            
            # 执行搜索
            print("Executing search...")
            print(f"Search parameters: {search_params}")
            print(f"Filter expression: {expr}")
            print(f"Similarity threshold: {max_distance}")
            
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="diagnosis_embedding",
                param=search_params,
                limit=200,  # 增加返回数量
                expr=expr,
                output_fields=["patient_id", "diagnosis", "birth_year", "diagnosis_time"]
            )
            
            # 处理搜索结果
            hits = []
            if results and isinstance(results, list) and len(results) > 0:
                # 找到最小和最大距离用于归一化
                distances = [hit.distance for hit in results[0] if hasattr(hit, 'distance')]
                if distances:
                    min_dist = min(distances)
                    max_dist = max(distances)
                    dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
                    
                    for hit in results[0]:
                        if hasattr(hit, 'entity') and hasattr(hit, 'distance'):
                            age = current_year - hit.entity.get('birth_year')
                            # 计算归一化的相似度分数 (1 - (distance - min_dist) / dist_range)
                            similarity = 1 - (hit.distance - min_dist) / dist_range
                            
                            # 只添加满足相似度阈值的结果
                            if similarity >= max_distance:
                                hits.append({
                                    'patient_id': hit.entity.get('patient_id'),
                                    'age': age,
                                    'diagnosis': hit.entity.get('diagnosis'),
                                    'diagnosis_time': hit.entity.get('diagnosis_time'),
                                    'similarity': similarity,
                                    'distance': hit.distance
                                })
            
            # 按相似度降序排序
            hits.sort(key=lambda x: x['similarity'], reverse=True)
            print(f"Found {len(hits)} matching patients after filtering")
            return hits
            
        except Exception as e:
            print(f"Error searching similar patients: {e}")
            return []

def main():
    total_start = time.time()
    print("\nInitializing patient search system...")
    
    try:
        # 检查命令行参数
        if len(sys.argv) < 2:
            print("\nUsage: python patient_search.py <query_text> [min_age] [max_distance]")
            print("Example: python patient_search.py \"发热咳嗽\" 60 0.8")
            return
        
        # 获取查询文本
        query = sys.argv[1]
        
        # 获取可选参数
        min_age = 60  # 默认最小年龄
        max_distance = 0.8  # 默认最大距离
        
        if len(sys.argv) > 2:
            try:
                min_age = int(sys.argv[2])
            except ValueError:
                print(f"Invalid min_age value: {sys.argv[2]}, using default value 60")
        
        if len(sys.argv) > 3:
            try:
                max_distance = float(sys.argv[3])
            except ValueError:
                print(f"Invalid max_distance value: {sys.argv[3]}, using default value 0.8")
        
        # 检查集合是否存在
        if not utility.has_collection("patient_records"):
            print("No existing collection found. Please run generate_data.py first.")
            return
            
        patient_search = PatientSearch()
        
        # 验证数据
        print("\nVerifying data...")
        collection = Collection("patient_records")
        collection.load()
        num_entities = collection.num_entities
        print(f"Found {num_entities} records in the collection")
        
        if num_entities == 0:
            print("No data found in collection. Please run generate_data.py first.")
            return
        
        print("\nStarting patient search...")
        print(f"Query text: {query}")
        print(f"Minimum age: {min_age}")
        print(f"Maximum distance: {max_distance}")
        
        results = patient_search.search_similar_patients(query, min_age=min_age, max_distance=max_distance)
        
        if results:
            print(f"\nFound {len(results)} matching patients:")
            for hit in results:
                print(f"\nPatient ID: {hit['patient_id']}")
                print(f"Age: {hit['age']} years old")
                print(f"Diagnosis: {hit['diagnosis']}")
                print(f"Diagnosis Time: {hit['diagnosis_time']}")
                print(f"Similarity Score: {hit['similarity']:.4f}")
                print(f"Distance: {hit['distance']:.4f}")
        else:
            print("\nNo matching results found.")
        
    except Exception as e:
        print(f"Error in main program: {e}")
    finally:
        print(f"\nTotal program execution time: {time.time() - total_start:.2f} seconds")

if __name__ == "__main__":
    main() 