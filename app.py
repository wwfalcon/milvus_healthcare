from flask import Flask, render_template, request, jsonify
from patient_search import PatientSearch
import time
from pymilvus import Collection, connections, utility
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

def verify_milvus_connection():
    """验证 Milvus 连接和集合状态"""
    try:
        try:
            connections.connect(alias="default", host='localhost', port='19530')
            logger.info("Successfully connected to Milvus")
        except Exception as e:
            logger.warning(f"Connection already exists: {e}")
            
        try:
            collection = Collection("patient_records")
            collection.load()
            
            # 检查集合中是否有数据
            count = collection.num_entities
            logger.info(f"Collection 'patient_records' has {count} records")
            
            if count == 0:
                logger.warning("Collection 'patient_records' is empty")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error checking collection: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying Milvus connection: {e}")
        return False

def get_all_records():
    """获取所有患者记录，按诊断时间降序排列"""
    try:
        collection = Collection("patient_records")
        collection.load()
        
        # 执行查询，按诊断时间降序排序
        results = collection.query(
            expr="",  # 空表达式表示查询所有记录
            output_fields=["patient_id", "birth_year", "diagnosis", "diagnosis_time"],
            order_by="diagnosis_time",
            limit=100,  # 限制返回的记录数
            offset=0
        )
        
        # 处理结果
        current_year = 2025  # 根据测试数据设置
        processed_results = []
        for result in results:
            age = current_year - result['birth_year']
            processed_results.append({
                'patient_id': result['patient_id'],
                'age': age,
                'diagnosis': result['diagnosis'],
                'diagnosis_time': result['diagnosis_time']
            })
        
        collection.release()
        return processed_results
        
    except Exception as e:
        logger.error(f"Error retrieving all records: {e}")
        return []

# 初始化患者搜索系统
print("Initializing patient search system...")
try:
    if not verify_milvus_connection():
        logger.error("Failed to verify Milvus connection or empty collection")
    patient_search = PatientSearch()
    logger.info("Patient search system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize patient search system: {e}")
    raise

def get_patient_statistics():
    """获取患者统计信息"""
    try:
        # 验证连接状态
        try:
            connections.connect(alias="default", host='localhost', port='19530')
        except Exception as e:
            logger.warning(f"Connection already exists: {e}")
            
        # 加载集合
        collection = Collection("patient_records")
        collection.load()
        
        # 获取所有患者数据
        expr = "birth_year > 0"  # 确保只获取有效数据
        results = collection.query(
            expr=expr,
            output_fields=["birth_year", "diagnosis_time"]
        )
        
        logger.info(f"Retrieved {len(results)} patient records")
        
        if not results:
            logger.warning("No patient records found in collection")
            raise Exception("No patient records found")
        
        # 计算当前年份
        current_year = 2025  # 根据测试数据设置
        
        # 计算年龄分布
        age_distribution = {
            "0-20": 0,
            "21-40": 0,
            "41-60": 0,
            "61-80": 0,
            "81+": 0
        }
        
        # 记录处理的记录数
        processed_count = 0
        error_count = 0
        
        for patient in results:
            try:
                birth_year = patient.get("birth_year")
                if birth_year and isinstance(birth_year, (int, float)):
                    age = current_year - birth_year
                    processed_count += 1
                    
                    if age <= 20:
                        age_distribution["0-20"] += 1
                    elif age <= 40:
                        age_distribution["21-40"] += 1
                    elif age <= 60:
                        age_distribution["41-60"] += 1
                    elif age <= 80:
                        age_distribution["61-80"] += 1
                    else:
                        age_distribution["81+"] += 1
                else:
                    error_count += 1
                    logger.warning(f"Invalid birth_year value: {birth_year}")
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing patient record: {e}")
                continue
        
        logger.info(f"Processed {processed_count} records successfully")
        logger.info(f"Encountered {error_count} errors while processing")
        logger.info(f"Age distribution calculated: {age_distribution}")
        
        # 计算诊断时间分布（按天）
        diagnosis_dates = {}
        for patient in results:
            try:
                diagnosis_time = patient.get("diagnosis_time")
                if diagnosis_time:
                    date = diagnosis_time.split()[0]  # 只取日期部分
                    diagnosis_dates[date] = diagnosis_dates.get(date, 0) + 1
            except Exception as e:
                logger.error(f"Error processing diagnosis date: {e}")
                continue
        
        # 按日期排序
        sorted_dates = sorted(diagnosis_dates.items())
        
        stats = {
            "total_patients": len(results),
            "age_distribution": age_distribution,
            "diagnosis_dates": dict(sorted_dates)
        }
        
        logger.info(f"Final statistics: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        # 返回默认值
        default_stats = {
            "total_patients": 0,
            "age_distribution": {
                "0-20": 0,
                "21-40": 0,
                "41-60": 0,
                "61-80": 0,
                "81+": 0
            },
            "diagnosis_dates": {}
        }
        logger.info("Returning default statistics due to error")
        return default_stats

@app.route('/')
def index():
    # 获取统计信息
    stats = get_patient_statistics()
    
    # 获取所有记录（按诊断时间降序排序）
    initial_records = get_all_records()
    
    # 渲染模板，传递统计信息和初始记录
    return render_template('index.html', 
                         stats=stats,
                         initial_records={
                             'total_results': len(initial_records),
                             'results': initial_records
                         })

def get_default_statistics():
    """返回默认的统计信息"""
    return {
        "total_patients": 0,
        "age_distribution": {
            "0-20": 0,
            "21-40": 0,
            "41-60": 0,
            "61-80": 0,
            "81+": 0
        },
        "diagnosis_dates": {}
    }

@app.route('/search', methods=['POST'])
def search():
    try:
        # 验证 Milvus 连接
        if not verify_milvus_connection():
            raise Exception("Milvus connection verification failed")
            
        # 获取查询参数
        query = request.form.get('query', '')
        min_age = int(request.form.get('min_age', 60))
        max_distance = float(request.form.get('max_distance', 0.8))
        
        if not query:
            # 如果没有查询文本，返回所有记录
            all_records = get_all_records()
            return jsonify({
                'query': '',
                'min_age': min_age,
                'max_distance': max_distance,
                'search_time': "0.00",
                'total_results': len(all_records),
                'results': all_records
            })
        
        logger.info(f"Processing search request - query: {query}, min_age: {min_age}, max_distance: {max_distance}")
        
        # 执行搜索
        start_time = time.time()
        results = patient_search.search_similar_patients(
            query_symptoms=query,
            min_age=min_age,
            max_distance=max_distance
        )
        search_time = time.time() - start_time
        
        logger.info(f"Search completed in {search_time:.2f}s, found {len(results)} results")
        
        # 准备返回结果
        response = {
            'query': query,
            'min_age': min_age,
            'max_distance': max_distance,
            'search_time': f"{search_time:.2f}",
            'total_results': len(results),
            'results': results
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in search route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 