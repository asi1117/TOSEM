import pandas as pd
import glob
import os
import time
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from top2vec import Top2Vec
import numpy as np

input_folder = r''
# 输出文件夹路径
output_folder = r''
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 一致性计算函数
def compute_coherence(topics, texts, dictionary):
    coherence_model = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v',
        processes=1  # 强制单过程模式
    )
    return coherence_model.get_coherence()

# 多样性计算函数
def compute_diversity(topics, num_keywords=10):
    unique_words = set()
    total_words = 0
    for topic in topics:
        topic_words = topic[:num_keywords]
        unique_words.update(topic_words)
        total_words += len(topic_words)
    return len(unique_words) / total_words

# 覆盖率计算函数
def compute_coverage(assignments):
    covered_documents = sum(1 for topic in assignments if topic >= 0)
    return covered_documents / len(assignments)

# 尝试读取 CSV 文件
def read_csv_with_encoding(file_path, encodings=['utf-8', 'ISO-8859-1', 'latin1']):
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            print(f"Failed to read {file_path} with encoding {encoding}")
    raise ValueError(f"Unable to read {file_path} with provided encodings")

if __name__ == '__main__':
    # 查找文件夹中所有的 CSV 文件
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    # 加载预训练 SentenceTransformer 模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # 遍历所有的 CSV 文件并处理数据
    for file in tqdm(csv_files, desc="Processing files"):
        try:
            # 使用改进的读取函数
            df = read_csv_with_encoding(file)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

        # 检查 Content 列是否存在
        if 'Content' in df.columns:
            df['processed_review'] = df['Content'].apply(remove_custom_stopwords)
            texts = df['processed_review'].dropna().tolist()

            # 开始计时
            start_time = time.time()

            top2vec_model = Top2Vec(
                documents=texts,
                embedding_model='distiluse-base-multilingual-cased',
                # min_cluster_size=2,  # 调低最小聚类大小
                speed='deep-learn'
            )

            # 提取主题的前10个关键词
            topic_list = []
            topics_words = top2vec_model.topic_words  # 获取关键词矩阵
            if topics_words is not None and len(topics_words) > 0:
                for topic_num in range(min(50, len(topics_words))):
                    topic_list.append(list(topics_words[topic_num][:10]))  # 获取每个主题的前10个关键词

            # 调试输出以确认格式
            print("Formatted Topics for Coherence Calculation (Top 10 keywords):", topic_list[:3])
            print(f"Number of topics: {len(topic_list)}")  # 输出话题数量

            # 准备数据计算一致性、多样性和覆盖率
            dictionary = corpora.Dictionary([text.split() for text in texts])
            coherence = compute_coherence(topic_list, [text.split() for text in texts], dictionary) if topic_list else 0
            diversity = compute_diversity(topic_list) if topic_list else 0

            try:
                doc_ids = list(range(len(texts)))  # 使用连续的行数作为文档 ID
                result = top2vec_model.get_documents_topics(doc_ids=doc_ids)
                if len(result) >= 2:
                    assignments = result[0]
                else:
                    raise ValueError("Unexpected result format from get_documents_topics")
                print(f"Assignments sample: {assignments[:5000]}")
            except Exception as e:
                print(f"Error assigning document topics: {e}")
                assignments = []  # 如果分配失败，使用空列表
            #使用行数生成唯一字符串 ID


            # 计算覆盖率
            coverage = compute_coverage(assignments)
            print(f"Coverage: {coverage:.4f}")

            # 结束计时
            end_time = time.time()
            runtime = end_time - start_time

            # 输出结果
            print(f"文件: {os.path.basename(file)}")
            print(f"一致性: {coherence:.4f}")
            print(f"多样性: {diversity:.4f}")
            print(f"覆盖率: {coverage:.4f}")
            print(f"运行时间: {runtime:.2f} 秒")

            # 保存关键词到文件
            output_data = [{"Topic": topic_num, "Keywords": ", ".join(words)} for topic_num, words in enumerate(topic_list)]
            output_file = os.path.join(output_folder, os.path.basename(file).replace('.csv', '_Top2Vec_keywords.csv'))
            pd.DataFrame(output_data).to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f"关键词已保存到 {output_file}")
        else:
            print(f"文件: {os.path.basename(file)} 不包含 'Content' 列，已跳过")


    print("所有文件已处理并保存到新的文件夹。")
