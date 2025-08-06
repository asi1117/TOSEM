import pandas as pd
import glob
import os
import time
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
print(torch.cuda.is_available())
from umap import UMAP
from torch.utils.data import DataLoader


# 输入文件夹路径
input_folder = r''
# 输出文件夹路径
output_folder = r''
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 停用词删除函数
def remove_custom_stopwords(text):
    if not isinstance(text, str):
        return text
    words = text.split()
    words = [word for word in words if word.lower() not in custom_stop_words]
    return ' '.join(words)

# 一致性计算函数
def compute_coherence(topics, texts, dictionary):
    coherence_model = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v',
        processes=1  # 强制单进程模式
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
            return pd.read_csv(file_path, encoding=encoding, low_memory=False)  # 添加 low_memory=False
        except UnicodeDecodeError:
            print(f"Failed to read {file_path} with encoding {encoding}")
        except pd.errors.ParserError as e:
            print(f"Parser error reading {file_path}: {e}")
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
        if 'review' in df.columns:
            df['processed_review'] = df['review'].apply(remove_custom_stopwords)
            texts = df['processed_review'].dropna().tolist()

            # 开始计时
            start_time = time.time()

            # 生成句向量
            embeddings = model.encode(texts, show_progress_bar=True)
            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.1, metric='cosine')
            topic_model = BERTopic(umap_model=umap_model, nr_topics=50)
            # 使用 BERTopic 模型提取主题
            # topic_model = BERTopic(nr_topics=50)
            topics, probs = topic_model.fit_transform(texts, embeddings)

            # 提取每个主题的关键词
            topic_info = topic_model.get_topic_info()
            topics_words = {
                topic: [word for word, _ in topic_model.get_topic(topic)] for topic in topic_info['Topic'] if topic >= 0
            }

            # 准备数据计算一致性、多样性和覆盖率
            dictionary = corpora.Dictionary([text.split() for text in texts])
            topic_list = list(topics_words.values())
            coherence = compute_coherence(topic_list, [text.split() for text in texts], dictionary)
            diversity = compute_diversity(topic_list)
            coverage = compute_coverage(topics)

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
            output_data = [{"Topic": topic, "Keywords": ", ".join(words)} for topic, words in topics_words.items()]
            output_file = os.path.join(output_folder, os.path.basename(file).replace('.csv', '_BERTopic_keywords.csv'))
            pd.DataFrame(output_data).to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f"关键词已保存到 {output_file}")
        else:
            print(f"文件: {os.path.basename(file)} 不包含 'Content' 列，已跳过")

    print("所有文件已处理并保存到新的文件夹。")
