import pandas as pd
import glob
import os
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm
import time

# 输入文件夹路径
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
        processes=1
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
def compute_coverage(doc_topic_matrix, threshold=0.01):
    covered_documents = sum(1 for doc in doc_topic_matrix if any(prob > threshold for prob in doc))
    return covered_documents / len(doc_topic_matrix)

if __name__ == '__main__':
    # 设置设备为 GPU 或 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # 查找文件夹中所有的 CSV 文件
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    # 遍历所有的 CSV 文件并处理数据
    for file in tqdm(csv_files, desc="Processing files"):
        try:
            # 计时开始
            start_time = time.time()
            # 尝试使用 UTF-8 解码，失败后改用 ISO-8859-1 解码
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding='ISO-8859-1')
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

        # 检查 Content 列是否存在
        if 'Content' in df.columns:
            # 删除自定义停用词
            total_rows = len(df)
            df['processed_review'] = ''
            for i, row in df.iterrows():
                df.at[i, 'processed_review'] = remove_custom_stopwords(row['Content'])
                if (i + 1) % 5000 == 0:
                    print(f"Processed {i + 1}/{total_rows} rows in file: {os.path.basename(file)}")

            # 使用 TfidfVectorizer 转换为词频矩阵
            texts = df['processed_review'].dropna().apply(lambda x: str(x).split()).tolist()
            dictionary = corpora.Dictionary(texts)
            corpus = [' '.join(text) for text in texts]
            vectorizer = TfidfVectorizer(max_features=5000)
            term_doc_matrix = vectorizer.fit_transform(corpus)
            term_doc_matrix = normalize(term_doc_matrix, norm='l1', axis=1)

            # 将数据转移到 GPU 或 CPU
            term_doc_tensor = torch.tensor(term_doc_matrix.toarray(), dtype=torch.float32).to(device)

            # 使用 TruncatedSVD 实现 LSA
            topic_count = 50  # 固定话题数量
            lsa_model = TruncatedSVD(n_components=topic_count, random_state=42)
            lsa_model.fit(term_doc_tensor.cpu().numpy())

            # 提取话题关键词
            feature_names = vectorizer.get_feature_names_out()
            topics = [[feature_names[i] for i in topic.argsort()[-10:][::-1]] for topic in lsa_model.components_]

            # 计算一致性、多样性和覆盖率
            coherence = compute_coherence(topics, texts, dictionary)
            doc_topic_matrix = lsa_model.transform(term_doc_tensor.cpu().numpy())
            diversity = compute_diversity(topics)
            coverage = compute_coverage(doc_topic_matrix, threshold=0.01)

            # 计时结束
            end_time = time.time()
            runtime = end_time - start_time

            # 输出结果
            print(f"文件: {os.path.basename(file)}")
            print(f"一致性: {coherence:.4f}")
            print(f"多样性: {diversity:.4f}")
            print(f"覆盖率: {coverage:.4f}")
            print(f"运行时间: {runtime:.2f} 秒")

            # 保存关键词到文件
            keywords = [{"Topic": idx, "Keywords": ", ".join(topic)} for idx, topic in enumerate(topics)]
            output_file = os.path.join(output_folder, os.path.basename(file).replace('.csv', '_LSA_keywords.csv'))
            pd.DataFrame(keywords).to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f"关键词已保存到 {output_file}")
        else:
            print(f"文件: {os.path.basename(file)} 不包含 'Content' 列，已跳过")

    print("所有文件已处理并保存到新的文件夹。")
