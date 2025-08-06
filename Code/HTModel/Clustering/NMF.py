import pandas as pd
import glob
import os
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm
import time

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# pip install pandas tqdm scikit-learn gensim
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
def compute_coverage(doc_topic_matrix, threshold=0.01):
    covered_documents = sum(1 for doc in doc_topic_matrix if any(prob > threshold for prob in doc))
    return covered_documents / len(doc_topic_matrix)


if __name__ == '__main__':
    # 查找文件夹中所有的 CSV 文件
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    # 遍历所有的 CSV 文件并处理数据
    for file in tqdm(csv_files, desc="Processing files"):
        try:
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

            # 使用 CountVectorizer 转换为词频矩阵
            texts = df['processed_review'].dropna().apply(lambda x: str(x).split()).tolist()
            dictionary = corpora.Dictionary(texts)
            corpus = [' '.join(text) for text in texts]
            vectorizer = CountVectorizer(max_features=5000)
            term_doc_matrix = vectorizer.fit_transform(corpus).toarray()
            term_doc_matrix = normalize(term_doc_matrix, norm='l1', axis=1)

            # 将数据转换为 PyTorch 张量并传输到 GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            term_doc_tensor = torch.tensor(term_doc_matrix, dtype=torch.float32).to(device)

            # 设置 NMF 模型的参数
            topic_count = 50  # 固定话题数量
            W = torch.rand(term_doc_tensor.shape[0], topic_count, device=device, requires_grad=True)
            H = torch.rand(topic_count, term_doc_tensor.shape[1], device=device, requires_grad=True)

            # 使用 PyTorch 实现 NMF
            optimizer = torch.optim.Adam([W, H], lr=0.01)
            num_epochs = 200

            start_time = time.time()
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                reconstruction = torch.mm(W, H)
                loss = torch.nn.functional.mse_loss(reconstruction, term_doc_tensor)
                loss.backward()
                optimizer.step()

            end_time = time.time()
            runtime = end_time - start_time

            # 从 H 中提取话题关键词
            H_cpu = H.cpu().detach().numpy()
            feature_names = vectorizer.get_feature_names_out()
            topics = [[feature_names[i] for i in topic.argsort()[-10:][::-1]] for topic in H_cpu]

            # 计算一致性、多样性和覆盖率
            coherence = compute_coherence(topics, texts, dictionary)
            diversity = compute_diversity(topics)
            coverage = compute_coverage(W.cpu().detach().numpy(), threshold=0.01)

            # 输出结果
            print(f"文件: {os.path.basename(file)}")
            print(f"一致性: {coherence:.4f}")
            print(f"多样性: {diversity:.4f}")
            print(f"覆盖率: {coverage:.4f}")
            print(f"运行时间: {runtime:.2f} 秒")

            # 保存关键词到文件
            keywords = [{"Topic": idx, "Keywords": ", ".join(topic)} for idx, topic in enumerate(topics)]
            output_file = os.path.join(output_folder, os.path.basename(file).replace('.csv', '_NMF_keywords.csv'))
            pd.DataFrame(keywords).to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f"关键词已保存到 {output_file}")
        else:
            print(f"文件: {os.path.basename(file)} 不包含 'Content' 列，已跳过")

    print("所有文件已处理并保存到新的文件夹。")
