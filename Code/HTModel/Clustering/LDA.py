import pandas as pd
import glob
import os
import torch
from gensim import corpora
from torch import nn
from tqdm import tqdm
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from gensim.models.coherencemodel import CoherenceModel

# 输入文件夹路径
input_folder = r''
# 输出文件夹路径
output_folder = r''
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# LDA 模型类 (基于 PyTorch)
class LDAModel(nn.Module):
    def __init__(self, vocab_size, num_topics):
        super(LDAModel, self).__init__()
        self.topic_word_dist = nn.Parameter(torch.rand(num_topics, vocab_size))
        self.doc_topic_dist = None

    def forward(self, doc_term_matrix):
        topic_word_dist = torch.softmax(self.topic_word_dist, dim=1)
        doc_topic_dist = torch.softmax(doc_term_matrix @ topic_word_dist.T, dim=1)
        self.doc_topic_dist = doc_topic_dist
        return doc_topic_dist @ topic_word_dist

# 一致性计算函数
def compute_coherence(lda_model, texts, vectorizer):
    topics = [[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]] for topic in lda_model.topic_word_dist.cpu().detach().numpy()]
    dictionary = corpora.Dictionary(texts)
    coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v', processes=1)
    return coherence_model.get_coherence()

# 多样性计算函数
def compute_diversity(lda_model, num_keywords=10):
    topics = lda_model.topic_word_dist.cpu().detach().numpy()
    unique_words = set()
    total_words = 0
    for topic in topics:
        topic_words = topic.argsort()[-num_keywords:]
        unique_words.update(topic_words)
        total_words += len(topic_words)
    return len(unique_words) / total_words

# 覆盖率计算函数
def compute_coverage(lda_model, doc_term_matrix, threshold=0.01):
    doc_topic_matrix = lda_model.doc_topic_dist.cpu().detach().numpy()
    covered_documents = sum(1 for doc in doc_topic_matrix if any(prob > threshold for prob in doc))
    return covered_documents / len(doc_term_matrix)


if __name__ == "__main__":
    # 查找文件夹中所有的 CSV 文件
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    # 遍历所有的 CSV 文件并处理数据
    for file in tqdm(csv_files, desc="Processing files"):
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='ISO-8859-1')

        # 检查 Content 列是否存在
        if 'Content' in df.columns:
            # 删除自定义停用词
            df['processed_review'] = df['Content'].apply(remove_custom_stopwords)

            # 使用 CountVectorizer 转换为词频矩阵
            corpus = df['processed_review'].dropna().tolist()
            vectorizer = CountVectorizer(max_features=5000)
            term_doc_matrix = vectorizer.fit_transform(corpus).toarray()
            term_doc_matrix = normalize(term_doc_matrix, norm='l1', axis=1)

            # 转换为 PyTorch 张量并检测 GPU 是否可用
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            term_doc_tensor = torch.tensor(term_doc_matrix, dtype=torch.float32).to(device)

            # 设置 LDA 模型的参数
            vocab_size = term_doc_tensor.shape[1]
            topic_count = 50

            # 定义 LDA 模型
            lda_model = LDAModel(vocab_size=vocab_size, num_topics=topic_count).to(device)
            optimizer = torch.optim.Adam(lda_model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()

            # 训练 LDA 模型
            start_time = time.time()
            for epoch in range(200):
                optimizer.zero_grad()
                reconstructed = lda_model(term_doc_tensor)
                loss = loss_fn(reconstructed, term_doc_tensor)
                loss.backward()
                optimizer.step()
            end_time = time.time()
            runtime = end_time - start_time

            # 提取关键词
            topic_word_dist = torch.softmax(lda_model.topic_word_dist, dim=1).cpu().detach().numpy()
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic_dist in enumerate(topic_word_dist):
                top_keywords = [feature_names[i] for i in topic_dist.argsort()[-10:][::-1]]
                topics.append({"Topic": topic_idx, "Keywords": ", ".join(top_keywords)})

            # 保存关键词到文件
            output_file = os.path.join(output_folder, os.path.basename(file).replace('.csv', '_LDA_keywords.csv'))
            pd.DataFrame(topics).to_csv(output_file, index=False, encoding='utf-8-sig')

            # 计算一致性、多样性和覆盖率
            texts = [text.split() for text in corpus]
            coherence = compute_coherence(lda_model, texts, vectorizer)
            diversity = compute_diversity(lda_model, num_keywords=10)
            coverage = compute_coverage(lda_model, term_doc_matrix,threshold=0.01)

            # 输出结果
            print(f"文件: {os.path.basename(file)}")
            print(f"一致性: {coherence:.4f}")
            print(f"多样性: {diversity:.4f}")
            print(f"覆盖率: {coverage:.4f}")
            print(f"运行时间: {runtime:.2f} 秒")
            print(f"关键词已保存到 {output_file}\n")
        else:
            print(f"文件: {os.path.basename(file)} 不包含 'Content' 列，已跳过")

    print("所有文件已处理并保存到新的文件夹。")
