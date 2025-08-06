import pandas as pd
import glob
import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm

# 输入文件夹路径
input_folder = r''
# 输出文件夹路径
output_folder = r''
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 初始化停用词集合和词干提取器
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
additional_stopwords = []  # Add custom stopwords here
stop_words.update(additional_stopwords)

# 定义清理函数
def clean_and_stem_text(text):
    if not isinstance(text, str):
        return text
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize, remove stopwords, and apply stemming
    text = ' '.join(stemmer.stem(word) for word in text.split() if word.lower() not in stop_words)
    return text

# 查找文件夹中所有的CSV文件
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

# 遍历所有的CSV文件并处理数据
for file in tqdm(csv_files, desc="Processing files"):
    df = pd.read_csv(file)

    # 检查review列是否存在
    if 'Content' in df.columns:
        # 清理并提取词干
        df['Content'] = df['Content'].apply(clean_and_stem_text)

        # 输出文件路径
        output_file = os.path.join(output_folder, os.path.basename(file))

        # 保存处理后的数据框到新的CSV文件
        df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"文件: {os.path.basename(file)} 已处理并保存到 {output_file}")
    else:
        print(f"文件: {os.path.basename(file)} 不包含 'review' 列，已跳过")

print("所有文件已处理并保存到新的文件夹。")
