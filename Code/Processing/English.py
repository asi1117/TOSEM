import pandas as pd
import glob
import os
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# 确保检测结果是可重复的
DetectorFactory.seed = 0

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

# 输入文件夹路径
input_folder = r''
# 输出文件夹路径
output_folder = r''
non_english_output_folder = r''
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(non_english_output_folder):
    os.makedirs(non_english_output_folder)

# 查找文件夹中所有的CSV文件
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

# 遍历所有的CSV文件并处理数据
for file in csv_files:
    df = pd.read_csv(file)

    # 检查Content列是否存在
    if 'Content' in df.columns:
        # 去除前的行数
        initial_row_count = len(df)

        # 过滤Content列中非英文字符的行
        df_english = df[df['Content'].apply(lambda x: is_english(str(x)))]
        df_non_english = df[~df['Content'].apply(lambda x: is_english(str(x)))]

        # 去除后的行数
        filtered_row_count = len(df_english)

        # 输出文件路径
        english_output_file = os.path.join(output_folder, os.path.basename(file))
        non_english_output_file = os.path.join(non_english_output_folder, os.path.basename(file))

        # 保存过滤后的数据框到新的CSV文件
        df_english.to_csv(english_output_file, index=False, encoding='utf-8-sig')
        df_non_english.to_csv(non_english_output_file, index=False, encoding='utf-8-sig')

        # 打印非英文内容
        print(f"文件: {os.path.basename(file)} 非英文内容:")
        for content in df_non_english['Content']:
            print(content)

        print(f"文件: {os.path.basename(file)}")
        print(f"去除前行数: {initial_row_count}")
        print(f"去除后行数: {filtered_row_count}")
        print(f"已处理并保存到 {english_output_file}")
        print(f"非英文内容已保存到 {non_english_output_file}\n")
    else:
        print(f"文件: {os.path.basename(file)} 不包含 'Content' 列，已跳过")

print("所有文件已处理并保存到新的文件夹。")
