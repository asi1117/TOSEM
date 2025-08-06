import os
import pandas as pd
import openai

openai.api_key = ""

def gpt_label_topics(input_file, output_file):
    """
    For each topic in the input CSV, calls GPT to generate a concise label.
    Writes the results to the output CSV.
    """
    data = pd.read_csv(input_file)
    topic_keywords = data[['Dominant_Topic', 'Top_Keywords']].values.tolist()

    results = []
    for topic, keywords in topic_keywords:
        # Compose the prompt with the current topic's keywords
        prompt = (
            f"This analysis categorizes VR developer discussions into 50 topics. "
            f"One topic is defined by the following keywords: {keywords}. "
            f"Generate a concise, semantically meaningful label that best represents this topic. "
            f"Use the format:\nTopic: <topic label>."
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            content = response['choices'][0]['message']['content'].strip()
            gpt_label = "Unknown"
            definition = ""

            # Extract Topic label and optional definition from the response
            if "Topic:" in content and "Definition:" in content:
                parts = content.split("Definition:")
                gpt_label = parts[0].replace("Topic:", "").strip()
                definition = parts[1].strip()
            elif "Topic:" in content:
                gpt_label = content.replace("Topic:", "").strip()
            else:
                gpt_label = content.strip()

            results.append({
                "Topic": topic,
                "Keywords": keywords,
                "GPT_Label": gpt_label,
                "Definition": definition
            })
        except Exception as e:
            print(f"Error with GPT API: {e}")
            results.append({
                "Topic": topic,
                "Keywords": keywords,
                "GPT_Label": "Unknown",
                "Definition": "Unknown"
            })

    pd.DataFrame(results).to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"GPT labels saved to {output_file}")

def process_all_csv_files(input_folder, output_folder):
    """
    Processes all CSV files in the input folder, applies GPT labeling,
    and writes results to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, f"processed2_{file_name}")
            print(f"Processing file: {file_name}")
            gpt_label_topics(input_file, output_file)

input_folder = r''
output_folder = r''

process_all_csv_files(input_folder, output_folder)
