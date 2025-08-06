import pandas as pd

def convert_reviews_to_lowercase(file_path, output_path):
    """
    Convert all content in the 'review' column to lowercase and save to a new CSV file.

    Args:
        file_path (str): The path to the input CSV file.
        output_path (str): The path to save the output CSV file.

    Returns:
        None
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path, encoding='utf-8')

        # Convert the 'review' column to lowercase
        if 'review' in data.columns:
            data['review'] = data['review'].str.lower()

        # Save the updated DataFrame to a new CSV file
        data.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"All reviews converted to lowercase. Updated file saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = r""  # Replace with the path to your file
output_path = r""  # Replace with the output file path
convert_reviews_to_lowercase(file_path, output_path)
