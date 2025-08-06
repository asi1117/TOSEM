import pandas as pd

def remove_duplicates_by_review(file_path, output_path):
    """
    Remove duplicate rows based on the 'review' column and save the result to a new file.

    Args:
        file_path (str): The path to the input CSV file.
        output_path (str): The path to save the output CSV file.

    Returns:
        None
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path, encoding='utf-8')

        # Remove duplicate rows based on the 'review' column
        data_deduplicated = data.drop_duplicates(subset=['Content'], keep='first')

        # Save the deduplicated DataFrame to a new CSV file
        data_deduplicated.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"Duplicates removed. Deduplicated file saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = r""  # Replace with the path to your file
output_path = r""  # Replace with the output file path
remove_duplicates_by_review(file_path, output_path)
