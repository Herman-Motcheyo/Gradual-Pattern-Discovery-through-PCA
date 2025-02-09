import pandas as pd


def load_dataset(file_path, file_format='csv'):
    """
    Loads a dataset from a file.

    Parameters:
    file_path (str): The path to the file.
    file_format (str): The format of the file ('csv' or 'excel').

    Returns:
    DataFrame: The DataFrame containing the loaded data.
    """
    try:
        if file_format == 'csv':
            df = pd.read_csv(file_path)
        elif file_format == 'excel':
            df = pd.read_excel(file_path)
        elif file_format =='xls':
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'excel'.")
        
        print(f"Dataset successfully loaded. Dimensions: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
