import pandas as pd
import numpy as np
import sys
sys.path.append('../')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .load_data import load_dataset
from ..visualizations.code_plots.Visualize_prepro import plot_box_plot, plot_all_distributions
import joblib

def rename_columns(df, new_cols_dictionnary = None):
    """Rename the columns of the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame whose columns need to be renamed.
    new_cols_dictionnary: The dictionnary with new column's name

    Returns:
    DataFrame: The DataFrame with renamed columns.
    """

    try:
        
        df = df.rename(columns=new_cols_dictionnary)
        
        print("Columns renamed successfully.")
        return df
    except Exception as e:
      print(f"Error renaming columns: {e}")
      return df



def replace_outliers_iqr(df, column):
    """
    Replace outliers in the specified column of the DataFrame using IQR method.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to process.

    Returns:
    pd.DataFrame: DataFrame with outliers replaced by median.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    median = df[column].median()

    df[column] = np.where(df[column] < lower_bound, median, df[column])
    df[column] = np.where(df[column] > upper_bound, median, df[column])

    return df



def standardize_data(df, numerical_columns=None, exclude_columns=None, scaler=None, scaler_filename=None):
    """
    Standardizes the data in a DataFrame or Series such that each feature has a mean of 0 and a standard deviation of 1.
    Optionally saves the scaler to a file.

    Parameters:
    df (DataFrame or Series): The data to be standardized. Can be either a DataFrame or a Series.
    numerical_columns (list): List of numerical columns to be standardized. If None, all numerical columns will be standardized. Default is None.
    exclude_columns (list): List of columns to exclude from standardization. Default is None.
    scaler (StandardScaler): A pre-initialized StandardScaler. If None, a new StandardScaler will be created. Default is None.
    scaler_filename (str): The filename to save the scaler. If None, the scaler will not be saved. Default is None.

    Returns:
    DataFrame or Series: The standardized data.
    """
    try:
        if scaler is None:
            scaler = StandardScaler()

        if isinstance(df, pd.DataFrame):
            # Handle DataFrame
            if numerical_columns is None:
                numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

            if exclude_columns is not None:
                numerical_columns = [col for col in numerical_columns if col not in exclude_columns]

            print(f"Columns to be standardized: {numerical_columns}")

            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        elif isinstance(df, pd.Series):
            # Handle Series
            df = pd.DataFrame(df)  
            df = scaler.fit_transform(df)
            df = pd.Series(df.flatten())  

        else:
            raise TypeError("Input must be a pandas DataFrame or Series.")

        if scaler_filename:
            joblib.dump(scaler, scaler_filename)
            print(f"Scaler saved to {scaler_filename}")

        print("Data standardized successfully.")
        return df

    except Exception as e:
        print(f"Error standardizing data: {e}")
        return df


def apply_standardization(df, scaler_filename, numerical_columns=None, exclude_columns=None):
    """
    Applies the saved scaler to the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the data to be standardized.
    scaler_filename (str): The filename from which to load the scaler.
    numerical_columns (list): List of numerical columns to be standardized. If None, all numerical columns will be standardized. Default is None.
    exclude_columns (list): List of columns to exclude from standardization. Default is None.

    Returns:
    DataFrame: The DataFrame with standardized numerical columns.
    """
    try:
        # Load the scaler
        scaler = joblib.load(scaler_filename)
        print(f"Scaler loaded from {scaler_filename}")

        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if exclude_columns is not None:
            numerical_columns = [col for col in numerical_columns if col not in exclude_columns]

        # Logging the columns to be standardized
        print(f"Columns to be standardized: {numerical_columns}")

        # Transform the numerical columns
        df[numerical_columns] = scaler.transform(df[numerical_columns])

        print("Data standardized successfully.")
        return df
    except Exception as e:
        print(f"Error applying standardization: {e}")
        return df


def drop_duplicated_rows(df, subset=None, keep='first'):
    """
    Drops duplicated rows from a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame from which duplicated rows need to be dropped.
    subset (list): List of columns to consider for identifying duplicates. If None, consider all columns. Default is None.
    keep (str): Determines which duplicates (if any) to keep. 'first' keeps the first occurrence, 'last' keeps the last occurrence, False drops all duplicates. Default is 'first'.

    Returns:
    DataFrame: The DataFrame with duplicated rows dropped.
    """
    try:
        df_dedup = df.drop_duplicates(subset=subset, keep=keep)
        
        print(f"Duplicated rows dropped. Remaining rows: {len(df_dedup)}")
        return df_dedup
    except Exception as e:
        print(f"Error dropping duplicated rows: {e}")
        return df


def export_to_excel(df, file_path, sheet_name='Sheet1', index=False):
    """
    Exports a pandas DataFrame to an Excel file, with options for sheet name and index inclusion.

    Parameters:
    df (pd.DataFrame): The DataFrame to be exported. Must be a pandas DataFrame.
    file_path (str): The path to the Excel file where the DataFrame will be saved. Must be a valid file path.
    sheet_name (str, optional): The name of the sheet in the Excel file. Default is 'Sheet1'. Must be a string.
    index (bool, optional): Whether to include the DataFrame's index in the Excel file. Default is False. Must be a boolean.

    Returns:
    None: This function does not return a value.

    Raises:
    ValueError: If `df` is not a pandas DataFrame.
    TypeError: If `file_path` is not a string or `sheet_name` is not a string.
    """
    
    if not isinstance(file_path, str):
        raise TypeError("The 'file_path' parameter must be a string.")
    
    if not isinstance(sheet_name, str):
        raise TypeError("The 'sheet_name' parameter must be a string.")
    
    if not isinstance(index, bool):
        raise TypeError("The 'index' parameter must be a boolean.")
    
    try:
        df.to_excel(file_path, sheet_name=sheet_name, index=index, engine='openpyxl')
        print(f"DataFrame successfully exported to {file_path} in sheet '{sheet_name}'.")
    except Exception as e:
        print(f"Error exporting DataFrame to Excel: {e}")
        raise
