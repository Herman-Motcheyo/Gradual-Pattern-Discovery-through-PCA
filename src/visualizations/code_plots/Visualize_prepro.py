from pathlib import Path
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


def plot_box_plot(df, path_to_save=None):
    """
    Plot box plot for each numeric column of the DataFrame.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the data to analyze.
    path_to_save (str, optional): The file path to save the resulting boxplot figure. If None, the plot will be displayed but not saved.

    Returns:
    None
    """
    if df.empty:
        raise ValueError("The DataFrame is empty. Please provide a DataFrame with data.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in the DataFrame.")

    num_cols = len(numeric_cols)
    num_rows = (num_cols + 2) // 3  # Calculate the number of rows needed for subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()

    # Save the figure if a path is provided
    if path_to_save:
        output_path = Path(path_to_save)
        if not output_path.suffix in ['.png', '.jpg', '.jpeg', '.pdf','.eps']:
            raise ValueError("Unsupported file extension. Supported extensions are: .png, .jpg, .jpeg, .pdf, .eps")
        plt.savefig(output_path)
        print(f"Figure saved to {output_path}")
    else:
        plt.show()



def plot_all_distributions(df, path_to_save=None):
    """
    Plots the distribution of each numeric column in the DataFrame in a grid of subplots.

    Args:
    df (pd.DataFrame): The DataFrame containing the data to visualize.
    path_to_save (str, optional): The file path to save the resulting figure. If None, the plot will be displayed but not saved.

    Returns:
    None
    """
    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. Please provide a DataFrame with data.")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in the DataFrame.")
    
    num_cols = len(numeric_cols)
    num_rows = (num_cols + 2) // 3  
    
   
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], ax=axes[i], kde=True,  alpha=0.5)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel('')

    
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()

    if path_to_save:
        output_path = Path(path_to_save)
        if not output_path.suffix in ['.png', '.jpg', '.jpeg', '.pdf','.eps']:
            raise ValueError("Unsupported file extension. Supported extensions are: .png, .jpg, .jpeg, .pdf, .eps")
        plt.savefig(output_path)
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


def plot_correlation_heatmap(df, title="Correlation Heatmap", figsize=(10, 6), cmap="coolwarm"):
    """
    Displays a correlation heatmap for the numeric columns of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.
    - title (str): The title of the heatmap.
    - figsize (tuple): Size of the figure.
    - cmap (str): Color palette for the heatmap.
    """
    numeric_df = df.select_dtypes(exclude=('object'))
    
    plt.figure(figsize=figsize)
    sns.heatmap(numeric_df.corr(), annot=True, cmap=cmap)
    plt.title(title)
    plt.show()
