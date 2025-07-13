import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

def plot_correlation_coefficients(df, target_column, save_path=None, alpha=0.05):
    """
    Load data from a file, compute Spearman and Kendall correlation coefficients of variables with a target variable,
    and plot the results. Optionally save the plot to a file.
    
    Parameters:
    file_path (str): Path to the data file (e.g., CSV file).
    target_column (str): The name of the target column in the DataFrame.
    save_path (str, optional): Path to save the plot image. If None, the plot is not saved. Default is None.
    alpha (float): Significance level for p-values. Default is 0.05.
    """
    
    variables = df.drop(target_column, axis=1)
    cible = df[target_column]
    
    # Compute Spearman correlation coefficients and p-values
    spearman_corrs = variables.apply(lambda x: spearmanr(x, cible)[0])
    spearman_p_values = variables.apply(lambda x: spearmanr(x, cible)[1])
    
    # Compute Kendall correlation coefficients and p-values
    kendall_corrs = variables.apply(lambda x: kendalltau(x, cible)[0])
    kendall_p_values = variables.apply(lambda x: kendalltau(x, cible)[1])
    
    corrs_df = pd.DataFrame({
        'Variable': spearman_corrs.index, 
        'Spearman': spearman_corrs.values, 
        'Spearman_p_values': spearman_p_values.values, 
        'Kendall': kendall_corrs.values,
        'Kendall_p_values': kendall_p_values.values
    })
    
    sns.lineplot(data=corrs_df, x='Variable', y='Spearman', marker='*', color='green',
                 markersize=10, label='Spearman', linewidth=2)
    sns.lineplot(data=corrs_df, x='Variable', y='Kendall', marker='+', color='blue',
                 markersize=12, label='Kendall', linewidth=2)
    
    # Add error bars for non-significant correlations
    for index, row in corrs_df.iterrows():
        # Error bars for Spearman correlations with p-value > alpha
        if row['Spearman_p_values'] > alpha:
            plt.errorbar(x=row['Variable'], y=row['Spearman'], yerr=0.05, color='green', 
                         fmt='o', capsize=5, capthick=2, linestyle='--')
        # Error bars for Kendall correlations with p-value > alpha
        if row['Kendall_p_values'] > alpha:
            plt.errorbar(x=row['Variable'], y=row['Kendall'], yerr=0.05, color='blue', 
                         fmt='o', capsize=5, capthick=2, linestyle='-.')
    
    
    plt.xticks(rotation=45) 
    plt.xlabel('Variables')  
    plt.ylabel('Correlation Coefficient')  
   
    plt.legend() 
    
    if save_path:
        plt.savefig(save_path)  
    else:
        plt.show() 