import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .CustomException import CustomError

class PcaGradualPattern:

    def __init__(self, dataframes):
        self.dataframes = dataframes

    def standardize_data(self, df, df_name):
        scaler = StandardScaler()
        df = df.dropna()
        
        df_numeric = df.select_dtypes(include=[np.number], exclude=["datetime64", "object"])
        
        if df_numeric.shape[1] == 0:
            error = CustomError(df_name, "Aucune colonne numérique trouvée")
            error.log_error()
            raise error

        return scaler.fit_transform(df_numeric)


    def apply_pca(self, df_scaled):
        """
        Apply PCA on the standardized data and return explained variance.
        
        Parameters:
        - df_scaled: numpy array of standardized data
        
        Returns:
        - explained_variance: array of explained variance ratios
        - cumulative_variance: array of cumulative explained variance ratios
        """
        pca = PCA()
        pca.fit(df_scaled)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        return explained_variance, cumulative_variance , pca

    def plot_variances(self, axes, explained_variance, cumulative_variance, df_name):
        """
        Plot cumulative and explained variance.
        
        Parameters:
        - axes: matplotlib Axes object for plotting
        - explained_variance: array of explained variance ratios
        - cumulative_variance: array of cumulative explained variance ratios
        - df_name: string, name of the dataframe being plotted
        """
        # Trace de la variance expliquée cumulée
        axes[0].plot(cumulative_variance, label=df_name)
        
        # Limitation des composants pour une meilleure lisibilité a 10
        num_components_to_display = min(10, len(explained_variance))
        bars = axes[1].bar(
            range(1, num_components_to_display + 1), 
            explained_variance[:num_components_to_display], 
            alpha=0.7, label=df_name
        )
        axes[1].bar_label(bars, fmt='%.2f', fontsize=8)


    def plot_pca_variance_final(self):
        """
        Main function to plot the cumulative and individual explained variance 
        for each DataFrame in the list of dataframes.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        
        for i, dfx in enumerate(self.dataframes):
            df = dfx["df"].dropna()
            df_scaled = self.standardize_data(df, dfx["name"])
            explained_variance, cumulative_variance, pca = self.apply_pca(df_scaled)
            self.plot_variances(axes, explained_variance, cumulative_variance, dfx["name"])
        
           
        axes[0].set_xlabel('Number of Principal Components')
        axes[0].set_ylabel('Cumulative Explained Variance')
        axes[0].set_title('Cumulative Explained Variance by PCA')
        axes[0].legend()
        axes[0].grid(True)
    
        axes[1].set_xlabel('Principal Components')
        axes[1].set_ylabel('Percentage of Explained Variance')
        axes[1].set_title('Explained Variance by Each Principal Component')
        axes[1].legend()
    
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        #plt.savefig("pca_improve_advance.png")
        plt.savefig("pca_improve_advance.jpeg")
        plt.show()

    def apply_pca_and_get_features_importance(self, threshold):
        solutions_for_dfs = []
        for i, dfx in enumerate(self.dataframes):
            selected_features = {}

            df = dfx["df"].dropna() 
            df = df.select_dtypes(include=[np.number], exclude=["datetime64", "object"])
            df_scaled = self.standardize_data(df, dfx["name"])
            explained_variance, cumulative_variance, pca = self.apply_pca(df_scaled)
            # calcul des charges pour chaque composantes
            loading = pd.DataFrame(pca.components_.T, index = df.columns,columns=[f'Composante {i+1}' for i in range(len(pca.components_))] )
            
            n_components = np.argmax(cumulative_variance >= threshold) + 1
            print(f"Nombre de composantes pour expliquer { threshold * 100}% de la variance: {n_components}")

            for i in range(n_components): # # Pour chaque composante principale retenue
                # Trier les charges de la composante en valeur absolue
                sorted_loadings = loading.iloc[:, i].abs().sort_values(ascending =False)

                #Calculer la somme cumulée des charges
                cumulative_influence = sorted_loadings.cumsum() / sorted_loadings.sum()

                # Trouer le nombre minimal de variables pour atteindre le seuil
                num_vars = (cumulative_influence >= threshold).values.argmax()+1

                # Sélectionner les variables jusqu'à ce nombre
                selected_features[f'Composante {i+1}'] = sorted_loadings.index[:num_vars].tolist()
            features = set()

            for vars_ in selected_features.values():
                features.update(vars_)

            features = list(features)

            solutions_for_dfs.append(
                {  "df_name": dfx["name"],
                    "df_loadings": loading.iloc[:,:n_components],
                    "n_components": n_components,
                    "features": features
                }
            )
        return solutions_for_dfs


    def extract_significant_variables(self,df_loading, threshold):
        """
        Extrait les noms et les valeurs des variables dont le loading est supérieur ou égal au seuil dans chaque colonne d'un DataFrame.
        Args:
            df (pd.DataFrame): DataFrame contenant les loadings.
            threshold (float): Seuil à partir duquel une variable est considérée comme significative (en valeur absolue).
        Returns:
            pd.DataFrame: Un DataFrame récapitulatif avec le nom des variables et le nombre de variables sélectionnées par colonne.
        """
        selected_variables = []
    
        for col in df_loading.columns:
            # Filtrer les valeurs qui sont supérieures ou égales au seuil en valeur absolue
            mask = df_loading[col].abs() >= threshold
            if mask.any():
                # Récupérer les noms des variables répondant au critère dans cette colonne
                variables = df_loading.index[mask].tolist()
                for va in variables:
                    selected_variables.append(va)

        return selected_variables

    def process_multiple_dfs(self,dfs_loadings, threshold=0.5):
        """
            Applique la fonction de sélection sur plusieurs DataFrames et combine les résultats.
            Args:
            dfs (list of pd.DataFrame): Liste de DataFrames contenant les loadings à analyser.
            threshold (float): Seuil pour la sélection des variables.
            Returns:
            pd.DataFrame: DataFrame combiné avec les noms des variables sélectionnées et leur nombre par DataFrame et colonne.
        """
        all_results = []
    
        for i, df in enumerate(dfs_loadings):
            # Appliquer la fonction d'extraction sur chaque DataFrame de la liste
            summary_df = self.extract_significant_variables(df, threshold)
            all_results.append(summary_df)
    
        return all_results

    
    def show_size_final_features(self, datasets, num_selected_features, num_initial_components,num_original_variables):
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.2  
        x = range(len(datasets))

         
        bars_selected = ax.bar([p - width for p in x], num_selected_features, width=width, label="Selected features")
        bars_components = ax.bar(x, num_initial_components, width=width, label="Initial components")
        bars_original = ax.bar([p + width for p in x], num_original_variables, width=width, label="Original feature")

        # Ajouter les valeurs au-dessus de chaque barre
        for bars in [bars_selected, bars_components, bars_original]:
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, int(yval), ha='center', va='bottom')

       
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Numbers of features/Components")
        ax.set_title("Numbers of features vs Components")
        ax.legend()

        plt.savefig("Selecte features using pca.eps")
        plt.savefig("Selecte features using pca.jpeg")
        plt.tight_layout()
        plt.show()