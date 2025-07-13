import ast
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


def transform_to_dict(unique_array):

    result_dict = {'0': 'Null'}

    for item in unique_array:
        if item != '0':  
            variable_list = ast.literal_eval(item)
            
            for var in variable_list:
                # Diviser en numéro et nom, puis ajouter au dictionnaire
                num, name = map(str.strip, var.split(':'))
                result_dict[num] = name
    
    return result_dict


def count_common_Motifs(df1, df2):
   
    common_counts = []

    unique_Supports = set(df1['Support']).intersection(set(df2['Support']))
    print(unique_Supports)
    for Support in unique_Supports:
        compteur = 0
        Motifs1 = df1[df1['Support'] == Support]['Motifs'].iloc[0] 
        Motifs2 = df2[df2['Support'] == Support]['Motifs'].iloc[0]  # Liste d'ensembles

        for motif_2 in Motifs2:
            print(motif_2)
            for motif_1 in Motifs1:
                if motif_2 == motif_1:
                    compteur  = compteur + 1
        
        common_counts.append({'Support': Support, 'Pattern': compteur})
    return common_counts



def transform_column(df_column, variable_mapping):

    col_transform = []

    for x in df_column:
        t = x.strip("[]")
        a = t.split("}, {")
        
        result = [
            set(item.strip("{}").replace("'", "").split(", ")) for item in a
        ]
        
        transformed_line = []  
        for group in result:
            new_group = set()  
            for element in group:
                numero = ''.join(filter(str.isdigit, element))
                mapped_element = variable_mapping.get(numero, numero) + (element[-1] if not numero == element else "")
                new_group.add(mapped_element)
            transformed_line.append(new_group)  

        col_transform.append(transformed_line)

    return col_transform

def plot_memory(df, file_name ):

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="Support",
        y="Mémoire maximale (Ko)",
        hue="Dataset",
        markers=True, 
        style="Dataset", 
        linewidth=2,  
        markersize=10,  
    )

    plt.ylabel("Memory usage(Ko)")
    plt.xlabel('Minimum support threshold')

    plt.tight_layout()
    plt.savefig("../../data/article/"+file_name+".eps")
    plt.savefig("../../data/article/"+file_name+".jpeg")
    plt.show()
    plt.close()


def plot_number_pattern(df, file_name ):

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="Support",
        y="Size",
        hue="Dataset",
        markers=True, 
        style="Dataset", 
        linewidth=2,  
        markersize=10,  
    )

    plt.ylabel('Number of patterns')
    plt.xlabel('Minimum support threshold')

    plt.tight_layout()
    plt.savefig("../../data/article/"+file_name+".eps")
    plt.savefig("../../data/article/"+file_name+".jpeg")
    plt.show()
    plt.close()


def plot_time(df, file_name ):

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="Support",
        y="Temps d'exécution (s)",
        hue="Dataset",
        markers=True, 
        style="Dataset", 
        linewidth=2,  
        markersize=10,  
    )

    plt.ylabel("Time (s)")
    plt.xlabel('Minimum support threshold')

    plt.tight_layout()
    plt.savefig("../../data/article/"+file_name+".eps")
    plt.savefig("../../data/article/"+file_name+".jpeg")
    plt.show()
    plt.close()



def create_motif_score_dict_last(df):
    motif_score_dict = {}
    test = []

    for idx, row in df.iterrows():
        chaine_array = row['Supports motifs'] 
        motifs = row['Motifs']

        
        # S'assurer que motifs est une liste
        if isinstance(motifs, int):
            motifs = [motifs]  # Convertir en liste si c'est un entier
        elif not isinstance(motifs, list):
            motifs = list(motifs)  
        
        try:
            # Essayer de convertir la chaîne en liste
            supports = ast.literal_eval(chaine_array)
            #print(supports )
        except Exception as e:
            # Si la conversion échoue, essayer d'ajouter un crochet fermant
            try:
                supports = ast.literal_eval(chaine_array + "]")
            except Exception as e:
                print(f"Erreur lors de l'évaluation de la chaîne {chaine_array}: {e}")
                continue 
        
        #print("Supports:", supports)
        #print("Motifs:", motifs)
        
        if isinstance(motifs, int) or isinstance(supports, int):
            motif_score_dict[f'{motifs}'] = supports

        else:
            for motif, score in zip(motifs, supports):
                motif_str = str(motif)  
                motif_score_dict[motif_str] = score

    return motif_score_dict

#pour convertir une clé (chaîne) en frozenset
def safe_convert(key):
    try:
        result = ast.literal_eval(key)
        if isinstance(result, set):
            return frozenset(result)  # Utiliser frozenset pour éviter les problèmes d'ordre
        elif isinstance(result, list) and len(result) == 1 and isinstance(result[0], set):
            return frozenset(result[0])  # Cas particulier où l'ensemble est dans une liste
    except Exception as e:
        raise ValueError(f"Format inattendu pour la clé : {key}, erreur : {e}")




def compare_supports(dic_1_frozen, dic_2_frozen, common_keys):
    """
    Compare les supports des motifs entre deux dictionnaires après réduction de dimension.
    
    Arguments:
        dic_1_frozen (dict): Dictionnaire des supports avant réduction.
        dic_2_frozen (dict): Dictionnaire des supports apres réduction.
        common_keys (set): Ensemble des motifs communs entre les deux dictionnaires.

    Retourne:
        DataFrame contenant les supports et leurs différences.
    """

    # Liste pour stocker les résultats de la comparaison
    support_comparison = []

    for key in common_keys:
        support_pca = dic_2_frozen[key]
        support_original = dic_1_frozen[key]
        
        support_comparison.append({
            "Motif": key,
            "Support PCA": support_pca,
            "Support before PCA": support_original,
            "Difference": support_original - support_pca
        })

    # Convertir en DataFrame
    df_support = pd.DataFrame(support_comparison)

    # Trier par différence de support
    df_support = df_support.sort_values(by="Difference", ascending=False)

    # --- Visualisation ---
    plt.figure(figsize=(12, 5))

    # 1. Scatter plot : Comparaison des supports
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=df_support["Support before PCA"], y=df_support["Support PCA"])
    plt.xlabel("Support before PCA")
    plt.ylabel("Support PCA")
    plt.title("Supports comparisons before and before and after PCA")

    plt.tight_layout()
    plt.show()

    return df_support




def plot_comparison_sizes(original, pca, dataset_valid, title="Comparison of Sizes and Common Elements"):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # First axis (left) for "Size" curves
    sns.lineplot(data=original, x="Support", y="Size", label="Original (total size)", color="green", linestyle="-", linewidth=2, ax=ax1)
    sns.lineplot(data=pca, x="Support", y="Size", label="PCA (total size)", color="orange", linestyle="-.", linewidth=2, ax=ax1)

    # Configure the first axis (left)
    ax1.set_xlabel("Support", fontsize=14)
    ax1.set_ylabel("Size", fontsize=14, color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_title(title, fontsize=16)

    # Second axis (right) for "Pattern" of common elements
    ax2 = ax1.twinx()
    sns.lineplot(data=dataset_valid, x="Support", y="Pattern", label="Common Elements", color="blue", linestyle="--", linewidth=2, ax=ax2, marker="o")

    # Configure the second axis (right)
    ax2.set_ylabel("Number of Elements", fontsize=14, color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Add the legend (merge both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)

    # Add a grid for better readability
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Display the graph
    plt.tight_layout()
    plt.show()
