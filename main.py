import time
import tracemalloc
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from src.models.graank2 import algorithm_gradual_gradual  


def process_support(dataset, support):
    """
    Fonction pour exécuter l'algorithme sur un dataset avec un support donné.
    Elle sauvegarde le résultat dans un fichier suffixé par le support.
    """
    file_results = f"Cargo_clean_par_{support}.csv"

    with open(file_results, mode='w', newline="", encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Dataset", "Support", "Size", "Temps d'exécution (s)", "Mémoire maximale (Ko)",
                         "Titre variables", "Supports motifs", "Motifs détectés"])

        tracemalloc.start()
        start_time = time.time()
        len_motifs, title, supports_motifs, motifs_detect = algorithm_gradual_gradual(dataset, support)

        end_time = time.time()
        memory_usage, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed_time = end_time - start_time
        memory_usage = memory_usage / 1024  # Convertir en Ko
        writer.writerow([
            dataset,
            support,
            len_motifs,
            elapsed_time,
            memory_usage,
            title,
            supports_motifs,
            motifs_detect
        ])
    print(f"Support {support} pour le dataset {dataset} terminé. Résultats enregistrés dans {file_results}")


def experimentation_parallel(dataframes, supports):
    """
    Exécuter l'algorithme pour chaque support en parallèle.
    """
    with ProcessPoolExecutor() as executor:
        futures = []
        for df_name in dataframes:
            for support in supports:
                futures.append(executor.submit(process_support, df_name, support))
        
        # Attendre que tous les processus soient terminés
        for future in futures:
            future.result()


if __name__ == "__main__":
    datasets = [
        "./data/Clean/Cargo_2000.csv"
    ]
    supports = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.38,0.34,0.3,0.28,0.24,0.2,0.18,0.14,0.1,0.08,0.04]
    print("Début expérimentation")
    experimentation_parallel(datasets, supports)
    print("Fin expérimentation")
