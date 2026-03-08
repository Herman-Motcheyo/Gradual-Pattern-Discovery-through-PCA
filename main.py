import time
import tracemalloc
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from src.models.graank2 import algorithm_gradual_gradual  


def process_support(dataset, support):

    file_results = f"chickenpox_par_{support}.csv"

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
        "./data/Clean/Chickenpox.csv",
    ]
    supports = [0.3]
    print("Début expérimentation")
    experimentation_parallel(datasets, supports)
    print("Fin expérimentation")
