import time
import tracemalloc
import csv
from datetime import datetime
from graank2 import algorithm_gradual_gradual  


def experimentation(dataframes, supports):
    print("Début de l'expérimentation")

    file_results = "results_gradual.csv"
        
    with open(file_results, mode='w', newline="", encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Dataset", "Support", "Size","Temps d'exécution (s)",  "Mémoire maximale (Ko)","Titre variables","Supports motifs", "Motifs detectés"])

        for df_name in dataframes:
            for support in supports:
                tracemalloc.start()
                start_time = time.time()
                len_motifs, title, supports_motifs, motifs_detect = algorithm_gradual_gradual(df_name, support)

                end_time = time.time()
                memory_usage, _ = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                elapsed_time = end_time - start_time
                memory_usage = memory_usage / 1024  # Convertir en Ko
                writer.writerow([
                        df_name,    
                        support, 
                        len_motifs, 
                        elapsed_time,
                        memory_usage, 
                        title,       
                        supports_motifs  
                        ,motifs_detect
                    ])
            print(f"Dataset {df_name} terminé")

datasets = [
        #"../../data/Clean/HCV.csv",
        #"../../data/AfterPCA/HCV.csv",
        #"../../data/Clean/AirQualityUCI.csv",
         #"../../data/AfterPCA/Air_quality.csv",
        #"../../data/Clean/Cargo_2000.csv",
        #"../../data/AfterPCA/Cargo_2000.csv",
     	"../../data/Clean/Chickenpox.csv",
     	"../../data/AfterPCA/Chickenpox.csv"
       
    ]
supports = [0.1,0.14,0.18,0.16,0.2,0.24,0.28,0.3,0,34,0.38,0.4,0.5,0.6,0.7]
print("test")
experimentation(datasets, supports)
