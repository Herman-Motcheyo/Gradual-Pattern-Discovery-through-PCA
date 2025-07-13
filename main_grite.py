import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from src.models.grite  import grite3  

datasets = {
    "hcv_clean": pd.read_csv("./data/Clean/HCV.csv"),
    "hcv_pca": pd.read_csv("./data/AfterPCA/HCV.csv"),
    "airquality_pca": pd.read_csv("./data/AfterPCA/Air_quality.csv"),
    "cargo2000_pca": pd.read_csv("./data/AfterPCA/Cargo_2000.csv"),
    "chickenpox_pca": pd.read_csv("./data/AfterPCA/Chickenpox.csv"),
    "airquality_clean": pd.read_csv("./data/Clean/Air_quality.csv"),
    "cargo2000_clean": pd.read_csv("./data/Clean/Cargo_2000.csv"),
    "chickenpox_clean": pd.read_csv("./data/Clean/Chickenpox.csv"),
    
}
min_freqs = [0.04, 0.08, 0.1, 0.14, 0.16, 0.18, 0.2, 0.24, 0.28, 0.3, 0.34, 0.38, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
output_dir = "grite_outputs"
os.makedirs(output_dir, exist_ok=True)

def process_dataset(args):
    name, df, freq = args
    print(f"Traitement de {name} avec min_freq={freq}...")

    results = grite3(df, min_freq=freq) 

    df_result = pd.DataFrame(results["motifs"])
    df_result["sup_min"] = freq
    df_result["data"] = name
    df_result["execution_time"] = results["execution_time"]
    df_result["total"] = results["total"]
    df_result["memory_usage"] = results["memory_usage"]

    file_name = f"grite_{name}_sup{str(freq).replace('.', '_')}.txt"
    file_path = os.path.join(output_dir, file_name)
    df_result.to_csv(file_path, sep="\t", index=False)

    return f"{name} avec support {freq} terminé. Fichier: {file_path}"

if __name__ == "__main__":
    # Créer toutes les combinaisons (dataset, support)
    tasks = [(name, df, freq) for name, df in datasets.items() for freq in min_freqs]

    max_workers = min(45, multiprocessing.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_dataset, tasks)

        for res in results:
            print(res)
