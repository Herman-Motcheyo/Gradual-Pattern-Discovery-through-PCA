import subprocess
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import sys


#"./data/herman/Clean/HCV.csv",
#"./data/herman/Clean/air_quality.csv",
#"./data/herman/Clean/Chickenpox.csv",
DATASETS = [
   
    "./data/herman/Clean/Cargo_2000.csv",
]
#  [ "ga","pso", "prs"]
SUPPORTS = [0.3, 0.34, 0.38, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
ALGORITHMS = [ "ga"] 
MAX_WORKERS = 10      
LOG_DIR = Path("results1") # Changé pour correspondre à votre dossier de résultats

# ──────────────────────────────────────────────────────────────────────────────

def make_log_path(algo: str, dataset: str, support: float) -> Path:
    """Génère le chemin du fichier log incluant l'algorithme."""
    dataset_path = Path(dataset)
    stem = dataset_path.stem                 
    support_str = str(support).replace(".", "_")
    return LOG_DIR / f"res_{algo}_{stem}_s_{support_str}.txt"


def run_experiment(algo: str, dataset: str, support: float) -> dict:
    """Lance une expérimentation pour un algo, un dataset et un support."""
    cmd = ["python", "src/main.py", "-a", algo, "-f", dataset, "-s", str(support)]
    
    log_path = make_log_path(algo, dataset, support)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    label = f"[{algo.upper()} | {Path(dataset).stem} | s={support}]"

    print(f"  START  {label}")

    with open(log_path, "w") as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )

        end = datetime.now()
        duration = (end - start).total_seconds()
        
        log_file.write(f"\nRun-time: {duration} seconds\n")
        log_file.write(f"Return Code: {result.returncode}\n")

    status = " OK " if result.returncode == 0 else " ERR"
    print(f"{status} DONE  {label}  ({duration:.1f}s)  → {log_path.name}")

    return {
        "algo": algo,
        "dataset": dataset,
        "support": support,
        "returncode": result.returncode,
        "duration": duration,
        "log": str(log_path),
    }


def main():
    LOG_DIR.mkdir(exist_ok=True)

    experiments = list(itertools.product(ALGORITHMS, DATASETS, SUPPORTS))
    total = len(experiments)

    print(f"\n{'='*60}")
    print(f"  EXPÉRIMENTATIONS  —  {total} jobs  ({MAX_WORKERS} en parallèle)")
    print(f"  Algos   : {ALGORITHMS}")
    print(f"  Datasets: {len(DATASETS)}")
    print(f"  Supports: {SUPPORTS}")
    print(f"{'='*60}\n")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # On passe 'algo', 'ds', et 'sup' à submit
        futures = {
            executor.submit(run_experiment, algo, ds, sup): (algo, ds, sup)
            for algo, ds, sup in experiments
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                algo, ds, sup = futures[future]
                print(f" EXCEPTION [{algo} | {ds} | s={sup}]: {e}")
                results.append({"algo": algo, "dataset": ds, "support": sup, "returncode": -1, "duration": 0, "log": ""})

    # ─── RÉSUMÉ FINAL ─────────────────────────────────────────────────────────
    success = [r for r in results if r["returncode"] == 0]
    failed  = [r for r in results if r["returncode"] != 0]

    summary_path = LOG_DIR / "summary_execution.txt"
    with open(summary_path, "w") as f:
        f.write(f"RÉSUMÉ — {datetime.now().isoformat()}\n")
        f.write(f"Total : {total}  |  {len(success)}  |   {len(failed)}\n\n")
        
        for r in sorted(results, key=lambda x: (x["algo"], x["dataset"], x["support"])):
            status = "OK " if r["returncode"] == 0 else "ERR"
            f.write(f"  [{status}] {r['algo'].upper()} - {Path(r['dataset']).stem} - s={r['support']} ({r['duration']:.1f}s)\n")

    print(f"\n{'='*60}")
    print(f"  TERMINÉ :  {len(success)}/{total}")
    print(f"  Résultats dans le dossier : {LOG_DIR}")
    print(f"{'='*60}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()