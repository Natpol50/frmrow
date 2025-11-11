import argparse
import csv
import time
import random
import numpy as np
from pathlib import Path
from typing import List
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solvermanager.constructors import get_constructor
from filemanagers.instancefilemanager import InstanceFileManager
from solvermanager.routemanager import RouteEvaluator

#!/usr/bin/env python3
"""
perf_contrcut_test.py

Script de comparaison des constructeurs définis dans `constructors.py`.

Usage (exemples):
    python perf_contrcut_test.py --data-dir data --instances C101 R101 \
        --constructors nearest_neighbor random savings --reps 3 --out results.csv

Si --instances n'est pas fourni, le script essaie d'interroger
InstanceFileManager.list_instances() / discover_instances() ou
scan le répertoire data pour des fichiers reconnus.

Résultats CSV:
    instance,constructor,rep,time_s,total_cost,feasible,n_routes,coverage_rate,error

Note: adapte les chemins si nécessaire selon votre projet.
"""

# Ajoute le parent du projet au PYTHONPATH si nécessaire (optionnel)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# random_constructor peut être appelé indirectement ; on fixe les seeds avant l'appel

def discover_instances(manager: InstanceFileManager, data_dir: Path) -> List[str]:
    # Tentatives: méthodes du manager puis scan fichiers
    for attr in ("list_instances", "discover_instances", "available_instances"):
        if hasattr(manager, attr):
            fn = getattr(manager, attr)
            try:
                ids = fn()
                if ids:
                    return list(ids)
            except Exception:
                pass

    # Fallback: scan fichiers du dossier data (extensions usuelles)
    exts = {".vrp", ".txt", ".dat", ".csv"}
    ids = []
    for p in data_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            ids.append(p.stem)
    return ids

def evaluate_solution(solution, instance, evaluator: RouteEvaluator):
    """Retourne (feasible:bool, total_cost:float, n_routes:int, coverage_rate:float, error:str|None)"""
    try:
        total_cost = 0.0
        customers_in_solution = []
        feasible = True
        for route in solution.routes:
            ok, cost = evaluator.evaluate_route_fast(route)
            if not ok:
                feasible = False
            total_cost += cost
            customers_in_solution.extend(route)
        expected_customers = set(range(1, instance.dimension))
        covered = set(customers_in_solution)
        coverage_rate = len(covered & expected_customers) / max(1, len(expected_customers))
        n_routes = solution.get_n_routes()
        # If some expected customers missing => not feasible
        if covered != expected_customers:
            feasible = False
        return feasible, float(total_cost), int(n_routes), float(coverage_rate), None
    except Exception as e:
        return False, float("inf"), -1, 0.0, str(e)

def main():
    parser = argparse.ArgumentParser(description="Compare VRP constructors performance")
    parser.add_argument("--data-dir", "-d", type=str, default="data", help="Répertoire des instances")
    parser.add_argument("--instances", "-i", nargs="*", help="Liste des noms d'instances (optionnel)")
    parser.add_argument("--constructors", "-c", nargs="*", help="Constructeurs à tester",
                        default=[
                            "nearest_neighbor",
                            "random",
                            "savings",
                            "savings_parallel",
                            "savings_sequential",
                            "insertion_cheapest",
                            "insertion_nearest",
                            "insertion_farthest",
                        ])
    parser.add_argument("--reps", "-r", type=int, default=3, help="Répétitions par combinaison")
    parser.add_argument("--out", "-o", type=str, default="perf_constructors_results.csv", help="Fichier CSV de sortie")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manager = InstanceFileManager(str(data_dir))

    if args.instances:
        instance_names = args.instances
    else:
        all_instances = discover_instances(manager, data_dir)
        if not all_instances:
            print("Aucune instance trouvée. Spécifiez --instances ou vérifiez --data-dir.")
            return
        k = min(3, len(all_instances))
        instance_names = random.sample(all_instances, k)
        print(f"Instances sélectionnées aléatoirement ({k}): {instance_names}")

    constructors = args.constructors
    reps = args.reps
    out_path = Path(args.out)

    # Prépare le CSV
    fieldnames = [
        "instance", "constructor", "rep", "time_s", "total_cost",
        "feasible", "n_routes", "coverage_rate", "error"
    ]
    with out_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for inst_name in instance_names:
            try:
                instance = manager.load_instance(inst_name)
            except Exception as e:
                print(f"Impossible de charger instance {inst_name}: {e}")
                # Écrit une ligne d'erreur pour visibilité
                writer.writerow({
                    "instance": inst_name, "constructor": "", "rep": "", "time_s": "",
                    "total_cost": "", "feasible": False, "n_routes": "", "coverage_rate": "", "error": f"load_error:{e}"
                })
                continue

            for ctor_name in constructors:
                # Récupère la fonction constructeur (peut raise ValueError)
                try:
                    ctor_fn = get_constructor(ctor_name)
                except Exception as e:
                    print(f"Constructeur inconnu {ctor_name}: {e}")
                    writer.writerow({
                        "instance": inst_name, "constructor": ctor_name, "rep": "", "time_s": "",
                        "total_cost": "", "feasible": False, "n_routes": "", "coverage_rate": "", "error": f"constructor_error:{e}"
                    })
                    continue

                for rep in range(1, reps + 1):
                    # Fixe les seeds pour reproductibilité des constructeurs stochastiques
                    seed = rep + 1000
                    random.seed(seed)
                    np.random.seed(seed)

                    evaluator = RouteEvaluator(instance)
                    start = time.perf_counter()
                    try:
                        solution = ctor_fn(instance, evaluator)
                        elapsed = time.perf_counter() - start
                        feasible, total_cost, n_routes, coverage_rate, err = evaluate_solution(solution, instance, evaluator)
                        writer.writerow({
                            "instance": inst_name,
                            "constructor": ctor_name,
                            "rep": rep,
                            "time_s": f"{elapsed:.6f}",
                            "total_cost": f"{total_cost:.6f}",
                            "feasible": str(feasible),
                            "n_routes": n_routes,
                            "coverage_rate": f"{coverage_rate:.4f}",
                            "error": err or ""
                        })
                        print(f"[{inst_name}] {ctor_name} rep{rep}: time={elapsed:.3f}s cost={total_cost:.2f} feasible={feasible} routes={n_routes} cov={coverage_rate:.2f}")
                    except Exception as e:
                        elapsed = time.perf_counter() - start
                        writer.writerow({
                            "instance": inst_name,
                            "constructor": ctor_name,
                            "rep": rep,
                            "time_s": f"{elapsed:.6f}",
                            "total_cost": "",
                            "feasible": False,
                            "n_routes": "",
                            "coverage_rate": 0.0,
                            "error": str(e)
                        })
                        print(f"[{inst_name}] {ctor_name} rep{rep} ERROR: {e}")

    print(f"Résultats écrits dans {out_path}")

if __name__ == "__main__":
    main()