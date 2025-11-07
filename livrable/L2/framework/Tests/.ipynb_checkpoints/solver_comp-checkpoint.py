#!/usr/bin/env python3
"""
Exemple complet - Comparaison LocalSearch vs SimulatedAnnealing

Teste les deux solvers sur une instance et compare les résultats.

Usage:
    python example_comparison.py
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

# ajoute le dossier parent
sys.path.insert(0, str(project_root))

added = {str(project_root.resolve())}
for p in project_root.rglob('*'):
    try:
        if not p.is_dir():
            continue
        if p.name.startswith('.'):
            continue
        sp = str(p.resolve())
        if sp not in added:
            sys.path.insert(0, sp)
            added.add(sp)
    except (OSError, PermissionError):
        # ignore folders we cannot access
        continue
from solvermanager.solvermanager_exe import SolverManager
from solvermanager.solvers.localsearch import LocalSearchConfig
from solvermanager.solvers.annealing import SimulatedAnnealingConfig


def main():
    """Lance une comparaison complète des solvers."""
    
    print("\n" + "="*70)
    print(" "*15 + "VRP SOLVER COMPARISON")
    print("="*70 + "\n")
    
    # 1. Initialise le manager
    manager = SolverManager()
    
    # 2. Liste les instances disponibles
    instances = manager.list_instances()
    print(f"Available instances: {instances[:5]}...")  # Affiche les 5 premières
    
    # Choisit une petite instance pour le test
    instance_name = "R101" if "R101" in instances else instances[0]
    print(f"Testing on: {instance_name}\n")
    
    # 3. Configure les solvers
    configs = {
        'local_search': LocalSearchConfig(
            name='local_search_first',
            strategy='first_improvement',
            operators=['relocate', '2opt', 'exchange'],
            max_iterations=500000,
            max_iterations_no_improvement=5000,
            verbose=True
        ),
        
        'local_search': LocalSearchConfig(
            name='local_search_best',
            strategy='best_improvement',
            operators=['relocate', '2opt'],
            max_iterations=10000,  # Moins d'itérations car plus lent
            max_iterations_no_improvement=5000,
            verbose=True
        ),
        
        'simulated_annealing': SimulatedAnnealingConfig(
            name='simulated_annealing_hot',
            initial_temperature=2000.0,  # Température élevée
            cooling_rate=0.99999,
            max_iterations=1000000,
            max_iterations_no_improvement=5000,
            verbose=True
        ),
        
        'simulated_annealing': SimulatedAnnealingConfig(
            name='simulated_annealing_cold',
            initial_temperature=500.0,  # Température plus basse
            cooling_rate=0.9999999,
            max_iterations=10000000,
            max_iterations_no_improvement=5000,
            verbose=True
        ),
    }
    
    # 4. Compare les solvers (3 runs chacun)
    print("Running comparison with 3 runs per solver...\n")
    comparison = manager.compare_solvers(
        instance_name=instance_name,
        solver_configs=configs,
        constructor='savings_parallel',
        n_runs=3
    )
    
    # 5. Analyse détaillée
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70 + "\n")
    
    # Trouve le meilleur
    best_solver = min(comparison.items(), key=lambda x: x[1]['avg_cost'])
    print(f"🏆 BEST SOLVER: {best_solver[0]}")
    print(f"   Average cost: {best_solver[1]['avg_cost']:.2f}")
    print(f"   Best cost: {best_solver[1]['best_cost']:.2f}")
    print(f"   Average time: {best_solver[1]['avg_time']:.2f}s\n")
    
    # Compare LocalSearch vs SimulatedAnnealing
    ls_solvers = {k: v for k, v in comparison.items() if 'local_search' in k}
    sa_solvers = {k: v for k, v in comparison.items() if 'simulated_annealing' in k}
    
    if ls_solvers and sa_solvers:
        ls_best = min(ls_solvers.values(), key=lambda x: x['avg_cost'])
        sa_best = min(sa_solvers.values(), key=lambda x: x['avg_cost'])
        
        print("LocalSearch vs SimulatedAnnealing:")
        print(f"  LocalSearch best avg:     {ls_best['avg_cost']:.2f}")
        print(f"  SimulatedAnnealing best avg: {sa_best['avg_cost']:.2f}")
        
        if ls_best['avg_cost'] < sa_best['avg_cost']:
            improvement = (sa_best['avg_cost'] - ls_best['avg_cost']) / sa_best['avg_cost'] * 100
            print(f"  → LocalSearch is {improvement:.1f}% better")
        else:
            improvement = (ls_best['avg_cost'] - sa_best['avg_cost']) / ls_best['avg_cost'] * 100
            print(f"  → SimulatedAnnealing is {improvement:.1f}% better")
    
    print("\n" + "="*70)
    print("Results saved in 'results/' directory")
    print("="*70 + "\n")


def test_single_run():
    """Exemple de run unique avec affichage détaillé."""
    
    print("\n" + "="*70)
    print(" "*20 + "SINGLE RUN EXAMPLE")
    print("="*70 + "\n")
    
    manager = SolverManager()
    
    # LocalSearch avec verbose
    print("Testing LocalSearch with verbose output:\n")
    config_ls = LocalSearchConfig(
        strategy='first_improvement',
        max_iterations=300000,
        max_iterations_no_improvement=5000,
        verbose=True
    )
    
    results_ls = manager.run_experiment(
        instance_name="R101",
        solver_name="local_search",
        config=config_ls,
        constructor="savings_parallel",
        seed=42
    )
    
    # SimulatedAnnealing avec verbose
    print("\nTesting SimulatedAnnealing with verbose output:\n")
    config_sa = SimulatedAnnealingConfig(
        initial_temperature=1000.0,
        cooling_rate=0.995,
        max_iterations=5000,
        verbose=True
    )
    
    results_sa = manager.run_experiment(
        instance_name="R101",
        solver_name="simulated_annealing",
        config=config_sa,
        constructor="savings_parallel",
        seed=42
    )
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"LocalSearch:         {results_ls.cost:.2f} ({results_ls.time_seconds:.2f}s)")
    print(f"SimulatedAnnealing:  {results_sa.cost:.2f} ({results_sa.time_seconds:.2f}s)")
    print("="*70 + "\n")


def test_batch():
    """Exemple de batch sur plusieurs instances."""
    
    print("\n" + "="*70)
    print(" "*20 + "BATCH EXAMPLE")
    print("="*70 + "\n")
    
    manager = SolverManager()
    
    # Sélectionne quelques petites instances
    instances = manager.list_instances()[:3]  # Les 3 premières
    
    configs = {
        'local_search': LocalSearchConfig(
            strategy='first_improvement',
            max_iterations=20000,
            verbose=False
        ),
        'simulated_annealing': SimulatedAnnealingConfig(
            initial_temperature=1000.0,
            max_iterations=30000,
            verbose=False
        )
    }
    
    # Lance le batch avec 2 seeds
    results = manager.run_batch(
        instance_names=instances,
        solver_configs=configs,
        constructor='savings_parallel',
        seeds=[42, 123],
        save_results=True
    )
    
    # Affiche le résumé
    print("\n" + "="*70)
    print("BATCH RESULTS SUMMARY")
    print("="*70)
    for instance_name, instance_results in results.items():
        avg_cost = sum(r.cost for r in instance_results) / len(instance_results)
        print(f"{instance_name}: Avg cost = {avg_cost:.2f} ({len(instance_results)} runs)")
    print("="*70 + "\n")


if __name__ == "__main__":
    
    # Option 1: Comparaison complète
    main()
    
    # Option 2: Run unique avec verbose
    # test_single_run()
    
    # Option 3: Batch sur plusieurs instances
    # test_batch()