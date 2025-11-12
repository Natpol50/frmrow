"""
SolverManager - Orchestrateur d'expériences VRP - Asha Geyon 2025

Gère le pipeline complet pour lancer des expériences VRP:
1. Charge l'instance
2. Crée l'évaluateur
3. Construit la solution initiale
4. Lance le solver
5. Sauvegarde les résultats

Usage:
    from solver_manager import SolverManager
    from local_search_solver import LocalSearchConfig
    
    manager = SolverManager(data_dir="data", results_dir="results")
    
    config = LocalSearchConfig(max_iterations=10000, verbose=True)
    results = manager.run_experiment(
        instance_name="C101",
        solver_class="local_search",
        config=config,
        constructor="nearest_neighbor",
        seed=42
    )
"""


from typing import Optional, Dict, Any, List
from dataclasses import asdict
import time
import sys
from pathlib import Path
import json
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Imports des modules du framework
from filemanagers.instancefilemanager import InstanceFileManager
from routemanager import RouteEvaluator
from solutionclass import Solution
import constructors
from filemanagers.runfilemanager import RunFileManager, Config, Results, ConvergencePoint

# Imports des solvers
from solvers.localsearch import LocalSearchSolver, LocalSearchConfig
from solvers.annealing import SimulatedAnnealingSolver, SimulatedAnnealingConfig
from solvers.alns import ALNSSolver, ALNSConfig

class SolverManager:
    """
    Orchestrateur pour lancer des expériences VRP.
    
    Gère le pipeline complet:
    - Chargement d'instance
    - Construction de solution initiale
    - Résolution avec solver
    - Sauvegarde des résultats
    
    Supporte plusieurs solvers et constructeurs.
    """
    
    # Registry des solvers disponibles
    SOLVERS = {
        'local_search': (LocalSearchSolver, LocalSearchConfig),
        'simulated_annealing': (SimulatedAnnealingSolver, SimulatedAnnealingConfig),
        'alns': (ALNSSolver, ALNSConfig),
    }
    
    def __init__(self, 
                 data_dir: str = "/data",
                 results_dir: str = "/results"):
        """
        Initialise le manager.
        
        Args:
            data_dir: Répertoire des instances
            results_dir: Répertoire des résultats
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Crée les managers
        self.instance_manager = InstanceFileManager(data_dir)
        self.run_manager = RunFileManager(results_dir)
        
        # Cache pour éviter de recharger les instances
        self._instance_cache = {}
        self._evaluator_cache = {}
    
    def run_experiment(self,
                      instance_name: str,
                      solver_name: str,
                      config: Any,
                      constructor: str = "nearest_neighbor",
                      seed: Optional[int] = None,
                      save_results: bool = True,
                      force_recompute: bool = False) -> Results:
        """
        Lance UNE expérience complète.

        Args:
            instance_name: Nom de l'instance (ex: "C101")
            solver_name: Nom du solver ('local_search', 'simulated_annealing')
            config: Configuration du solver (LocalSearchConfig, etc.)
            constructor: Nom du constructeur ('nearest_neighbor', 'savings', etc.)
            seed: Graine aléatoire (pour reproductibilité)
            save_results: Si True, sauvegarde dans RunFileManager
            force_recompute: Si True, force la recomputation même si un run existant est présent

        Returns:
            Results avec tous les résultats

        Raises:
            ValueError: Si solver_name ou constructor inconnu
        """
        print(f"\n{'='*60}")
        print(f"Experiment: {instance_name} + {solver_name}")
        print(f"{'='*60}")

        # 1. Charge l'instance
        print(f"[1/6] Loading instance '{instance_name}'...")
        instance = self._get_instance(instance_name)
        print(f"      → {instance.dimension-1} clients, capacity={instance.capacity}")

        # 2. Crée l'évaluateur
        print(f"[2/6] Creating evaluator...")
        evaluator = self._get_evaluator(instance_name, instance)

        # 3. Construit la solution initiale
        print(f"[3/6] Building initial solution with '{constructor}'...")
        initial_solution = self._construct_initial_solution(
            instance, evaluator, constructor, seed
        )
        print(f"      → Cost: {initial_solution.total_cost:.2f}, "
              f"Vehicles: {initial_solution.n_vehicles_used}")

        # Prepare run config (used to identify saved runs)
        run_config = Config(
            instance_name=instance_name,
            solver_name=solver_name,
            seed=seed or 0,
            parameters=asdict(config)
        )

        # If not forcing recompute, try to load existing results
        if not force_recompute:
            try:
                existing_run = self.run_manager.load_run(run_config)
                existing_results = existing_run.results  # ← Extrais juste la partie results
            except FileNotFoundError:
                existing_results = None
                print(f"[LOAD] No existing results found.")
        
            if existing_results:
                print(f"[LOAD] Found existing results — loaded")
                print(f"{'='*60}\n")
                return existing_results

        # 4. Crée et lance le solver
        print(f"[4/6] Solving with '{solver_name}'...")
        start_time = time.time()

        solver = self._create_solver(
            solver_name, instance, evaluator, initial_solution, config
        )

        best_solution = solver.solve()
        elapsed_time = time.time() - start_time

        print(f"      → Cost: {best_solution.total_cost:.2f}, "
              f"Vehicles: {best_solution.n_vehicles_used}")
        print(f"      → Time: {elapsed_time:.2f}s")
        print(f"      → Improvement: {initial_solution.total_cost - best_solution.total_cost:.2f} "
              f"({(initial_solution.total_cost - best_solution.total_cost) / initial_solution.total_cost * 100:.1f}%)")

        print(f"[5/6] Verifying full VRP solution...")
        is_valid, message = best_solution.is_valid_vrp(instance, evaluator)
        if is_valid:
            print(f"      → Solution is VALID")
        else:
            print(f"      → Solution is INVALID: {message}")

        # 5. Prépare les résultats
        print(f"[6/6] Preparing results...")

        # Convertit convergence
        convergence = []
        for point in solver.get_convergence_history():
            convergence.append(ConvergencePoint(
                iteration=point.iteration,
                cost=point.cost
            ))

        results = Results(
            time_seconds=elapsed_time,
            n_iterations=getattr(solver, 'iteration', 0),
            cost=best_solution.total_cost,
            solution=best_solution.routes,
            convergence=convergence,
            additional_info={
                'initial_cost': initial_solution.total_cost,
                'n_vehicles': best_solution.n_vehicles_used,
                'improvement': initial_solution.total_cost - best_solution.total_cost,
                'improvement_pct': (initial_solution.total_cost - best_solution.total_cost)
                                  / initial_solution.total_cost * 100,
                'constructor': constructor,
                **(solver.get_statistics() if hasattr(solver, 'get_statistics') else {})
            }
        )

        # Sauvegarde si demandé
        if save_results:
            try:
                self.run_manager.add_run(run_config, results)
                print(f"      → Results saved")
            except Exception as e:
                print(f"      → Failed to save results: {e}")
        else:
            print(f"      → Results NOT saved (save_results=False)")

        print(f"{'='*60}\n")

        return results
    
    def run_batch(self,
                 instance_names: List[str],
                 solver_configs: Dict[str, Any],
                 constructor: str = "nearest_neighbor",
                 seeds: Optional[List[int]] = None,
                 save_results: bool = True) -> Dict[str, List[Results]]:
        """
        Lance un batch d'expériences.
        
        Args:
            instance_names: Liste d'instances à tester
            solver_configs: Dict {solver_name: config}
            constructor: Constructeur à utiliser
            seeds: Liste de graines (None = une seule run sans seed)
            save_results: Si True, sauvegarde les résultats
            
        Returns:
            Dict {instance_name: [results]}
        """
        if seeds is None:
            seeds = [None]
        
        all_results = {}
        
        total_runs = len(instance_names) * len(solver_configs) * len(seeds)
        current_run = 0
        
        print(f"\n{'='*60}")
        print(f"BATCH EXPERIMENT")
        print(f"Instances: {len(instance_names)}")
        print(f"Solvers: {len(solver_configs)}")
        print(f"Seeds: {len(seeds)}")
        print(f"Total runs: {total_runs}")
        print(f"{'='*60}\n")
        
        for instance_name in instance_names:
            all_results[instance_name] = []
            
            for solver_name, config in solver_configs.items():
                for seed in seeds:
                    current_run += 1
                    print(f"[Run {current_run}/{total_runs}]")
                    
                    try:
                        results = self.run_experiment(
                            instance_name=instance_name,
                            solver_name=solver_name,
                            config=config,
                            constructor=constructor,
                            seed=seed,
                            save_results=save_results
                        )
                        all_results[instance_name].append(results)
                    
                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue
        
        return all_results
    
    def compare_solvers(self,
                       instance_name: str,
                       solver_configs: Dict[str, Any],
                       constructor: str = "nearest_neighbor",
                       n_runs: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Compare plusieurs solvers sur une instance.
        
        Args:
            instance_name: Instance à tester
            solver_configs: Dict {solver_name: config}
            constructor: Constructeur à utiliser
            n_runs: Nombre de runs par solver
            
        Returns:
            Dict {solver_name: stats}
        """
        print(f"\n{'='*60}")
        print(f"SOLVER COMPARISON on {instance_name}")
        print(f"{'='*60}\n")
        
        comparison = {}
        
        for solver_name, config in solver_configs.items():
            print(f"Testing {solver_name}...")
            
            costs = []
            times = []
            
            for run in range(n_runs):
                results = self.run_experiment(
                    instance_name=instance_name,
                    solver_name=solver_name,
                    config=config,
                    constructor=constructor,
                    seed=run,
                    save_results=False
                )
                
                costs.append(results.cost)
                times.append(results.time_seconds)
            
            comparison[solver_name] = {
                'avg_cost': sum(costs) / len(costs),
                'best_cost': min(costs),
                'worst_cost': max(costs),
                'avg_time': sum(times) / len(times),
                'n_runs': n_runs
            }
        
        # Affiche le résumé
        print(f"\n{'='*60}")
        print(f"COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"{'Solver':<25s} {'Avg Cost':>12s} {'Best':>12s} {'Worst':>12s} {'Avg Time':>10s}")
        print(f"{'-'*60}")
        
        for solver_name, stats in comparison.items():
            print(f"{solver_name:<25s} "
                  f"{stats['avg_cost']:>12.2f} "
                  f"{stats['best_cost']:>12.2f} "
                  f"{stats['worst_cost']:>12.2f} "
                  f"{stats['avg_time']:>10.2f}s")
        
        print(f"{'='*60}\n")
        
        return comparison
    
    # ========================================================================
    # Méthodes internes
    # ========================================================================
    
    def _get_instance(self, instance_name: str):
        """Charge l'instance (avec cache)."""
        if instance_name not in self._instance_cache:
            self._instance_cache[instance_name] = \
                self.instance_manager.load_instance(instance_name)
        return self._instance_cache[instance_name]
    
    def _get_evaluator(self, instance_name: str, instance):
        """Crée l'évaluateur (avec cache)."""
        if instance_name not in self._evaluator_cache:
            self._evaluator_cache[instance_name] = RouteEvaluator(instance)
        return self._evaluator_cache[instance_name]
    
    def _construct_initial_solution(self, instance, evaluator, constructor_name, seed):
        """Construit la solution initiale."""
        constructor = constructors.get_constructor(constructor_name)
        
        # Applique le seed si fourni et si c'est random_constructor
        if constructor_name == 'random' and seed is not None:
            return constructor(instance, evaluator, seed=seed)
        else:
            return constructor(instance, evaluator)
    
    def _create_solver(self, solver_name, instance, evaluator, initial_solution, config):
        """Crée le solver."""
        if solver_name not in self.SOLVERS:
            raise ValueError(
                f"Unknown solver: {solver_name}. "
                f"Available: {list(self.SOLVERS.keys())}"
            )
        
        solver_class, config_class = self.SOLVERS[solver_name]
        
        # Vérifie que config est du bon type

        
        return solver_class(instance, evaluator, initial_solution, config)
    
    def list_instances(self) -> List[str]:
        """Liste les instances disponibles."""
        # Charge l'index s'il existe, sinon scanne
        instances = self.instance_manager.scan_instances()
        
        # Si scan_instances retourne None (index existe déjà),
        # charge l'index manuellement
        if instances is None:
            index_path = Path(self.data_dir) / "instance_index.json"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                    instances = list(index_data.keys())
            else:
                # Force le rescan si vraiment aucun index
                instances = self.instance_manager.scan_instances(force_rescan=True)
        
        return instances if instances is not None else []
    
    def list_solvers(self) -> List[str]:
        """Liste les solvers disponibles."""
        return list(self.SOLVERS.keys())