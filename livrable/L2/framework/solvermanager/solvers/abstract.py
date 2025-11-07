"""
AbstractSolver - Classe abstraite pour les solvers VRP - Asha Geyon 2025

Définit l'interface commune pour tous les algorithmes de résolution.
Tous les solvers (LocalSearch, SimulatedAnnealing, GeneticAlgorithm, etc.)
héritent de cette classe et implémentent la méthode solve().

Architecture:
- __init__: Initialisation commune
- solve(): Méthode abstraite à implémenter
- Méthodes helper: _should_stop(), _record_convergence(), etc.

Usage:
    class MyCustomSolver(AbstractSolver):
        def solve(self):
            # Implémentation personnalisée
            return self.best_solution
    
    solver = MyCustomSolver(instance, evaluator, initial_solution, config)
    solution = solver.solve()
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solutionclass import Solution
from routemanager import RouteEvaluator


@dataclass
class SolverConfig:
    """
    Configuration pour un solver.
    
    Attributs communs à tous les solvers:
        max_time: Temps max en secondes (None = illimité)
        max_iterations: Nombre max d'itérations (None = illimité)
        max_iterations_no_improvement: Arrêt si pas d'amélioration (None = désactivé)
        record_convergence: Si True, enregistre l'historique de convergence
        convergence_interval: Enregistre tous les N itérations
        verbose: Si True, affiche la progression
        seed: Graine aléatoire (pour reproductibilité)
    
    Attributs spécifiques peuvent être ajoutés par les solvers enfants.
    """
    # Critères d'arrêt
    max_time: Optional[float] = None
    max_iterations: Optional[int] = None
    max_iterations_no_improvement: Optional[int] = None
    
    # Logging et tracking
    record_convergence: bool = True
    convergence_interval: int = 1
    verbose: bool = False
    
    # Reproductibilité
    seed: Optional[int] = None
    
    # Métadonnées
    name: Optional[str] = None


@dataclass
class ConvergencePoint:
    """Point dans l'historique de convergence."""
    iteration: int
    time_elapsed: float
    cost: float
    n_vehicles: int


class AbstractSolver(ABC):
    """
    Classe abstraite pour tous les solvers VRP.
    
    Fournit l'infrastructure commune:
    - Gestion du temps et des itérations
    - Tracking de la convergence
    - Critères d'arrêt
    - Interface standardisée
    
    Les classes filles doivent implémenter solve().
    """
    
    def __init__(self,
                 instance,
                 evaluator: RouteEvaluator,
                 initial_solution: Solution,
                 config: Optional[SolverConfig] = None):
        """
        Initialise le solver.
        
        Args:
            instance: Instance VRP à résoudre
            evaluator: RouteEvaluator pour vérifier faisabilité
            initial_solution: Solution de départ
            config: Configuration du solver
        """
        self.instance = instance
        self.evaluator = evaluator
        self.initial_solution = initial_solution.copy()
        self.config = config or SolverConfig()
        
        # État du solver
        self.current_solution = initial_solution.copy()
        self.best_solution = initial_solution.copy()
        self.best_cost = initial_solution.total_cost
        
        # Tracking
        self.iteration = 0
        self.iterations_no_improvement = 0
        self.start_time = None
        self.convergence_history: List[ConvergencePoint] = []
        
        # Métriques
        self.total_evaluations = 0
        self.accepted_moves = 0
        self.rejected_moves = 0
    
    @abstractmethod
    def solve(self) -> Solution:
        """
        Résout le problème VRP.
        
        DOIT être implémentée par les classes filles.
        
        Returns:
            Meilleure solution trouvée
        """
        pass
    
    # ========================================================================
    # Méthodes helper pour les classes filles
    # ========================================================================
    
    def _start_solving(self):
        """
        Initialise le solve (appelé au début de solve()).
        
        Démarre le chronomètre et enregistre le point initial.
        """
        self.start_time = time.time()
        self.iteration = 0
        self.iterations_no_improvement = 0
        
        if self.config.record_convergence:
            self._record_convergence()
        
        if self.config.verbose:
            print(f"[{self.__class__.__name__}] Starting solve...")
            print(f"  Initial cost: {self.best_cost:.2f}")
            print(f"  Initial vehicles: {self.best_solution.n_vehicles_used}")
    
    def _should_stop(self) -> bool:
        """
        Vérifie si le solver doit s'arrêter.
        
        Returns:
            True si un critère d'arrêt est atteint
        """
        # Critère de temps
        if self.config.max_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed >= self.config.max_time:
                if self.config.verbose:
                    print(f"\n[Stop] Time limit reached: {elapsed:.2f}s")
                return True
        
        # Critère d'itérations
        if self.config.max_iterations is not None:
            if self.iteration >= self.config.max_iterations:
                if self.config.verbose:
                    print(f"\n[Stop] Iteration limit reached: {self.iteration}")
                return True
        
        # Critère de stagnation
        if self.config.max_iterations_no_improvement is not None:
            if self.iterations_no_improvement >= self.config.max_iterations_no_improvement:
                if self.config.verbose:
                    print(f"\n[Stop] No improvement for {self.iterations_no_improvement} iterations")
                return True
        
        return False
    
    def _update_best(self, new_solution: Solution) -> bool:
        """
        Met à jour la meilleure solution si amélioration.
        
        Args:
            new_solution: Nouvelle solution candidate
            
        Returns:
            True si amélioration
        """
        if new_solution.total_cost < self.best_cost:
            self.best_solution = new_solution.copy()
            self.best_cost = new_solution.total_cost
            self.iterations_no_improvement = 0
            
            if self.config.verbose:
                elapsed = time.time() - self.start_time
                print(f"  [Iter {self.iteration:6d}] New best: {self.best_cost:.2f} "
                      f"(vehicles: {self.best_solution.n_vehicles_used}, "
                      f"time: {elapsed:.1f}s)")
            
            return True
        
        self.iterations_no_improvement += 1
        return False
    
    def _record_convergence(self):
        """Enregistre un point dans l'historique de convergence."""
        if not self.config.record_convergence:
            return
        
        if self.iteration % self.config.convergence_interval == 0:
            point = ConvergencePoint(
                iteration=self.iteration,
                time_elapsed=time.time() - self.start_time,
                cost=self.best_cost,
                n_vehicles=self.best_solution.n_vehicles_used
            )
            self.convergence_history.append(point)
    
    def _finish_solving(self) -> Solution:
        """
        Finalise le solve (appelé à la fin de solve()).
        
        Returns:
            Meilleure solution trouvée
        """
        elapsed = time.time() - self.start_time
        
        if self.config.verbose:
            print(f"\n[{self.__class__.__name__}] Solving finished")
            print(f"  Final cost: {self.best_cost:.2f}")
            print(f"  Iterations: {self.iteration}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Accepted moves: {self.accepted_moves}")
            print(f"  Rejected moves: {self.rejected_moves}")
            print(f"  Acceptance rate: {self.accepted_moves / max(self.iteration, 1) * 100:.1f}%")
        
        # Enregistre le point final si besoin
        if self.config.record_convergence:
            if len(self.convergence_history) == 0 or \
               self.convergence_history[-1].iteration != self.iteration:
                self._record_convergence()
        
        return self.best_solution
    
    def _accept_move(self):
        """Comptabilise un mouvement accepté."""
        self.accepted_moves += 1
    
    def _reject_move(self):
        """Comptabilise un mouvement rejeté."""
        self.rejected_moves += 1
    
    # ========================================================================
    # Méthodes publiques
    # ========================================================================
    
    def get_convergence_history(self) -> List[ConvergencePoint]:
        """
        Retourne l'historique de convergence.
        
        Returns:
            Liste de ConvergencePoint
        """
        return self.convergence_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du solve.
        
        Returns:
            Dictionnaire avec métriques
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            'solver_name': self.__class__.__name__,
            'initial_cost': self.initial_solution.total_cost,
            'final_cost': self.best_cost,
            'improvement': self.initial_solution.total_cost - self.best_cost,
            'improvement_pct': (self.initial_solution.total_cost - self.best_cost) / 
                              self.initial_solution.total_cost * 100,
            'iterations': self.iteration,
            'time_seconds': elapsed,
            'accepted_moves': self.accepted_moves,
            'rejected_moves': self.rejected_moves,
            'acceptance_rate': self.accepted_moves / max(self.iteration, 1),
            'iterations_per_second': self.iteration / max(elapsed, 0.001),
            'n_vehicles': self.best_solution.n_vehicles_used
        }
    
    def get_solution(self) -> Solution:
        """
        Retourne la meilleure solution trouvée.
        
        Returns:
            Meilleure solution
        """
        return self.best_solution
    
    def __repr__(self) -> str:
        """Représentation textuelle du solver."""
        return (f"{self.__class__.__name__}("
                f"instance={self.instance.name}, "
                f"initial_cost={self.initial_solution.total_cost:.2f})")