"""
SimulatedAnnealingSolver - Recuit simulé pour VRP - Asha Geyon 2025

Implémentation du recuit simulé (Simulated Annealing) pour VRP.
Accepte des mouvements détériorants avec une probabilité qui décroît
avec la température.

Principe:
1. Part d'une température élevée
2. Génère des voisins aléatoires
3. Accepte toujours les améliorations
4. Accepte les détériorations selon critère de Metropolis: P = exp(-delta/T)
5. Refroidit progressivement
6. S'arrête quand température trop basse

CORRECTIONS APPLIQUÉES :
- Bug copie : Ajout de .copy() lors de l'acceptation des mouvements
- Bug position : Gestion correcte du cas "même route" dans relocate
- Validation : Vérification que les neighbors sont valides
- Paramètres : Valeurs par défaut plus appropriées

Usage:
    from simulated_annealing_solver import SimulatedAnnealingSolver, SimulatedAnnealingConfig
    
    config = SimulatedAnnealingConfig(
        initial_temperature=2000.0,
        cooling_rate=0.999,
        max_iterations=50000
    )
    
    solver = SimulatedAnnealingSolver(instance, evaluator, initial_solution, config)
    solution = solver.solve()
"""

from dataclasses import dataclass
from typing import Optional
import random
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solvers.abstract import AbstractSolver, SolverConfig
from solutionclass import Solution
from routemanager import RouteEvaluator
from operators import try_relocate, try_2opt_intra, try_exchange


@dataclass
class SimulatedAnnealingConfig(SolverConfig):
    """
    Configuration pour SimulatedAnnealingSolver.
    
    Attributs spécifiques:
        initial_temperature: Température initiale (plus élevée = plus d'exploration)
        cooling_rate: Taux de refroidissement (0 < cooling_rate < 1)
                     Typique: 0.99-0.999
        min_temperature: Température minimale (critère d'arrêt)
        reheating: Si True, réchauffe périodiquement
        reheating_interval: Nombre d'itérations entre réchauffages
        reheating_factor: Facteur de réchauffage (T *= factor)
        validate_neighbors: Si True, rejette les voisins invalides
    """
    name: str = "simulated_annealing"
    initial_temperature: float = 2000.0  # ← Augmenté (était 1000.0)
    cooling_rate: float = 0.999  # ← Plus lent (était 0.995)
    min_temperature: float = 0.001  # ← Plus bas (était 0.01)
    reheating: bool = False
    reheating_interval: int = 1000
    reheating_factor: float = 1.5
    validate_neighbors: bool = True  # ← NOUVEAU : validation des voisins


class SimulatedAnnealingSolver(AbstractSolver):
    """
    Solver par recuit simulé (Simulated Annealing).
    
    Principe du recuit simulé:
    - Température élevée → accepte beaucoup de détériorations (exploration)
    - Température basse → accepte peu de détériorations (exploitation)
    - Permet d'échapper aux optima locaux
    
    Critère d'acceptation de Metropolis:
    - Si delta < 0 (amélioration) → toujours accepter
    - Si delta > 0 (détérioration) → accepter avec probabilité exp(-delta/T)
    """
    
    def __init__(self,
                 instance,
                 evaluator: RouteEvaluator,
                 initial_solution: Solution,
                 config: Optional[SimulatedAnnealingConfig] = None):
        """
        Initialise le solver.
        
        Args:
            instance: Instance VRP
            evaluator: RouteEvaluator
            initial_solution: Solution de départ
            config: Configuration (SimulatedAnnealingConfig)
        """
        super().__init__(instance, evaluator, initial_solution, 
                        config or SimulatedAnnealingConfig())
        
        # Vérifie que config est bien SimulatedAnnealingConfig
        if not isinstance(self.config, SimulatedAnnealingConfig):
            self.config = SimulatedAnnealingConfig(**self.config.__dict__)
        
        # État du recuit
        self.temperature = self.config.initial_temperature
        
        # Statistiques supplémentaires
        self.invalid_neighbors = 0
        
        # Initialise la graine aléatoire si fournie
        if self.config.seed is not None:
            random.seed(self.config.seed)
    
    def solve(self) -> Solution:
        """
        Résout par recuit simulé.
        
        Returns:
            Meilleure solution trouvée
        """
        self._start_solving()
        
        while not self._should_stop():
            self.iteration += 1
            
            # Génère un voisin aléatoire
            neighbor = self._generate_random_neighbor()
            
            # Si neighbor est None (invalide), on continue
            if neighbor is None:
                self.invalid_neighbors += 1
                continue
            
            # Calcule le delta de coût
            delta = neighbor.total_cost - self.current_solution.total_cost
            
            # Critère d'acceptation de Metropolis
            if delta < 0:
                # Amélioration → toujours accepter
                self.current_solution = neighbor.copy()  # ← FIX : .copy() ajouté
                self._accept_move()
                self._update_best(self.current_solution)
            elif random.random() < math.exp(-delta / self.temperature):
                # Détérioration → accepter selon probabilité
                self.current_solution = neighbor.copy()  # ← FIX : .copy() ajouté
                self._accept_move()
            else:
                # Rejeter
                self._reject_move()
            
            # Refroidissement
            self.temperature *= self.config.cooling_rate
            
            # Réchauffage périodique (optionnel)
            if self.config.reheating:
                if self.iteration % self.config.reheating_interval == 0:
                    self.temperature *= self.config.reheating_factor
                    if self.config.verbose:
                        print(f"  [Iter {self.iteration:6d}] Reheating: T={self.temperature:.2f}")
            
            # Arrêt si température trop basse
            if self.temperature < self.config.min_temperature:
                if self.config.verbose:
                    print(f"  [Iter {self.iteration:6d}] Min temperature reached")
                break
            
            # Enregistre convergence
            self._record_convergence()
            
            # Affichage périodique
            if self.config.verbose and self.iteration % 1000 == 0:
                print(f"  [Iter {self.iteration:6d}] T={self.temperature:.2f}, "
                      f"Current={self.current_solution.total_cost:.2f}, "
                      f"Best={self.best_cost:.2f}")
        
        return self._finish_solving()
    
    def _generate_random_neighbor(self) -> Optional[Solution]:
        """
        Génère un voisin aléatoire en appliquant un opérateur aléatoire.
        
        Returns:
            Solution voisine (ou None si invalide)
        """
        neighbor = self.current_solution.copy()
        
        # Choisit un opérateur aléatoire
        operator = random.choice(['relocate', '2opt', 'exchange'])
        
        if operator == 'relocate':
            self._try_random_relocate(neighbor)
        elif operator == '2opt':
            self._try_random_2opt(neighbor)
        elif operator == 'exchange':
            self._try_random_exchange(neighbor)
        
        # ✅ VALIDATION : vérifie que le voisin est valide
        if self.config.validate_neighbors:
            is_valid, error = neighbor.is_valid_vrp(self.instance, self.evaluator)
            if not is_valid:
                return None  # Rejette le voisin invalide
        
        return neighbor
    
    def _try_random_relocate(self, solution: Solution):
        """
        Essaie un relocate aléatoire.
        
        ✅ FIX : Gestion correcte du cas "même route"
        """
        customers = list(solution.get_customers())
        if not customers:
            return
        
        customer = random.choice(customers)
        from_route_idx = solution.get_route_of_customer(customer)
        to_route_idx = random.randint(0, solution.get_n_routes() - 1)
        to_route = solution.routes[to_route_idx]
        
        # ✅ CORRECTION : Calcul correct de la position max
        if from_route_idx == to_route_idx:
            # Même route : le client sera retiré d'abord, donc max = len - 1
            max_position = len(to_route) - 1
            if max_position < 0:
                return  # Route d'un seul client, skip
            position = random.randint(0, max_position)
        else:
            # Routes différentes : position peut aller jusqu'à len(to_route)
            position = random.randint(0, len(to_route))
        
        # Accepte même si pas d'amélioration (pour SA)
        try_relocate(solution, self.evaluator, customer, 
                    from_route_idx, to_route_idx, position, accept_equal=True)
    
    def _try_random_2opt(self, solution: Solution):
        """Essaie un 2-opt aléatoire."""
        if solution.get_n_routes() == 0:
            return
        
        route_idx = random.randint(0, solution.get_n_routes() - 1)
        route = solution.routes[route_idx]
        
        if len(route) < 3:
            return
        
        i = random.randint(0, len(route) - 2)
        j = random.randint(i + 2, len(route))
        
        # Accepte même si pas d'amélioration
        try_2opt_intra(solution, self.evaluator, route_idx, i, j, accept_equal=True)
    
    def _try_random_exchange(self, solution: Solution):
        """Essaie un exchange aléatoire."""
        customers = list(solution.get_customers())
        if len(customers) < 2:
            return
        
        customer1, customer2 = random.sample(customers, 2)
        
        # Vérifie qu'ils sont dans des routes différentes
        route1 = solution.get_route_of_customer(customer1)
        route2 = solution.get_route_of_customer(customer2)
        
        if route1 != route2:
            try_exchange(solution, self.evaluator, customer1, customer2, accept_equal=True)
    
    def get_statistics(self):
        """Retourne les statistiques avec infos sur température."""
        stats = super().get_statistics()
        stats['final_temperature'] = self.temperature
        stats['initial_temperature'] = self.config.initial_temperature
        stats['cooling_rate'] = self.config.cooling_rate
        stats['invalid_neighbors'] = self.invalid_neighbors
        if self.iteration > 0:
            stats['invalid_neighbor_rate'] = self.invalid_neighbors / self.iteration
        return stats