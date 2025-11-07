"""
LocalSearchSolver - Recherche locale pour VRP - Asha Geyon 2025

Implémentation de descente par recherche locale.
Utilise les opérateurs relocate, 2-opt, exchange, cross pour améliorer la solution.

Stratégies:
- first_improvement: S'arrête dès qu'une amélioration est trouvée
- best_improvement: Teste tout le voisinage et prend le meilleur
- random: Teste les voisins dans un ordre aléatoire

Usage:
    from local_search_solver import LocalSearchSolver, LocalSearchConfig
    
    config = LocalSearchConfig(
        strategy='first_improvement',
        operators=['relocate', '2opt'],
        max_iterations=10000
    )
    
    solver = LocalSearchSolver(instance, evaluator, initial_solution, config)
    solution = solver.solve()
"""

from dataclasses import dataclass
from typing import Optional, List
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solvers.abstract import AbstractSolver, SolverConfig
from solutionclass import Solution
from routemanager import RouteEvaluator
from operators import (
    try_relocate, try_2opt_intra, try_exchange, 
    try_cross, try_swap_intra
)


@dataclass
class LocalSearchConfig(SolverConfig):
    """
    Configuration pour LocalSearchSolver.
    
    Attributs spécifiques:
        strategy: 'first_improvement' ou 'best_improvement' ou 'random'
        operators: Liste d'opérateurs à utiliser
                  ['relocate', '2opt', 'exchange', 'cross', 'swap']
        neighborhood_size: Pour random, taille du voisinage à explorer
        shuffle_customers: Si True, mélange l'ordre des clients
        shuffle_routes: Si True, mélange l'ordre des routes
    """
    name: str = "local_search"
    strategy: str = 'first_improvement'
    operators: List[str] = None
    neighborhood_size: Optional[int] = None
    shuffle_customers: bool = True
    shuffle_routes: bool = True
    
    def __post_init__(self):
        """Valide et initialise les valeurs par défaut."""
        if self.operators is None:
            self.operators = ['relocate', '2opt', 'exchange']
        
        if self.strategy not in ['first_improvement', 'best_improvement', 'random']:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. "
                f"Use 'first_improvement', 'best_improvement', or 'random'"
            )
        
        valid_operators = ['relocate', '2opt', 'exchange', 'cross', 'swap']
        for op in self.operators:
            if op not in valid_operators:
                raise ValueError(
                    f"Unknown operator: {op}. "
                    f"Valid operators: {valid_operators}"
                )


class LocalSearchSolver(AbstractSolver):
    """
    Solver par recherche locale (descente).
    
    Principe:
    1. Part d'une solution initiale
    2. Explore le voisinage (relocate, 2-opt, exchange, etc.)
    3. Accepte les améliorations
    4. S'arrête à l'optimum local
    
    Stratégies:
    - first_improvement: Accepte la première amélioration trouvée (rapide)
    - best_improvement: Explore tout et prend le meilleur (lent mais mieux)
    - random: Explore aléatoirement (entre les deux)
    """
    
    def __init__(self,
                 instance,
                 evaluator: RouteEvaluator,
                 initial_solution: Solution,
                 config: Optional[LocalSearchConfig] = None):
        """
        Initialise le solver.
        
        Args:
            instance: Instance VRP
            evaluator: RouteEvaluator
            initial_solution: Solution de départ
            config: Configuration (LocalSearchConfig)
        """
        super().__init__(instance, evaluator, initial_solution, config or LocalSearchConfig())
        
        # Vérifie que config est bien LocalSearchConfig
        if not isinstance(self.config, LocalSearchConfig):
            self.config = LocalSearchConfig(**self.config.__dict__)
        
        # Initialise la graine aléatoire si fournie
        if self.config.seed is not None:
            random.seed(self.config.seed)
    
    def solve(self) -> Solution:
        """
        Résout par recherche locale.
        
        Returns:
            Meilleure solution trouvée
        """
        self._start_solving()
        
        while not self._should_stop():
            self.iteration += 1
            
            # Applique la stratégie choisie
            if self.config.strategy == 'first_improvement':
                improved = self._first_improvement_iteration()
            elif self.config.strategy == 'best_improvement':
                improved = self._best_improvement_iteration()
            else:  # random
                improved = self._random_iteration()
            
            # Met à jour la meilleure solution si amélioration
            if improved:
                self._update_best(self.current_solution)
            
            # Enregistre la convergence
            self._record_convergence()
            
            # Si aucune amélioration trouvée → optimum local atteint
            if not improved:
                if self.config.verbose:
                    print(f"  [Iter {self.iteration:6d}] Local optimum reached")
                break
        
        return self._finish_solving()
    
    # ========================================================================
    # Stratégies d'exploration
    # ========================================================================
    
    def _first_improvement_iteration(self) -> bool:
        """
        Stratégie first improvement.
        
        S'arrête dès qu'une amélioration est trouvée.
        
        Returns:
            True si amélioration trouvée
        """
        # Pour chaque opérateur
        for operator in self.config.operators:
            if operator == 'relocate':
                if self._try_all_relocate_first():
                    return True
            
            elif operator == '2opt':
                if self._try_all_2opt_first():
                    return True
            
            elif operator == 'exchange':
                if self._try_all_exchange_first():
                    return True
            
            elif operator == 'cross':
                if self._try_all_cross_first():
                    return True
            
            elif operator == 'swap':
                if self._try_all_swap_first():
                    return True
        
        return False
    
    def _best_improvement_iteration(self) -> bool:
        """
        Stratégie best improvement.
        
        Explore tout le voisinage et prend la meilleure amélioration.
        
        Returns:
            True si amélioration trouvée
        """
        best_move = None
        best_delta = 0
        best_operator = None
        
        # Pour chaque opérateur
        for operator in self.config.operators:
            if operator == 'relocate':
                move, delta = self._find_best_relocate()
                if delta < best_delta:
                    best_delta = delta
                    best_move = move
                    best_operator = 'relocate'
            
            elif operator == '2opt':
                move, delta = self._find_best_2opt()
                if delta < best_delta:
                    best_delta = delta
                    best_move = move
                    best_operator = '2opt'
            
            elif operator == 'exchange':
                move, delta = self._find_best_exchange()
                if delta < best_delta:
                    best_delta = delta
                    best_move = move
                    best_operator = 'exchange'
        
        # Applique le meilleur mouvement
        if best_move is not None:
            self._apply_move(best_operator, best_move)
            self._accept_move()
            return True
        
        return False
    
    def _random_iteration(self) -> bool:
        """
        Stratégie random.
        
        Explore un sous-ensemble aléatoire du voisinage.
        
        Returns:
            True si amélioration trouvée
        """
        # Détermine la taille du voisinage à explorer
        if self.config.neighborhood_size is None:
            n_customers = self.current_solution.get_n_customers()
            neighborhood_size = n_customers * 2
        else:
            neighborhood_size = self.config.neighborhood_size
        
        # Sélectionne un opérateur aléatoire et essaie plusieurs mouvements
        for _ in range(neighborhood_size):
            operator = random.choice(self.config.operators)
            
            if operator == 'relocate':
                if self._try_random_relocate():
                    return True
            elif operator == '2opt':
                if self._try_random_2opt():
                    return True
            elif operator == 'exchange':
                if self._try_random_exchange():
                    return True
        
        return False
    
    # ========================================================================
    # Implémentations des opérateurs - First Improvement
    # ========================================================================
    
    def _try_all_relocate_first(self) -> bool:
        """Essaie relocate sur tous les clients (first improvement)."""
        customers = list(self.current_solution.get_customers())
        
        if self.config.shuffle_customers:
            random.shuffle(customers)
        
        for customer in customers:
            from_route_idx = self.current_solution.get_route_of_customer(customer)
            
            routes = list(range(self.current_solution.get_n_routes()))
            if self.config.shuffle_routes:
                random.shuffle(routes)
            
            for to_route_idx in routes:
                to_route = self.current_solution.routes[to_route_idx]
                
                for position in range(len(to_route) + 1):
                    if try_relocate(self.current_solution, self.evaluator,
                                  customer, from_route_idx, to_route_idx, position):
                        self._accept_move()
                        return True
        
        return False
    
    def _try_all_2opt_first(self) -> bool:
        """Essaie 2-opt sur toutes les routes (first improvement)."""
        routes = list(range(self.current_solution.get_n_routes()))
        
        if self.config.shuffle_routes:
            random.shuffle(routes)
        
        for route_idx in routes:
            route = self.current_solution.routes[route_idx]
            
            if len(route) < 3:
                continue
            
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route) + 1):
                    if try_2opt_intra(self.current_solution, self.evaluator,
                                    route_idx, i, j):
                        self._accept_move()
                        return True
        
        return False
    
    def _try_all_exchange_first(self) -> bool:
        """Essaie exchange sur toutes les paires de clients (first improvement)."""
        customers = list(self.current_solution.get_customers())
        
        if self.config.shuffle_customers:
            random.shuffle(customers)
        
        for i, customer1 in enumerate(customers):
            for customer2 in customers[i+1:]:
                if try_exchange(self.current_solution, self.evaluator,
                              customer1, customer2):
                    self._accept_move()
                    return True
        
        return False
    
    def _try_all_cross_first(self) -> bool:
        """Essaie cross sur toutes les paires de routes (first improvement)."""
        n_routes = self.current_solution.get_n_routes()
        
        for route1_idx in range(n_routes):
            for route2_idx in range(route1_idx + 1, n_routes):
                route1 = self.current_solution.routes[route1_idx]
                route2 = self.current_solution.routes[route2_idx]
                
                for pos1 in range(len(route1)):
                    for pos2 in range(len(route2)):
                        if try_cross(self.current_solution, self.evaluator,
                                   route1_idx, route2_idx, pos1, pos2):
                            self._accept_move()
                            return True
        
        return False
    
    def _try_all_swap_first(self) -> bool:
        """Essaie swap intra-route (first improvement)."""
        routes = list(range(self.current_solution.get_n_routes()))
        
        if self.config.shuffle_routes:
            random.shuffle(routes)
        
        for route_idx in routes:
            route = self.current_solution.routes[route_idx]
            
            if len(route) < 2:
                continue
            
            for i in range(len(route)):
                for j in range(i + 1, len(route)):
                    if try_swap_intra(self.current_solution, self.evaluator,
                                    route_idx, i, j):
                        self._accept_move()
                        return True
        
        return False
    
    # ========================================================================
    # Implémentations des opérateurs - Best Improvement
    # ========================================================================
    
    def _find_best_relocate(self):
        """Trouve le meilleur relocate."""
        best_move = None
        best_delta = 0
        
        for customer in self.current_solution.get_customers():
            from_route_idx = self.current_solution.get_route_of_customer(customer)
            old_from_cost = self.current_solution.route_costs[from_route_idx]
            
            for to_route_idx in range(self.current_solution.get_n_routes()):
                to_route = self.current_solution.routes[to_route_idx]
                old_to_cost = self.current_solution.route_costs[to_route_idx]
                
                for position in range(len(to_route) + 1):
                    # Simule le mouvement
                    from_route = [c for c in self.current_solution.routes[from_route_idx] 
                                 if c != customer]
                    new_to_route = to_route.copy()
                    new_to_route.insert(position, customer)
                    
                    from_feasible, from_cost = self.evaluator.evaluate_route_fast(from_route)
                    to_feasible, to_cost = self.evaluator.evaluate_route_fast(new_to_route)
                    
                    if not (from_feasible and to_feasible):
                        continue
                    
                    if to_route_idx == from_route_idx:
                        delta = to_cost - old_from_cost
                    else:
                        delta = (from_cost + to_cost) - (old_from_cost + old_to_cost)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (customer, from_route_idx, to_route_idx, position)
        
        return best_move, best_delta
    
    def _find_best_2opt(self):
        """Trouve le meilleur 2-opt."""
        best_move = None
        best_delta = 0
        
        for route_idx in range(self.current_solution.get_n_routes()):
            route = self.current_solution.routes[route_idx]
            old_cost = self.current_solution.route_costs[route_idx]
            
            if len(route) < 3:
                continue
            
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route) + 1):
                    new_route = route.copy()
                    new_route[i+1:j] = reversed(new_route[i+1:j])
                    
                    is_feasible, new_cost = self.evaluator.evaluate_route_fast(new_route)
                    
                    if not is_feasible:
                        continue
                    
                    delta = new_cost - old_cost
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (route_idx, i, j)
        
        return best_move, best_delta
    
    def _find_best_exchange(self):
        """Trouve le meilleur exchange."""
        best_move = None
        best_delta = 0
        
        customers = list(self.current_solution.get_customers())
        
        for i, customer1 in enumerate(customers):
            route1_idx = self.current_solution.get_route_of_customer(customer1)
            pos1 = self.current_solution.get_position_of_customer(customer1)
            
            for customer2 in customers[i+1:]:
                route2_idx = self.current_solution.get_route_of_customer(customer2)
                pos2 = self.current_solution.get_position_of_customer(customer2)
                
                if route1_idx == route2_idx:
                    continue
                
                # Simule l'échange
                route1 = self.current_solution.routes[route1_idx].copy()
                route2 = self.current_solution.routes[route2_idx].copy()
                
                route1[pos1] = customer2
                route2[pos2] = customer1
                
                feasible1, cost1 = self.evaluator.evaluate_route_fast(route1)
                feasible2, cost2 = self.evaluator.evaluate_route_fast(route2)
                
                if not (feasible1 and feasible2):
                    continue
                
                old_cost = (self.current_solution.route_costs[route1_idx] + 
                           self.current_solution.route_costs[route2_idx])
                new_cost = cost1 + cost2
                delta = new_cost - old_cost
                
                if delta < best_delta:
                    best_delta = delta
                    best_move = (customer1, customer2)
        
        return best_move, best_delta
    
    def _apply_move(self, operator: str, move):
        """Applique un mouvement."""
        if operator == 'relocate':
            customer, from_idx, to_idx, position = move
            try_relocate(self.current_solution, self.evaluator,
                        customer, from_idx, to_idx, position)
        
        elif operator == '2opt':
            route_idx, i, j = move
            try_2opt_intra(self.current_solution, self.evaluator,
                          route_idx, i, j)
        
        elif operator == 'exchange':
            customer1, customer2 = move
            try_exchange(self.current_solution, self.evaluator,
                        customer1, customer2)
    
    # ========================================================================
    # Implémentations des opérateurs - Random
    # ========================================================================
    
    def _try_random_relocate(self) -> bool:
        """Essaie un relocate aléatoire."""
        customers = list(self.current_solution.get_customers())
        if not customers:
            return False
        
        customer = random.choice(customers)
        from_route_idx = self.current_solution.get_route_of_customer(customer)
        to_route_idx = random.randint(0, self.current_solution.get_n_routes() - 1)
        to_route = self.current_solution.routes[to_route_idx]
        position = random.randint(0, len(to_route))
        
        if try_relocate(self.current_solution, self.evaluator,
                       customer, from_route_idx, to_route_idx, position):
            self._accept_move()
            return True
        
        return False
    
    def _try_random_2opt(self) -> bool:
        """Essaie un 2-opt aléatoire."""
        if self.current_solution.get_n_routes() == 0:
            return False
        
        route_idx = random.randint(0, self.current_solution.get_n_routes() - 1)
        route = self.current_solution.routes[route_idx]
        
        if len(route) < 3:
            return False
        
        i = random.randint(0, len(route) - 2)
        j = random.randint(i + 2, len(route))
        
        if try_2opt_intra(self.current_solution, self.evaluator, route_idx, i, j):
            self._accept_move()
            return True
        
        return False
    
    def _try_random_exchange(self) -> bool:
        """Essaie un exchange aléatoire."""
        customers = list(self.current_solution.get_customers())
        if len(customers) < 2:
            return False
        
        customer1, customer2 = random.sample(customers, 2)
        
        if try_exchange(self.current_solution, self.evaluator, customer1, customer2):
            self._accept_move()
            return True
        
        return False