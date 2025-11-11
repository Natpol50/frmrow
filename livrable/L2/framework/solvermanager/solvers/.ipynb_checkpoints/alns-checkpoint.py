"""
ALNSSolver - Adaptive Large Neighborhood Search pour VRP - Asha Geyon 2025

Implémentation de l'ALNS (Adaptive Large Neighborhood Search) pour VRP.
Combine destruction/reconstruction avec apprentissage adaptatif des opérateurs.

Principe:
1. Sélectionne un opérateur destroy selon les poids adaptatifs
2. Sélectionne un opérateur repair selon les poids adaptatifs
3. Applique : nouvelle_solution = repair(destroy(solution_courante))
4. Accepte/rejette selon critère (type Simulated Annealing)
5. Met à jour les poids selon la performance des opérateurs
6. Répète jusqu'au critère d'arrêt

Mécanisme adaptatif:
- Chaque opérateur a un poids qui évolue
- Les bons opérateurs voient leur poids augmenter (intensification)
- Tous gardent une probabilité minimale (diversification)
- Roulette wheel selection pour choisir

Référence:
Ropke, S., & Pisinger, D. (2006). An Adaptive Large Neighborhood Search 
Heuristic for the Pickup and Delivery Problem with Time Windows.

Usage:
    from alns import ALNSSolver, ALNSConfig
    
    config = ALNSConfig(
        destroy_operators=['random', 'worst', 'shaw'],
        repair_operators=['greedy', 'regret'],
        max_iterations=10000
    )
    
    solver = ALNSSolver(instance, evaluator, initial_solution, config)
    solution = solver.solve()
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
import random
import math
import numpy as np
import time
import sys
from pathlib import Path

# Imports framework
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solvermanager.solvers.abstract import AbstractSolver, SolverConfig
from solvermanager.solutionclass import Solution
from solvermanager.routemanager import RouteEvaluator


@dataclass
class ALNSConfig(SolverConfig):
    """
    Configuration pour ALNSSolver.
    
    Paramètres destroy/repair:
        destroy_operators: Liste des opérateurs de destruction à utiliser
                          ['random', 'worst', 'shaw', 'route']
        repair_operators: Liste des opérateurs de réparation à utiliser
                         ['greedy', 'regret2', 'regret3']
        min_destroy_size: Nombre minimum de clients à retirer
        max_destroy_size: Nombre maximum de clients à retirer
        destroy_size_ratio: Ratio de clients à retirer (si min/max non spécifiés)
    
    Paramètres adaptatifs:
        initial_weight: Poids initial de chaque opérateur
        weight_decay: Facteur de décroissance des poids (0 < decay < 1)
        weight_update_interval: Nombre d'itérations entre mises à jour
        score_new_best: Score si nouvelle meilleure solution globale
        score_better: Score si amélioration de la solution courante
        score_accepted: Score si solution acceptée (sans amélioration)
        score_rejected: Score si solution rejetée
    
    Paramètres acceptation:
        initial_temperature: Température initiale (type SA)
        cooling_rate: Taux de refroidissement
        min_temperature: Température minimale
        
    Paramètres Shaw removal:
        shaw_relatedness_weight_distance: Poids pour distance dans relatedness
        shaw_relatedness_weight_time: Poids pour temps dans relatedness
        shaw_relatedness_weight_demand: Poids pour demande dans relatedness
        shaw_removal_randomness: Paramètre de randomisation (>0, typique: 3-10)
    """
    name: str = "alns"
    
    # Opérateurs
    destroy_operators: List[str] = field(default_factory=lambda: ['random', 'worst', 'shaw'])
    repair_operators: List[str] = field(default_factory=lambda: ['greedy', 'regret2'])
    
    # Taille de destruction
    min_destroy_size: Optional[int] = None
    max_destroy_size: Optional[int] = None
    destroy_size_ratio: float = 0.3  # 30% des clients
    
    # Poids adaptatifs
    initial_weight: float = 1.0
    weight_decay: float = 0.95
    weight_update_interval: int = 100
    score_new_best: float = 33.0
    score_better: float = 9.0
    score_accepted: float = 3.0
    score_rejected: float = 0.0
    
    # Acceptation (SA-like)
    initial_temperature: float = 1000.0
    cooling_rate: float = 0.9995
    min_temperature: float = 0.01
    
    # Shaw removal parameters
    shaw_relatedness_weight_distance: float = 1.0
    shaw_relatedness_weight_time: float = 0.5
    shaw_relatedness_weight_demand: float = 0.1
    shaw_removal_randomness: float = 6.0
    
    def __post_init__(self):
        """Valide les paramètres."""
        # Valide destroy operators
        valid_destroy = ['random', 'worst', 'shaw', 'route']
        for op in self.destroy_operators:
            if op not in valid_destroy:
                raise ValueError(f"Unknown destroy operator: {op}. Valid: {valid_destroy}")
        
        # Valide repair operators
        valid_repair = ['greedy', 'regret2', 'regret3']
        for op in self.repair_operators:
            if op not in valid_repair:
                raise ValueError(f"Unknown repair operator: {op}. Valid: {valid_repair}")
        
        # Au moins un opérateur de chaque type
        if len(self.destroy_operators) == 0:
            raise ValueError("Need at least one destroy operator")
        if len(self.repair_operators) == 0:
            raise ValueError("Need at least one repair operator")


class OperatorWeightManager:
    """
    Gère les poids adaptatifs des opérateurs.
    
    Chaque opérateur accumule un score selon ses performances.
    Périodiquement, les poids sont mis à jour et les scores réinitialisés.
    """
    
    def __init__(self, operators: List[str], initial_weight: float, decay: float):
        """
        Args:
            operators: Liste des noms d'opérateurs
            initial_weight: Poids initial pour chaque opérateur
            decay: Facteur de décroissance (0 < decay < 1)
        """
        self.operators = operators
        self.decay = decay
        
        # Poids et scores
        self.weights = {op: initial_weight for op in operators}
        self.scores = {op: 0.0 for op in operators}
        self.usage_count = {op: 0 for op in operators}
        
    def select_operator(self) -> str:
        """
        Sélectionne un opérateur selon la méthode de la roulette.
        
        La probabilité de choisir un opérateur est proportionnelle à son poids.
        """
        total_weight = sum(self.weights.values())
        probabilities = [self.weights[op] / total_weight for op in self.operators]
        
        return np.random.choice(self.operators, p=probabilities)
    
    def record_score(self, operator: str, score: float):
        """Enregistre un score pour un opérateur."""
        self.scores[operator] += score
        self.usage_count[operator] += 1
    
    def update_weights(self):
        """
        Met à jour les poids selon les scores accumulés.
        
        Principe:
        - weight_new = decay * weight_old + (1 - decay) * score_avg
        - Assure que tous les poids restent > 0 (diversification)
        """
        for op in self.operators:
            # Score moyen depuis dernière mise à jour
            if self.usage_count[op] > 0:
                avg_score = self.scores[op] / self.usage_count[op]
            else:
                avg_score = 0.0
            
            # Mise à jour avec décroissance
            self.weights[op] = self.decay * self.weights[op] + (1 - self.decay) * avg_score
            
            # Assure un poids minimal (diversification)
            self.weights[op] = max(self.weights[op], 0.01)
        
        # Reset scores
        self.scores = {op: 0.0 for op in self.operators}
        self.usage_count = {op: 0 for op in self.operators}
    
    def get_statistics(self) -> Dict[str, float]:
        """Retourne les poids actuels pour logging."""
        return self.weights.copy()


class ALNSSolver(AbstractSolver):
    """
    Solver par Adaptive Large Neighborhood Search.
    
    Principe:
    - Détruit une partie de la solution (remove customers)
    - Reconstruit en réinsérant les clients removed
    - Apprend quels opérateurs marchent bien
    - Équilibre exploration (diversité opérateurs) et exploitation (meilleurs opérateurs)
    
    Architecture:
    - Hérite de AbstractSolver pour l'infrastructure commune
    - Utilise OperatorWeightManager pour l'apprentissage adaptatif
    - Implémente destroy/repair operators comme méthodes
    - Critère d'acceptation type Simulated Annealing
    """
    
    def __init__(self,
                 instance,
                 evaluator: RouteEvaluator,
                 initial_solution: Solution,
                 config: Optional[ALNSConfig] = None):
        """
        Initialise le solver ALNS.
        
        Args:
            instance: Instance VRP
            evaluator: RouteEvaluator pour validation
            initial_solution: Solution de départ
            config: Configuration ALNS
        """
        super().__init__(instance, evaluator, initial_solution, 
                        config or ALNSConfig())
        
        # Vérifie que config est bien ALNSConfig
        if not isinstance(self.config, ALNSConfig):
            self.config = ALNSConfig(**self.config.__dict__)
        
        # Weight managers pour destroy et repair
        self.destroy_weights = OperatorWeightManager(
            self.config.destroy_operators,
            self.config.initial_weight,
            self.config.weight_decay
        )
        self.repair_weights = OperatorWeightManager(
            self.config.repair_operators,
            self.config.initial_weight,
            self.config.weight_decay
        )
        
        # État SA pour acceptation
        self.temperature = self.config.initial_temperature
        
        # Statistiques
        self.accepted_moves = 0
        self.rejected_moves = 0
        self.new_best_count = 0
        
        # Cache pour Shaw removal (calculs coûteux)
        self._relatedness_cache = {}
        
        # Seed aléatoire
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
    
    def solve(self) -> Solution:
        """
        Résout par ALNS.

        Utilise les fonctions d'AbstractSolver pour accepter/rejeter/mettre à jour la meilleure
        solution (_accept_move, _reject_move, _update_best). L'itération est incrémentée
        au début de la boucle (comme dans le recuit simulé exemple) afin que les mises à jour
        périodiques et les logs utilisent une itération cohérente.
        """
        self._start_solving()

        # Calcule la taille de destruction
        n_customers = self.current_solution.get_n_customers()
        if self.config.min_destroy_size is None:
            self.min_destroy = max(1, int(n_customers * 0.1))
        else:
            self.min_destroy = self.config.min_destroy_size

        if self.config.max_destroy_size is None:
            self.max_destroy = max(1, int(n_customers * self.config.destroy_size_ratio))
        else:
            self.max_destroy = self.config.max_destroy_size

        # Init counter for iterations without improvement (used by _should_stop)
        if not hasattr(self, "iterations_no_improvement"):
            self.iterations_no_improvement = 0

        # Affichage initial si verbose
        if self.config.verbose:
            print("\n" + "="*70)
            print("ALNS SOLVER - Starting optimization")
            print("="*70)
            print(f"Initial cost: {self.initial_solution.total_cost:.2f}")
            print(f"Vehicles: {self.initial_solution.n_vehicles_used}")
            print(f"Customers: {n_customers}")
            print(f"Destroy operators: {self.config.destroy_operators}")
            print(f"Repair operators: {self.config.repair_operators}")
            print(f"Destroy size: {self.min_destroy}-{self.max_destroy} customers ({self.config.destroy_size_ratio*100:.0f}%)")
            print(f"Initial temperature: {self.config.initial_temperature:.2f}")
            print(f"Cooling rate: {self.config.cooling_rate}")
            print(f"Weight update interval: {self.config.weight_update_interval}")
            print("="*70 + "\n")

        # Boucle principale
        while not self._should_stop():
            # Incrémente l'itération en début (cohérent avec l'exemple SA)
            self.iteration += 1

            # flag pour savoir si on a trouvé une nouvelle meilleure solution cette itération
            new_best_this_iter = False

            # Sélectionne opérateurs
            destroy_op = self.destroy_weights.select_operator()
            repair_op = self.repair_weights.select_operator()

            # Applique destroy + repair
            destroyed, removed_customers = self._apply_destroy(destroy_op)

            # Si échec du destroy (rare), on rejette le move et continue
            if destroyed is None or len(removed_customers) == 0:
                # On peut enregistrer un score rejeté pour les opérateurs
                try:
                    self.destroy_weights.record_score(destroy_op, self.config.score_rejected)
                    self.repair_weights.record_score(repair_op, self.config.score_rejected)
                except Exception:
                    pass
                self._reject_move()

                # increment iterations-without-improvement (aucune amélioration possible ici)
                self.iterations_no_improvement += 1

                self._record_convergence()
                continue

            repaired = self._apply_repair(repair_op, destroyed, removed_customers)

            # Si échec du repair, rejette et enregistre score
            if repaired is None:
                self.destroy_weights.record_score(destroy_op, self.config.score_rejected)
                self.repair_weights.record_score(repair_op, self.config.score_rejected)
                self._reject_move()

                # increment iterations-without-improvement (aucune amélioration possible ici)
                self.iterations_no_improvement += 1

                self._record_convergence()
                continue

            # Évalue la nouvelle solution
            new_cost = repaired.total_cost
            current_cost = self.current_solution.total_cost
            delta = new_cost - current_cost

            # Détermine le score selon le résultat et critère d'acceptation SA-like
            if new_cost < self.best_solution.total_cost:
                score = self.config.score_new_best
                accept = True
            elif delta < 0:
                score = self.config.score_better
                accept = True
            else:
                # Détérioration : critère SA
                accept_prob = math.exp(-delta / max(self.temperature, 1e-12))
                accept = random.random() < accept_prob
                score = self.config.score_accepted if accept else self.config.score_rejected

            # Enregistre les scores pour les opérateurs
            self.destroy_weights.record_score(destroy_op, score)
            self.repair_weights.record_score(repair_op, score)

            # Accepte ou rejette (utilise helpers de AbstractSolver)
            if accept:
                self.current_solution = repaired
                self._accept_move()

                # Met à jour meilleure solution via la fonction dédiée
                if new_cost < self.best_solution.total_cost:
                    # _update_best doit gérer la copie / enregistrement du timestamp, etc.
                    self._update_best(self.current_solution)
                    self.new_best_count += 1
                    new_best_this_iter = True

                    # reset iterations-without-improvement lorsque on trouve une nouvelle meilleure
                    self.iterations_no_improvement = 0

                    if self.config.verbose:
                        improvement = self.initial_solution.total_cost - new_cost
                        improvement_pct = 100 * improvement / self.initial_solution.total_cost
                        print(f"  [Iter {self.iteration:6d}] *** NEW BEST: {new_cost:.2f} "
                              f"(improvement: {improvement:.2f}, {improvement_pct:.2f}%) "
                              f"[{destroy_op}+{repair_op}] Time = {(time.time() - self.start_time):.2f}s")
            else:
                self._reject_move()

            # Si aucune nouvelle meilleure solution cette itération, on incrémente le compteur
            if not new_best_this_iter:
                self.iterations_no_improvement += 1

            # Refroidit
            self.temperature = max(
                self.config.min_temperature,
                self.temperature * self.config.cooling_rate
            )

            # Met à jour poids périodiquement
            if self.iteration % self.config.weight_update_interval == 0:
                self.destroy_weights.update_weights()
                self.repair_weights.update_weights()

                # Affiche stats périodiquement
                if self.config.verbose:
                    total_moves = self.accepted_moves + self.rejected_moves
                    accept_rate = 100 * self.accepted_moves / total_moves if total_moves > 0 else 0.0
                    print(f"  [Iter {self.iteration:6d}] Time = {(time.time() - self.start_time):.2f}s T={self.temperature:.2f}, "
                          f"Current={self.current_solution.total_cost:.2f}, "
                          f"Best={self.best_solution.total_cost:.2f}, "
                          f"Accept={accept_rate:.1f}%")

            # Enregistre convergence et continue
            self._record_convergence()

        return self._finish_solving()
    
    # ========================================================================
    # DESTROY OPERATORS
    # ========================================================================
    
    def _apply_destroy(self, operator: str) -> Tuple[Optional[Solution], Set[int]]:
        """
        Applique un opérateur de destruction.
        
        Args:
            operator: Nom de l'opérateur ('random', 'worst', 'shaw', 'route')
        
        Returns:
            (solution_détruite, ensemble_clients_retirés)
            Retourne (None, set()) en cas d'échec
        """
        # Détermine combien de clients retirer
        n_to_remove = random.randint(self.min_destroy, self.max_destroy)
        n_to_remove = min(n_to_remove, self.current_solution.get_n_customers())
        
        if n_to_remove == 0:
            return None, set()
        
        # Applique l'opérateur
        if operator == 'random':
            return self._destroy_random(n_to_remove)
        elif operator == 'worst':
            return self._destroy_worst(n_to_remove)
        elif operator == 'shaw':
            return self._destroy_shaw(n_to_remove)
        elif operator == 'route':
            return self._destroy_route(n_to_remove)
        else:
            raise ValueError(f"Unknown destroy operator: {operator}")
    
    def _destroy_random(self, n_to_remove: int) -> Tuple[Solution, Set[int]]:
        """
        Retire n clients au hasard.
        
        Simple mais efficace pour diversification.
        """
        # Clone la solution
        solution = self.current_solution.copy()
        
        # Sélectionne clients au hasard
        all_customers = list(solution.get_customers())
        removed = set(random.sample(all_customers, n_to_remove))
        
        # Retire de la solution
        for customer in removed:
            route_idx = solution.get_route_of_customer(customer)
            route = solution.routes[route_idx]
            position = route.index(customer)
            
            # Retire le client
            new_route = route[:position] + route[position+1:]
            
            # Réévalue la route
            if len(new_route) > 0:
                eval_result = self.evaluator.evaluate_route(new_route)
                solution.routes[route_idx] = new_route
                solution.route_costs[route_idx] = eval_result.cost
            else:
                # Route vide : on la garde vide pour l'instant
                solution.routes[route_idx] = []
                solution.route_costs[route_idx] = 0.0
        
        # Recalcule coût total
        solution.total_cost = sum(solution.route_costs)
        
        return solution, removed
    
    def _destroy_worst(self, n_to_remove: int) -> Tuple[Solution, Set[int]]:
        """
        Retire les n clients les plus coûteux (savings removal).
        
        Pour chaque client, calcule son "coût de retrait" = saving si on le retire.
        Retire ceux avec le plus grand saving (= coûtent le plus cher).
        
        Principe:
        cost(client) = dist(pred, client) + dist(client, succ) - dist(pred, succ)
        """
        solution = self.current_solution.copy()
        
        # Calcule le coût de chaque client
        customer_costs = []
        
        for customer in solution.get_customers():
            route_idx = solution.get_route_of_customer(customer)
            route = solution.routes[route_idx]
            position = route.index(customer)
            
            # Prédécesseur et successeur
            if position == 0:
                pred = self.instance.depot
            else:
                pred = route[position - 1]
            
            if position == len(route) - 1:
                succ = self.instance.depot
            else:
                succ = route[position + 1]
            
            # Coût = ce qu'on perd en l'ayant dans la route
            dist_matrix = self.instance.distance_matrix
            cost_with = dist_matrix[pred, customer] + dist_matrix[customer, succ]
            cost_without = dist_matrix[pred, succ]
            
            customer_cost = cost_with - cost_without
            customer_costs.append((customer, customer_cost))
        
        # Trie par coût décroissant et prend les n pires
        customer_costs.sort(key=lambda x: x[1], reverse=True)
        removed = set(c for c, _ in customer_costs[:n_to_remove])
        
        # Retire de la solution
        for customer in removed:
            route_idx = solution.get_route_of_customer(customer)
            route = solution.routes[route_idx]
            position = route.index(customer)
            
            new_route = route[:position] + route[position+1:]
            
            if len(new_route) > 0:
                eval_result = self.evaluator.evaluate_route(new_route)
                solution.routes[route_idx] = new_route
                solution.route_costs[route_idx] = eval_result.cost
            else:
                solution.routes[route_idx] = []
                solution.route_costs[route_idx] = 0.0
        
        solution.total_cost = sum(solution.route_costs)
        
        return solution, removed
    
    def _destroy_shaw(self, n_to_remove: int) -> Tuple[Solution, Set[int]]:
        """
        Shaw removal : retire des clients "similaires" (related).
        
        Principe de Shaw (1998):
        1. Choisir un client seed au hasard
        2. Calculer relatedness entre seed et tous les autres
        3. Retirer seed + ses n-1 plus proches selon relatedness
        
        Relatedness entre clients i et j:
        R(i,j) = w_dist * dist(i,j) + w_time * |time_i - time_j| + w_demand * |demand_i - demand_j|
        
        Plus R est petit, plus les clients sont similaires.
        On ajoute du randomness pour pas toujours prendre les plus proches.
        """
        solution = self.current_solution.copy()
        
        # Choisis un seed au hasard
        all_customers = list(solution.get_customers())
        seed = random.choice(all_customers)
        
        # Calcule relatedness entre seed et tous les autres
        relatedness_scores = []
        for customer in all_customers:
            if customer == seed:
                continue
            
            rel = self._compute_relatedness(seed, customer)
            relatedness_scores.append((customer, rel))
        
        # Trie par relatedness croissant (plus proches d'abord)
        relatedness_scores.sort(key=lambda x: x[1])
        
        # Sélection avec randomness
        # On utilise une probabilité proportionnelle à 1/rank^p
        # où p = shaw_removal_randomness
        n_candidates = len(relatedness_scores)
        n_to_select = n_to_remove - 1  # -1 car on a déjà le seed
        n_to_select = min(n_to_select, n_candidates)
        
        removed = {seed}
        
        if n_to_select > 0:
            # Calcule probabilités
            p = self.config.shaw_removal_randomness
            probabilities = np.array([(1.0 / (i + 1) ** p) for i in range(n_candidates)])
            probabilities = probabilities / probabilities.sum()
            
            # Sélectionne n_to_select clients
            selected_indices = np.random.choice(
                n_candidates,
                size=n_to_select,
                replace=False,
                p=probabilities
            )
            
            for idx in selected_indices:
                customer, _ = relatedness_scores[idx]
                removed.add(customer)
        
        # Retire de la solution
        for customer in removed:
            route_idx = solution.get_route_of_customer(customer)
            route = solution.routes[route_idx]
            position = route.index(customer)
            
            new_route = route[:position] + route[position+1:]
            
            if len(new_route) > 0:
                eval_result = self.evaluator.evaluate_route(new_route)
                solution.routes[route_idx] = new_route
                solution.route_costs[route_idx] = eval_result.cost
            else:
                solution.routes[route_idx] = []
                solution.route_costs[route_idx] = 0.0
        
        solution.total_cost = sum(solution.route_costs)
        
        return solution, removed
    
    def _compute_relatedness(self, customer1: int, customer2: int) -> float:
        """
        Calcule la relatedness entre deux clients selon Shaw.
        
        Plus le score est bas, plus les clients sont similaires.
        """
        # Check cache
        key = (min(customer1, customer2), max(customer1, customer2))
        if key in self._relatedness_cache:
            return self._relatedness_cache[key]
        
        # Distance
        dist = self.instance.distance_matrix[customer1, customer2]
        dist_normalized = dist / self.instance.distance_matrix.max()
        
        # Time windows (si disponible)
        if self.instance.is_vrptw():
            time1 = (self.instance.ready_times[customer1] + self.instance.due_dates[customer1]) / 2
            time2 = (self.instance.ready_times[customer2] + self.instance.due_dates[customer2]) / 2
            time_diff = abs(time1 - time2)
            time_range = self.instance.due_dates.max() - self.instance.ready_times.min()
            time_normalized = time_diff / time_range if time_range > 0 else 0.0
        else:
            time_normalized = 0.0
        
        # Demand
        demand1 = self.instance.demands[customer1]
        demand2 = self.instance.demands[customer2]
        demand_diff = abs(demand1 - demand2)
        demand_normalized = demand_diff / self.instance.capacity
        
        # Relatedness avec poids
        relatedness = (
            self.config.shaw_relatedness_weight_distance * dist_normalized +
            self.config.shaw_relatedness_weight_time * time_normalized +
            self.config.shaw_relatedness_weight_demand * demand_normalized
        )
        
        # Cache le résultat
        self._relatedness_cache[key] = relatedness
        
        return relatedness
    
    def _destroy_route(self, n_to_remove: int) -> Tuple[Solution, Set[int]]:
        """
        Retire tous les clients d'une ou plusieurs routes au hasard.
        
        Utile pour forcer une reconstruction complète de certaines routes.
        """
        solution = self.current_solution.copy()
        
        # Filtre les routes non vides
        non_empty_routes = [
            (idx, route) for idx, route in enumerate(solution.routes)
            if len(route) > 0
        ]
        
        if len(non_empty_routes) == 0:
            return solution, set()
        
        # Mélange et prend des routes jusqu'à avoir n_to_remove clients
        random.shuffle(non_empty_routes)
        
        removed = set()
        for route_idx, route in non_empty_routes:
            removed.update(route)
            
            # Vide la route
            solution.routes[route_idx] = []
            solution.route_costs[route_idx] = 0.0
            
            if len(removed) >= n_to_remove:
                break
        
        # Si on a trop retiré, on en remet certains
        if len(removed) > n_to_remove:
            excess = len(removed) - n_to_remove
            to_keep = random.sample(list(removed), excess)
            
            # Réinsère ceux qu'on garde dans des routes non vides de solution
            # (on ne peut pas utiliser self.current_solution car les routes ont changé)
            for customer in to_keep:
                # Trouve une route non vide pour réinsérer
                non_empty_routes = [i for i, r in enumerate(solution.routes) if len(r) > 0]
                if non_empty_routes:
                    # Réinsère dans une route aléatoire
                    route_idx = random.choice(non_empty_routes)
                    solution.routes[route_idx].append(customer)
                    
                    # Réévalue la route
                    eval_result = self.evaluator.evaluate_route(solution.routes[route_idx])
                    solution.route_costs[route_idx] = eval_result.cost
            
            # Retire les clients réinsérés de l'ensemble removed
            removed = removed - set(to_keep)
        
        solution.total_cost = sum(solution.route_costs)
        
        return solution, removed
    
    # ========================================================================
    # REPAIR OPERATORS
    # ========================================================================
    
    def _apply_repair(self, operator: str, solution: Solution, 
                     removed: Set[int]) -> Optional[Solution]:
        """
        Applique un opérateur de réparation.
        
        Args:
            operator: Nom de l'opérateur ('greedy', 'regret2', 'regret3')
            solution: Solution détruite (avec clients manquants)
            removed: Ensemble des clients à réinsérer
        
        Returns:
            Solution réparée ou None si échec
        """
        if operator == 'greedy':
            return self._repair_greedy(solution, removed)
        elif operator == 'regret2':
            return self._repair_regret(solution, removed, k=2)
        elif operator == 'regret3':
            return self._repair_regret(solution, removed, k=3)
        else:
            raise ValueError(f"Unknown repair operator: {operator}")
    
    def _repair_greedy(self, solution: Solution, removed: Set[int]) -> Optional[Solution]:
        """
        Réparation gloutonne : insère chaque client à la meilleure position.
        
        Pour chaque client non inséré:
        1. Teste toutes les positions dans toutes les routes
        2. Choisit la position avec le plus petit coût d'insertion
        3. Insère le client
        4. Répète jusqu'à insérer tous les clients
        """
        solution = solution.copy()
        uninserted = list(removed)
        random.shuffle(uninserted)  # Randomize pour diversité
        
        while len(uninserted) > 0:
            best_customer = None
            best_route_idx = None
            best_position = None
            best_cost_increase = float('inf')
            
            # Pour chaque client non inséré
            for customer in uninserted:
                # Teste insertion dans chaque route à chaque position
                for route_idx in range(len(solution.routes)):
                    route = solution.routes[route_idx]
                    current_cost = solution.route_costs[route_idx]
                    
                    # Teste chaque position
                    for position in range(len(route) + 1):
                        # Crée nouvelle route avec customer inséré
                        new_route = route[:position] + [customer] + route[position:]
                        
                        # Évalue
                        eval_result = self.evaluator.evaluate_route(new_route)
                        
                        if eval_result.is_feasible:
                            cost_increase = eval_result.cost - current_cost
                            
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_customer = customer
                                best_route_idx = route_idx
                                best_position = position
            
            # Si aucune insertion faisable, échec
            if best_customer is None:
                return None
            
            # Applique la meilleure insertion
            route = solution.routes[best_route_idx]
            new_route = route[:best_position] + [best_customer] + route[best_position:]
            eval_result = self.evaluator.evaluate_route(new_route)
            
            solution.routes[best_route_idx] = new_route
            solution.route_costs[best_route_idx] = eval_result.cost
            
            uninserted.remove(best_customer)
        
        # Recalcule coût total
        solution.total_cost = sum(solution.route_costs)
        
        return solution
    
    def _repair_regret(self, solution: Solution, removed: Set[int], k: int) -> Optional[Solution]:
        """
        Réparation par regret-k.
        
        Principe du regret:
        - Pour chaque client, on calcule les k meilleures insertions
        - Le regret = différence entre meilleure et k-ième meilleure
        - On insère le client avec le plus grand regret (= celui qui perdrait le plus à attendre)
        
        Intuition: Si un client a regret=100, c'est qu'il a une très bonne position
        (coût X) mais que sa 2ème meilleure position est bien pire (coût X+100).
        Si on attend, on risque de perdre la bonne position.
        Alors que si regret=1, peu importe quand on l'insère.
        
        Args:
            k: Nombre de meilleures positions à considérer (typique: 2 ou 3)
        """
        solution = solution.copy()
        uninserted = list(removed)
        
        while len(uninserted) > 0:
            best_customer = None
            best_route_idx = None
            best_position = None
            best_regret = -float('inf')
            
            # Pour chaque client, calcule son regret
            for customer in uninserted:
                # Trouve les k meilleures insertions possibles
                insertion_costs = []
                
                for route_idx in range(len(solution.routes)):
                    route = solution.routes[route_idx]
                    current_cost = solution.route_costs[route_idx]
                    
                    for position in range(len(route) + 1):
                        new_route = route[:position] + [customer] + route[position:]
                        eval_result = self.evaluator.evaluate_route(new_route)
                        
                        if eval_result.is_feasible:
                            cost_increase = eval_result.cost - current_cost
                            insertion_costs.append((cost_increase, route_idx, position))
                
                # Si aucune insertion possible pour ce client, échec
                if len(insertion_costs) == 0:
                    return None
                
                # Trie par coût croissant
                insertion_costs.sort(key=lambda x: x[0])
                
                # Calcule le regret = différence entre meilleure et k-ième meilleure
                if len(insertion_costs) >= k:
                    regret = insertion_costs[k-1][0] - insertion_costs[0][0]
                else:
                    # Si moins de k insertions possibles, prend la pire - la meilleure
                    regret = insertion_costs[-1][0] - insertion_costs[0][0]
                
                # Met à jour le meilleur selon regret
                # (en cas d'égalité, on favorise le plus petit coût)
                if regret > best_regret or (regret == best_regret and insertion_costs[0][0] < best_cost_increase):
                    best_regret = regret
                    best_customer = customer
                    best_cost_increase = insertion_costs[0][0]
                    best_route_idx = insertion_costs[0][1]
                    best_position = insertion_costs[0][2]
            
            # Insère le client avec plus grand regret
            route = solution.routes[best_route_idx]
            new_route = route[:best_position] + [best_customer] + route[best_position:]
            eval_result = self.evaluator.evaluate_route(new_route)
            
            solution.routes[best_route_idx] = new_route
            solution.route_costs[best_route_idx] = eval_result.cost
            
            uninserted.remove(best_customer)
        
        solution.total_cost = sum(solution.route_costs)
        
        return solution