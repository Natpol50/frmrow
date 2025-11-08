"""
Route Evaluator - Évaluation rapide d'une route CVRPTW - Asha Geyon 2025

Ce module fournit les outils pour évaluer si une route est faisable
et calculer son coût, en tenant compte des contraintes de capacité
et des fenêtres temporelles.

Architecture:
- RouteEvaluator : Classe principale pour évaluer une route
- Utilise NumPy pour les calculs vectorisés quand possible
- Validation incrémentale : ne touche qu'UNE route à la fois

Usage typique:
    evaluator = RouteEvaluator(instance)
    is_feasible, cost, violations = evaluator.evaluate_route(route)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import timeit
import statistics


@dataclass
class RouteEvaluation:
    """
    Résultat de l'évaluation d'une route.
    
    Attributes:
        is_feasible: True si la route respecte toutes les contraintes
        cost: Coût total de la route (distance)
        total_demand: Demande totale de la route
        violations: Dictionnaire des violations détectées
        arrival_times: Temps d'arrivée à chaque client (pour debug)
    """
    is_feasible: bool
    cost: float
    total_demand: int
    violations: Dict[str, List[str]]
    arrival_times: Optional[np.ndarray] = None
    
    def get_violation_summary(self) -> str:
        """Retourne un résumé textuel des violations pour le debug/analyse."""
        if self.is_feasible:
            return "Route faisable"
        
        summary = []
        for vtype, msgs in self.violations.items():
            summary.append(f"{vtype}: {len(msgs)} violations")
        return ", ".join(summary)


class RouteEvaluator:
    """
    Évaluateur de routes pour CVRPTW.
    
    Évalue si une route respecte les contraintes de capacité et de time windows,
    et calcule son coût total. Conçu pour être appelé dans une boucle critique
    (recherche locale), donc optimisé pour la vitesse.
    
    Attributes:
        instance: Instance CVRPTW (avec matrice de distances, demands, etc.)
        depot: Indice du dépôt (généralement 0)
        capacity: Capacité maximale d'un véhicule
        distance_matrix: Matrice de distances pré-calculée
        demands: Vecteur des demandes
        ready_times: Vecteur des temps d'ouverture (time windows)
        due_dates: Vecteur des temps de fermeture (time windows)
        service_times: Vecteur des temps de service
    """
    
    def __init__(self, instance):
        """
        Initialise l'évaluateur avec une instance.
        
        Args:
            instance: Instance CVRPTW (doit avoir les attributs nécessaires)
        """
        self.instance = instance
        self.depot = instance.depot
        self.capacity = instance.capacity
        
        self.distance_matrix = instance.distance_matrix
        self.demands = instance.demands
        
        # Time windows (peuvent être None si pas VRPTW)
        self.ready_times = instance.ready_times
        self.due_dates = instance.due_dates
        self.service_times = instance.service_times
        
        self.has_time_windows = instance.is_vrptw()
    
    def evaluate_route(self, route: List[int], 
                       return_details: bool = False) -> RouteEvaluation:
        """
        Évalue complètement une route.
        
        Une route est une liste d'indices de clients (SANS le dépôt au début/fin).
        Exemple: [3, 7, 12, 5] signifie dépôt → 3 → 7 → 12 → 5 → dépôt
        
        Args:
            route: Liste des indices de clients dans l'ordre de visite
            return_details: Si True, inclut les temps d'arrivée dans le résultat
            
        Returns:
            RouteEvaluation avec faisabilité, coût, et violations éventuelles
        """
        violations = {
            'capacity': [],
            'time_windows': []
        }
        
        # Cas particulier : route vide
        if len(route) == 0:
            return RouteEvaluation(
                is_feasible=True,
                cost=0.0,
                total_demand=0,
                violations=violations
            )
        
        # 1. Vérification de la capacité (ultra-rapide)
        total_demand = self._check_capacity(route, violations)
        
        # 2. Calcul du coût (distances)
        cost = self._compute_cost(route)
        
        # 3. Vérification des time windows (si applicable)
        arrival_times = None
        if self.has_time_windows:
            arrival_times = self._check_time_windows(route, violations)
        
        is_feasible = all(len(v) == 0 for v in violations.values())
        
        return RouteEvaluation(
            is_feasible=is_feasible,
            cost=cost,
            total_demand=total_demand,
            violations=violations,
            arrival_times=arrival_times if return_details else None
        )
    
    def evaluate_route_fast(self, route: List[int]) -> Tuple[bool, float]:
        """
        Version ultra-rapide pour la boucle critique : retourne juste (faisable, coût).
        
        Pas de détails sur les violations, juste True/False + coût.
        Si impossible, le coût est infini et le calcul s'arrête dès la première violation.
        
        A utiliser dans les opérateurs de voisinage.
        
        Args:
            route: Liste des indices de clients
            
        Returns:
            (is_feasible, cost)
        """
        if len(route) == 0:
            return True, 0.0
        
        # Check capacité
        total_demand = np.sum(self.demands[route])
        if total_demand > self.capacity:
            return False, float('inf')
        
        # Calcul coût
        cost = self._compute_cost(route)
        
        # Check time windows
        if self.has_time_windows:
            if not self._check_time_windows_fast(route):
                return False, float('inf')
        
        return True, cost
    
    def _check_capacity(self, route: List[int], 
                       violations: Dict[str, List[str]]) -> int:
        """
        Vérifie la contrainte de capacité.
        
        Utilise NumPy pour la somme vectorisée (ultra-rapide).
        
        Args:
            route: Liste des indices de clients
            violations: Dictionnaire pour enregistrer les violations
            
        Returns:
            Demande totale de la route
        """
        total_demand = int(np.sum(self.demands[route]))
        
        if total_demand > self.capacity:
            violations['capacity'].append(
                f"Capacité dépassée: {total_demand} > {self.capacity}"
            )
        
        return total_demand
    
    def _compute_cost(self, route: List[int]) -> float:
        """
        Calcule le coût total de la route (somme des distances).
        
        Route complète : dépôt → route[0] → route[1] → ... → route[n-1] → dépôt
        
        Args:
            route: Liste des indices de clients
            
        Returns:
            Coût total (distance totale)
        """
        if len(route) == 0:
            return 0.0
        
        cost = 0.0
        
        # Dépôt → premier client
        cost += self.distance_matrix[self.depot, route[0]]
        
        # Entre clients consécutifs
        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i], route[i + 1]]
        
        # Dernier client → dépôt
        cost += self.distance_matrix[route[-1], self.depot]
        
        return cost
    
    def _compute_cost_v2(self, route: List[int]) -> float:
        """
        Calcule le coût total de la route (somme des distances).
        
        Route complète : dépôt → route[0] → route[1] → ... → route[n-1] → dépôt
        
        Args:
            route: Liste des indices de clients
            
        Returns:
            Coût total (distance totale)
        """
        if len(route) == 0:
            return 0.0
    
        route_arr = np.concatenate(([self.depot], np.asarray(route, dtype=int), [self.depot]))
        cost = np.sum(self.distance_matrix[route_arr[:-1], route_arr[1:]])
        return float(cost)
    

    def _check_time_windows(self, route: List[int], 
                           violations: Dict[str, List[str]]) -> np.ndarray:
        """
        Simule le parcours de la route pour vérifier les time windows.
        
        Algorithme:
        1. On part du dépôt à t=0
        2. Pour chaque client:
           - On calcule le temps d'arrivée (départ précédent + distance)
           - Si on arrive trop tôt → on attend jusqu'à ready_time
           - Si on arrive trop tard (> due_date) → VIOLATION
           - On sert le client pendant service_time
           - On part vers le suivant
        3. On vérifie qu'on rentre au dépôt avant sa due_date
        
        Args:
            route: Liste des indices de clients
            violations: Dictionnaire pour enregistrer les violations
            
        Returns:
            Array des temps d'arrivée à chaque client (pour debug)
        """
        n = len(route)
        arrival_times = np.zeros(n + 2)  # +2 pour dépôt départ et retour
        
        current_time = 0.0
        current_location = self.depot
        
        # Dépôt de départ
        arrival_times[0] = 0.0
        
        # Visite de chaque client
        for i, customer in enumerate(route):
            # Temps de trajet
            travel_time = self.distance_matrix[current_location, customer]
            arrival_time = current_time + travel_time
            
            # Si on arrive trop tôt, on attend
            if arrival_time < self.ready_times[customer]:
                arrival_time = self.ready_times[customer]
            
            # Vérification de la due_date
            if arrival_time > self.due_dates[customer]:
                violations['time_windows'].append(
                    f"Client {customer}: arrivée à t={arrival_time:.2f} "
                    f"> due_date={self.due_dates[customer]:.2f}"
                )
            
            arrival_times[i + 1] = arrival_time
            
            # Temps de service
            departure_time = arrival_time + self.service_times[customer]
            
            # Prépare pour le prochain
            current_time = departure_time
            current_location = customer
        
        # Retour au dépôt
        travel_time = self.distance_matrix[current_location, self.depot]
        arrival_depot = current_time + travel_time
        arrival_times[-1] = arrival_depot
        
        # Vérification du retour au dépôt
        if arrival_depot > self.due_dates[self.depot]:
            violations['time_windows'].append(
                f"Retour dépôt: arrivée à t={arrival_depot:.2f} "
                f"> due_date={self.due_dates[self.depot]:.2f}"
            )
        
        return arrival_times
    
    def _check_time_windows_fast(self, route: List[int]) -> bool:
        """
        Version rapide sans détails : retourne juste True/False.
        
        Même logique que _check_time_windows mais s'arrête dès la première violation.
        Utilisé dans evaluate_route_fast() pour la boucle critique.
        
        Args:
            route: Liste des indices de clients
            
        Returns:
            True si les time windows sont respectées, False sinon
        """
        current_time = 0.0
        current_location = self.depot
        
        for customer in route:
            # Temps de trajet
            travel_time = self.distance_matrix[current_location, customer]
            arrival_time = current_time + travel_time
            
            # Attente si nécessaire
            if arrival_time < self.ready_times[customer]:
                arrival_time = self.ready_times[customer]
            
            # Vérification (early exit si violation)
            if arrival_time > self.due_dates[customer]:
                return False
            
            # Service et départ
            departure_time = arrival_time + self.service_times[customer]
            current_time = departure_time
            current_location = customer
        
        # Retour au dépôt
        travel_time = self.distance_matrix[current_location, self.depot]
        arrival_depot = current_time + travel_time
        
        if arrival_depot > self.due_dates[self.depot]:
            return False
        
        return True
    
    def compute_route_cost_only(self, route: List[int]) -> float:
        """
        Calcule uniquement le coût sans vérifier les contraintes.
        
        Utile quand on sait déjà que la route est faisable et qu'on veut
        juste comparer des coûts rapidement.
        
        Args:
            route: Liste des indices de clients
            
        Returns:
            Coût total de la route
        """
        return self._compute_cost(route)
    
    def get_route_slack(self, route: List[int]) -> np.ndarray:
        """
        Calcule le "slack" temporel à chaque client.
        
        Le slack est la marge de temps disponible : combien de temps on pourrait
        arriver plus tard sans violer la time window.
        
        Utile pour l'optimisation : on peut réarranger les clients avec beaucoup
        de slack sans risquer de violations.
        
        Args:
            route: Liste des indices de clients
            
        Returns:
            Array du slack à chaque position (due_date - arrival_time)
        """
        if not self.has_time_windows:
            return np.full(len(route), float('inf'))
        
        n = len(route)
        slacks = np.zeros(n)
        
        current_time = 0.0
        current_location = self.depot
        
        for i, customer in enumerate(route):
            travel_time = self.distance_matrix[current_location, customer]
            arrival_time = current_time + travel_time
            
            if arrival_time < self.ready_times[customer]:
                arrival_time = self.ready_times[customer]
            
            # Slack = combien de temps on pourrait arriver plus tard
            slacks[i] = self.due_dates[customer] - arrival_time
            
            departure_time = arrival_time + self.service_times[customer]
            current_time = departure_time
            current_location = customer
        
        return slacks