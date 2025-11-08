"""
Solution - Structure de données pour une solution VRP - Asha Geyon 2025

Cette classe représente une solution complète à un problème VRP.
Elle est conçue pour être modifiée rapidement (des milliers de fois/seconde)
tout en maintenant la cohérence entre ses différentes structures de données.

Architecture:
- routes: List[List[int]] - Structure principale
- route_costs: List[float] - Coût de chaque route (cache)
- client_to_route: Dict[int, int] - Mapping client → route_idx
- client_position: Dict[int, int] - Mapping client → position dans route

La classe garantit la cohérence : toute modification met à jour automatiquement
toutes les structures. Si une incohérence est détectée, une exception est levée.

Usage:
    solution = Solution(instance)
    solution.add_route([3, 7, 12])
    solution.relocate_customer(7, 0, 1, 2)  # Client 7 : route 0 → route 1, pos 2
    solution.validate_consistency()  # Vérifie la cohérence interne
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np


class Solution:
    """
    Structure de données pour une solution VRP.
    
    Maintient plusieurs représentations de la même information pour
    des accès rapides selon le contexte (lecture par route, recherche par client, etc.)
    
    Attributes:
        routes: Liste des routes [route_0, route_1, ...] où chaque route est [client_i, ...]
        route_costs: Coût de chaque route (synchronisé avec routes)
        total_cost: Coût total de la solution (somme des route_costs)
        n_vehicles_used: Nombre de véhicules utilisés (= nombre de routes non-vides)
        
        # Mappings inverses pour accès O(1)
        client_to_route: {client_id: route_idx}
        client_position: {client_id: position_in_route}
        
        # Métadonnées (optionnelles)
        instance_name: Nom de l'instance
        algo_name: Nom de l'algo ayant généré cette solution
    """
    
    def __init__(self, instance_name: Optional[str] = None):
        """
        Initialise une solution vide.
        
        Args:
            instance_name: Nom de l'instance (pour traçabilité)
        """
        self.routes: List[List[int]] = []
        self.route_costs: List[float] = []
        self.total_cost: float = 0.0
        self.n_vehicles_used: int = 0
        
        # Mappings inverses
        self.client_to_route: Dict[int, int] = {}
        self.client_position: Dict[int, int] = {}
        
        # Métadonnées
        self.instance_name = instance_name
        self.algo_name: Optional[str] = None
    
    # ========================================================================
    # Méthodes de construction / modification de base
    # ========================================================================
    
    def add_route(self, route: List[int], cost: float = 0.0):
        """
        Ajoute une nouvelle route à la solution.
        
        Met à jour automatiquement tous les mappings.
        
        Args:
            route: Liste des clients dans l'ordre de visite
            cost: Coût de la route (si déjà calculé, sinon 0.0)
            
        Raises:
            ValueError: Si un client de la route est déjà dans une autre route
        """
        # Vérifie que les clients ne sont pas déjà assignés
        for client in route:
            if client in self.client_to_route:
                raise ValueError(
                    f"Client {client} déjà dans la route {self.client_to_route[client]}"
                )
        
        route_idx = len(self.routes)
        
        # Ajoute la route
        self.routes.append(route.copy())
        self.route_costs.append(cost)
        self.total_cost += cost
        
        if len(route) > 0:
            self.n_vehicles_used += 1
        
        # Met à jour les mappings
        for position, client in enumerate(route):
            self.client_to_route[client] = route_idx
            self.client_position[client] = position
    
    def remove_route(self, route_idx: int):
        """
        Retire une route de la solution.
        
        Met à jour automatiquement tous les mappings.
        ATTENTION: Change les indices des routes suivantes !
        
        Args:
            route_idx: Indice de la route à retirer
            
        Raises:
            IndexError: Si route_idx invalide
        """
        if route_idx >= len(self.routes):
            raise IndexError(f"Route index {route_idx} out of range")
        
        route = self.routes[route_idx]
        cost = self.route_costs[route_idx]
        
        # Retire des mappings
        for client in route:
            del self.client_to_route[client]
            del self.client_position[client]
        
        # Retire la route
        self.routes.pop(route_idx)
        self.route_costs.pop(route_idx)
        self.total_cost -= cost
        
        if len(route) > 0:
            self.n_vehicles_used -= 1
        
        # Met à jour les indices des routes suivantes dans les mappings
        for client in self.client_to_route:
            if self.client_to_route[client] > route_idx:
                self.client_to_route[client] -= 1
    
    def set_route(self, route_idx: int, new_route: List[int], new_cost: float):
        """
        Remplace complètement une route existante.
        
        Plus efficace que remove + add car ne change pas les indices.
        
        Args:
            route_idx: Indice de la route à remplacer
            new_route: Nouvelle liste de clients
            new_cost: Nouveau coût
            
        Raises:
            IndexError: Si route_idx invalide
            ValueError: Si un client de new_route est dans une autre route
        """
        if route_idx >= len(self.routes):
            raise IndexError(f"Route index {route_idx} out of range")
        
        old_route = self.routes[route_idx]
        old_cost = self.route_costs[route_idx]
        
        # Vérifie que les nouveaux clients ne sont pas ailleurs
        for client in new_route:
            if client in self.client_to_route:
                existing_route = self.client_to_route[client]
                if existing_route != route_idx:
                    raise ValueError(
                        f"Client {client} déjà dans la route {existing_route}"
                    )
        
        # Retire les anciens clients des mappings
        for client in old_route:
            if client not in new_route:  # Seulement si pas dans la nouvelle
                del self.client_to_route[client]
                del self.client_position[client]
        
        # Met à jour la route
        self.routes[route_idx] = new_route.copy()
        self.route_costs[route_idx] = new_cost
        self.total_cost += (new_cost - old_cost)
        
        # Ajuste n_vehicles_used
        was_empty = len(old_route) == 0
        is_empty = len(new_route) == 0
        
        if was_empty and not is_empty:
            self.n_vehicles_used += 1
        elif not was_empty and is_empty:
            self.n_vehicles_used -= 1
        
        # Met à jour les mappings pour les nouveaux clients
        for position, client in enumerate(new_route):
            self.client_to_route[client] = route_idx
            self.client_position[client] = position
    
    def update_route_cost(self, route_idx: int, new_cost: float):
        """
        Met à jour le coût d'une route (après réévaluation par exemple).
        
        Args:
            route_idx: Indice de la route
            new_cost: Nouveau coût
        """
        if route_idx >= len(self.routes):
            raise IndexError(f"Route index {route_idx} out of range")
        
        old_cost = self.route_costs[route_idx]
        self.route_costs[route_idx] = new_cost
        self.total_cost += (new_cost - old_cost)
    
    # ========================================================================
    # Opérations de haut niveau (pour les opérateurs de voisinage)
    # ========================================================================
    
    def relocate_customer(self, customer: int, 
                         from_route_idx: int, 
                         to_route_idx: int, 
                         to_position: int,
                         new_from_cost: float,
                         new_to_cost: float):
        """
        Déplace un client d'une route à une autre.
        
        Opération atomique : met à jour toutes les structures en une fois.
        
        Args:
            customer: ID du client à déplacer
            from_route_idx: Route source
            to_route_idx: Route destination
            to_position: Position d'insertion dans la route destination
            new_from_cost: Nouveau coût de la route source (après retrait)
            new_to_cost: Nouveau coût de la route destination (après insertion)
            
        Raises:
            ValueError: Si le client n'est pas dans from_route ou si to_position invalide
        """
        # Vérifie cohérence
        if customer not in self.client_to_route:
            raise ValueError(f"Client {customer} not in solution")
        
        actual_route = self.client_to_route[customer]
        if actual_route != from_route_idx:
            raise ValueError(
                f"Client {customer} is in route {actual_route}, not {from_route_idx}"
            )
        
        # Retire de la route source
        from_route = self.routes[from_route_idx]
        from_position = self.client_position[customer]
        from_route.pop(from_position)
        
        # Met à jour les positions des clients suivants dans from_route
        for i in range(from_position, len(from_route)):
            self.client_position[from_route[i]] = i
        
        # Insère dans la route destination
        to_route = self.routes[to_route_idx]
        
        if to_position < 0 or to_position > len(to_route):
            raise ValueError(f"Invalid to_position {to_position} for route of size {len(to_route)}")
        
        to_route.insert(to_position, customer)
        
        # Met à jour les positions dans to_route
        for i in range(to_position, len(to_route)):
            client_i = to_route[i]
            self.client_to_route[client_i] = to_route_idx
            self.client_position[client_i] = i
        
        # Met à jour les coûts
        old_total = self.route_costs[from_route_idx] + self.route_costs[to_route_idx]
        self.route_costs[from_route_idx] = new_from_cost
        self.route_costs[to_route_idx] = new_to_cost
        new_total = new_from_cost + new_to_cost
        
        self.total_cost += (new_total - old_total)
        
        # Ajuste n_vehicles_used si nécessaire
        if len(from_route) == 0:  # Route source maintenant vide
            self.n_vehicles_used -= 1
        if len(to_route) == 1:  # Route destination était vide
            self.n_vehicles_used += 1
    
    def exchange_customers(self, customer1: int, customer2: int,
                          new_cost_route1: float, new_cost_route2: float):
        """
        Échange deux clients entre leurs routes respectives.
        
        Args:
            customer1: Premier client
            customer2: Deuxième client
            new_cost_route1: Nouveau coût de la route de customer1
            new_cost_route2: Nouveau coût de la route de customer2
            
        Raises:
            ValueError: Si les clients sont dans la même route ou non présents
        """
        if customer1 not in self.client_to_route:
            raise ValueError(f"Customer {customer1} not in solution")
        if customer2 not in self.client_to_route:
            raise ValueError(f"Customer {customer2} not in solution")
        
        route1_idx = self.client_to_route[customer1]
        route2_idx = self.client_to_route[customer2]
        
        if route1_idx == route2_idx:
            raise ValueError("Cannot exchange customers from the same route")
        
        pos1 = self.client_position[customer1]
        pos2 = self.client_position[customer2]
        
        # Échange dans les routes
        self.routes[route1_idx][pos1] = customer2
        self.routes[route2_idx][pos2] = customer1
        
        # Met à jour les mappings
        self.client_to_route[customer1] = route2_idx
        self.client_to_route[customer2] = route1_idx
        self.client_position[customer1] = pos2
        self.client_position[customer2] = pos1
        
        # Met à jour les coûts
        old_total = self.route_costs[route1_idx] + self.route_costs[route2_idx]
        self.route_costs[route1_idx] = new_cost_route1
        self.route_costs[route2_idx] = new_cost_route2
        new_total = new_cost_route1 + new_cost_route2
        
        self.total_cost += (new_total - old_total)
    
    def reverse_segment(self, route_idx: int, start: int, end: int, new_cost: float):
        """
        Inverse un segment dans une route (2-opt intra-route).
        
        Args:
            route_idx: Indice de la route
            start: Début du segment (inclus)
            end: Fin du segment (exclus)
            new_cost: Nouveau coût après inversion
            
        Raises:
            ValueError: Si indices invalides
        """
        if route_idx >= len(self.routes):
            raise IndexError(f"Route index {route_idx} out of range")
        
        route = self.routes[route_idx]
        
        if start < 0 or end > len(route) or start >= end:
            raise ValueError(f"Invalid segment [{start}:{end}] for route of size {len(route)}")
        
        # Inverse le segment
        route[start:end] = reversed(route[start:end])
        
        # Met à jour les positions des clients dans le segment inversé
        for i in range(start, end):
            self.client_position[route[i]] = i
        
        # Met à jour le coût
        old_cost = self.route_costs[route_idx]
        self.route_costs[route_idx] = new_cost
        self.total_cost += (new_cost - old_cost)
    
    # ========================================================================
    # Méthodes de requête (lecture)
    # ========================================================================
    
    def get_route_of_customer(self, customer: int) -> Optional[int]:
        """
        Retourne l'indice de la route contenant le client.
        
        Args:
            customer: ID du client
            
        Returns:
            Indice de la route ou None si non trouvé
        """
        return self.client_to_route.get(customer)
    
    def get_position_of_customer(self, customer: int) -> Optional[int]:
        """
        Retourne la position du client dans sa route.
        
        Args:
            customer: ID du client
            
        Returns:
            Position dans la route ou None si non trouvé
        """
        return self.client_position.get(customer)
    
    def get_customers(self) -> Set[int]:
        """Retourne l'ensemble de tous les clients dans la solution."""
        return set(self.client_to_route.keys())
    
    def get_n_routes(self) -> int:
        """Retourne le nombre total de routes (incluant vides)."""
        return len(self.routes)
    
    def get_n_customers(self) -> int:
        """Retourne le nombre total de clients dans la solution."""
        return len(self.client_to_route)
    
    def is_empty(self) -> bool:
        """Vérifie si la solution est vide (aucune route)."""
        return len(self.routes) == 0
    
    # ========================================================================
    # Validation et cohérence
    # ========================================================================
    
    def validate_consistency(self) -> Tuple[bool, List[str]]:
        """
        Vérifie la cohérence interne de la solution.
        
        Vérifie que:
        - client_to_route correspond aux routes
        - client_position correspond aux routes
        - route_costs et total_cost sont cohérents
        - n_vehicles_used est correct
        - Pas de doublons de clients
        
        Returns:
            (is_consistent, list_of_errors)
        """
        errors = []
        
        # 1. Vérifie que chaque client dans routes est dans les mappings
        all_customers_in_routes = set()
        
        for route_idx, route in enumerate(self.routes):
            for position, customer in enumerate(route):
                # Détecte doublons
                if customer in all_customers_in_routes:
                    errors.append(f"Customer {customer} appears multiple times")
                all_customers_in_routes.add(customer)
                
                # Vérifie client_to_route
                if customer not in self.client_to_route:
                    errors.append(f"Customer {customer} in routes but not in client_to_route")
                elif self.client_to_route[customer] != route_idx:
                    errors.append(
                        f"Customer {customer} in route {route_idx} but "
                        f"client_to_route says {self.client_to_route[customer]}"
                    )
                
                # Vérifie client_position
                if customer not in self.client_position:
                    errors.append(f"Customer {customer} in routes but not in client_position")
                elif self.client_position[customer] != position:
                    errors.append(
                        f"Customer {customer} at position {position} but "
                        f"client_position says {self.client_position[customer]}"
                    )
        
        # 2. Vérifie que chaque client dans les mappings est dans routes
        for customer, route_idx in self.client_to_route.items():
            if customer not in all_customers_in_routes:
                errors.append(f"Customer {customer} in mappings but not in routes")
        
        # 3. Vérifie total_cost
        expected_total = sum(self.route_costs)
        if abs(self.total_cost - expected_total) > 1e-6:
            errors.append(
                f"total_cost={self.total_cost} but sum(route_costs)={expected_total}"
            )
        
        # 4. Vérifie n_vehicles_used
        expected_vehicles = sum(1 for route in self.routes if len(route) > 0)
        if self.n_vehicles_used != expected_vehicles:
            errors.append(
                f"n_vehicles_used={self.n_vehicles_used} but actual={expected_vehicles}"
            )
        
        # 5. Vérifie cohérence taille des listes
        if len(self.routes) != len(self.route_costs):
            errors.append(
                f"len(routes)={len(self.routes)} but len(route_costs)={len(self.route_costs)}"
            )
        
        return len(errors) == 0, errors
    
    def assert_consistency(self):
        """
        Assertion de cohérence : lève une exception si incohérente.
        
        Utilisé en mode debug pour détecter les bugs.
        
        Raises:
            AssertionError: Si la solution est incohérente
        """
        is_consistent, errors = self.validate_consistency()
        if not is_consistent:
            error_msg = "Solution inconsistency detected:\n" + "\n".join(f"  - {e}" for e in errors)
            raise AssertionError(error_msg)
    
    # ========================================================================
    # Conversion et affichage
    # ========================================================================
    
    def to_dict(self) -> dict:
        """Convertit la solution en dictionnaire (pour JSON)."""
        return {
            'routes': self.routes,
            'route_costs': self.route_costs,
            'total_cost': self.total_cost,
            'n_vehicles_used': self.n_vehicles_used,
            'instance_name': self.instance_name,
            'algo_name': self.algo_name
        }
    
    def copy(self) -> 'Solution':
        """Crée une copie profonde de la solution."""
        new_solution = Solution(self.instance_name)
        new_solution.algo_name = self.algo_name
        
        for route, cost in zip(self.routes, self.route_costs):
            new_solution.add_route(route.copy(), cost)
        
        return new_solution
    
    def __repr__(self) -> str:
        """Représentation textuelle de la solution."""
        return (
            f"Solution(n_routes={len(self.routes)}, "
            f"n_customers={self.get_n_customers()}, "
            f"n_vehicles={self.n_vehicles_used}, "
            f"cost={self.total_cost:.2f})"
        )
    
    def __str__(self) -> str:
        """Affichage détaillé de la solution."""
        lines = [f"Solution: {self.total_cost:.2f}"]
        for i, (route, cost) in enumerate(zip(self.routes, self.route_costs)):
            if len(route) > 0:
                lines.append(f"  Route {i+1}: {route} (cost={cost:.2f})")
        return "\n".join(lines)
    

    # ========================================================================
    # Vérification full VRP
    # ========================================================================

    def is_valid_vrp(self, instance, evaluator) -> Tuple[bool, str]:
        """
        Vérifie qu'une solution VRP est valide.
        Version SIMPLE et RAPIDE.
        
        Vérifie :
        1. Cohérence interne de la structure
        2. Toutes les routes sont évaluables (capacité + temps OK)
        3. Tous les clients visités exactement une fois
        
        Args:
            instance: Instance VRP
            evaluator: RouteEvaluator
            
        Returns:
            (is_valid, error_message)
            Si is_valid=True, error_message est vide
            Si is_valid=False, error_message explique pourquoi
        """
        # 1. Cohérence interne
        is_consistent, errors = self.validate_consistency()
        if not is_consistent:
            return False, f"Internal inconsistency: {errors[0]}"
        
        # 2. Toutes les routes évaluables
        for route_idx, route in enumerate(self.routes):
            if len(route) == 0:
                continue
            
            try:
                if not evaluator.evaluate_route(route).is_feasible:
                    return False, f"Route {route_idx} is invalid (cost={evaluator.evaluate_route(route).cost:.0f})"
            except Exception as e:
                return False, f"Route {route_idx} cannot be evaluated: {str(e)}"
        
        # 3. Tous les clients visités exactement une fois
        clients_in_solution = set()
        for route in self.routes:
            for customer in route:
                if customer in clients_in_solution:
                    return False, f"Customer {customer} appears multiple times"
                clients_in_solution.add(customer)
        
        expected_clients = set(range(1, instance.dimension))
        
        missing = expected_clients - clients_in_solution
        if missing:
            return False, f"Missing customers: {sorted(list(missing)[:5])}"
        
        extra = clients_in_solution - expected_clients
        if extra:
            return False, f"Extra customers (not in instance): {sorted(list(extra)[:5])}"
        
        return True, ""


    def assert_valid_vrp(self, instance, evaluator):
        """
        Version assertion : lève une exception si invalide.
        Pratique pour mode debug.
        """
        is_valid, error = self.is_valid_vrp(instance, evaluator)
        if not is_valid:
            raise AssertionError(f"Invalid VRP solution: {error}")