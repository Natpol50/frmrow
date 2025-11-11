"""
Operators - Opérateurs de voisinage pour VRP - Asha Geyon 2025

Ce module fournit les opérateurs de voisinage classiques pour modifier
une solution VRP. Chaque opérateur essaie une modification et l'applique
seulement si elle est faisable et améliore la solution.

Opérateurs disponibles:
- try_relocate: Déplace un client vers une autre route
- try_2opt_intra: 2-opt dans UNE route
- try_exchange: Échange deux clients entre routes
- try_cross: 2-opt inter-routes

Usage:
    from operators import try_relocate
    
    if try_relocate(solution, evaluator, customer=5, from_idx=0, to_idx=1, position=2):
        print("Amélioration trouvée!")
"""

from typing import Optional, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solvermanager.solutionclass import Solution
from solvermanager.routemanager import RouteEvaluator


def try_relocate(solution: Solution, 
                evaluator: RouteEvaluator,
                customer: int,
                from_route_idx: int,
                to_route_idx: int,
                to_position: int,
                accept_equal: bool = False) -> bool:
    """
    Essaie de déplacer un client d'une route à une autre.
    
    Algorithme:
    1. Crée les routes modifiées (sans/avec le client)
    2. Évalue avec RouteEvaluator
    3. Si faisable ET meilleur → applique
    4. Retourne True si appliqué, False sinon
    
    Args:
        solution: Solution à modifier
        evaluator: RouteEvaluator pour vérifier faisabilité
        customer: ID du client à déplacer
        from_route_idx: Route source
        to_route_idx: Route destination
        to_position: Position d'insertion dans la route destination
        accept_equal: Si True, accepte les mouvements à coût égal
        
    Returns:
        True si le mouvement a été appliqué, False sinon
    """
    # Vérifie que le client est bien dans from_route
    if solution.get_route_of_customer(customer) != from_route_idx:
        return False
    
    # Crée les nouvelles routes
    from_route = [c for c in solution.routes[from_route_idx] if c != customer]
    to_route = solution.routes[to_route_idx].copy()
    
    # Vérifie position valide
    if to_position < 0 or to_position > len(to_route):
        return False
    
    to_route.insert(to_position, customer)
    
    # Évalue les deux routes
    from_feasible, from_cost = evaluator.evaluate_route_fast(from_route)
    to_feasible, to_cost = evaluator.evaluate_route_fast(to_route)
    
    if not (from_feasible and to_feasible):
        return False
    
    # Calcule le delta de coût
    old_cost = solution.route_costs[from_route_idx] + solution.route_costs[to_route_idx]
    new_cost = from_cost + to_cost
    
    # Accepte si amélioration (ou égalité si accept_equal)
    if new_cost < old_cost or (accept_equal and new_cost == old_cost):
        solution.relocate_customer(
            customer=customer,
            from_route_idx=from_route_idx,
            to_route_idx=to_route_idx,
            to_position=to_position,
            new_from_cost=from_cost,
            new_to_cost=to_cost
        )
        return True
    
    return False


def try_2opt_intra(solution: Solution,
                  evaluator: RouteEvaluator,
                  route_idx: int,
                  i: int,
                  j: int,
                  accept_equal: bool = False) -> bool:
    """
    Essaie un 2-opt intra-route (inverse un segment).
    
    Algorithme:
    1. Inverse le segment [i+1:j] de la route
    2. Évalue la nouvelle route
    3. Si faisable ET meilleur → applique
    
    Args:
        solution: Solution à modifier
        evaluator: RouteEvaluator
        route_idx: Indice de la route à modifier
        i: Début du segment (le segment inversé commence à i+1)
        j: Fin du segment (exclus)
        accept_equal: Si True, accepte les mouvements à coût égal
        
    Returns:
        True si appliqué, False sinon
    """
    route = solution.routes[route_idx]
    
    # Vérifie indices valides
    if i < 0 or j > len(route) or i >= j - 1:
        return False
    
    # Crée la route modifiée
    new_route = route.copy()
    new_route[i+1:j] = reversed(new_route[i+1:j])
    
    # Évalue
    is_feasible, new_cost = evaluator.evaluate_route_fast(new_route)
    
    if not is_feasible:
        return False
    
    old_cost = solution.route_costs[route_idx]
    
    if new_cost < old_cost or (accept_equal and new_cost == old_cost):
        solution.reverse_segment(route_idx, i+1, j, new_cost)
        return True
    
    return False


def try_exchange(solution: Solution,
                evaluator: RouteEvaluator,
                customer1: int,
                customer2: int,
                accept_equal: bool = False) -> bool:
    """
    Essaie d'échanger deux clients entre leurs routes.
    
    Algorithme:
    1. Échange les clients dans leurs routes respectives
    2. Évalue les deux routes modifiées
    3. Si faisable ET meilleur → applique
    
    Args:
        solution: Solution à modifier
        evaluator: RouteEvaluator
        customer1: Premier client
        customer2: Deuxième client
        accept_equal: Si True, accepte les mouvements à coût égal
        
    Returns:
        True si appliqué, False sinon
    """
    route1_idx = solution.get_route_of_customer(customer1)
    route2_idx = solution.get_route_of_customer(customer2)
    
    if route1_idx is None or route2_idx is None:
        return False
    
    if route1_idx == route2_idx:
        return False  # Même route
    
    # Crée les routes modifiées
    pos1 = solution.get_position_of_customer(customer1)
    pos2 = solution.get_position_of_customer(customer2)
    
    route1 = solution.routes[route1_idx].copy()
    route2 = solution.routes[route2_idx].copy()
    
    route1[pos1] = customer2
    route2[pos2] = customer1
    
    # Évalue
    feasible1, cost1 = evaluator.evaluate_route_fast(route1)
    feasible2, cost2 = evaluator.evaluate_route_fast(route2)
    
    if not (feasible1 and feasible2):
        return False
    
    old_cost = solution.route_costs[route1_idx] + solution.route_costs[route2_idx]
    new_cost = cost1 + cost2
    
    if new_cost < old_cost or (accept_equal and new_cost == old_cost):
        solution.exchange_customers(customer1, customer2, cost1, cost2)
        return True
    
    return False


def try_cross(solution: Solution,
             evaluator: RouteEvaluator,
             route1_idx: int,
             route2_idx: int,
             pos1: int,
             pos2: int,
             accept_equal: bool = False) -> bool:
    """
    Essaie un 2-opt* inter-routes (cross).
    
    Algorithme:
    Coupe deux routes et recroise:
    - Route1: [0, a, b, c, d, 0] coupée après b
    - Route2: [0, e, f, g, h, 0] coupée après f
    - Nouvelle1: [0, a, b, g, h, 0]
    - Nouvelle2: [0, e, f, c, d, 0]
    
    Args:
        solution: Solution à modifier
        evaluator: RouteEvaluator
        route1_idx: Première route
        route2_idx: Deuxième route
        pos1: Position de coupe dans route1 (après pos1)
        pos2: Position de coupe dans route2 (après pos2)
        accept_equal: Si True, accepte les mouvements à coût égal
        
    Returns:
        True si appliqué, False sinon
    """
    if route1_idx == route2_idx:
        return False
    
    route1 = solution.routes[route1_idx]
    route2 = solution.routes[route2_idx]
    
    # Vérifie positions valides
    if pos1 < 0 or pos1 >= len(route1):
        return False
    if pos2 < 0 or pos2 >= len(route2):
        return False
    
    # Crée les nouvelles routes
    new_route1 = route1[:pos1+1] + route2[pos2+1:]
    new_route2 = route2[:pos2+1] + route1[pos1+1:]
    
    # Évalue
    feasible1, cost1 = evaluator.evaluate_route_fast(new_route1)
    feasible2, cost2 = evaluator.evaluate_route_fast(new_route2)
    
    if not (feasible1 and feasible2):
        return False
    
    old_cost = solution.route_costs[route1_idx] + solution.route_costs[route2_idx]
    new_cost = cost1 + cost2
    
    if new_cost < old_cost or (accept_equal and new_cost == old_cost):
        # Applique les modifications
        solution.set_route(route1_idx, new_route1, cost1)
        solution.set_route(route2_idx, new_route2, cost2)
        return True
    
    return False


def try_swap_intra(solution: Solution,
                  evaluator: RouteEvaluator,
                  route_idx: int,
                  i: int,
                  j: int,
                  accept_equal: bool = False) -> bool:
    """
    Essaie d'échanger deux clients dans la MÊME route.
    
    Args:
        solution: Solution à modifier
        evaluator: RouteEvaluator
        route_idx: Indice de la route
        i: Position du premier client
        j: Position du deuxième client
        accept_equal: Si True, accepte les mouvements à coût égal
        
    Returns:
        True si appliqué, False sinon
    """
    route = solution.routes[route_idx]
    
    if i < 0 or j >= len(route) or i >= j:
        return False
    
    # Crée la route modifiée
    new_route = route.copy()
    new_route[i], new_route[j] = new_route[j], new_route[i]
    
    # Évalue
    is_feasible, new_cost = evaluator.evaluate_route_fast(new_route)
    
    if not is_feasible:
        return False
    
    old_cost = solution.route_costs[route_idx]
    
    if new_cost < old_cost or (accept_equal and new_cost == old_cost):
        solution.set_route(route_idx, new_route, new_cost)
        return True
    
    return False


def find_best_relocate(solution: Solution,
                      evaluator: RouteEvaluator,
                      customer: int) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Trouve la meilleure position pour relocate un client.
    
    Explore toutes les routes et positions possibles et retourne la meilleure.
    
    Args:
        solution: Solution
        evaluator: RouteEvaluator
        customer: Client à déplacer
        
    Returns:
        (improved, best_route_idx, best_position) ou (False, None, None) si aucune amélioration
    """
    from_route_idx = solution.get_route_of_customer(customer)
    if from_route_idx is None:
        return False, None, None
    
    best_delta = 0
    best_route_idx = None
    best_position = None
    
    old_from_cost = solution.route_costs[from_route_idx]
    
    # Teste toutes les routes
    for to_route_idx in range(solution.get_n_routes()):
        to_route = solution.routes[to_route_idx]
        
        # Teste toutes les positions
        for position in range(len(to_route) + 1):
            # Crée les nouvelles routes
            from_route = [c for c in solution.routes[from_route_idx] if c != customer]
            new_to_route = to_route.copy()
            new_to_route.insert(position, customer)
            
            # Évalue
            from_feasible, from_cost = evaluator.evaluate_route_fast(from_route)
            to_feasible, to_cost = evaluator.evaluate_route_fast(new_to_route)
            
            if not (from_feasible and to_feasible):
                continue
            
            # Calcule delta
            if to_route_idx == from_route_idx:
                # Même route
                delta = to_cost - old_from_cost
            else:
                old_to_cost = solution.route_costs[to_route_idx]
                delta = (from_cost + to_cost) - (old_from_cost + old_to_cost)
            
            if delta < best_delta:
                best_delta = delta
                best_route_idx = to_route_idx
                best_position = position
    
    if best_route_idx is not None:
        # Applique le meilleur mouvement
        success = try_relocate(solution, evaluator, customer, 
                             from_route_idx, best_route_idx, best_position)
        return success, best_route_idx, best_position
    
    return False, None, None


def find_best_2opt_intra(solution: Solution,
                        evaluator: RouteEvaluator,
                        route_idx: int) -> bool:
    """
    Trouve le meilleur 2-opt dans une route.
    
    Explore tous les segments possibles et applique le meilleur.
    
    Args:
        solution: Solution
        evaluator: RouteEvaluator
        route_idx: Indice de la route
        
    Returns:
        True si une amélioration a été trouvée
    """
    route = solution.routes[route_idx]
    
    if len(route) < 3:
        return False
    
    best_delta = 0
    best_i = None
    best_j = None
    
    old_cost = solution.route_costs[route_idx]
    
    for i in range(len(route) - 1):
        for j in range(i + 2, len(route) + 1):
            # Crée la route modifiée
            new_route = route.copy()
            new_route[i+1:j] = reversed(new_route[i+1:j])
            
            # Évalue
            is_feasible, new_cost = evaluator.evaluate_route_fast(new_route)
            
            if not is_feasible:
                continue
            
            delta = new_cost - old_cost
            
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j
    
    if best_i is not None:
        return try_2opt_intra(solution, evaluator, route_idx, best_i, best_j)
    
    return False