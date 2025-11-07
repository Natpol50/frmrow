"""
Constructors - Fonctions pour créer des solutions initiales VRP - Asha Geyon 2025

Ce module fournit différentes méthodes pour construire une solution initiale
à partir d'une instance VRP. Une solution initiale est nécessaire pour démarrer
les algorithmes d'optimisation (recherche locale, métaheuristiques, etc.)

Constructeurs disponibles:
- nearest_neighbor: Glouton, rapide, qualité moyenne
- random: Totalement aléatoire, qualité mauvaise, utile pour tests
- savings: Clarke & Wright, lent mais meilleure qualité

Usage:
    from constructors import nearest_neighbor
    from instance_file_manager import InstanceFileManager
    from route_evaluator import RouteEvaluator
    
    instance = InstanceFileManager("data").load_instance("C101")
    evaluator = RouteEvaluator(instance)
    solution = nearest_neighbor(instance, evaluator)
"""

import numpy as np
from typing import List, Tuple, Set
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solvermanager.solutionclass import Solution
from solvermanager.routemanager import RouteEvaluator
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def nearest_neighbor(instance, evaluator: RouteEvaluator, 
                    start_from_depot: bool = True) -> Solution:
    """
    Constructeur glouton Nearest Neighbor (plus proche voisin).
    
    Algorithme:
    1. Part du dépôt
    2. À chaque étape, ajoute le client non visité le plus proche
    3. Si ajout impossible (capacité/TW) → nouvelle route
    4. Répète jusqu'à ce que tous les clients soient visités
    
    Complexité: O(n²) où n = nombre de clients
    Qualité: Moyenne (bon point de départ pour optimisation)
    
    Args:
        instance: Instance VRP
        evaluator: RouteEvaluator pour vérifier faisabilité
        start_from_depot: Si True, commence toujours au dépôt (recommandé)
        
    Returns:
        Solution construite (garantie faisable)
    """
    solution = Solution(instance.name)
    solution.algo_name = "nearest_neighbor_constructor"
    
    # Clients à visiter
    unvisited = set(range(1, instance.dimension))  # Exclut le dépôt (0)
    depot = instance.depot
    
    while unvisited:
        # Nouvelle route
        current_route = []
        current_location = depot
        route_demand = 0
        route_time = 0.0  # Pour VRPTW
        
        while unvisited:
            # Trouve le plus proche client non visité
            best_customer = None
            best_distance = float('inf')
            
            for customer in unvisited:
                distance = instance.distance_matrix[current_location, customer]
                if distance < best_distance:
                    best_distance = distance
                    best_customer = customer
            
            if best_customer is None:
                break
            
            # Essaie d'ajouter ce client
            candidate_route = current_route + [best_customer]
            
            # Vérifie faisabilité
            is_feasible, cost = evaluator.evaluate_route_fast(candidate_route)
            
            if is_feasible:
                # Accepte
                current_route.append(best_customer)
                unvisited.remove(best_customer)
                current_location = best_customer
                route_demand += instance.demands[best_customer]
            else:
                # Ne peut pas ajouter → commence nouvelle route
                break
        
        # Ajoute la route si non vide
        if current_route:
            # Calcule le coût final
            _, route_cost = evaluator.evaluate_route_fast(current_route)
            solution.add_route(current_route, cost=route_cost)
    
    return solution


def random_constructor(instance, evaluator: RouteEvaluator, 
                      seed: int = 3,
                      max_attempts: int = 100) -> Solution:
    """
    Constructeur aléatoire - insère les clients dans un ordre aléatoire.
    
    Algorithme:
    1. Mélange les clients aléatoirement
    2. Essaie de les insérer dans l'ordre mélangé
    3. Si impossible → nouvelle route
    
    Complexité: O(n²)
    Qualité: Mauvaise (mais utile pour tests/benchmarks)
    
    Args:
        instance: Instance VRP
        evaluator: RouteEvaluator
        seed: Graine aléatoire (pour reproductibilité)
        max_attempts: Nombre max de tentatives pour construire une solution
        
    Returns:
        Solution construite (garantie faisable)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    solution = Solution(instance.name)
    solution.algo_name = "random_constructor"
    
    # Clients dans un ordre aléatoire
    customers = list(range(1, instance.dimension))
    random.shuffle(customers)
    
    current_route = []
    
    for customer in customers:
        # Essaie d'ajouter à la route actuelle
        candidate_route = current_route + [customer]
        is_feasible, cost = evaluator.evaluate_route_fast(candidate_route)
        
        if is_feasible:
            current_route.append(customer)
        else:
            # Route pleine → sauvegarde et commence nouvelle
            if current_route:
                _, route_cost = evaluator.evaluate_route_fast(current_route)
                solution.add_route(current_route, cost=route_cost)
            
            # Nouvelle route avec ce client
            current_route = [customer]
    
    # Ajoute la dernière route
    if current_route:
        _, route_cost = evaluator.evaluate_route_fast(current_route)
        solution.add_route(current_route, cost=route_cost)
    
    return solution


def savings_constructor(instance, evaluator: RouteEvaluator,
                       parallel: bool = True) -> Solution:
    """
    Constructeur Clarke & Wright amélioré:
    - Calcul vectorisé des savings (numpy)
    - Évaluation des options de fusion en parallèle (ThreadPool)
    - Mappage customer->route pour éviter les recherches linéaires
    """

    solution = Solution(instance.name)
    solution.algo_name = "savings_constructor"

    depot = instance.depot
    n_customers = instance.dimension - 1  # sans dépôt
    # Étape 1: une route par client + mapping client->route
    for customer in range(1, instance.dimension):
        _, cost = evaluator.evaluate_route_fast([customer])
        solution.add_route([customer], cost=cost)
    customer_to_route = {customer: idx for idx, customer in enumerate(range(1, instance.dimension))}

    # Étape 2: calcul vectorisé des savings
    idxs = np.arange(1, instance.dimension)
    if n_customers > 0:
        D_sub = instance.distance_matrix[np.ix_(idxs, idxs)]
        d0_vec = instance.distance_matrix[depot, idxs]
        savings_matrix = d0_vec[:, None] + d0_vec[None, :] - D_sub
        a, b = np.triu_indices(n_customers, k=1)
        savings_values = savings_matrix[a, b]
        savings_list = list(zip(savings_values.tolist(), idxs[a].tolist(), idxs[b].tolist()))
    else:
        savings_list = []

    # Étape 3: tri décroissant
    savings_list.sort(reverse=True, key=lambda x: x[0])

    # Pool threads pour évaluer rapidement les options de fusion
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    executor = ThreadPoolExecutor(max_workers=max_workers)

    # Fonction utilitaire pour évaluer une route (wrapper)
    def eval_route(route):
        return evaluator.evaluate_route_fast(route)

    # Étape 4: parcours des savings et tentative de fusion
    for saving_value, i, j in savings_list:
        # Rechercher routes actuelles via mapping (très rapide)
        route_i_idx = customer_to_route.get(i)
        route_j_idx = customer_to_route.get(j)

        if route_i_idx is None or route_j_idx is None:
            continue
        if route_i_idx == route_j_idx:
            continue

        route_i = solution.routes[route_i_idx]
        route_j = solution.routes[route_j_idx]

        # Générer options de fusion selon mode
        if parallel:
            merge_options = [
                route_i + route_j,
                route_i + route_j[::-1],
                route_j + route_i,
                route_j[::-1] + route_i
            ]
        else:
            # Séquentiel: i et j doivent être aux extrémités
            if not route_i or not route_j:
                continue
            i_at_end = (route_i[-1] == i)
            i_at_start = (route_i[0] == i)
            j_at_end = (route_j[-1] == j)
            j_at_start = (route_j[0] == j)
            if not ((i_at_end or i_at_start) and (j_at_end or j_at_start)):
                continue
            if i_at_end and j_at_start:
                merge_options = [route_i + route_j]
            elif i_at_start and j_at_end:
                merge_options = [route_j + route_i]
            elif i_at_end and j_at_end:
                merge_options = [route_i + route_j[::-1]]
            elif i_at_start and j_at_start:
                merge_options = [route_i[::-1] + route_j]
            else:
                continue

        # Évaluer les options en parallèle
        futures = {executor.submit(eval_route, opt): opt for opt in merge_options}
        best_merged = None
        best_cost = float('inf')
        for fut in as_completed(futures):
            is_feasible, cost = fut.result()
            if is_feasible and cost < best_cost:
                best_cost = cost
                best_merged = futures[fut]

        if best_merged is not None:
            # Réaliser la fusion: retirer la route avec l'indice le plus élevé d'abord
            # pour éviter de décaler l'autre index
            ri = route_i_idx
            rj = route_j_idx
            if rj > ri:
                solution.remove_route(rj)
                solution.set_route(ri, best_merged, best_cost)
                new_idx = ri
                removed_idx = rj
            else:
                solution.remove_route(rj)
                # si rj < ri, l'index de ri décale de -1
                solution.set_route(ri - 1, best_merged, best_cost)
                new_idx = ri - 1
                removed_idx = rj

            # Mettre à jour mapping customer->route
            # 1) clients de la nouvelle route pointent vers new_idx
            for c in best_merged:
                customer_to_route[c] = new_idx
            # 2) pour les clients dans des routes avec index > removed_idx: décrémenter l'indice
            for cust, r_idx in list(customer_to_route.items()):
                if cust in best_merged:
                    continue
                if r_idx > removed_idx:
                    customer_to_route[cust] = r_idx - 1

    executor.shutdown(wait=True)
    return solution


def sequential_insertion(instance, evaluator: RouteEvaluator,
                        criterion: str = 'cheapest') -> Solution:
    """
    Constructeur par insertion séquentielle.
    
    Algorithme:
    1. Commence avec des routes vides
    2. Pour chaque client non inséré:
       - Trouve la meilleure position d'insertion (selon critère)
       - Insère à cette position
    
    Critères d'insertion:
    - 'cheapest': Position qui augmente le moins le coût
    - 'nearest': Plus proche d'un client déjà dans une route
    - 'farthest': Plus loin du dépôt (pour équilibrer les routes)
    
    Complexité: O(n²k) où n = clients, k = routes
    Qualité: Bonne (souvent meilleure que nearest neighbor)
    
    Args:
        instance: Instance VRP
        evaluator: RouteEvaluator
        criterion: Critère d'insertion ('cheapest', 'nearest', 'farthest')
        
    Returns:
        Solution construite
    """
    solution = Solution(instance.name)
    solution.algo_name = f"sequential_insertion_{criterion}"
    
    depot = instance.depot
    unvisited = set(range(1, instance.dimension))
    
    # Commence avec une route vide
    solution.add_route([], cost=0.0)
    
    while unvisited:
        best_customer = None
        best_route_idx = None
        best_position = None
        best_increase = float('inf')
        
        # Pour chaque client non visité
        for customer in unvisited:
            # Pour chaque route existante
            for route_idx in range(solution.get_n_routes()):
                route = solution.routes[route_idx]
                old_cost = solution.route_costs[route_idx]
                
                # Essaie toutes les positions d'insertion
                for position in range(len(route) + 1):
                    # Crée la route candidate
                    candidate = route.copy()
                    candidate.insert(position, customer)
                    
                    # Évalue
                    is_feasible, new_cost = evaluator.evaluate_route_fast(candidate)
                    
                    if not is_feasible:
                        continue
                    
                    # Calcule l'augmentation de coût
                    if criterion == 'cheapest':
                        increase = new_cost - old_cost
                    elif criterion == 'nearest':
                        # Distance au plus proche voisin dans la route
                        if len(route) == 0:
                            increase = instance.distance_matrix[depot, customer]
                        else:
                            distances = [instance.distance_matrix[c, customer] 
                                       for c in route]
                            increase = min(distances)
                    elif criterion == 'farthest':
                        # Distance au dépôt (négatif pour trier)
                        increase = -instance.distance_matrix[depot, customer]
                    else:
                        raise ValueError(f"Unknown criterion: {criterion}")
                    
                    # Garde le meilleur
                    if increase < best_increase:
                        best_increase = increase
                        best_customer = customer
                        best_route_idx = route_idx
                        best_position = position
        
        # Si aucune insertion possible → nouvelle route
        if best_customer is None:
            # Prend n'importe quel client restant
            customer = unvisited.pop()
            _, cost = evaluator.evaluate_route_fast([customer])
            solution.add_route([customer], cost=cost)
        else:
            # Insère le meilleur client
            route = solution.routes[best_route_idx]
            new_route = route.copy()
            new_route.insert(best_position, best_customer)
            _, new_cost = evaluator.evaluate_route_fast(new_route)
            solution.set_route(best_route_idx, new_route, new_cost)
            unvisited.remove(best_customer)
    
    # Retire les routes vides
    routes_to_remove = []
    for idx in range(solution.get_n_routes()):
        if len(solution.routes[idx]) == 0:
            routes_to_remove.append(idx)
    
    for idx in reversed(routes_to_remove):
        solution.remove_route(idx)
    
    return solution


def get_constructor(name: str):
    """
    Factory pour obtenir un constructeur par son nom.
    
    Args:
        name: Nom du constructeur ('nearest_neighbor', 'random', 'savings', etc.)
        
    Returns:
        Fonction constructeur
        
    Raises:
        ValueError: Si le constructeur n'existe pas
    """
    constructors = {
        'nearest_neighbor': nearest_neighbor,
        'random': random_constructor,
        'savings': savings_constructor,
        'savings_parallel': lambda i, e: savings_constructor(i, e, parallel=True),
        'savings_sequential': lambda i, e: savings_constructor(i, e, parallel=False),
        'insertion_cheapest': lambda i, e: sequential_insertion(i, e, 'cheapest'),
        'insertion_nearest': lambda i, e: sequential_insertion(i, e, 'nearest'),
        'insertion_farthest': lambda i, e: sequential_insertion(i, e, 'farthest'),
    }
    
    if name not in constructors:
        raise ValueError(
            f"Unknown constructor: {name}. "
            f"Available: {list(constructors.keys())}"
        )
    
    return constructors[name]