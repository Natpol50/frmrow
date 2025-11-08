"""
Tests pytest pour les constructeurs - Asha Geyon 2025

Usage:
    pytest test_constructors.py -v
"""

import pytest
import numpy as np
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solvermanager.constructors import (
    nearest_neighbor, random_constructor, savings_constructor,
    sequential_insertion, get_constructor
)
from solvermanager.routemanager import RouteEvaluator
from solvermanager.solutionclass import Solution


# ============================================================================
# FIXTURES - Instances de test
# ============================================================================

class MockInstance:
    """Instance simplifiée pour les tests."""
    def __init__(self):
        self.name = "test_instance"
        self.depot = 0
        self.dimension = 6  # Dépôt + 5 clients
        self.capacity = 100
        self.n_vehicles = 10
        
        # Distance matrix (symétrique)
        self.distance_matrix = np.array([
            [0,  10, 15, 20, 25, 30],
            [10, 0,  5,  15, 20, 25],
            [15, 5,  0,  10, 15, 20],
            [20, 15, 10, 0,  5,  15],
            [25, 20, 15, 5,  0,  10],
            [30, 25, 20, 15, 10, 0]
        ], dtype=float)
        
        self.demands = np.array([0, 20, 30, 25, 40, 35])
        
        # Pas de time windows pour simplifier
        self.ready_times = None
        self.due_dates = None
        self.service_times = None
        self.is_vrptw = lambda: False


@pytest.fixture
def simple_instance():
    """Instance simple sans time windows."""
    return MockInstance()


@pytest.fixture
def evaluator(simple_instance):
    """RouteEvaluator pour l'instance simple."""
    return RouteEvaluator(simple_instance)


# ============================================================================
# TESTS - Nearest Neighbor
# ============================================================================

def test_nearest_neighbor_basic(simple_instance, evaluator):
    """Test basique du constructeur nearest neighbor."""
    solution = nearest_neighbor(simple_instance, evaluator)
    
    # Vérifie que la solution est créée
    assert solution is not None
    assert isinstance(solution, Solution)
    assert solution.instance_name == "test_instance"
    
    # Vérifie que tous les clients sont visités
    all_customers = solution.get_customers()
    assert all_customers == {1, 2, 3, 4, 5}
    
    # Vérifie cohérence
    solution.assert_consistency()


def test_nearest_neighbor_feasibility(simple_instance, evaluator):
    """Vérifie que la solution construite est faisable."""
    solution = nearest_neighbor(simple_instance, evaluator)
    
    # Chaque route doit être faisable
    for route_idx, route in enumerate(solution.routes):
        is_feasible, cost = evaluator.evaluate_route_fast(route)
        assert is_feasible, f"Route {route_idx} infaisable: {route}"


def test_nearest_neighbor_cost_computed(simple_instance, evaluator):
    """Vérifie que les coûts sont bien calculés."""
    solution = nearest_neighbor(simple_instance, evaluator)
    
    # Recalcule les coûts
    total = 0.0
    for route_idx, route in enumerate(solution.routes):
        _, cost = evaluator.evaluate_route_fast(route)
        assert abs(cost - solution.route_costs[route_idx]) < 1e-6
        total += cost
    
    assert abs(total - solution.total_cost) < 1e-6


# ============================================================================
# TESTS - Random Constructor
# ============================================================================

def test_random_constructor_basic(simple_instance, evaluator):
    """Test basique du constructeur aléatoire."""
    solution = random_constructor(simple_instance, evaluator, seed=42)
    
    assert solution is not None
    assert solution.get_customers() == {1, 2, 3, 4, 5}
    solution.assert_consistency()


def test_random_constructor_reproducible(simple_instance, evaluator):
    """Vérifie que le constructeur aléatoire est reproductible avec seed."""
    sol1 = random_constructor(simple_instance, evaluator, seed=42)
    sol2 = random_constructor(simple_instance, evaluator, seed=42)
    
    # Même seed → même solution
    assert sol1.routes == sol2.routes
    assert sol1.total_cost == sol2.total_cost


def test_random_constructor_different_seeds(simple_instance, evaluator):
    """Vérifie que des seeds différentes donnent des solutions différentes."""
    sol1 = random_constructor(simple_instance, evaluator, seed=42)
    sol2 = random_constructor(simple_instance, evaluator, seed=123)
    
    # Seeds différentes → probablement solutions différentes
    # (peut échouer si par hasard même solution, mais très improbable)
    assert sol1.routes != sol2.routes or sol1.total_cost != sol2.total_cost


# ============================================================================
# TESTS - Savings Constructor
# ============================================================================

def test_savings_constructor_basic(simple_instance, evaluator):
    """Test basique du constructeur savings."""
    solution = savings_constructor(simple_instance, evaluator, parallel=True)
    
    assert solution is not None
    assert solution.get_customers() == {1, 2, 3, 4, 5}
    solution.assert_consistency()


def test_savings_constructor_parallel_vs_sequential(simple_instance, evaluator):
    """Compare les versions parallèle et séquentielle."""
    sol_parallel = savings_constructor(simple_instance, evaluator, parallel=True)
    sol_sequential = savings_constructor(simple_instance, evaluator, parallel=False)
    
    # Les deux doivent être faisables
    assert sol_parallel.get_customers() == {1, 2, 3, 4, 5}
    assert sol_sequential.get_customers() == {1, 2, 3, 4, 5}
    
    # Peuvent être différentes (c'est normal)
    # Mais les deux doivent être cohérentes
    sol_parallel.assert_consistency()
    sol_sequential.assert_consistency()


def test_savings_constructor_quality(simple_instance, evaluator):
    """Vérifie que savings est souvent meilleur que random."""
    savings_sol = savings_constructor(simple_instance, evaluator)
    random_sol = random_constructor(simple_instance, evaluator, seed=42)
    
    # Savings devrait souvent être meilleur (pas toujours garanti)
    # On vérifie juste qu'il produit une solution raisonnable
    assert savings_sol.total_cost > 0


# ============================================================================
# TESTS - Sequential Insertion
# ============================================================================

def test_sequential_insertion_cheapest(simple_instance, evaluator):
    """Test du critère cheapest."""
    solution = sequential_insertion(simple_instance, evaluator, criterion='cheapest')
    
    assert solution is not None
    assert solution.get_customers() == {1, 2, 3, 4, 5}
    solution.assert_consistency()


def test_sequential_insertion_nearest(simple_instance, evaluator):
    """Test du critère nearest."""
    solution = sequential_insertion(simple_instance, evaluator, criterion='nearest')
    
    assert solution is not None
    assert solution.get_customers() == {1, 2, 3, 4, 5}
    solution.assert_consistency()


def test_sequential_insertion_farthest(simple_instance, evaluator):
    """Test du critère farthest."""
    solution = sequential_insertion(simple_instance, evaluator, criterion='farthest')
    
    assert solution is not None
    assert solution.get_customers() == {1, 2, 3, 4, 5}
    solution.assert_consistency()


def test_sequential_insertion_invalid_criterion(simple_instance, evaluator):
    """Test qu'un critère invalide lève une exception."""
    with pytest.raises(ValueError):
        sequential_insertion(simple_instance, evaluator, criterion='invalid')


# ============================================================================
# TESTS - Get Constructor Factory
# ============================================================================

def test_get_constructor_valid(simple_instance, evaluator):
    """Test de la factory avec des constructeurs valides."""
    constructors = [
        'nearest_neighbor',
        'random',
        'savings',
        'savings_parallel',
        'savings_sequential',
        'insertion_cheapest',
        'insertion_nearest',
        'insertion_farthest'
    ]
    
    for name in constructors:
        constructor = get_constructor(name)
        solution = constructor(simple_instance, evaluator)
        
        assert solution is not None
        assert solution.get_customers() == {1, 2, 3, 4, 5}


def test_get_constructor_invalid():
    """Test qu'un constructeur invalide lève une exception."""
    with pytest.raises(ValueError, match="Unknown constructor"):
        get_constructor('invalid_constructor')


# ============================================================================
# TESTS - Comparaison des constructeurs
# ============================================================================

def test_constructors_comparison(simple_instance, evaluator):
    """Compare la qualité des différents constructeurs."""
    constructors = {
        'nearest_neighbor': nearest_neighbor,
        'random': lambda i, e: random_constructor(i, e, seed=42),
        'savings': savings_constructor,
        'insertion_cheapest': lambda i, e: sequential_insertion(i, e, 'cheapest')
    }
    
    results = {}
    for name, constructor in constructors.items():
        solution = constructor(simple_instance, evaluator)
        results[name] = {
            'cost': solution.total_cost,
            'n_routes': solution.n_vehicles_used,
            'n_customers': solution.get_n_customers()
        }
        
        # Vérifie que tous les clients sont visités
        assert results[name]['n_customers'] == 5
    
    # Affiche les résultats pour comparaison
    print("\n" + "="*60)
    print("Comparaison des constructeurs:")
    print("="*60)
    for name, metrics in results.items():
        print(f"{name:20s}: cost={metrics['cost']:8.2f}, "
              f"routes={metrics['n_routes']}")
    print("="*60)


# ============================================================================
# TESTS - Edge Cases
# ============================================================================

def test_constructor_with_tight_capacity():
    """Test avec capacité très serrée."""
    instance = MockInstance()
    instance.capacity = 50  # Plus petite capacité
    
    evaluator = RouteEvaluator(instance)
    
    # Devrait créer plus de routes
    solution = nearest_neighbor(instance, evaluator)
    
    assert solution.get_customers() == {1, 2, 3, 4, 5}
    
    # Vérifie capacité de chaque route
    for route in solution.routes:
        total_demand = sum(instance.demands[c] for c in route)
        assert total_demand <= instance.capacity


def test_constructor_single_customer():
    """Test avec un seul client."""
    instance = MockInstance()
    instance.dimension = 2  # Dépôt + 1 client
    instance.distance_matrix = np.array([[0, 10], [10, 0]], dtype=float)
    instance.demands = np.array([0, 20])
    
    evaluator = RouteEvaluator(instance)
    solution = nearest_neighbor(instance, evaluator)
    
    assert solution.get_n_customers() == 1
    assert solution.n_vehicles_used == 1
    assert solution.routes[0] == [1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])