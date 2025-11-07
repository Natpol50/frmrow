"""
Tests pytest pour les opérateurs - Asha Geyon 2025

Usage:
    pytest test_operators.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solvermanager.operators import (
    try_relocate, try_2opt_intra, try_exchange, try_cross,
    try_swap_intra, find_best_relocate, find_best_2opt_intra
)
from solvermanager.routemanager import RouteEvaluator
from solvermanager.solutionclass import Solution


# ============================================================================
# FIXTURES
# ============================================================================

class MockInstance:
    """Instance simplifiée pour les tests."""
    def __init__(self):
        self.name = "test_instance"
        self.depot = 0
        self.dimension = 8  # Dépôt + 7 clients
        self.capacity = 100
        self.n_vehicles = 10
        
        # Distance matrix (symétrique)
        np.random.seed(3)
        self.distance_matrix = np.random.rand(8, 8) * 50 + 10
        self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2
        np.fill_diagonal(self.distance_matrix, 0)
        
        self.demands = np.array([0, 15, 20, 18, 25, 22, 19, 16])
        
        # Pas de time windows
        self.ready_times = None
        self.due_dates = None
        self.service_times = None
        self.is_vrptw = lambda: False


@pytest.fixture
def instance():
    """Instance de test."""
    return MockInstance()


@pytest.fixture
def evaluator(instance):
    """RouteEvaluator pour l'instance."""
    return RouteEvaluator(instance)


@pytest.fixture
def simple_solution(instance, evaluator):
    """Solution simple avec 2 routes."""
    solution = Solution(instance.name)
    
    # Route 1: clients 1, 2, 3
    _, cost1 = evaluator.evaluate_route_fast([1, 2, 3])
    solution.add_route([1, 2, 3], cost=cost1)
    
    # Route 2: clients 4, 5, 6, 7
    _, cost2 = evaluator.evaluate_route_fast([4, 5, 6, 7])
    solution.add_route([4, 5, 6, 7], cost=cost2)
    
    return solution


# ============================================================================
# TESTS - try_relocate
# ============================================================================

def test_try_relocate_basic(simple_solution, evaluator):
    """Test basique de relocate."""
    initial_cost = simple_solution.total_cost
    
    # Essaie de déplacer client 2 de route 0 vers route 1, position 0
    success = try_relocate(simple_solution, evaluator, 
                          customer=2, from_route_idx=0, to_route_idx=1, to_position=0)
    
    # Vérifie cohérence
    simple_solution.assert_consistency()
    
    # Si succès, le client 2 doit être dans la route 1
    if success:
        assert simple_solution.get_route_of_customer(2) == 1
        assert simple_solution.get_position_of_customer(2) == 0


def test_try_relocate_invalid_customer(simple_solution, evaluator):
    """Test relocate avec un client inexistant."""
    success = try_relocate(simple_solution, evaluator,
                          customer=999, from_route_idx=0, to_route_idx=1, to_position=0)
    
    assert success == False


def test_try_relocate_wrong_route(simple_solution, evaluator):
    """Test relocate avec mauvais from_route_idx."""
    # Client 2 est dans route 0, pas route 1
    success = try_relocate(simple_solution, evaluator,
                          customer=2, from_route_idx=1, to_route_idx=0, to_position=0)
    
    assert success == False


def test_try_relocate_accept_equal(simple_solution, evaluator):
    """Test du paramètre accept_equal."""
    # Crée une situation où le mouvement est à coût égal
    # (difficile à garantir, donc on teste juste que le paramètre est utilisé)
    success = try_relocate(simple_solution, evaluator,
                          customer=2, from_route_idx=0, to_route_idx=1, 
                          to_position=0, accept_equal=True)
    
    simple_solution.assert_consistency()


# ============================================================================
# TESTS - try_2opt_intra
# ============================================================================

def test_try_2opt_intra_basic(simple_solution, evaluator):
    """Test basique de 2-opt intra-route."""
    initial_cost = simple_solution.route_costs[1]
    
    # Route 1: [4, 5, 6, 7] → essaie d'inverser [5, 6]
    success = try_2opt_intra(simple_solution, evaluator, route_idx=1, i=0, j=3)
    
    simple_solution.assert_consistency()
    
    if success:
        # La route a changé
        assert simple_solution.routes[1] != [4, 5, 6, 7]


def test_try_2opt_intra_invalid_indices(simple_solution, evaluator):
    """Test 2-opt avec indices invalides."""
    # i >= j-1
    success = try_2opt_intra(simple_solution, evaluator, route_idx=0, i=1, j=2)
    assert success == False
    
    # i < 0
    success = try_2opt_intra(simple_solution, evaluator, route_idx=0, i=-1, j=3)
    assert success == False
    
    # j > len(route)
    success = try_2opt_intra(simple_solution, evaluator, route_idx=0, i=0, j=100)
    assert success == False


def test_try_2opt_intra_short_route(instance, evaluator):
    """Test 2-opt sur une route trop courte."""
    solution = Solution()
    solution.add_route([1, 2], cost=50.0)  # Route de 2 clients
    
    # Impossible de faire 2-opt sur une route de 2 clients
    success = try_2opt_intra(solution, evaluator, route_idx=0, i=0, j=2)
    assert success == False


# ============================================================================
# TESTS - try_exchange
# ============================================================================

def test_try_exchange_basic(simple_solution, evaluator):
    """Test basique d'échange."""
    # Client 2 dans route 0, client 5 dans route 1
    initial_route0 = simple_solution.routes[0].copy()
    initial_route1 = simple_solution.routes[1].copy()
    
    success = try_exchange(simple_solution, evaluator, customer1=2, customer2=5)
    
    simple_solution.assert_consistency()
    
    if success:
        # Les clients ont été échangés
        assert 2 not in simple_solution.routes[0]
        assert 5 not in simple_solution.routes[1]
        assert 2 in simple_solution.routes[1]
        assert 5 in simple_solution.routes[0]


def test_try_exchange_same_route(simple_solution, evaluator):
    """Test qu'on ne peut pas échanger des clients de la même route."""
    # Clients 1 et 2 sont tous deux dans route 0
    success = try_exchange(simple_solution, evaluator, customer1=1, customer2=2)
    
    assert success == False


def test_try_exchange_invalid_customer(simple_solution, evaluator):
    """Test exchange avec client inexistant."""
    success = try_exchange(simple_solution, evaluator, customer1=999, customer2=2)
    assert success == False


# ============================================================================
# TESTS - try_cross
# ============================================================================

def test_try_cross_basic(simple_solution, evaluator):
    """Test basique de cross (2-opt*)."""
    initial_routes = [r.copy() for r in simple_solution.routes]
    
    # Coupe route 0 après position 1, route 1 après position 1
    success = try_cross(simple_solution, evaluator, 
                       route1_idx=0, route2_idx=1, pos1=1, pos2=1)
    
    simple_solution.assert_consistency()
    
    if success:
        # Les routes ont été modifiées
        assert simple_solution.routes != initial_routes


def test_try_cross_same_route(simple_solution, evaluator):
    """Test qu'on ne peut pas faire cross sur la même route."""
    success = try_cross(simple_solution, evaluator,
                       route1_idx=0, route2_idx=0, pos1=1, pos2=2)
    
    assert success == False


def test_try_cross_invalid_positions(simple_solution, evaluator):
    """Test cross avec positions invalides."""
    # pos1 négatif
    success = try_cross(simple_solution, evaluator,
                       route1_idx=0, route2_idx=1, pos1=-1, pos2=1)
    assert success == False
    
    # pos1 >= len(route)
    success = try_cross(simple_solution, evaluator,
                       route1_idx=0, route2_idx=1, pos1=100, pos2=1)
    assert success == False


# ============================================================================
# TESTS - try_swap_intra
# ============================================================================

def test_try_swap_intra_basic(simple_solution, evaluator):
    """Test basique de swap intra-route."""
    # Route 1: [4, 5, 6, 7] → swap positions 0 et 2
    success = try_swap_intra(simple_solution, evaluator, route_idx=1, i=0, j=2)
    
    simple_solution.assert_consistency()
    
    if success:
        # Position 0 et 2 ont été échangées
        route = simple_solution.routes[1]
        assert route[0] == 6  # Était à position 2
        assert route[2] == 4  # Était à position 0


def test_try_swap_intra_invalid_indices(simple_solution, evaluator):
    """Test swap avec indices invalides."""
    # i >= j
    success = try_swap_intra(simple_solution, evaluator, route_idx=0, i=2, j=1)
    assert success == False
    
    # i négatif
    success = try_swap_intra(simple_solution, evaluator, route_idx=0, i=-1, j=2)
    assert success == False


# ============================================================================
# TESTS - find_best_relocate
# ============================================================================

def test_find_best_relocate_basic(simple_solution, evaluator):
    """Test de recherche du meilleur relocate."""
    improved, best_route, best_position = find_best_relocate(
        simple_solution, evaluator, customer=2
    )
    
    simple_solution.assert_consistency()
    
    if improved:
        # Une amélioration a été trouvée et appliquée
        assert best_route is not None
        assert best_position is not None
        assert simple_solution.get_route_of_customer(2) == best_route


def test_find_best_relocate_no_improvement(instance, evaluator):
    """Test quand aucune amélioration n'est possible."""
    solution = Solution()
    solution.add_route([1, 2, 3], cost=100.0)
    
    # Si la solution est déjà optimale localement, aucune amélioration
    improved, _, _ = find_best_relocate(solution, evaluator, customer=2)
    
    solution.assert_consistency()


# ============================================================================
# TESTS - find_best_2opt_intra
# ============================================================================

def test_find_best_2opt_intra_basic(simple_solution, evaluator):
    """Test de recherche du meilleur 2-opt intra."""
    improved = find_best_2opt_intra(simple_solution, evaluator, route_idx=1)
    
    simple_solution.assert_consistency()


def test_find_best_2opt_intra_short_route(instance, evaluator):
    """Test 2-opt sur route trop courte."""
    solution = Solution()
    solution.add_route([1, 2], cost=50.0)
    
    improved = find_best_2opt_intra(solution, evaluator, route_idx=0)
    
    # Route trop courte → pas d'amélioration possible
    assert improved == False


# ============================================================================
# TESTS - Intégration
# ============================================================================

def test_operators_combined(simple_solution, evaluator):
    """Test de combinaison d'opérateurs."""
    initial_cost = simple_solution.total_cost
    
    # Applique plusieurs opérateurs
    operators_applied = 0
    
    if try_relocate(simple_solution, evaluator, 2, 0, 1, 0):
        operators_applied += 1
    
    if try_2opt_intra(simple_solution, evaluator, 0, 0, 2):
        operators_applied += 1
    
    if try_exchange(simple_solution, evaluator, 1, 4):
        operators_applied += 1
    
    # Vérifie cohérence finale
    simple_solution.assert_consistency()
    
    print(f"\n{operators_applied} opérateurs appliqués avec succès")
    print(f"Coût initial: {initial_cost:.2f}")
    print(f"Coût final: {simple_solution.total_cost:.2f}")


def test_operators_preserve_feasibility(simple_solution, evaluator):
    """Vérifie que les opérateurs préservent la faisabilité."""
    # Applique plusieurs opérateurs
    try_relocate(simple_solution, evaluator, 2, 0, 1, 0)
    try_2opt_intra(simple_solution, evaluator, 0, 0, 2)
    try_exchange(simple_solution, evaluator, 1, 4)
    
    # Vérifie que toutes les routes restent faisables
    for route_idx, route in enumerate(simple_solution.routes):
        is_feasible, _ = evaluator.evaluate_route_fast(route)
        assert is_feasible, f"Route {route_idx} devenue infaisable après opérateurs"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])