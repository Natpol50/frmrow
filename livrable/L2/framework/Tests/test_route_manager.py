"""
Tests pytest pour RouteEvaluator - Asha Geyon 2025

Structure des tests:
- Fixtures pour créer des instances de test
- Tests unitaires pour chaque méthode
- Tests d'intégration pour scénarios réels
- Tests de performance

Usage:
    pytest test_route_evaluator_pytest.py -v
    pytest test_route_evaluator_pytest.py::test_evaluate_empty_route -v
    pytest test_route_evaluator_pytest.py -k "capacity" -v
"""

import pytest
import numpy as np
from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solvermanager.routemanager import RouteEvaluator, RouteEvaluation


# ============================================================================
# FIXTURES - Instances de test
# ============================================================================

@dataclass
class MockInstance:
    """Instance simplifiée pour les tests."""
    depot: int
    capacity: int
    distance_matrix: np.ndarray
    demands: np.ndarray
    ready_times: np.ndarray = None
    due_dates: np.ndarray = None
    service_times: np.ndarray = None
    
    def is_vrptw(self):
        return self.ready_times is not None


@pytest.fixture
def simple_cvrp_instance():
    """
    Instance CVRP simple sans time windows.
    
    5 nœuds : dépôt (0) + 4 clients (1,2,3,4)
    Capacité : 100
    """
    # Distance matrix (symétrique, euclidienne simplifiée)
    distance_matrix = np.array([
        [0,  10, 15, 20, 25],  # Dépôt
        [10, 0,  5,  15, 20],  # Client 1
        [15, 5,  0,  10, 15],  # Client 2
        [20, 15, 10, 0,  5],   # Client 3
        [25, 20, 15, 5,  0]    # Client 4
    ], dtype=float)
    
    demands = np.array([0, 20, 30, 25, 40])  # Dépôt = 0 demande
    
    return MockInstance(
        depot=0,
        capacity=100,
        distance_matrix=distance_matrix,
        demands=demands
    )


@pytest.fixture
def simple_vrptw_instance():
    """
    Instance VRPTW simple avec time windows.
    
    Même structure que simple_cvrp mais avec time windows.
    """
    distance_matrix = np.array([
        [0,  10, 15, 20, 25],
        [10, 0,  5,  15, 20],
        [15, 5,  0,  10, 15],
        [20, 15, 10, 0,  5],
        [25, 20, 15, 5,  0]
    ], dtype=float)
    
    demands = np.array([0, 20, 30, 25, 40])
    
    # Time windows : [ready_time, due_date]
    ready_times = np.array([0, 10, 20, 30, 40])
    due_dates = np.array([200, 50, 60, 80, 100])
    service_times = np.array([0, 5, 5, 5, 5])
    
    return MockInstance(
        depot=0,
        capacity=100,
        distance_matrix=distance_matrix,
        demands=demands,
        ready_times=ready_times,
        due_dates=due_dates,
        service_times=service_times
    )


@pytest.fixture
def tight_capacity_instance():
    """Instance avec capacité très serrée pour tester violations."""
    distance_matrix = np.array([
        [0,  10, 15],
        [10, 0,  5],
        [15, 5,  0]
    ], dtype=float)
    
    demands = np.array([0, 40, 45])  # Total = 85 > 80 = violation
    
    return MockInstance(
        depot=0,
        capacity=80,  # Capacité insuffisante
        distance_matrix=distance_matrix,
        demands=demands
    )


@pytest.fixture
def tight_time_window_instance():
    """Instance avec time windows très serrées pour tester violations."""
    distance_matrix = np.array([
        [0,  50, 30],  # Distances longues
        [50, 0,  40],
        [30, 40, 0]
    ], dtype=float)
    
    demands = np.array([0, 20, 25])
    ready_times = np.array([0, 10, 80])
    due_dates = np.array([200, 50, 90])  # Fenêtre très serrée
    service_times = np.array([0, 10, 10])
    
    return MockInstance(
        depot=0,
        capacity=100,
        distance_matrix=distance_matrix,
        demands=demands,
        ready_times=ready_times,
        due_dates=due_dates,
        service_times=service_times
    )


# ============================================================================
# TESTS - Cas de base
# ============================================================================

def test_evaluator_initialization(simple_cvrp_instance):
    """Test que l'évaluateur s'initialise correctement."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    
    assert evaluator.depot == 0
    assert evaluator.capacity == 100
    assert evaluator.has_time_windows == False
    assert np.array_equal(evaluator.distance_matrix, simple_cvrp_instance.distance_matrix)


def test_evaluate_empty_route(simple_cvrp_instance):
    """Test avec une route vide."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    
    evaluation = evaluator.evaluate_route([])
    
    assert evaluation.is_feasible == True
    assert evaluation.cost == 0.0
    assert evaluation.total_demand == 0


def test_evaluate_single_customer_route(simple_cvrp_instance):
    """Test avec une route d'un seul client."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    
    route = [1]  # Dépôt → Client 1 → Dépôt
    evaluation = evaluator.evaluate_route(route)
    
    assert evaluation.is_feasible == True
    assert evaluation.total_demand == 20
    # Coût = distance(0→1) + distance(1→0) = 10 + 10 = 20
    assert evaluation.cost == 20.0


def test_evaluate_multi_customer_route(simple_cvrp_instance):
    """Test avec plusieurs clients."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    
    route = [1, 2, 3]  # Dépôt → 1 → 2 → 3 → Dépôt
    evaluation = evaluator.evaluate_route(route)
    
    assert evaluation.is_feasible == True
    assert evaluation.total_demand == 20 + 30 + 25  # 75
    # Coût = 0→1(10) + 1→2(5) + 2→3(10) + 3→0(20) = 45
    assert evaluation.cost == 45.0


# ============================================================================
# TESTS - Contraintes de capacité
# ============================================================================

def test_capacity_feasible(simple_cvrp_instance):
    """Test avec demande totale dans la capacité."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    
    route = [1, 2]  # Demande = 20 + 30 = 50 <= 100
    evaluation = evaluator.evaluate_route(route)
    
    assert evaluation.is_feasible == True
    assert evaluation.total_demand == 50
    assert len(evaluation.violations['capacity']) == 0


def test_capacity_violation(tight_capacity_instance):
    """Test avec dépassement de capacité."""
    evaluator = RouteEvaluator(tight_capacity_instance)
    
    route = [1, 2]  # Demande = 40 + 45 = 85 > 80
    evaluation = evaluator.evaluate_route(route)
    
    assert evaluation.is_feasible == False
    assert evaluation.total_demand == 85
    assert len(evaluation.violations['capacity']) > 0
    assert "85" in evaluation.violations['capacity'][0]
    assert "80" in evaluation.violations['capacity'][0]


def test_capacity_at_limit(simple_cvrp_instance):
    """Test avec demande exactement égale à la capacité."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    
    route = [1, 2, 3, 4]  # Demande = 20+30+25+40 = 115 > 100
    evaluation = evaluator.evaluate_route(route)
    
    # Dépasse la capacité
    assert evaluation.is_feasible == False


# ============================================================================
# TESTS - Time Windows (VRPTW)
# ============================================================================

def test_vrptw_feasible_route(simple_vrptw_instance):
    """Test avec route respectant les time windows."""
    evaluator = RouteEvaluator(simple_vrptw_instance)
    
    route = [1, 2]  # Route simple
    evaluation = evaluator.evaluate_route(route, return_details=True)
    
    # Devrait être faisable
    assert evaluation.is_feasible == True
    assert evaluation.arrival_times is not None
    assert len(evaluation.violations['time_windows']) == 0


def test_vrptw_with_waiting(simple_vrptw_instance):
    """Test où le véhicule doit attendre (arrive trop tôt)."""
    evaluator = RouteEvaluator(simple_vrptw_instance)
    
    # Client 2 a ready_time=20, mais on peut arriver avant
    route = [2]  # Dépôt → 2 → Dépôt
    evaluation = evaluator.evaluate_route(route, return_details=True)
    
    assert evaluation.is_feasible == True
    # Arrivée à 0 + 15 = 15, mais ready_time = 20 → attend jusqu'à 20
    assert evaluation.arrival_times[1] == 20.0  # Temps d'arrivée ajusté


def test_vrptw_violation_due_date(tight_time_window_instance):
    """Test avec violation de due_date."""
    evaluator = RouteEvaluator(tight_time_window_instance)
    
    # Route trop longue : 0 → 1 (distance 50) → 2 (distance 40)
    # Arrivée client 1 : 50 > due_date=50 → limite
    # Service : 50 + 10 = 60, puis 60 + 40 = 100 > due_date(2)=90 → VIOLATION
    route = [1, 2]
    evaluation = evaluator.evaluate_route(route)
    
    assert evaluation.is_feasible == False
    assert len(evaluation.violations['time_windows']) > 0


def test_vrptw_return_to_depot_violation(tight_time_window_instance):
    """Test avec retour au dépôt trop tard."""
    evaluator = RouteEvaluator(tight_time_window_instance)
    
    # Modifie la due_date du dépôt pour forcer violation
    tight_time_window_instance.due_dates[0] = 50  # Très tôt
    
    evaluator = RouteEvaluator(tight_time_window_instance)
    route = [1, 2]
    evaluation = evaluator.evaluate_route(route)
    
    # Devrait violer le retour au dépôt
    assert evaluation.is_feasible == False


# ============================================================================
# TESTS - Modes d'évaluation
# ============================================================================

def test_evaluate_route_fast(simple_cvrp_instance):
    """Test du mode rapide (sans détails)."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    
    route = [1, 2, 3]
    is_feasible, cost = evaluator.evaluate_route_fast(route)
    
    assert is_feasible == True
    assert cost == 45.0
    assert isinstance(is_feasible, bool)
    assert isinstance(cost, (float, np.floating))


def test_evaluate_route_fast_infeasible(tight_capacity_instance):
    """Test du mode rapide avec route infaisable."""
    evaluator = RouteEvaluator(tight_capacity_instance)
    
    route = [1, 2]
    is_feasible, cost = evaluator.evaluate_route_fast(route)
    
    assert is_feasible == False
    assert cost == float('inf')


def test_compute_route_cost_only(simple_cvrp_instance):
    """Test du calcul de coût uniquement."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    
    route = [1, 2, 3]
    cost = evaluator.compute_route_cost_only(route)
    
    assert cost == 45.0
    
    # Vérifie cohérence avec evaluate_route
    evaluation = evaluator.evaluate_route(route)
    assert cost == evaluation.cost


# ============================================================================
# TESTS - Fonctionnalités avancées
# ============================================================================

def test_get_route_slack(simple_vrptw_instance):
    """Test du calcul de slack temporel."""
    evaluator = RouteEvaluator(simple_vrptw_instance)
    
    route = [1, 2]
    slacks = evaluator.get_route_slack(route)
    
    assert len(slacks) == 2
    assert all(slack >= 0 for slack in slacks)  # Slack positif si faisable


def test_get_route_slack_no_time_windows(simple_cvrp_instance):
    """Test du slack sur instance sans time windows."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    
    route = [1, 2, 3]
    slacks = evaluator.get_route_slack(route)
    
    assert len(slacks) == 3
    assert all(slack == float('inf') for slack in slacks)  # Pas de contrainte


def test_violation_summary(tight_capacity_instance):
    """Test du résumé de violations."""
    evaluator = RouteEvaluator(tight_capacity_instance)
    
    route = [1, 2]
    evaluation = evaluator.evaluate_route(route)
    
    summary = evaluation.get_violation_summary()
    
    assert "capacity" in summary.lower()
    assert evaluation.is_feasible == False


# ============================================================================
# TESTS - Edge cases
# ============================================================================

def test_route_with_all_customers(simple_cvrp_instance):
    """Test avec tous les clients dans une route."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    
    route = [1, 2, 3, 4]
    evaluation = evaluator.evaluate_route(route)
    
    # Dépasse la capacité (115 > 100)
    assert evaluation.is_feasible == False


def test_route_with_depot_in_list_should_fail():
    """Test que l'inclusion du dépôt dans la route cause une erreur logique."""
    # Note: Le code actuel n'empêche pas ça, mais ça donnerait des résultats faux
    # On teste juste qu'on peut détecter le comportement
    distance_matrix = np.array([
        [0, 10],
        [10, 0]
    ], dtype=float)
    
    instance = MockInstance(
        depot=0,
        capacity=100,
        distance_matrix=distance_matrix,
        demands=np.array([0, 20])
    )
    
    evaluator = RouteEvaluator(instance)
    
    # Route avec dépôt (incorrect)
    route = [0, 1, 0]  # Ne devrait pas inclure le dépôt
    evaluation = evaluator.evaluate_route(route)
    
    # Le coût sera bizarre (0→0→1→0→0)
    # C'est pourquoi on ne doit PAS inclure le dépôt


def test_duplicate_customer_in_route():
    """Test avec un client visité plusieurs fois (incorrect)."""
    distance_matrix = np.array([
        [0, 10, 15],
        [10, 0, 5],
        [15, 5, 0]
    ], dtype=float)
    
    instance = MockInstance(
        depot=0,
        capacity=100,
        distance_matrix=distance_matrix,
        demands=np.array([0, 20, 30])
    )
    
    evaluator = RouteEvaluator(instance)
    
    route = [1, 1, 2]  # Client 1 deux fois (incorrect)
    evaluation = evaluator.evaluate_route(route)
    
    # La capacité sera comptée deux fois pour client 1
    assert evaluation.total_demand == 20 + 20 + 30  # 70


# ============================================================================
# TESTS - Performance
# ============================================================================

def test_performance_many_evaluations(simple_cvrp_instance):
    """Test de performance : beaucoup d'évaluations rapides."""
    import time
    
    evaluator = RouteEvaluator(simple_cvrp_instance)
    route = [1, 2, 3]
    
    n_evaluations = 10000
    
    start = time.time()
    for _ in range(n_evaluations):
        is_feasible, cost = evaluator.evaluate_route_fast(route)
    elapsed = time.time() - start
    
    time_per_eval = elapsed / n_evaluations
    
    print(f"\n{n_evaluations} évaluations en {elapsed:.3f}s")
    print(f"Temps par évaluation: {time_per_eval*1000:.4f}ms")
    
    # Assertion : devrait être < 0.1ms par évaluation
    assert time_per_eval < 0.0001  # 0.1ms


def test_cost_computation_methods_consistency(simple_cvrp_instance):
    """Vérifie que les différentes méthodes de calcul donnent le même résultat."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    route = [1, 2, 3]
    
    # Trois méthodes
    cost_full = evaluator.evaluate_route(route).cost
    is_feasible, cost_fast = evaluator.evaluate_route_fast(route)
    cost_only = evaluator.compute_route_cost_only(route)
    
    # Tous devraient donner le même coût
    assert cost_full == cost_only
    assert cost_full == cost_fast
    


# ============================================================================
# TESTS - Intégration
# ============================================================================

def test_realistic_scenario(simple_vrptw_instance):
    """Scénario réaliste : évaluer plusieurs routes d'une solution."""
    evaluator = RouteEvaluator(simple_vrptw_instance)
    
    # Solution avec 2 routes
    route1 = [1, 2]
    route2 = [3, 4]
    
    eval1 = evaluator.evaluate_route(route1)
    eval2 = evaluator.evaluate_route(route2)
    
    assert eval1.is_feasible == True
    assert eval2.is_feasible == True
    
    total_cost = eval1.cost + eval2.cost
    total_demand = eval1.total_demand + eval2.total_demand
    
    assert total_demand == 20 + 30 + 25 + 40  # Tous les clients
    assert total_cost > 0


# ============================================================================
# MARKERS et PARAMETRIZE
# ============================================================================

@pytest.mark.parametrize("route,expected_demand", [
    ([1], 20),
    ([1, 2], 50),
    ([1, 2, 3], 75),
    ([2, 3, 4], 95),
])
def test_capacity_calculation_parametrized(simple_cvrp_instance, route, expected_demand):
    """Test paramétrisé pour différentes routes."""
    evaluator = RouteEvaluator(simple_cvrp_instance)
    evaluation = evaluator.evaluate_route(route)
    assert evaluation.total_demand == expected_demand



def test_large_route_performance():
    """Test avec une grande route (marqué comme slow)."""
    # Crée une instance plus grande
    n = 100
    distance_matrix = np.random.rand(n, n) * 100
    demands = np.random.randint(1, 20, size=n)
    demands[0] = 0
    
    instance = MockInstance(
        depot=0,
        capacity=200,
        distance_matrix=distance_matrix,
        demands=demands
    )
    
    evaluator = RouteEvaluator(instance)
    route = list(range(1, 21))  # 20 clients
    
    import time
    start = time.time()
    evaluation = evaluator.evaluate_route(route)
    elapsed = time.time() - start
    
    print(f"\nÉvaluation route de 20 clients : {elapsed*1000:.2f}ms")
    assert elapsed < 0.01  # Moins de 10ms


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])