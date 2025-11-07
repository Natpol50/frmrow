"""
Tests pytest pour la classe Solution - Mon bon pote Claude

Vérifie que toutes les opérations maintiennent la cohérence interne.

Usage:
    pytest test_solution.py -v
"""

import pytest
import sys
from pathlib import Path
# Ensure the parent folder (project root) is on sys.path so runfilemanager is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solvermanager.solutionclass import Solution

# ============================================================================
# TESTS - Construction de base
# ============================================================================

def test_solution_initialization():
    """Test de l'initialisation d'une solution vide."""
    sol = Solution("C101")
    
    assert sol.instance_name == "C101"
    assert sol.get_n_routes() == 0
    assert sol.get_n_customers() == 0
    assert sol.total_cost == 0.0
    assert sol.n_vehicles_used == 0
    assert sol.is_empty()


def test_add_single_route():
    """Test d'ajout d'une route simple."""
    sol = Solution()
    sol.add_route([1, 2, 3], cost=100.0)
    
    assert sol.get_n_routes() == 1
    assert sol.get_n_customers() == 3
    assert sol.total_cost == 100.0
    assert sol.n_vehicles_used == 1
    
    # Vérifie les mappings
    assert sol.get_route_of_customer(1) == 0
    assert sol.get_route_of_customer(2) == 0
    assert sol.get_route_of_customer(3) == 0
    
    assert sol.get_position_of_customer(1) == 0
    assert sol.get_position_of_customer(2) == 1
    assert sol.get_position_of_customer(3) == 2
    
    # Cohérence
    sol.assert_consistency()


def test_add_multiple_routes():
    """Test d'ajout de plusieurs routes."""
    sol = Solution()
    sol.add_route([1, 2], cost=50.0)
    sol.add_route([3, 4, 5], cost=75.0)
    sol.add_route([6], cost=25.0)
    
    assert sol.get_n_routes() == 3
    assert sol.get_n_customers() == 6
    assert sol.total_cost == 150.0
    assert sol.n_vehicles_used == 3
    
    # Vérifie que chaque client est bien assigné
    assert sol.get_route_of_customer(1) == 0
    assert sol.get_route_of_customer(3) == 1
    assert sol.get_route_of_customer(6) == 2
    
    sol.assert_consistency()


def test_add_empty_route():
    """Test d'ajout d'une route vide."""
    sol = Solution()
    sol.add_route([], cost=0.0)
    
    assert sol.get_n_routes() == 1
    assert sol.get_n_customers() == 0
    assert sol.n_vehicles_used == 0  # Route vide ne compte pas
    
    sol.assert_consistency()


def test_add_duplicate_customer_fails():
    """Test qu'on ne peut pas ajouter le même client deux fois."""
    sol = Solution()
    sol.add_route([1, 2, 3], cost=100.0)
    
    with pytest.raises(ValueError, match="déjà dans la route"):
        sol.add_route([4, 2, 5], cost=50.0)  # Client 2 déjà présent


# ============================================================================
# TESTS - Modification de routes
# ============================================================================

def test_set_route():
    """Test de remplacement d'une route."""
    sol = Solution()
    sol.add_route([1, 2, 3], cost=100.0)
    sol.add_route([4, 5], cost=50.0)
    
    # Remplace la route 0
    sol.set_route(0, [1, 6, 7], new_cost=120.0)
    
    assert sol.routes[0] == [1, 6, 7]
    assert sol.route_costs[0] == 120.0
    assert sol.total_cost == 170.0  # 120 + 50
    
    # Vérifie les mappings
    assert sol.get_route_of_customer(1) == 0
    assert sol.get_route_of_customer(6) == 0
    assert sol.get_route_of_customer(7) == 0
    assert sol.get_route_of_customer(2) is None  # Retiré
    assert sol.get_route_of_customer(3) is None  # Retiré
    
    sol.assert_consistency()


def test_update_route_cost():
    """Test de mise à jour du coût d'une route."""
    sol = Solution()
    sol.add_route([1, 2], cost=100.0)
    sol.add_route([3, 4], cost=50.0)
    
    sol.update_route_cost(0, 110.0)
    
    assert sol.route_costs[0] == 110.0
    assert sol.total_cost == 160.0
    
    sol.assert_consistency()


def test_remove_route():
    """Test de suppression d'une route."""
    sol = Solution()
    sol.add_route([1, 2], cost=100.0)
    sol.add_route([3, 4], cost=50.0)
    sol.add_route([5, 6], cost=75.0)
    
    # Retire la route du milieu
    sol.remove_route(1)
    
    assert sol.get_n_routes() == 2
    assert sol.get_n_customers() == 4
    assert sol.total_cost == 175.0
    assert sol.n_vehicles_used == 2
    
    # Vérifie que les clients de la route retirée ne sont plus là
    assert sol.get_route_of_customer(3) is None
    assert sol.get_route_of_customer(4) is None
    
    # Vérifie que les indices des routes suivantes ont été mis à jour
    assert sol.get_route_of_customer(5) == 1  # Était 2, maintenant 1
    assert sol.get_route_of_customer(6) == 1
    
    sol.assert_consistency()


# ============================================================================
# TESTS - Opérations de voisinage
# ============================================================================

def test_relocate_customer_same_route():
    """Test de relocate dans la même route (pas recommandé mais doit marcher)."""
    sol = Solution()
    sol.add_route([1, 2, 3, 4], cost=95.0)
    
    # Déplace client 2 de position 1 à position 3
    sol.relocate_customer(
        customer=2,
        from_route_idx=0,
        to_route_idx=0,
        to_position=3,
        new_from_cost=95.0,  # Même route, donc totalement le même coût
        new_to_cost=95.0
    )
    
    assert sol.routes[0] == [1, 3, 4, 2]
    assert sol.get_position_of_customer(2) == 3
    
    sol.assert_consistency()


def test_relocate_customer_between_routes():
    """Test de relocate entre deux routes."""
    sol = Solution()
    sol.add_route([1, 2, 3], cost=100.0)
    sol.add_route([4, 5], cost=50.0)
    
    # Déplace client 2 de route 0 à route 1
    sol.relocate_customer(
        customer=2,
        from_route_idx=0,
        to_route_idx=1,
        to_position=1,  # Entre 4 et 5
        new_from_cost=80.0,
        new_to_cost=70.0
    )
    
    assert sol.routes[0] == [1, 3]
    assert sol.routes[1] == [4, 2, 5]
    assert sol.total_cost == 150.0
    
    # Vérifie les mappings
    assert sol.get_route_of_customer(2) == 1
    assert sol.get_position_of_customer(2) == 1
    
    sol.assert_consistency()


def test_relocate_to_empty_route():
    """Test de relocate vers une route vide."""
    sol = Solution()
    sol.add_route([1, 2, 3], cost=100.0)
    sol.add_route([], cost=0.0)
    
    assert sol.n_vehicles_used == 1  # Route vide ne compte pas
    
    sol.relocate_customer(
        customer=2,
        from_route_idx=0,
        to_route_idx=1,
        to_position=0,
        new_from_cost=80.0,
        new_to_cost=20.0
    )
    
    assert sol.routes[0] == [1, 3]
    assert sol.routes[1] == [2]
    assert sol.n_vehicles_used == 2  # Maintenant 2 routes non-vides
    
    sol.assert_consistency()


def test_relocate_makes_route_empty():
    """Test de relocate qui vide une route."""
    sol = Solution()
    sol.add_route([1], cost=50.0)
    sol.add_route([2, 3], cost=75.0)
    
    assert sol.n_vehicles_used == 2
    
    sol.relocate_customer(
        customer=1,
        from_route_idx=0,
        to_route_idx=1,
        to_position=0,
        new_from_cost=0.0,
        new_to_cost=90.0
    )
    
    assert sol.routes[0] == []
    assert sol.routes[1] == [1, 2, 3]
    assert sol.n_vehicles_used == 1  # Une route est maintenant vide
    
    sol.assert_consistency()


def test_exchange_customers():
    """Test d'échange de deux clients."""
    sol = Solution()
    sol.add_route([1, 2, 3], cost=100.0)
    sol.add_route([4, 5, 6], cost=75.0)
    
    sol.exchange_customers(
        customer1=2,
        customer2=5,
        new_cost_route1=105.0,
        new_cost_route2=70.0
    )
    
    assert sol.routes[0] == [1, 5, 3]
    assert sol.routes[1] == [4, 2, 6]
    assert sol.total_cost == 175.0
    
    # Vérifie les mappings
    assert sol.get_route_of_customer(2) == 1
    assert sol.get_route_of_customer(5) == 0
    assert sol.get_position_of_customer(2) == 1
    assert sol.get_position_of_customer(5) == 1
    
    sol.assert_consistency()


def test_exchange_same_route_fails():
    """Test qu'on ne peut pas échanger deux clients de la même route."""
    sol = Solution()
    sol.add_route([1, 2, 3], cost=100.0)
    
    with pytest.raises(ValueError, match="same route"):
        sol.exchange_customers(1, 2, 100.0, 100.0)


def test_reverse_segment():
    """Test d'inversion de segment (2-opt intra)."""
    sol = Solution()
    sol.add_route([1, 2, 3, 4, 5], cost=100.0)
    
    # Inverse le segment [1:4] → clients 2, 3, 4
    sol.reverse_segment(
        route_idx=0,
        start=1,
        end=4,
        new_cost=95.0
    )
    
    assert sol.routes[0] == [1, 4, 3, 2, 5]
    assert sol.route_costs[0] == 95.0
    
    # Vérifie que les positions sont mises à jour
    assert sol.get_position_of_customer(2) == 3
    assert sol.get_position_of_customer(3) == 2
    assert sol.get_position_of_customer(4) == 1
    
    sol.assert_consistency()


# ============================================================================
# TESTS - Requêtes
# ============================================================================

def test_get_customers():
    """Test de récupération de tous les clients."""
    sol = Solution()
    sol.add_route([1, 2, 3], cost=100.0)
    sol.add_route([4, 5], cost=50.0)
    
    customers = sol.get_customers()
    
    assert customers == {1, 2, 3, 4, 5}


def test_get_route_and_position():
    """Test de récupération de route et position d'un client."""
    sol = Solution()
    sol.add_route([10, 20, 30], cost=100.0)
    
    assert sol.get_route_of_customer(20) == 0
    assert sol.get_position_of_customer(20) == 1
    
    # Client inexistant
    assert sol.get_route_of_customer(999) is None
    assert sol.get_position_of_customer(999) is None


# ============================================================================
# TESTS - Cohérence et validation
# ============================================================================

def test_validate_consistency_clean_solution():
    """Test de validation sur une solution propre."""
    sol = Solution()
    sol.add_route([1, 2, 3], cost=100.0)
    sol.add_route([4, 5], cost=50.0)
    
    is_consistent, errors = sol.validate_consistency()
    
    assert is_consistent
    assert len(errors) == 0


def test_detect_duplicate_customer():
    """Test de détection de doublon de client."""
    sol = Solution()
    sol.add_route([1, 2, 3], cost=100.0)
    
    # Force un doublon (normalement impossible via l'API publique)
    sol.routes.append([2, 4, 5])
    sol.route_costs.append(50.0)
    sol.client_to_route[4] = 1
    sol.client_to_route[5] = 1
    sol.client_position[4] = 1
    sol.client_position[5] = 2
    
    is_consistent, errors = sol.validate_consistency()
    
    assert not is_consistent
    assert any("appears multiple times" in e for e in errors)


def test_detect_mapping_mismatch():
    """Test de détection d'incohérence dans les mappings."""
    sol = Solution()
    sol.add_route([1, 2, 3], cost=100.0)
    
    # Corrompt le mapping
    sol.client_to_route[2] = 99  # Route inexistante
    
    is_consistent, errors = sol.validate_consistency()
    
    assert not is_consistent
    assert any("client_to_route says" in e for e in errors)


def test_detect_cost_mismatch():
    """Test de détection d'incohérence dans les coûts."""
    sol = Solution()
    sol.add_route([1, 2], cost=100.0)
    sol.add_route([3, 4], cost=50.0)
    
    # Corrompt le total_cost
    sol.total_cost = 999.0
    
    is_consistent, errors = sol.validate_consistency()
    
    assert not is_consistent
    assert any("total_cost" in e for e in errors)


# ============================================================================
# TESTS - Conversion et copie
# ============================================================================

def test_copy_solution():
    """Test de copie profonde."""
    sol1 = Solution("C101")
    sol1.add_route([1, 2, 3], cost=100.0)
    sol1.add_route([4, 5], cost=50.0)
    
    sol2 = sol1.copy()
    
    # Vérifie que c'est une vraie copie
    assert sol2.total_cost == sol1.total_cost
    assert sol2.routes == sol1.routes
    assert sol2.routes is not sol1.routes  # Pas le même objet
    
    # Modifie sol2
    sol2.update_route_cost(0, 110.0)
    
    # Vérifie que sol1 n'est pas affecté
    assert sol1.route_costs[0] == 100.0
    assert sol2.route_costs[0] == 110.0


def test_to_dict():
    """Test de conversion en dictionnaire."""
    sol = Solution("C101")
    sol.algo_name = "simulated_annealing"
    sol.add_route([1, 2], cost=100.0)
    
    data = sol.to_dict()
    
    assert data['instance_name'] == "C101"
    assert data['algo_name'] == "simulated_annealing"
    assert data['routes'] == [[1, 2]]
    assert data['route_costs'] == [100.0]
    assert data['total_cost'] == 100.0


def test_repr_and_str():
    """Test des représentations textuelles."""
    sol = Solution("C101")
    sol.add_route([1, 2, 3], cost=100.5)
    sol.add_route([4, 5], cost=50.3)
    
    repr_str = repr(sol)
    assert "n_routes=2" in repr_str
    assert "n_customers=5" in repr_str
    assert "150.8" in repr_str  # total cost
    
    str_str = str(sol)
    assert "Solution: 150.8" in str_str
    assert "Route 1:" in str_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])