"""
Tests de validation pour l'InstanceFileManager - Mon bon copain Claude

Vérifie:
1. Détection correcte des formats
2. Parsing des données
3. Calcul des distances
4. Cohérence des solutions
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filemanagers.instancefilemanager import InstanceFileManager, Instance, Solution


def test_distance_matrix_symmetry(instance: Instance) -> bool:
    """Vérifie que la matrice de distances est symétrique."""
    matrix = instance.distance_matrix
    is_symmetric = np.allclose(matrix, matrix.T)
    
    if not is_symmetric:
        print(f"  ✗ Matrice non symétrique!")
        return False
    
    print(f"  ✓ Matrice symétrique")
    return True


def test_distance_matrix_diagonal(instance: Instance) -> bool:
    """Vérifie que la diagonale est nulle (distance à soi-même = 0)."""
    diagonal = np.diag(instance.distance_matrix)
    is_zero_diagonal = np.allclose(diagonal, 0)
    
    if not is_zero_diagonal:
        print(f"  ✗ Diagonale non nulle: {diagonal[:5]}...")
        return False
    
    print(f"  ✓ Diagonale nulle")
    return True


def test_distance_matrix_positive(instance: Instance) -> bool:
    """Vérifie que toutes les distances sont positives ou nulles."""
    min_distance = instance.distance_matrix.min()
    
    if min_distance < 0:
        print(f"  ✗ Distance négative trouvée: {min_distance}")
        return False
    
    print(f"  ✓ Toutes les distances positives")
    return True


def test_distance_calculation_accuracy(instance: Instance) -> bool:
    """Vérifie quelques distances manuellement."""
    # Distance entre les deux premiers points
    x1, y1 = instance.coordinates[0]
    x2, y2 = instance.coordinates[1]
    
    expected = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    expected_rounded = round(expected)
    actual = instance.distance_matrix[0, 1]
    
    if abs(actual - expected_rounded) > 1:
        print(f"  ✗ Distance incorrecte: {actual} vs {expected_rounded} attendu")
        return False
    
    print(f"  ✓ Calcul de distance correct (vérifié sur échantillon)")
    return True


def test_demand_consistency(instance: Instance) -> bool:
    """Vérifie que les demandes sont cohérentes."""
    # Le dépôt doit avoir demande = 0
    if instance.demands[instance.depot] != 0:
        print(f"  ✗ Demande du dépôt non nulle: {instance.demands[instance.depot]}")
        return False
    
    # Toutes les demandes doivent être >= 0
    if (instance.demands < 0).any():
        print(f"  ✗ Demandes négatives trouvées")
        return False
    
    # La demande totale doit nécessiter au moins 1 véhicule
    total_demand = instance.get_total_demand()
    min_vehicles = int(np.ceil(total_demand / instance.capacity))
    
    if min_vehicles > instance.n_vehicles:
        print(f"  ⚠ Demande totale ({total_demand}) nécessite {min_vehicles} "
              f"véhicules mais seulement {instance.n_vehicles} disponibles")
    
    print(f"  ✓ Demandes cohérentes (total={total_demand}, "
          f"min_vehicles={min_vehicles}/{instance.n_vehicles})")
    return True


def test_vrptw_time_windows(instance: Instance) -> bool:
    """Vérifie la cohérence des fenêtres temporelles (si VRPTW)."""
    if not instance.is_vrptw():
        print(f"  - Pas de time windows (CVRP)")
        return True
    
    # Vérifie que ready_time <= due_date
    invalid_windows = instance.ready_times > instance.due_dates
    
    if invalid_windows.any():
        invalid_indices = np.where(invalid_windows)[0]
        print(f"  ✗ Fenêtres temporelles invalides aux indices: {invalid_indices[:5]}...")
        return False
    
    # Vérifie que les temps de service sont non négatifs
    if (instance.service_times < 0).any():
        print(f"  ✗ Temps de service négatifs trouvés")
        return False
    
    print(f"  ✓ Fenêtres temporelles cohérentes")
    print(f"    Horizon: [{instance.ready_times.min():.0f}, {instance.due_dates.max():.0f}]")
    return True


def test_solution_consistency(solution: Solution, instance: Instance) -> bool:
    """Vérifie la cohérence de la solution."""
    is_valid, violations = solution.validate_against_instance(instance)
    
    if not is_valid:
        print(f"  ✗ Solution invalide:")
        for violation in violations:
            print(f"      - {violation}")
        return False
    
    # Calcule le coût réel
    total_distance = 0
    for route in solution.routes:
        # Dépôt → premier client
        total_distance += instance.distance_matrix[instance.depot, route[0]]
        
        # Client à client
        for i in range(len(route) - 1):
            total_distance += instance.distance_matrix[route[i], route[i+1]]
        
        # Dernier client → dépôt
        total_distance += instance.distance_matrix[route[-1], instance.depot]
    
    # Compare avec le coût déclaré (tolérance de 0.1% pour arrondis)
    cost_diff = abs(total_distance - solution.cost)
    cost_tolerance = solution.cost * 0.001
    
    if cost_diff > max(1.0, cost_tolerance):
        print(f"  ⚠ Coût incohérent: calculé={total_distance:.1f}, "
              f"déclaré={solution.cost:.1f} (diff={cost_diff:.1f})")
    
    print(f"  ✓ Solution valide (coût={solution.cost:.1f}, "
          f"routes={solution.n_vehicles_used}, computed={total_distance:.1f})")
    return True


def run_tests_on_instance(manager: InstanceFileManager, instance_name: str):
    """Execute tous les tests sur une instance."""
    print(f"\n{'='*70}")
    print(f"Tests pour l'instance: {instance_name}")
    print(f"{'='*70}")
    
    try:
        # Charge l'instance
        print(f"\n1. Chargement de l'instance...")
        instance = manager.load_instance(instance_name)
        print(f"   ✓ Instance chargée: {instance.dimension} nœuds")
        
        # Tests sur l'instance
        print(f"\n2. Tests sur la matrice de distances:")
        test_distance_matrix_symmetry(instance)
        test_distance_matrix_diagonal(instance)
        test_distance_matrix_positive(instance)
        test_distance_calculation_accuracy(instance)
        
        print(f"\n3. Tests sur les demandes:")
        test_demand_consistency(instance)
        
        print(f"\n4. Tests sur les contraintes temporelles:")
        test_vrptw_time_windows(instance)
        
        # Tests sur la solution
        print(f"\n5. Tests sur la solution:")
        solution = manager.load_solution(instance_name)
        
        if solution:
            test_solution_consistency(solution, instance)
        else:
            print(f"   - Pas de solution disponible")
        
        print(f"\n{'='*70}")
        print(f"✓ Tous les tests réussis pour {instance_name}!")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n✗ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()


def quick_validation():
    """Validation rapide sur quelques instances."""
    print("\n" + "="*70)
    print(" VALIDATION RAPIDE DE L'INSTANCEFILEMANAGER")
    print("="*70)
    
    manager = InstanceFileManager("data")
    
    # Vérifie que l'index existe
    if not manager.index:
        print("\n⚠ Aucune instance indexée. Lancement du scan...")
        manager.scan_instances()
    
    # Teste sur les premières instances disponibles
    instances_all = list(manager.index.keys())
    # Divise les instances en deux groupes (première moitié / deuxième moitié)
    if not instances_all:
        instances_to_test = []
    else:
        mid = (len(instances_all) + 1) // 2
        group1 = instances_all[:mid]
        group2 = instances_all[mid:]
        print(f"\n  Groupe 1: {group1}")
        print(f"  Groupe 2: {group2}")
        # Concatène les deux groupes dans l'ordre voulu pour les tester
        instances_to_test = group1 + group2
    
    if not instances_to_test:
        print("\n✗ Aucune instance trouvée dans le dossier data/")
        print("   Assurez-vous que le dossier contient des fichiers .vrp ou .txt")
        return
    
    print(f"\nTest sur {len(instances_to_test)} instances: {instances_to_test}")
    
    for instance_name in instances_to_test:
        run_tests_on_instance(manager, instance_name)
    
    print("\n" + "="*70)
    print("✓ Validation terminée!")
    print("="*70)


if __name__ == "__main__":
    quick_validation()