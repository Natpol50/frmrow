import sys
import numpy as np
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filemanagers.instancefilemanager import InstanceFileManager, Instance, Solution

"""
Exemple d'utilisation de l'InstanceFileManager

Démontre:
1. Scan et indexation des instances
2. Chargement d'une instance avec matrice de distances
3. Chargement de la solution optimale
4. Validation de la solution
5. Calcul de statistiques
"""





def example_basic_usage():
    
    """Utilisation basique: scan, load, et accès aux données."""
    print("=" * 70)
    print("EXEMPLE 1: Utilisation basique")
    print("=" * 70)
    
    # Initialise le gestionnaire
    manager = InstanceFileManager("/data")
    
    # Scan des instances (une seule fois)
    print("\n1. Scan des instances...")
    manager.scan_instances(force_rescan=True)
    
    # Liste les instances disponibles
    print("\n2. Instances disponibles:")
    instances = manager.list_instances()
    print(f"   Total: {len(instances)} instances")
    
    # Liste par set
    hg_instances = manager.list_instances(set_name="HG")
    if hg_instances:
        print(f"   Set HG: {len(hg_instances)} instances")
    solomon_instances = manager.list_instances(set_name="Solomon")
    if solomon_instances:
        print(f"   Set Solomon: {len(solomon_instances)} instances")
    # Charge une instance
    print("\n3. Chargement de l'instance C1_2_1...")
    start = time.time()
    instance = manager.load_instance("C1_2_1")
    load_time = time.time() - start
    
    print(f"   ✓ Chargée en {load_time:.3f}s")
    print(f"   - Nom: {instance.name}")
    print(f"   - Dimension: {instance.dimension} nœuds")
    print(f"   - Capacité véhicule: {instance.capacity}")
    print(f"   - Véhicules disponibles: {instance.n_vehicles}")
    print(f"   - Total demande: {instance.get_total_demand()}")
    print(f"   - Type: {'VRPTW' if instance.is_vrptw() else 'CVRP'}")
    
    # Accès à la matrice de distances (déjà calculée!)
    print("\n4. Matrice de distances:")
    print(f"   Shape: {instance.distance_matrix.shape}")
    print(f"   Type: {instance.distance_matrix.dtype}")
    
    # Exemple de distances
    print(f"\n   Distance dépôt (0) → client 1: {instance.distance_matrix[0, 1]}")
    print(f"   Distance dépôt (0) → client 5: {instance.distance_matrix[0, 5]}")
    print(f"   Distance client 1 → client 5: {instance.distance_matrix[1, 5]}")
    
    # Charge la solution
    print("\n5. Chargement de la solution optimale...")
    solution = manager.load_solution("C1_2_1")
    
    if solution:
        print(f"   ✓ Solution trouvée")
        print(f"   - Coût: {solution.cost}")
        print(f"   - Véhicules utilisés: {solution.n_vehicles_used}")
        print(f"   - Nombre de routes: {len(solution.routes)}")
        print(f"\n   Première route: {solution.routes[0][:5]}... ({len(solution.routes[0])} clients)")
    else:
        print("   ✗ Pas de solution disponible")


def example_distance_calculation():
    """Démontre le calcul optimisé de distances."""
    print("\n" + "=" * 70)
    print("EXEMPLE 2: Performance du calcul de distances")
    print("=" * 70)
    
    manager = InstanceFileManager("/data", cache_distances=False)
    
    # Charge sans cache
    print("\n1. Premier chargement (calcul de la matrice)...")
    manager.clear_cache(memory_only=True)
    
    start = time.time()
    instance = manager.load_instance("C1_2_1", use_cache=False)
    first_load = time.time() - start
    print(f"   Temps: {first_load:.3f}s")
    
    # Recharge (avec cache)
    print("\n2. Deuxième chargement (depuis le cache)...")
    start = time.time()
    instance2 = manager.load_instance("C1_2_1", use_cache=True)
    second_load = time.time() - start
    print(f"   Temps: {second_load:.3f}s")
    print(f"   Speedup: {first_load/second_load:.1f}x")


def example_solution_validation():
    """Valide une solution contre les contraintes de l'instance."""
    print("\n" + "=" * 70)
    print("EXEMPLE 3: Validation de solution")
    print("=" * 70)
    
    manager = InstanceFileManager("/data")
    
    instance = manager.load_instance("C1_2_1")
    solution = manager.load_solution("C1_2_1")
    
    if solution:
        print(f"\n1. Validation de la solution {solution.instance_name}...")
        is_valid, violations = solution.validate_against_instance(instance)
        
        if is_valid:
            print("   ✓ Solution valide!")
        else:
            print("   ✗ Solution invalide:")
            for violation in violations:
                print(f"      - {violation}")
        
        # Statistiques des routes
        print(f"\n2. Statistiques des routes:")
        for i, route in enumerate(solution.routes, 1):
            route_demand = sum(instance.demands[c] for c in route)
            route_distance = calculate_route_distance(route, instance)
            
            print(f"   Route {i:2d}: {len(route):3d} clients, "
                  f"demande={route_demand:3d}/{instance.capacity}, "
                  f"distance={route_distance:.1f}")
            
            if i >= 5:  # Limite l'affichage
                print(f"   ... ({len(solution.routes) - i} routes restantes)")
                break


def calculate_route_distance(route: list, instance: Instance) -> float:
    """Calcule la distance totale d'une route (dépôt → clients → dépôt)."""
    if not route:
        return 0.0
    
    distance = instance.distance_matrix[instance.depot, route[0]]  # Dépôt → premier client
    
    for i in range(len(route) - 1):
        distance += instance.distance_matrix[route[i], route[i+1]]  # Client i → client i+1
    
    distance += instance.distance_matrix[route[-1], instance.depot]  # Dernier client → dépôt
    
    return float(distance)


def example_vrptw_features():
    """Démontre les fonctionnalités spécifiques aux VRPTW."""
    print("\n" + "=" * 70)
    print("EXEMPLE 4: Fonctionnalités VRPTW")
    print("=" * 70)
    
    manager = InstanceFileManager("/data")
    instance = manager.load_instance("C1_2_1")
    
    if instance.is_vrptw():
        print("\n1. Fenêtres temporelles:")
        print(f"   {'Client':<8} {'Ready':>8} {'Due':>8} {'Service':>8}")
        print("   " + "-" * 35)
        
        for i in range(min(10, instance.dimension)):
            print(f"   {i:<8} {instance.ready_times[i]:>8.0f} "
                  f"{instance.due_dates[i]:>8.0f} {instance.service_times[i]:>8.0f}")
        
        print(f"\n   Horizon de planification: [0, {instance.due_dates.max():.0f}]")


def example_batch_processing():
    """Traite plusieurs instances en batch."""
    print("\n" + "=" * 70)
    print("EXEMPLE 5: Traitement batch")
    print("=" * 70)
    
    manager = InstanceFileManager("/data")
    
    # Récupère toutes les instances HG
    hg_instances = manager.list_instances(set_name="HG")
    
    if not hg_instances:
        print("   Aucune instance HG trouvée")
        return
    
    print(f"\n1. Traitement de {len(hg_instances)} instances HG:")
    print(f"   {'Instance':<15} {'Clients':<8} {'Véhicules':<10} {'Demande totale':<15}")
    print("   " + "-" * 50)
    
    for name in hg_instances[:10]:  # Limite à 10 pour l'exemple
        instance = manager.load_instance(name)
        total_demand = instance.get_total_demand()
        
        print(f"   {name:<15} {instance.dimension:<8} "
              f"{instance.n_vehicles:<10} {total_demand:<15}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" DÉMONSTRATION: InstanceFileManager")
    print("=" * 70)
    
    try:
        example_basic_usage()
        example_distance_calculation()
        example_solution_validation()
        example_vrptw_features()
        example_batch_processing()
        
        print("\n" + "=" * 70)
        print("✓ Tous les exemples exécutés avec succès!")
        print("=" * 70)
    
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()