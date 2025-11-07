"""
Instance File Manager - Système de gestion et chargement des instances VRP

Ce module fournit un système optimisé pour scanner, indexer et charger
rapidement les instances VRP depuis un dossier structuré.

Supporte plusieurs formats:
- Format VRPLIB/TSPLIB (Augerat, Christofides, etc.)
- Format Solomon/Homberger (VRPTW tabulaire)

Architecture:
- Détection automatique du format
- Scan récursif des dossiers pour trouver les fichiers
- Index persistant pour accès O(1)
- Cache des matrices de distances (calcul coûteux fait une seule fois)

Structure attendue:
    data/
    ├── A/
    │   ├── A-n32-k5.vrp
    │   └── Solutions/
    │       └── A-n32-k5.sol
    ├── Solomon/
    │   ├── C101.txt
    │   └── Solutions/
    │       └── C101.sol
    └── ...

Usage:
    manager = InstanceFileManager("data")
    manager.scan_instances()  # Une fois au début
    
    instance = manager.load_instance("C1_2_1")
    distance_matrix = instance.distance_matrix  # Déjà calculée et cachée
    solution = manager.load_solution("C1_2_1")
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal
from dataclasses import dataclass, field
import re


@dataclass
class Instance:
    """
    Représentation unifiée d'une instance VRP.
    
    Attributes:
        name: Nom de l'instance (ex: "A-n32-k5" ou "C1_2_1")
        set_name: Nom du set (ex: "A", "Solomon", "HG")
        dimension: Nombre de nœuds (dépôt + clients)
        capacity: Capacité des véhicules
        n_vehicles: Nombre de véhicules disponibles
        coordinates: Coordonnées (x, y) de chaque nœud [dimension x 2]
        demands: Demande de chaque client [dimension]
        depot: Indice du dépôt (généralement 0)
        distance_matrix: Matrice de distances [dimension x dimension]
        
        # Spécifique VRPTW (optionnel)
        ready_times: Fenêtre temporelle - début [dimension] (None si pas VRPTW)
        due_dates: Fenêtre temporelle - fin [dimension] (None si pas VRPTW)
        service_times: Temps de service [dimension] (None si pas VRPTW)
    """
    name: str
    set_name: str
    dimension: int
    capacity: int
    n_vehicles: int
    coordinates: np.ndarray
    demands: np.ndarray
    distance_matrix: np.ndarray
    depot: int = 0
    
    # VRPTW optionnel
    ready_times: Optional[np.ndarray] = None
    due_dates: Optional[np.ndarray] = None
    service_times: Optional[np.ndarray] = None
    
    def get_client_indices(self) -> List[int]:
        """Retourne la liste des indices clients (sans le dépôt)."""
        return [i for i in range(self.dimension) if i != self.depot]
    
    def get_total_demand(self) -> int:
        """Retourne la demande totale de tous les clients."""
        return int(np.sum(self.demands[self.get_client_indices()]))
    
    def is_vrptw(self) -> bool:
        """Vérifie si l'instance a des contraintes de time windows."""
        return self.ready_times is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'instance en dictionnaire."""
        base = {
            'name': self.name,
            'set_name': self.set_name,
            'dimension': self.dimension,
            'capacity': self.capacity,
            'n_vehicles': self.n_vehicles,
            'coordinates': self.coordinates.tolist(),
            'demands': self.demands.tolist(),
            'depot': self.depot,
            'is_vrptw': self.is_vrptw()
        }
        
        if self.is_vrptw():
            base.update({
                'ready_times': self.ready_times.tolist(),
                'due_dates': self.due_dates.tolist(),
                'service_times': self.service_times.tolist()
            })
        
        return base


@dataclass
class Solution:
    """
    Représentation d'une solution VRP.
    
    Attributes:
        instance_name: Nom de l'instance associée
        cost: Coût total de la solution
        routes: Liste des routes (chaque route = liste d'indices clients)
        n_vehicles_used: Nombre de véhicules utilisés
    """
    instance_name: str
    cost: float
    routes: List[List[int]]
    n_vehicles_used: int
    
    @staticmethod
    def parse_from_file(filepath: Path, instance_name: str) -> 'Solution':
        """
        Parse un fichier .sol au format VRPLIB.
        
        Format attendu:
            Route #1: 5 3 7 8 10
            Route #2: 13 17 18 19
            Cost 827.3
            
        Args:
            filepath: Chemin du fichier .sol
            instance_name: Nom de l'instance
            
        Returns:
            Solution parsée
        """
        routes = []
        cost = None
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('Route'):
                    route_str = line.split(':', 1)[1].strip()
                    route = [int(x) for x in route_str.split()]
                    routes.append(route)
                
                elif line.startswith('Cost'):
                    cost_str = line.split()[1]
                    cost = float(cost_str)
        
        if cost is None:
            raise ValueError(f"No cost found in solution file: {filepath}")
        
        return Solution(
            instance_name=instance_name,
            cost=cost,
            routes=routes,
            n_vehicles_used=len(routes)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la solution en dictionnaire."""
        return {
            'instance_name': self.instance_name,
            'cost': self.cost,
            'routes': self.routes,
            'n_vehicles_used': self.n_vehicles_used
        }
    
    def validate_against_instance(self, instance: Instance) -> Tuple[bool, List[str]]:
        """
        Vérifie si la solution respecte les contraintes de l'instance.
        
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        # Vérifie que tous les clients sont visités exactement une fois
        visited = set()
        for route in self.routes:
            for customer in route:
                if customer in visited:
                    violations.append(f"Customer {customer} visited multiple times")
                visited.add(customer)
        
        expected_customers = set(instance.get_client_indices())
        missing = expected_customers - visited
        if missing:
            violations.append(f"Missing customers: {missing}")
        
        extra = visited - expected_customers
        if extra:
            violations.append(f"Unknown customers: {extra}")
        
        # Vérifie la capacité
        for i, route in enumerate(self.routes, 1):
            route_demand = sum(instance.demands[c] for c in route)
            if route_demand > instance.capacity:
                violations.append(
                    f"Route {i} exceeds capacity: {route_demand} > {instance.capacity}"
                )
        
        return len(violations) == 0, violations


class InstanceFileManager:
    """
    Gestionnaire de fichiers d'instances VRP avec support multi-format.
    
    Supporte:
    - Format VRPLIB/TSPLIB (avec sections DIMENSION, NODE_COORD_SECTION, etc.)
    - Format Solomon/Homberger (tabulaire avec colonnes XCOORD, YCOORD, etc.)
    
    Attributes:
        data_dir: Répertoire racine contenant les instances
        index_path: Chemin du fichier d'index
        cache_dir: Répertoire pour cache des matrices de distances
        index: Index chargé en mémoire {instance_name → metadata}
        instances_cache: Cache des instances chargées {name → Instance}
    """
    
    def __init__(self, data_dir: str = "data", cache_distances: bool = True):
        """
        Initialise le gestionnaire.
        
        Args:
            data_dir: Chemin du répertoire racine des données
            cache_distances: Si True, sauvegarde les matrices de distances sur disque
        """
        self.data_dir = Path(data_dir)
        self.cache_distances = cache_distances
        self.index_path = self.data_dir / "instance_index.json"
        self.cache_dir = self.data_dir / ".cache"
        
        if self.cache_distances:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index = self._load_index()
        self.instances_cache: Dict[str, Instance] = {}
    
    def scan_instances(self, force_rescan: bool = False):
        """
        Scanne récursivement le dossier data pour trouver toutes les instances.
        
        Détecte automatiquement les formats:
        - .vrp → Format VRPLIB
        - .txt → Format Solomon/Homberger
        - Autres extensions configurables
        
        Args:
            force_rescan: Si True, rescanne même si l'index existe
        """
        if not force_rescan and self.index:
            print(f"Index déjà existant avec {len(self.index)} instances")
            print("Utilisez force_rescan=True pour forcer un nouveau scan")
            return
        
        print(f"Scan de {self.data_dir}...")
        self.index = {}
        # Extensions supportées
        supported_extensions = ['.vrp', '.txt']
        
        # Scan récursif
        instance_files = []
        for ext in supported_extensions:
            instance_files.extend(self.data_dir.rglob(f"*{ext}"))
        
        # Filtre les fichiers dans Solutions/
        instance_files = [f for f in instance_files if 'solution' not in str(f).lower()]
        
        print(f"Trouvé {len(instance_files)} fichiers d'instances")
        
        for filepath in instance_files:
            try:
                # Détecte le format et parse les métadonnées
                format_type = self._detect_format(filepath)
                metadata = self._parse_instance_metadata(filepath, format_type)
                
                # Cherche le fichier solution associé
                solution_path = self._find_solution_file(filepath)
                
                # Ajoute à l'index
                self.index[metadata['name']] = {
                    'name': metadata['name'],
                    'set_name': metadata['set_name'],
                    'filepath': str(filepath),
                    'solution_filepath': str(solution_path) if solution_path else None,
                    'dimension': metadata['dimension'],
                    'capacity': metadata['capacity'],
                    'vehicles': metadata['vehicles'],
                    'format': format_type,
                    'is_vrptw': metadata.get('is_vrptw', False)
                }
                
                vrptw_str = " (VRPTW)" if metadata.get('is_vrptw') else ""
                print(f"  ✓ {metadata['name']} ({metadata['dimension']} nodes, "
                      f"{metadata['vehicles']} vehicles){vrptw_str}")
            
            except Exception as e:
                print(f"  ✗ Erreur lors du parsing de {filepath.name}: {e}")
        
        # Sauvegarde l'index
        self._save_index()
        print(f"\n✓ Index créé: {len(self.index)} instances indexées")
    
    def load_instance(self, instance_name: str, use_cache: bool = True) -> Instance:
        """
        Charge une instance par son nom.
        
        La matrice de distances est TOUJOURS calculée et incluse.
        Utilise le cache disque pour éviter les recalculs.
        
        Args:
            instance_name: Nom de l'instance (ex: "A-n32-k5" ou "C1_2_1")
            use_cache: Si True, utilise le cache en mémoire
            
        Returns:
            Instance chargée avec matrice de distances
            
        Raises:
            KeyError: Si l'instance n'existe pas dans l'index
            FileNotFoundError: Si le fichier n'existe pas
        """
        # Check cache mémoire
        if use_cache and instance_name in self.instances_cache:
            return self.instances_cache[instance_name]
        
        # Check index
        if instance_name not in self.index:
            raise KeyError(
                f"Instance '{instance_name}' not found in index. "
                f"Run scan_instances() first or check the name."
            )
        
        metadata = self.index[instance_name]
        filepath = Path(metadata['filepath'])
        
        if not filepath.exists():
            raise FileNotFoundError(f"Instance file not found: {filepath}")
        
        # Parse l'instance selon son format
        format_type = metadata['format']
        if format_type == 'solomon':
            instance = self._parse_solomon_format(filepath, metadata)
        elif format_type == 'vrplib':
            instance = self._parse_vrplib_format(filepath, metadata)
        else:
            raise ValueError(f"Unknown format: {format_type}")
        
        # Gère le cache de la matrice de distances
        if self.cache_distances:
            distance_matrix = self._load_distance_matrix_from_cache(instance_name)
            if distance_matrix is not None:
                instance.distance_matrix = distance_matrix
            else:
                # Calcule et sauvegarde
                instance.distance_matrix = self._compute_distance_matrix(instance.coordinates)
                self._save_distance_matrix_to_cache(instance_name, instance.distance_matrix)
        else:
            # Calcule directement sans sauvegarder
            instance.distance_matrix = self._compute_distance_matrix(instance.coordinates)
        
        # Ajoute au cache mémoire
        if use_cache:
            self.instances_cache[instance_name] = instance
        
        return instance
    
    def load_solution(self, instance_name: str) -> Optional[Solution]:
        """
        Charge la solution optimale/best known pour une instance.
        
        Args:
            instance_name: Nom de l'instance
            
        Returns:
            Solution si elle existe, None sinon
        """
        if instance_name not in self.index:
            raise KeyError(f"Instance '{instance_name}' not found in index")
        
        solution_filepath = self.index[instance_name].get('solution_filepath')
        
        if not solution_filepath:
            return None
        
        solution_path = Path(solution_filepath)
        if not solution_path.exists():
            return None
        
        return Solution.parse_from_file(solution_path, instance_name)
    
    def list_instances(self, set_name: Optional[str] = None) -> List[str]:
        """
        Liste toutes les instances disponibles, optionnellement filtrées par set.
        
        Args:
            set_name: Si fourni, filtre par nom de set (ex: "A", "B", "Solomon")
            
        Returns:
            Liste des noms d'instances
        """
        if set_name:
            return [name for name, meta in self.index.items() 
                    if meta['set_name'] == set_name]
        return list(self.index.keys())
    
    def get_instance_info(self, instance_name: str) -> Dict[str, Any]:
        """
        Retourne les métadonnées d'une instance sans la charger.
        
        Args:
            instance_name: Nom de l'instance
            
        Returns:
            Dictionnaire avec les métadonnées
        """
        if instance_name not in self.index:
            raise KeyError(f"Instance '{instance_name}' not found in index")
        
        return self.index[instance_name].copy()
    
    def clear_cache(self, memory_only: bool = False):
        """
        Vide le cache.
        
        Args:
            memory_only: Si True, vide seulement le cache mémoire (pas le disque)
        """
        self.instances_cache.clear()
        
        if not memory_only and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("Cache disque vidé")
    
    # ========================================================================
    # Méthodes de détection et parsing
    # ========================================================================
    
    def _detect_format(self, filepath: Path) -> Literal['solomon', 'vrplib']:
        """
        Détecte automatiquement le format du fichier.
        
        Stratégie:
        1. Lit les premières lignes
        2. Cherche des marqueurs de format
        
        Format Solomon: Commence par "VEHICLE" ou contient des colonnes tabulaires
        Format VRPLIB: Contient "DIMENSION :" ou "NODE_COORD_SECTION"
        """
        with open(filepath, 'r') as f:
            # Lit les 20 premières lignes
            lines = [f.readline().strip() for _ in range(20)]
        
        content = '\n'.join(lines)
        
        # Marqueurs VRPLIB
        if 'DIMENSION' in content and ':' in content:
            return 'vrplib'
        
        if 'NODE_COORD_SECTION' in content:
            return 'vrplib'
        
        # Marqueurs Solomon
        if 'VEHICLE' in content and 'CUSTOMER' in content:
            return 'solomon'
        
        if 'XCOORD' in content and 'YCOORD' in content:
            return 'solomon'
        
        # Par défaut, essaie Solomon (plus tolérant)
        return 'solomon'
    
    def _parse_instance_metadata(self, filepath: Path, format_type: str) -> Dict[str, Any]:
        """
        Parse les métadonnées minimales selon le format.
        """
        name = filepath.stem
        set_name = self._extract_set_name(name, filepath)
        
        if format_type == 'solomon':
            return self._parse_solomon_metadata(filepath, name, set_name)
        else:
            return self._parse_vrplib_metadata(filepath, name, set_name)
    
    def _parse_solomon_metadata(self, filepath: Path, name: str, set_name: str) -> Dict[str, Any]:
        """
        Parse les métadonnées d'un fichier format Solomon/Homberger.
        
        Format:
            VEHICLE
            NUMBER     CAPACITY
              50          200
            
            CUSTOMER
            CUST NO.  XCOORD.    YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME
             0      70         70          0          0       1351          0
             1      33         78         20        750        809         90
             ...
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        n_vehicles = None
        capacity = None
        dimension = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Parse VEHICLE section
            if 'VEHICLE' in line:
                # Saute la ligne d'en-tête (NUMBER CAPACITY)
                i += 1
                if i < len(lines):
                    i += 1
                    if i < len(lines):
                        parts = lines[i].split()
                        if len(parts) >= 2:
                            n_vehicles = int(parts[0])
                            capacity = int(parts[1])
            
            # Parse CUSTOMER section
            elif 'CUSTOMER' in line:
                # Saute la ligne d'en-tête
                i += 1
                # Compte les lignes de données
                i += 1
                while i < len(lines):
                    parts = lines[i].split()
                    if len(parts) >= 7:  # Ligne valide
                        dimension += 1
                        i += 1
                    else:
                        break
                break
            
            i += 1
        
        return {
            'name': name,
            'set_name': set_name,
            'dimension': dimension,
            'capacity': capacity or 0,
            'vehicles': n_vehicles or 0,
            'is_vrptw': True
        }
    
    def _parse_vrplib_metadata(self, filepath: Path, name: str, set_name: str) -> Dict[str, Any]:
        """Parse les métadonnées d'un fichier format VRPLIB."""
        dimension = None
        capacity = None
        vehicles = None
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('DIMENSION'):
                    dimension = int(line.split(':')[1].strip())
                elif line.startswith('CAPACITY'):
                    capacity = int(line.split(':')[1].strip())
                elif line.startswith('NODE_COORD_SECTION'):
                    break
        
        # Extrait le nombre de véhicules depuis le nom
        vehicles_match = re.search(r'-k(\d+)', name)
        if vehicles_match:
            vehicles = int(vehicles_match.group(1))
        
        if dimension is None:
            raise ValueError(f"DIMENSION not found in {filepath}")
        
        return {
            'name': name,
            'set_name': set_name,
            'dimension': dimension,
            'capacity': capacity or 0,
            'vehicles': vehicles or 0,
            'is_vrptw': False
        }
    
    def _parse_solomon_format(self, filepath: Path, metadata: Dict[str, Any]) -> Instance:
        """
        Parse complètement une instance au format Solomon/Homberger.
        
        Lit toutes les données: coordonnées, demandes, fenêtres temporelles.
        Calcule automatiquement la dimension en comptant les lignes.
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Trouve la section CUSTOMER
        customer_start = None
        for i, line in enumerate(lines):
            if 'CUSTOMER' in line:
                customer_start = i + 2  # Saute "CUSTOMER" et la ligne d'en-tête
                break
        
        if customer_start is None:
            raise ValueError("CUSTOMER section not found")
        
        # Parse toutes les lignes de données
        data_rows = []
        for line in lines[customer_start:]:
            parts = line.split()
            if len(parts) >= 7:  # Ligne valide
                data_rows.append([float(x) for x in parts])
            elif len(parts) > 0:  # Ligne non vide mais invalide
                break
        
        if not data_rows:
            raise ValueError("No customer data found")
        
        # Convertit en numpy arrays
        data = np.array(data_rows)
        
        dimension = len(data)
        coordinates = data[:, 1:3]  # Colonnes XCOORD, YCOORD
        demands = data[:, 3].astype(int)  # Colonne DEMAND
        ready_times = data[:, 4]  # Colonne READY TIME
        due_dates = data[:, 5]  # Colonne DUE DATE
        service_times = data[:, 6]  # Colonne SERVICE TIME
        
        return Instance(
            name=metadata['name'],
            set_name=metadata['set_name'],
            dimension=dimension,
            capacity=metadata['capacity'],
            n_vehicles=metadata['vehicles'],
            coordinates=coordinates,
            demands=demands,
            distance_matrix=None,
            depot=0,
            ready_times=ready_times,
            due_dates=due_dates,
            service_times=service_times
        )
    
    def _parse_vrplib_format(self, filepath: Path, metadata: Dict[str, Any]) -> Instance:
        """Parse complètement une instance au format VRPLIB."""
        dimension = metadata['dimension']
        
        coordinates = np.zeros((dimension, 2))
        demands = np.zeros(dimension)
        depot = 0
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        section = None
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('NAME') or line.startswith('COMMENT'):
                continue
            
            if 'NODE_COORD_SECTION' in line:
                section = 'COORD'
                continue
            elif 'DEMAND_SECTION' in line:
                section = 'DEMAND'
                continue
            elif 'DEPOT_SECTION' in line:
                section = 'DEPOT'
                continue
            elif 'EOF' in line:
                break
            
            if section == 'COORD':
                parts = line.split()
                if len(parts) >= 3:
                    idx = int(parts[0]) - 1
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates[idx] = [x, y]
            
            elif section == 'DEMAND':
                parts = line.split()
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1
                    demand = int(parts[1])
                    demands[idx] = demand
            
            elif section == 'DEPOT':
                depot_id = int(line)
                if depot_id != -1:
                    depot = depot_id - 1
        
        return Instance(
            name=metadata['name'],
            set_name=metadata['set_name'],
            dimension=dimension,
            capacity=metadata['capacity'],
            n_vehicles=metadata['vehicles'],
            coordinates=coordinates,
            demands=demands,
            distance_matrix=None, 
            depot=depot
        )
    
    def _compute_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice de distances euclidiennes de manière ultra-optimisée.
        
        Utilise numpy broadcasting pour calcul vectorisé.
        Complexité: O(n²) mais vectorisé (très rapide en pratique).
        
        Pour n=1000 clients: ~0.1s vs ~10s avec double boucle Python.
        
        Args:
            coordinates: Array [n, 2] des coordonnées
            
        Returns:
            Matrice de distances [n, n]
        """
        # Broadcasting: [n, 1, 2] - [1, n, 2] = [n, n, 2]
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        
        # Distance euclidienne: sqrt(sum((xi - xj)²))
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Arrondit selon convention TSPLIB (floor)
        distances = np.floor(distances * 10.0) / 10.0
        
        return distances
    
    # ========================================================================
    # Utilitaires
    # ========================================================================
    
    def _find_solution_file(self, instance_path: Path) -> Optional[Path]:
        """
        Cherche le fichier solution (.sol) associé à une instance.
        
        Stratégies multiples:
        1. Même dossier
        2. Sous-dossier Solutions/
        3. Dossier frère Solutions/
        4. Recherche récursive (dernier recours)
        """
        sol_name = instance_path.stem + '.sol'
        
        # 1. Même dossier
        sol_path = instance_path.parent / sol_name
        if sol_path.exists():
            return sol_path
        
        # 2. Sous-dossier Solutions/
        sol_path = instance_path.parent / 'Solutions' / sol_name
        if sol_path.exists():
            return sol_path
        
        # 3. Dossier frère "Solutions"
        if instance_path.parent.name.lower() in ['instances', 'instance']:
            sol_dir = instance_path.parent.parent / 'Solutions'
            sol_path = sol_dir / sol_name
            if sol_path.exists():
                return sol_path
        
        # 4. Recherche récursive
        base_dir = (instance_path.parent.parent 
                   if instance_path.parent.name.lower() in ['instances', 'instance'] 
                   else instance_path.parent)
        
        for sol_path in base_dir.rglob(sol_name):
            return sol_path
        
        return None
    
    def _extract_set_name(self, instance_name: str, filepath: Path = None) -> str:
        """
        Extrait le nom du set depuis le nom de l'instance ou le chemin.
        
        Patterns supportés:
        - A-n32-k5 → "Augerat1"
        - C1_2_1 → "HG" ou "Solomon"
        - C101 → "Solomon"
        - Custom002 → "Custom"
        """
        if filepath:
            parts = filepath.parts
            for part in parts:
                if part.upper() == "HG":
                    return "HG"
                elif part.lower() == "solomon":
                    return "Solomon"
                elif part.lower() == "custom":
                    return "Custom"
        
        # Pattern HG: C1_10_10
        if re.match(r'^[CRC]+\d+_\d+_\d+$', instance_name):
            return "HG"
        
        # Pattern Solomon: C101
        if re.match(r'^[RC]+\d+$', instance_name):
            return "Solomon"
        
        # Pattern classique: A-n32-k5
        match = re.match(r'^([A-Za-z]+)', instance_name)
        if match:
            return match.group(1)

        # Pattern Custom: Custom002
        if re.match(r'^Custom\d+$', instance_name):
            return "Custom"

        return instance_name.split('-')[0].split('_')[0]
    
    def _load_index(self) -> Dict[str, Any]:
        """Charge l'index depuis le fichier JSON."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Sauvegarde l'index dans le fichier JSON."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _load_distance_matrix_from_cache(self, instance_name: str) -> Optional[np.ndarray]:
        """Charge une matrice de distances depuis le cache disque."""
        cache_file = self.cache_dir / f"{instance_name}_distances.npy"
        
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                return None
        
        return None
    
    def _save_distance_matrix_to_cache(self, instance_name: str, matrix: np.ndarray):
        """Sauvegarde une matrice de distances dans le cache disque."""
        cache_file = self.cache_dir / f"{instance_name}_distances.npy"
        np.save(cache_file, matrix)