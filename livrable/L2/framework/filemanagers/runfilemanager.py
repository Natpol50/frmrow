"""
Run File Manager - Système de gestion des résultats expérimentaux - Asha Geyon 2025

Ce module fournit un système complet pour stocker, indexer et retrouver
les résultats d'expérimentations algorithmiques sous forme de fichiers JSON.

Architecture:
- Chaque run est stockée dans un fichier JSON unique nommé via timestamp & hash
- Un fichier d'index maintient les mappings hash→fichier et config→runs pour des accès ultra rapides
- Les données sont sérialisables (pour le hashage hein) et reconstituables complètement

"""

import json
import hashlib
import platform
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ConvergencePoint:
    """
    Point d'amélioration dans l'historique de convergence d'un algorithme.
    
    Attributs:
        iteration: Numéro de l'itération où l'amélioration a eu lieu
        cost: Coût de la solution à cette itération
    """
    iteration: int
    cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le point en dictionnaire pour sérialisation JSON."""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ConvergencePoint':
        """Reconstruit un ConvergencePoint depuis un dictionnaire.Pour charger depuis un JSON"""
        return ConvergencePoint(**data)


@dataclass
class Results:
    """
    Résultats complets d'une exécution algorithmique.
    
    Attributs:
        time_seconds: Temps d'exécution total en secondes
        n_iterations: Nombre total d'itérations effectués
        cost: Coût final de la solution trouvée
        solution: Liste des routes (chaque route est une liste d'indices de clients)
        convergence: Historique des améliorations (points où le coût total a diminué)
        additional_info: Métadonnées supplémentaires (coût initial, amélioration, statistiques solver, etc.)
    """
    time_seconds: float
    n_iterations: int
    cost: float
    solution: List[List[int]]
    convergence: List[ConvergencePoint] = field(default_factory=list)
    additional_info: Dict[str, Any] = field(default_factory=dict)  # ← NOUVEAU CHAMP
    
    def add_improvement(self, iteration: int, cost: float):
        """
        Ajoute un point d'amélioration à l'historique de convergence.
        
        Args:
            iteration: Numéro de l'itération
            cost: Nouveau coût amélioré
        """
        self.convergence.append(ConvergencePoint(iteration, cost))
    
    def get_final_cost(self) -> float:
        """Retourne le coût final (normalement, dernier point de convergence ou cost)."""
        if self.convergence:
            return self.convergence[-1].cost
        return self.cost
    
    def get_n_routes(self) -> int:
        """Retourne le nombre de routes dans la solution."""
        return len(self.solution)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit les résultats en dictionnaire pour sérialisation JSON."""
        return {
            'time_seconds': self.time_seconds,
            'n_iterations': self.n_iterations,
            'cost': self.cost,
            'solution': self.solution,
            'convergence': [cp.to_dict() for cp in self.convergence],
            'additional_info': self.additional_info  # ← AJOUT ICI
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Results':
        """Reconstruit des Results depuis un dictionnaire.Pour charger depuis un JSON"""
        convergence = [ConvergencePoint.from_dict(cp) for cp in data.get('convergence', [])]
        return Results(
            time_seconds=data['time_seconds'],
            n_iterations=data['n_iterations'],
            cost=data['cost'],
            solution=data['solution'],
            convergence=convergence,
            additional_info=data.get('additional_info', {})  # ← AJOUT ICI avec .get() pour rétro-compatibilité
        )


@dataclass
class Config:
    """
    Configuration complète d'une exécution algorithmique.
    
    Attributs:
        instance_name: Nom de l'instance (ex: "A-n32-k5")
        solver_name: Nom de l'algorithme (ex: "simulated_annealing")
        seed: Graine aléatoire pour la reproductibilité
        parameters: Dictionnaire des paramètres spécifiques à l'algorithme
    """
    instance_name: str
    solver_name: str
    seed: int
    parameters: Dict[str, Any]
    
    def compute_hash_run(self) -> str:
        """
        Calcule le hash complet incluant la seed.
        Identifie UNE exécution unique et spécifique.
        
        Returns:
            Hash SHA256 de la configuration complète, pour pouvoir rapidement retrouver la run.
        """
        data = {
            'instance_name': self.instance_name,
            'solver_name': self.solver_name,
            'seed': self.seed,
            'parameters': self._normalize_parameters(self.parameters)
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def compute_hash_config(self) -> str:
        """
        Calcule le hash sans la seed.
        Permet d'identifier un groupe d'exécutions (même config, seeds différentes) très rapidement, pour pouvoir comparer après une run.
        
        Returns:
            Hash SHA256 de la configuration sans la seed
        """
        data = {
            'instance_name': self.instance_name,
            'solver_name': self.solver_name,
            'parameters': self._normalize_parameters(self.parameters)
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    @staticmethod
    def _normalize_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise les paramètres pour éviter les problèmes de précision flottante.
        Arrondit les floats à 10 décimales pour garantir des hash cohérents.
        """
        normalized = {}
        for key, value in params.items():
            if isinstance(value, float):
                normalized[key] = round(value, 10)
            elif isinstance(value, dict):
                normalized[key] = Config._normalize_parameters(value)
            else:
                normalized[key] = value
        return normalized
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la config en dictionnaire pour sérialisation JSON."""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Config':
        """Reconstruit une Config depuis un dictionnaire, pour charger depuis un JSON"""
        return Config(**data)


@dataclass
class Metadata:
    """
    Métadonnées d'environnement pour une exécution.
    Permet la traçabilité et la reproductibilité complète.
    
    Attributs:
        user: Nom de l'utilisateur ayant lancé l'expérience (configurable)
        date: Date et heure de l'exécution (format ISO)
        hardware_id: Identifiant de la machine (nom d'hôte)
        cpu_arch: Modèle du processeur
        ram: Quantité de RAM disponible
        python_version: Version de Python utilisée
    """
    user: str
    date: str
    hardware_id: str
    cpu_arch: str
    ram: str
    python_version: str
    
    @staticmethod
    def capture_system_info(user: Optional[str] = None) -> 'Metadata':
        """
        Capture automatiquement les informations système actuelles.
        
        Args:
            user: Nom de l'utilisateur (optionnel, détecté automatiquement si non fourni)
            
        Returns:
            Metadata avec toutes les informations système
        """
        import getpass
        import psutil
        
        if user is None:
            user = getpass.getuser()
        
        return Metadata(
            user=user,
            date=datetime.now().isoformat(),
            hardware_id=platform.node(),
            cpu_arch=platform.processor() or platform.machine(),
            ram=f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
            python_version=sys.version.split()[0]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit les metadata en dictionnaire pour sérialisation JSON."""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Metadata':
        """Reconstruit des Metadata depuis un dictionnaire, pour charger depuuis un JSON (pas forcément utile, mais un petit plus)"""
        return Metadata(**data)


@dataclass
class Run:
    """
    Représentation complète d'une exécution expérimentale.
    Contient toutes les informations nécessaires pour reproduire et analyser une run.
    C'est exactement cette classe que l'on vas stocker.
    
    Attributs:
        metadata: Informations d'environnement
        hash_run: Hash unique identifiant cette exécution précise
        hash_config: Hash identifiant le groupe de configuration
        config: Configuration algorithmique utilisée
        results: Résultats obtenus
    """
    metadata: Metadata
    hash_run: str
    hash_config: str
    config: Config
    results: Results
    
    def save(self, filepath: Path):
        """
        Sauvegarde la run dans un fichier JSON.
        
        Args:
            filepath: Chemin du fichier de destination
        """
        data = self.to_dict()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load(filepath: Path) -> 'Run':
        """
        Charge une run depuis un fichier JSON.
        
        Args:
            filepath: Chemin du fichier à charger
            
        Returns:
            Run reconstruit depuis le fichier
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            json.JSONDecodeError: Si le fichier n'est pas un JSON valide
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Run.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la run complète en dictionnaire pour sérialisation JSON."""
        return {
            'metadata': self.metadata.to_dict(),
            'hash_run': self.hash_run,
            'hash_config': self.hash_config,
            'config': self.config.to_dict(),
            'results': self.results.to_dict()
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Run':
        """Reconstruit une run complète depuis un dictionnaire, poru charger depuis un JSON."""
        return Run(
            metadata=Metadata.from_dict(data['metadata']),
            hash_run=data['hash_run'],
            hash_config=data['hash_config'],
            config=Config.from_dict(data['config']),
            results=Results.from_dict(data['results'])
        )


class RunFileManager:
    """
    Gestionnaire de fichiers pour les runs expérimentales.
    
    Gère le stockage, l'indexation et la récupération des résultats d'expériences.
    Chaque run est stocké dans un fichier unique nommé par timestamp + hash.
    Un index maintient les mappings pour des accès rapides.
    
    Structure de l'index:
        {
            "hash_to_file": {
                "hash_run_complet": "YYYYMMDD_HHMMSS_XXXXXXXX.json"
            },
            "config_groups": {
                "hash_config": ["hash_run1", "hash_run2", ...]
            }
        }
    
    Attributs:
        results_dir: Répertoire contenant les fichiers de résultats
        index_path: Chemin du fichier d'index
        index: Index chargé en mémoire
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialise le gestionnaire de fichiers.
        
        Args:
            results_dir: Chemin du répertoire des résultats
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.results_dir / "index.json"
        # Ensure index attribute exists before any call to _save_index()
        self.index = {
            "hash_to_file": {},
            "config_groups": {}
        }
        if not self.index_path.exists():
            self._save_index()
        else:
            # Load existing index from disk
            self.index = self._load_index()
    
    def run_exists(self, config: Config) -> bool:
        """
        Vérifie si une run avec cette configuration existe déjà.
        
        Args:
            config: Configuration à vérifier
            
        Returns:
            True si la run existe, False sinon
        """
        hash_run = config.compute_hash_run()
        return hash_run in self.index["hash_to_file"]
    
    def load_run(self, config: Config) -> Run:
        """
        Charge une run existant basé sur sa configuration.
        
        Args:
            config: Configuration du run à charger
            
        Returns:
            Run chargé depuis le fichier
            
        Raises:
            FileNotFoundError: Si la run n'existe pas
        """
        hash_run = config.compute_hash_run()
        
        if hash_run not in self.index["hash_to_file"]:
            raise FileNotFoundError(
                f"Run avec hash {hash_run[:12]}... n'existe pas. "
                f"Config: {config.instance_name}/{config.solver_name}/seed={config.seed}"
            )
        
        filename = self.index["hash_to_file"][hash_run]
        filepath = self.results_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Fichier {filename} référencé dans l'index mais absent du filesystem. "
                f"Exécutez rebuild_index() pour corriger."
            )
        
        return Run.load(filepath)
    
    def add_run(self, config: Config, results: Results) -> Path:
        """
        Ajoute un nouveau run: crée le fichier et met à jour l'index.
        Si la run existe déjà, retourne son chemin sans le recréer.
        
        Args:
            config: Configuration du run
            results: Résultats obtenus
            
        Returns:
            Chemin du fichier créé ou existant
        """
        hash_run = config.compute_hash_run()
        hash_config = config.compute_hash_config()
        
        # Vérification d'existence
        if hash_run in self.index["hash_to_file"]:
            existing_file = self.index["hash_to_file"][hash_run]
            print(f"Run déjà existante, skipping: {existing_file}")
            return self.results_dir / existing_file
        
        # Génération du nom de fichier avec timestamp + hash court
        filename = self._generate_filename(hash_run)
        filepath = self.results_dir / filename
        
        # Création du run complet
        run = Run(
            metadata=Metadata.capture_system_info(),
            hash_run=hash_run,
            hash_config=hash_config,
            config=config,
            results=results
        )
        
        # Sauvegarde du fichier
        run.save(filepath)
        
        # Mise à jour de l'index
        self.index["hash_to_file"][hash_run] = filename
        
        if hash_config not in self.index["config_groups"]:
            self.index["config_groups"][hash_config] = []
        self.index["config_groups"][hash_config].append(hash_run)
        
        self._save_index()
        
        return filepath
    
    def load_config_runs(self, config: Config) -> List[Run]:
        """
        Charge tous les runs d'une même configuration (seeds différentes).
        
        Args:
            config: Configuration dont on veut tous les runs
            
        Returns:
            Liste de tous les runs partageant cette config
        """
        hash_config = config.compute_hash_config()
        
        if hash_config not in self.index["config_groups"]:
            return []
        
        hash_runs = self.index["config_groups"][hash_config]
        
        runs = []
        for hash_run in hash_runs:
            if hash_run not in self.index["hash_to_file"]:
                print(f"Warning: hash_run {hash_run[:12]}... dans config_groups mais pas dans hash_to_file")
                continue
            
            filename = self.index["hash_to_file"][hash_run]
            filepath = self.results_dir / filename
            
            if not filepath.exists():
                print(f"Warning: fichier {filename} référencé mais absent")
                continue
            
            runs.append(Run.load(filepath))
        
        return runs
    
    def delete_run(self, hash_run: str):
        """
        Supprime complètement une run: fichier + entrées dans l'index.
        
        Args:
            hash_run: Hash du run à supprimer
        """
        if hash_run not in self.index["hash_to_file"]:
            print(f"Run {hash_run[:12]}... n'existe pas dans l'index")
            return
        
        # Suppression du fichier
        filename = self.index["hash_to_file"][hash_run]
        filepath = self.results_dir / filename
        
        if filepath.exists():
            filepath.unlink()
            print(f"Fichier supprimé: {filename}")
        else:
            print(f"Warning: fichier {filename} déjà absent du filesystem")
        
        # Retrait de hash_to_file
        del self.index["hash_to_file"][hash_run]
        
        # Retrait de config_groups
        for hash_config, runs in list(self.index["config_groups"].items()):
            if hash_run in runs:
                runs.remove(hash_run)
                
                # Nettoyage des configs vides
                if not runs:
                    del self.index["config_groups"][hash_config]
                break
        
        self._save_index()
    
    def rebuild_index(self):
        """
        Reconstruit l'index complet depuis les fichiers existants.
        Utile en cas de corruption ou de modifications manuelles.
        """
        self.index = {
            "hash_to_file": {},
            "config_groups": {}
        }
        
        n_files = 0
        for filepath in self.results_dir.glob("*.json"):
            if filepath.name == "index.json":
                continue
            
            try:
                run = Run.load(filepath)
                
                # Ajout à hash_to_file
                self.index["hash_to_file"][run.hash_run] = filepath.name
                
                # Ajout à config_groups
                if run.hash_config not in self.index["config_groups"]:
                    self.index["config_groups"][run.hash_config] = []
                self.index["config_groups"][run.hash_config].append(run.hash_run)
                
                n_files += 1
            except Exception as e:
                print(f"Erreur lors du chargement de {filepath.name}: {e}")
        
        self._save_index()
        
        n_configs = len(self.index["config_groups"])
        print(f"Index reconstruit: {n_configs} configs, {n_files} runs")
    
    def _load_index(self) -> Dict[str, Any]:
        """
        Charge l'index depuis le fichier JSON.
        Crée un index vide si le fichier n'existe pas.
        """
        if self.index_path.exists():
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "hash_to_file": {},
            "config_groups": {}
        }
    
    def _save_index(self):
        """Sauvegarde l'index dans le fichier JSON."""
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
    
    def _generate_filename(self, hash_run: str) -> str:
        """
        Génère un nom de fichier unique basé sur timestamp + hash court.
        
        Format: YYYYMMDD_HHMMSS_XXXXXXXX.json
        où XXXXXXXX = 8 premiers caractères du hash
        
        Args:
            hash_run: Hash complet du run
            
        Returns:
            Nom de fichier généré
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_short = hash_run[:8]
        return f"{timestamp}_{hash_short}.json"