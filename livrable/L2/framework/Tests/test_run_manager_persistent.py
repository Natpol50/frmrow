"""
Tests unitaires complets pour le système RunFileManager. - Mon bon copain Claude

Couvre tous les cas d'usage, edge cases et scénarios d'erreur.
Utilise pytest avec fixtures pour isoler les tests.

Exécution:
    pytest test_run_manager.py -v
    pytest test_run_manager.py -v --cov=run_manager  # avec coverage
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import time
import sys
from pathlib import Path
# Ensure the parent folder (project root) is on sys.path so runfilemanager is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filemanagers.runfilemanager import (
    ConvergencePoint, Results, Config, Metadata, Run, RunFileManager
)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def perm_dir():
    """
    Crée un répertoire pour les tests, qui reste après exécution.
    Les fichiers ne sont PAS supprimés pour inspection post-test.
    """
    base_dir = Path("tests_output")
    base_dir.mkdir(exist_ok=True)

    # Sous-dossier unique pour chaque session de test
    temp = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    temp.mkdir(parents=True, exist_ok=True)

    yield temp  # On fournit ce chemin aux tests

    # Pas de suppression — on garde les fichiers pour inspection
    print(f"[INFO] Fichiers de test conservés dans : {temp}")


@pytest.fixture
def simple_config():
    """Configuration simple pour tests de base."""
    return Config(
        instance_name="A-n32-k5",
        algo_name="simulated_annealing",
        seed=42,
        parameters={"T_init": 1000, "alpha": 0.95}
    )


@pytest.fixture
def complex_config():
    """Configuration complexe avec paramètres imbriqués."""
    return Config(
        instance_name="X-n101-k25",
        algo_name="tabu_search",
        seed=123,
        parameters={
            "tabu_tenure": 10,
            "intensification": {
                "threshold": 0.8,
                "iterations": 100
            },
            "diversification": {
                "probability": 0.1,
                "strength": 0.5
            }
        }
    )


@pytest.fixture
def simple_results():
    """Résultats simples pour tests de base."""
    return Results(
        time_seconds=45.2,
        n_iterations=1000,
        cost=812.4,
        solution=[[1, 2, 3], [4, 5, 6]],
        convergence=[
            ConvergencePoint(0, 1000),
            ConvergencePoint(100, 900),
            ConvergencePoint(500, 812.4)
        ]
    )


@pytest.fixture
def manager(perm_dir):
    """Gestionnaire de runs avec répertoire temporaire."""
    return RunFileManager(str(perm_dir / "results"))


# ============================================================================
# TESTS: ConvergencePoint
# ============================================================================

class TestConvergencePoint:
    """Tests pour la classe ConvergencePoint."""
    
    def test_creation(self):
        """Test création basique d'un point de convergence."""
        point = ConvergencePoint(iteration=100, cost=850.5)
        assert point.iteration == 100
        assert point.cost == 850.5
    
    def test_to_dict(self):
        """Test sérialisation en dictionnaire."""
        point = ConvergencePoint(iteration=50, cost=900.0)
        data = point.to_dict()
        
        assert data == {"iteration": 50, "cost": 900.0}
        assert isinstance(data, dict)
    
    def test_from_dict(self):
        """Test désérialisation depuis dictionnaire."""
        data = {"iteration": 75, "cost": 825.3}
        point = ConvergencePoint.from_dict(data)
        
        assert point.iteration == 75
        assert point.cost == 825.3
    
    def test_round_trip(self):
        """Test sérialisation/désérialisation complète."""
        original = ConvergencePoint(iteration=200, cost=780.9)
        reconstructed = ConvergencePoint.from_dict(original.to_dict())
        
        assert reconstructed.iteration == original.iteration
        assert reconstructed.cost == original.cost


# ============================================================================
# TESTS: Results
# ============================================================================

class TestResults:
    """Tests pour la classe Results."""
    
    def test_creation_minimal(self):
        """Test création avec paramètres minimaux."""
        results = Results(
            time_seconds=10.5,
            n_iterations=500,
            cost=800.0,
            solution=[[1, 2, 3]]
        )
        
        assert results.time_seconds == 10.5
        assert results.n_iterations == 500
        assert results.cost == 800.0
        assert results.solution == [[1, 2, 3]]
        assert results.convergence == []
    
    def test_add_improvement(self):
        """Test ajout de points d'amélioration."""
        results = Results(
            time_seconds=20.0,
            n_iterations=1000,
            cost=750.0,
            solution=[[1, 2]]
        )
        
        results.add_improvement(100, 900)
        results.add_improvement(500, 750)
        
        assert len(results.convergence) == 2
        assert results.convergence[0].iteration == 100
        assert results.convergence[0].cost == 900
        assert results.convergence[1].cost == 750
    
    def test_get_final_cost_with_convergence(self):
        """Test get_final_cost avec historique de convergence."""
        results = Results(
            time_seconds=30.0,
            n_iterations=2000,
            cost=800.0,
            solution=[[1]],
            convergence=[
                ConvergencePoint(0, 1000),
                ConvergencePoint(1000, 800)
            ]
        )
        
        assert results.get_final_cost() == 800
    
    def test_get_final_cost_without_convergence(self):
        """Test get_final_cost sans historique."""
        results = Results(
            time_seconds=10.0,
            n_iterations=100,
            cost=850.0,
            solution=[[1, 2]]
        )
        
        assert results.get_final_cost() == 850.0
    
    def test_get_n_routes(self):
        """Test comptage du nombre de routes."""
        results = Results(
            time_seconds=15.0,
            n_iterations=500,
            cost=700.0,
            solution=[[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        )
        
        assert results.get_n_routes() == 3
    
    def test_to_dict(self):
        """Test sérialisation complète."""
        results = Results(
            time_seconds=25.5,
            n_iterations=1500,
            cost=820.0,
            solution=[[1, 2], [3, 4]],
            convergence=[ConvergencePoint(0, 900), ConvergencePoint(100, 820)]
        )
        
        data = results.to_dict()
        
        assert data['time_seconds'] == 25.5
        assert data['n_iterations'] == 1500
        assert data['cost'] == 820.0
        assert data['solution'] == [[1, 2], [3, 4]]
        assert len(data['convergence']) == 2
        assert data['convergence'][0]['iteration'] == 0
    
    def test_from_dict(self):
        """Test désérialisation complète."""
        data = {
            'time_seconds': 30.0,
            'n_iterations': 2000,
            'cost': 750.0,
            'solution': [[1], [2], [3]],
            'convergence': [
                {'iteration': 0, 'cost': 1000},
                {'iteration': 500, 'cost': 750}
            ]
        }
        
        results = Results.from_dict(data)
        
        assert results.time_seconds == 30.0
        assert results.cost == 750.0
        assert len(results.convergence) == 2
        assert results.convergence[1].cost == 750
    
    def test_round_trip(self, simple_results):
        """Test sérialisation/désérialisation complète."""
        reconstructed = Results.from_dict(simple_results.to_dict())
        
        assert reconstructed.time_seconds == simple_results.time_seconds
        assert reconstructed.cost == simple_results.cost
        assert len(reconstructed.convergence) == len(simple_results.convergence)


# ============================================================================
# TESTS: Config
# ============================================================================

class TestConfig:
    """Tests pour la classe Config."""
    
    def test_creation(self, simple_config):
        """Test création basique."""
        assert simple_config.instance_name == "A-n32-k5"
        assert simple_config.algo_name == "simulated_annealing"
        assert simple_config.seed == 42
        assert simple_config.parameters["T_init"] == 1000
    
    def test_compute_hash_run_deterministic(self, simple_config):
        """Test que le hash_run est déterministe."""
        hash1 = simple_config.compute_hash_run()
        hash2 = simple_config.compute_hash_run()
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 = 64 caractères hex
    
    def test_compute_hash_run_different_seeds(self):
        """Test que seeds différentes donnent des hash_run différents."""
        config1 = Config("A-n32-k5", "SA", 42, {"alpha": 0.95})
        config2 = Config("A-n32-k5", "SA", 99, {"alpha": 0.95})
        
        assert config1.compute_hash_run() != config2.compute_hash_run()
    
    def test_compute_hash_config_ignores_seed(self):
        """Test que hash_config ignore la seed."""
        config1 = Config("A-n32-k5", "SA", 42, {"alpha": 0.95})
        config2 = Config("A-n32-k5", "SA", 99, {"alpha": 0.95})
        
        assert config1.compute_hash_config() == config2.compute_hash_config()
    
    def test_compute_hash_config_different_params(self):
        """Test que paramètres différents donnent des hash_config différents."""
        config1 = Config("A-n32-k5", "SA", 42, {"alpha": 0.95})
        config2 = Config("A-n32-k5", "SA", 42, {"alpha": 0.99})
        
        assert config1.compute_hash_config() != config2.compute_hash_config()
    
    def test_normalize_parameters_floats(self):
        """Test normalisation des floats pour éviter problèmes de précision."""
        config1 = Config("A-n32-k5", "SA", 42, {"alpha": 0.95})
        config2 = Config("A-n32-k5", "SA", 42, {"alpha": 0.9500000000000001})
        
        # Les deux devraient avoir le même hash après normalisation
        assert config1.compute_hash_config() == config2.compute_hash_config()
    
    def test_normalize_parameters_nested(self, complex_config):
        """Test normalisation de paramètres imbriqués."""
        hash1 = complex_config.compute_hash_run()
        
        # Crée une config identique avec float légèrement différent
        params_copy = {
            "tabu_tenure": 10,
            "intensification": {
                "threshold": 0.8000000000001,  # Légère différence
                "iterations": 100
            },
            "diversification": {
                "probability": 0.1,
                "strength": 0.5
            }
        }
        config2 = Config(
            complex_config.instance_name,
            complex_config.algo_name,
            complex_config.seed,
            params_copy
        )
        
        # Devrait avoir le même hash après normalisation
        assert config2.compute_hash_run() == hash1
    
    def test_to_dict(self, simple_config):
        """Test sérialisation."""
        data = simple_config.to_dict()
        
        assert data['instance_name'] == "A-n32-k5"
        assert data['seed'] == 42
        assert 'parameters' in data
    
    def test_from_dict(self):
        """Test désérialisation."""
        data = {
            'instance_name': "B-n31-k5",
            'algo_name': "tabu_search",
            'seed': 123,
            'parameters': {"tenure": 5}
        }
        
        config = Config.from_dict(data)
        
        assert config.instance_name == "B-n31-k5"
        assert config.seed == 123
        assert config.parameters["tenure"] == 5
    
    def test_round_trip(self, complex_config):
        """Test sérialisation/désérialisation complète."""
        reconstructed = Config.from_dict(complex_config.to_dict())
        
        assert reconstructed.compute_hash_run() == complex_config.compute_hash_run()
        assert reconstructed.compute_hash_config() == complex_config.compute_hash_config()


# ============================================================================
# TESTS: Metadata
# ============================================================================

class TestMetadata:
    """Tests pour la classe Metadata."""
    
    def test_capture_system_info(self):
        """Test capture automatique des infos système."""
        metadata = Metadata.capture_system_info(user="test_user")
        
        assert metadata.user == "test_user"
        assert metadata.hardware_id is not None
        assert metadata.cpu_arch is not None
        assert metadata.python_version is not None
        assert "GB" in metadata.ram
        
        # Vérifie format ISO de la date
        datetime.fromisoformat(metadata.date)
    
    def test_capture_system_info_auto_user(self):
        """Test détection automatique de l'utilisateur."""
        metadata = Metadata.capture_system_info()
        
        assert metadata.user is not None
        assert len(metadata.user) > 0
    
    def test_to_dict(self):
        """Test sérialisation."""
        metadata = Metadata(
            user="alice",
            date="2025-11-04T14:30:00",
            hardware_id="machine1",
            cpu_arch="Intel i7",
            ram="16GB",
            python_version="3.11.2"
        )
        
        data = metadata.to_dict()
        
        assert data['user'] == "alice"
        assert data['cpu_arch'] == "Intel i7"
    
    def test_from_dict(self):
        """Test désérialisation."""
        data = {
            'user': "bob",
            'date': "2025-11-04T15:00:00",
            'hardware_id': "machine2",
            'cpu_arch': "AMD Ryzen",
            'ram': "32GB",
            'python_version': "3.10.5"
        }
        
        metadata = Metadata.from_dict(data)
        
        assert metadata.user == "bob"
        assert metadata.ram == "32GB"
    
    def test_round_trip(self):
        """Test sérialisation/désérialisation."""
        original = Metadata.capture_system_info(user="charlie")
        reconstructed = Metadata.from_dict(original.to_dict())
        
        assert reconstructed.user == original.user
        assert reconstructed.hardware_id == original.hardware_id


# ============================================================================
# TESTS: Run
# ============================================================================

class TestRun:
    """Tests pour la classe Run."""
    
    def test_creation(self, simple_config, simple_results):
        """Test création d'un run complet."""
        metadata = Metadata.capture_system_info(user="test")
        hash_run = simple_config.compute_hash_run()
        hash_config = simple_config.compute_hash_config()
        
        run = Run(
            metadata=metadata,
            hash_run=hash_run,
            hash_config=hash_config,
            config=simple_config,
            results=simple_results
        )
        
        assert run.hash_run == hash_run
        assert run.hash_config == hash_config
        assert run.config.seed == 42
        assert run.results.cost == 812.4
    
    def test_save_and_load(self, perm_dir, simple_config, simple_results):
        """Test sauvegarde et chargement depuis fichier."""
        metadata = Metadata.capture_system_info(user="test")
        run = Run(
            metadata=metadata,
            hash_run=simple_config.compute_hash_run(),
            hash_config=simple_config.compute_hash_config(),
            config=simple_config,
            results=simple_results
        )
        
        filepath = perm_dir / "test_run.json"
        run.save(filepath)
        
        assert filepath.exists()
        
        loaded_run = Run.load(filepath)
        
        assert loaded_run.hash_run == run.hash_run
        assert loaded_run.config.seed == run.config.seed
        assert loaded_run.results.cost == run.results.cost
    
    def test_save_creates_directory(self, perm_dir, simple_config, simple_results):
        """Test que save crée les répertoires manquants."""
        metadata = Metadata.capture_system_info(user="test")
        run = Run(
            metadata=metadata,
            hash_run=simple_config.compute_hash_run(),
            hash_config=simple_config.compute_hash_config(),
            config=simple_config,
            results=simple_results
        )
        
        nested_path = perm_dir / "nested" / "path" / "run.json"
        run.save(nested_path)
        
        assert nested_path.exists()
    
    def test_load_nonexistent_file(self, perm_dir):
        """Test chargement d'un fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            Run.load(perm_dir / "nonexistent.json")
    
    def test_load_invalid_json(self, perm_dir):
        """Test chargement d'un fichier JSON invalide."""
        bad_file = perm_dir / "bad.json"
        bad_file.write_text("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            Run.load(bad_file)
    
    def test_to_dict_complete(self, simple_config, simple_results):
        """Test sérialisation complète."""
        metadata = Metadata.capture_system_info(user="test")
        run = Run(
            metadata=metadata,
            hash_run=simple_config.compute_hash_run(),
            hash_config=simple_config.compute_hash_config(),
            config=simple_config,
            results=simple_results
        )
        
        data = run.to_dict()
        
        assert 'metadata' in data
        assert 'hash_run' in data
        assert 'hash_config' in data
        assert 'config' in data
        assert 'results' in data
        
        assert data['config']['seed'] == 42
        assert data['results']['cost'] == 812.4
    
    def test_from_dict_complete(self):
        """Test désérialisation complète."""
        data = {
            'metadata': {
                'user': "test",
                'date': "2025-11-04T14:00:00",
                'hardware_id': "machine",
                'cpu_arch': "Intel",
                'ram': "16GB",
                'python_version': "3.11"
            },
            'hash_run': "abc123",
            'hash_config': "def456",
            'config': {
                'instance_name': "A-n32-k5",
                'algo_name': "SA",
                'seed': 42,
                'parameters': {}
            },
            'results': {
                'time_seconds': 10.0,
                'n_iterations': 100,
                'cost': 800.0,
                'solution': [[1, 2]],
                'convergence': []
            }
        }
        
        run = Run.from_dict(data)
        
        assert run.hash_run == "abc123"
        assert run.config.seed == 42
        assert run.results.cost == 800.0


# ============================================================================
# TESTS: RunFileManager
# ============================================================================

class TestRunFileManager:
    """Tests pour la classe RunFileManager."""
    
    def test_init_creates_directory(self, perm_dir):
        """Test que l'init crée le répertoire de résultats."""
        manager = RunFileManager(str(perm_dir / "new_results"))
        
        print(manager.results_dir)
        assert manager.results_dir.exists()
        assert manager.index_path.exists()
    
    def test_init_loads_existing_index(self, perm_dir):
        """Test que l'init charge un index existant."""
        results_dir = perm_dir / "results"
        results_dir.mkdir()
        
        # Crée un index pré-existant
        index_data = {
            "hash_to_file": {"abc": "file1.json"},
            "config_groups": {"def": ["abc"]}
        }
        index_path = results_dir / "index.json"
        with open(index_path, 'w') as f:
            json.dump(index_data, f)
        
        manager = RunFileManager(str(results_dir))
        
        assert "abc" in manager.index["hash_to_file"]
        assert manager.index["hash_to_file"]["abc"] == "file1.json"
    
    def test_run_exists_false(self, manager, simple_config):
        """Test run_exists retourne False pour run inexistant."""
        assert not manager.run_exists(simple_config)
    
    def test_add_run_creates_file(self, manager, simple_config, simple_results):
        """Test que add_run crée bien le fichier."""
        filepath = manager.add_run(simple_config, simple_results)
        
        assert filepath.exists()
        assert filepath.suffix == ".json"
    
    def test_add_run_updates_index(self, manager, simple_config, simple_results):
        """Test que add_run met à jour l'index correctement."""
        manager.add_run(simple_config, simple_results)
        
        hash_run = simple_config.compute_hash_run()
        hash_config = simple_config.compute_hash_config()
        
        assert hash_run in manager.index["hash_to_file"]
        assert hash_config in manager.index["config_groups"]
        assert hash_run in manager.index["config_groups"][hash_config]
    
    def test_add_run_filename_format(self, manager, simple_config, simple_results):
        """Test le format du nom de fichier généré."""
        filepath = manager.add_run(simple_config, simple_results)
        
        filename = filepath.name
        parts = filename.replace(".json", "").split("_")
        
        # Format: YYYYMMDD_HHMMSS_XXXXXXXX.json
        assert len(parts) == 3
        assert len(parts[0]) == 8  # Date YYYYMMDD
        assert len(parts[1]) == 6  # Heure HHMMSS
        assert len(parts[2]) == 8  # Hash court
    
    def test_run_exists_true_after_add(self, manager, simple_config, simple_results):
        """Test run_exists retourne True après ajout."""
        manager.add_run(simple_config, simple_results)
        
        assert manager.run_exists(simple_config)
    
    def test_add_run_twice_skips_second(self, manager, simple_config, simple_results):
        """Test qu'ajouter deux fois le même run skip le second."""
        filepath1 = manager.add_run(simple_config, simple_results)
        filepath2 = manager.add_run(simple_config, simple_results)
        
        assert filepath1 == filepath2
        
        # Vérifie qu'il n'y a qu'un seul fichier
        json_files = list(manager.results_dir.glob("*.json"))
        # -1 pour l'index.json
        assert len([f for f in json_files if f.name != "index.json"]) == 1
    
    def test_load_run_success(self, manager, simple_config, simple_results):
        """Test chargement d'un run existant."""
        manager.add_run(simple_config, simple_results)
        
        loaded_run = manager.load_run(simple_config)
        
        assert loaded_run.config.seed == simple_config.seed
        assert loaded_run.results.cost == simple_results.cost
    
    def test_load_run_nonexistent(self, manager, simple_config):
        """Test chargement d'un run inexistant."""
        with pytest.raises(FileNotFoundError):
            manager.load_run(simple_config)
    
    def test_load_config_runs_multiple_seeds(self, manager, simple_results):
        """Test chargement de plusieurs runs avec seeds différentes."""
        base_params = {"T_init": 1000, "alpha": 0.95}
        
        # Ajoute 3 runs avec même config, seeds différentes
        for seed in [42, 99, 123]:
            config = Config("A-n32-k5", "SA", seed, base_params)
            manager.add_run(config, simple_results)
        
        # Charge tous les runs de cette config
        config = Config("A-n32-k5", "SA", 42, base_params)
        runs = manager.load_config_runs(config)
        
        assert len(runs) == 3
        seeds = {run.config.seed for run in runs}
        assert seeds == {42, 99, 123}
    
    def test_load_config_runs_empty(self, manager, simple_config):
        """Test load_config_runs retourne liste vide si aucun run."""
        runs = manager.load_config_runs(simple_config)
        
        assert runs == []
    
    def test_delete_run_removes_file(self, manager, simple_config, simple_results):
        """Test que delete_run supprime le fichier."""
        filepath = manager.add_run(simple_config, simple_results)
        hash_run = simple_config.compute_hash_run()
        
        manager.delete_run(hash_run)
        
        assert not filepath.exists()
    
    def test_delete_run_updates_index(self, manager, simple_config, simple_results):
        """Test que delete_run met à jour l'index."""
        manager.add_run(simple_config, simple_results)
        hash_run = simple_config.compute_hash_run()
        hash_config = simple_config.compute_hash_config()
        
        manager.delete_run(hash_run)
        
        assert hash_run not in manager.index["hash_to_file"]
        assert hash_config not in manager.index["config_groups"]
    
    def test_delete_run_keeps_other_runs(self, manager, simple_results):
        """Test que delete_run ne supprime que le run ciblé."""
        config1 = Config("A-n32-k5", "SA", 42, {"alpha": 0.95})
        config2 = Config("A-n32-k5", "SA", 99, {"alpha": 0.95})
        
        manager.add_run(config1, simple_results)
        manager.add_run(config2, simple_results)
        
        hash_run1 = config1.compute_hash_run()
        manager.delete_run(hash_run1)
        
        # config2 doit toujours exister
        assert manager.run_exists(config2)
        assert not manager.run_exists(config1)
    
    def test_delete_run_nonexistent(self, manager):
        """Test suppression d'un run inexistant (ne doit pas crasher)."""
        manager.delete_run("nonexistent_hash")
        # Ne doit pas lever d'exception
    
    def test_rebuild_index_from_files(self, manager, simple_config, simple_results):
        """Test reconstruction de l'index depuis les fichiers."""
        # Ajoute un run
        manager.add_run(simple_config, simple_results)
        
        # Corrompt l'index
        manager.index = {"hash_to_file": {}, "config_groups": {}}
        
        # Rebuild
        manager.rebuild_index()
        
        # Vérifie que le run est de nouveau dans l'index
        hash_run = simple_config.compute_hash_run()
        assert hash_run in manager.index["hash_to_file"]
    
    def test_rebuild_index_multiple_files(self, manager, simple_results):
        """Test rebuild avec plusieurs fichiers."""
        configs = [
            Config("A-n32-k5", "SA", 42, {"alpha": 0.95}),
            Config("A-n32-k5", "SA", 99, {"alpha": 0.95}),
            Config("B-n31-k5", "Tabu", 42, {"tenure": 10})
        ]
        
        for config in configs:
            manager.add_run(config, simple_results)
        
        # Corrompt l'index
        manager.index = {"hash_to_file": {}, "config_groups": {}}
        
        # Rebuild
        manager.rebuild_index()
        
        # Vérifie que tous les runs sont restaurés
        for config in configs:
            assert manager.run_exists(config)
    
    def test_rebuild_index_counts(self, manager, simple_results, capsys):
        """Test que rebuild affiche les bons compteurs."""
        config1 = Config("A-n32-k5", "SA", 42, {"alpha": 0.95})
        config2 = Config("A-n32-k5", "SA", 99, {"alpha": 0.95})
        
        manager.add_run(config1, simple_results)
        manager.add_run(config2, simple_results)
        
        manager.index = {"hash_to_file": {}, "config_groups": {}}
        manager.rebuild_index()
        
        captured = capsys.readouterr()
        assert "1 configs" in captured.out
        assert "2 runs" in captured.out
    
    def test_concurrent_access_same_config(self, manager, simple_config, simple_results):
        """Test que l'ajout simultané de la même config est idempotent."""
        # Simule deux ajouts rapides de la même config
        filepath1 = manager.add_run(simple_config, simple_results)
        filepath2 = manager.add_run(simple_config, simple_results)
        
        assert filepath1 == filepath2
        
        # Vérifie qu'il n'y a qu'une seule entrée dans l'index
        hash_run = simple_config.compute_hash_run()
        hash_config = simple_config.compute_hash_config()
        
        assert manager.index["config_groups"][hash_config].count(hash_run) == 1
    
    def test_index_persistence(self, perm_dir, simple_config, simple_results):
        """Test que l'index persiste entre deux instances du manager."""
        # Premier manager: ajoute un run
        manager1 = RunFileManager(str(perm_dir / "results"))
        manager1.add_run(simple_config, simple_results)
        
        # Deuxième manager: devrait charger l'index existant
        manager2 = RunFileManager(str(perm_dir / "results"))
        
        assert manager2.run_exists(simple_config)
        
        # Vérifie que les index sont identiques
        assert manager1.index == manager2.index


# ============================================================================
# TESTS D'INTÉGRATION
# ============================================================================

class TestIntegration:
    """Tests d'intégration simulant des workflows réels."""
    
    def test_complete_workflow(self, manager):
        """Test un workflow complet: ajout, chargement, analyse."""
        # Crée plusieurs configs
        configs = [
            Config("A-n32-k5", "SA", seed, {"T_init": 1000, "alpha": 0.95})
            for seed in [42, 99, 123, 456, 789]
        ]
        
        # Simule des résultats avec coûts variables
        costs = [812.4, 808.1, 815.2, 810.5, 807.9]
        
        for config, cost in zip(configs, costs):
            results = Results(
                time_seconds=45.0,
                n_iterations=1000,
                cost=cost,
                solution=[[1, 2], [3, 4]],
                convergence=[ConvergencePoint(0, 1000), ConvergencePoint(500, cost)]
            )
            manager.add_run(config, results)
        
        # Charge tous les runs de cette config
        runs = manager.load_config_runs(configs[0])
        
        assert len(runs) == 5
        
        # Analyse statistique
        run_costs = [run.results.cost for run in runs]
        mean_cost = sum(run_costs) / len(run_costs)
        
        assert 807 < mean_cost < 816
    
    def test_multi_algo_comparison(self, manager):
        """Test comparaison de plusieurs algos sur même instance."""
        instance = "A-n32-k5"
        algos = ["SA", "Tabu", "ALNS"]
        
        for algo in algos:
            for seed in [42, 99, 123]:
                config = Config(instance, algo, seed, {"param": 1.0})
                results = Results(
                    time_seconds=30.0,
                    n_iterations=500,
                    cost=800.0 + (algos.index(algo) * 10),  # Coûts différents par algo
                    solution=[[1, 2]],
                    convergence=[]
                )
                manager.add_run(config, results)
        
        # Vérifie qu'on peut charger les runs de chaque algo
        for algo in algos:
            config = Config(instance, algo, 42, {"param": 1.0})
            runs = manager.load_config_runs(config)
            assert len(runs) == 3
    
    def test_large_scale_simulation(self, manager):
        """Test avec un grand nombre de runs."""
        n_instances = 5
        n_algos = 3
        n_seeds = 10
        
        for i in range(n_instances):
            for j in range(n_algos):
                for seed in range(n_seeds):
                    config = Config(
                        f"instance_{i}",
                        f"algo_{j}",
                        seed,
                        {"param": 1.0}
                    )
                    results = Results(
                        time_seconds=10.0,
                        n_iterations=100,
                        cost=800.0,
                        solution=[[1]],
                        convergence=[]
                    )
                    manager.add_run(config, results)
        
        # Vérifie les compteurs
        total_runs = n_instances * n_algos * n_seeds
        assert len(manager.index["hash_to_file"]) == total_runs
        
        # Vérifie qu'on peut charger n'importe quel run
        test_config = Config("instance_2", "algo_1", 5, {"param": 1.0})
        assert manager.run_exists(test_config)


# ============================================================================
# TESTS DE PERFORMANCE (optionnels)
# ============================================================================

class TestPerformance:
    """Tests de performance (peuvent être skip si trop lents)."""
    
    def test_rebuild_index_performance(self, manager):
        """Test performance du rebuild avec beaucoup de fichiers."""
        
        # Crée 100 runs
        for i in range(100):
            config = Config(f"instance_{i % 10}", "SA", i, {})
            results = Results(10.0, 100, 800.0, [[1]], [])
            manager.add_run(config, results)
        
        # Mesure le temps de rebuild
        manager.index = {"hash_to_file": {}, "config_groups": {}}
        
        start = time.time()
        manager.rebuild_index()
        duration = time.time() - start
        
        # Devrait prendre moins de 1 seconde pour 100 fichiers
        assert duration < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
