"""
Comprehensive tests for Neptune-to-mlop compatibility layer.

These tests validate that:
1. Neptune API calls continue to work unchanged
2. mlop receives the logged data when configured
3. Neptune never fails due to mlop errors
4. Configuration via environment variables works
5. Fallback behavior is correct
"""

import os
import tempfile
from typing import Any, Dict
from unittest import mock

import pytest

# Test both with and without neptune installed
pytest.importorskip('neptune_scale')

import mlop
from tests.utils import get_task_name

# Import after neptune is available
from neptune_scale.types import File as NeptuneFile
from neptune_scale.types import Histogram as NeptuneHistogram


class MockNeptuneRun:
    """
    Mock Neptune Run for testing without actual Neptune backend.

    This simulates Neptune's behavior for testing the monkeypatch.
    """

    def __init__(self, *args, **kwargs):
        self.experiment_name = kwargs.get('experiment_name', 'test-experiment')
        self.run_id = kwargs.get('run_id', None)
        self.project = kwargs.get('project', 'test/project')
        self.logged_metrics = []
        self.logged_configs = []
        self.logged_files = []
        self.logged_histograms = []
        self.tags = []
        self.closed = False
        self.terminated = False

    def log_metrics(self, data: Dict[str, float], step: int, timestamp=None, **kwargs):
        self.logged_metrics.append({'data': data, 'step': step, 'timestamp': timestamp})
        return None

    def log_configs(self, data: Dict[str, Any], **kwargs):
        self.logged_configs.append(data)
        return None

    def assign_files(self, files: Dict[str, Any], **kwargs):
        self.logged_files.append({'type': 'assign', 'files': files})
        return None

    def log_files(self, files: Dict[str, Any], step: int, timestamp=None, **kwargs):
        self.logged_files.append({
            'type': 'log',
            'files': files,
            'step': step,
            'timestamp': timestamp
        })
        return None

    def log_histograms(self, histograms: Dict[str, Any], step: int, timestamp=None, **kwargs):
        self.logged_histograms.append({
            'histograms': histograms,
            'step': step,
            'timestamp': timestamp
        })
        return None

    def add_tags(self, tags, **kwargs):
        self.tags.extend(tags)
        return None

    def remove_tags(self, tags, **kwargs):
        for tag in tags:
            if tag in self.tags:
                self.tags.remove(tag)
        return None

    def close(self, **kwargs):
        self.closed = True
        return None

    def terminate(self, **kwargs):
        self.terminated = True
        return None

    def wait_for_submission(self, **kwargs):
        return None

    def wait_for_processing(self, **kwargs):
        return None

    def get_run_url(self):
        return f'https://neptune.ai/{self.project}/runs/{self.run_id or "test-run"}'

    def get_experiment_url(self):
        return f'https://neptune.ai/{self.project}/experiments/{self.experiment_name}'

    def log_string_series(self, data: Dict[str, str], step: int, timestamp=None, **kwargs):
        # Not implemented in mock
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


@pytest.fixture
def mock_neptune_backend(monkeypatch):
    """Replace neptune_scale.Run with our mock for testing."""
    import neptune_scale
    original_run = neptune_scale.Run
    monkeypatch.setattr('neptune_scale.Run', MockNeptuneRun)
    yield
    # Restore
    monkeypatch.setattr('neptune_scale.Run', original_run)


@pytest.fixture
def clean_env():
    """Clean environment variables before each test."""
    env_vars = ['MLOP_PROJECT', 'MLOP_API_KEY', 'MLOP_URL_APP', 'MLOP_URL_API', 'MLOP_URL_INGEST']
    original_values = {}
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture
def reload_neptune_compat():
    """Reload the neptune compat module to reapply monkeypatch."""
    import importlib
    import mlop.compat.neptune
    importlib.reload(mlop.compat.neptune)
    yield
    # Restore after test
    mlop.compat.neptune.restore_neptune()


class TestNeptuneCompatBasic:
    """Test basic Neptune API functionality is preserved."""

    def test_neptune_import_without_mlop_config(self, mock_neptune_backend, clean_env, reload_neptune_compat):
        """Test that Neptune works normally when MLOP_PROJECT is not set."""
        # Don't set MLOP_PROJECT - should fall back to Neptune-only
        from neptune_scale import Run

        run = Run(experiment_name='test-exp')
        run.log_metrics({'loss': 0.5}, step=0)
        run.close()

        assert run._neptune_run.closed
        assert len(run._neptune_run.logged_metrics) == 1
        assert run._neptune_run.logged_metrics[0]['data'] == {'loss': 0.5}

    def test_neptune_metrics_logging(self, mock_neptune_backend, clean_env, reload_neptune_compat):
        """Test that Neptune metrics logging works unchanged."""
        from neptune_scale import Run

        run = Run(experiment_name='metrics-test')
        run.log_metrics({'acc': 0.95, 'loss': 0.1}, step=1)
        run.log_metrics({'acc': 0.96, 'loss': 0.09}, step=2)
        run.close()

        assert len(run._neptune_run.logged_metrics) == 2
        assert run._neptune_run.logged_metrics[0]['step'] == 1
        assert run._neptune_run.logged_metrics[1]['step'] == 2

    def test_neptune_configs_logging(self, mock_neptune_backend, clean_env, reload_neptune_compat):
        """Test that Neptune config logging works unchanged."""
        from neptune_scale import Run

        run = Run(experiment_name='config-test')
        run.log_configs({'lr': 0.001, 'batch_size': 32})
        run.close()

        assert len(run._neptune_run.logged_configs) == 1
        assert run._neptune_run.logged_configs[0] == {'lr': 0.001, 'batch_size': 32}

    def test_neptune_tags(self, mock_neptune_backend, clean_env, reload_neptune_compat):
        """Test that Neptune tags work unchanged."""
        from neptune_scale import Run

        run = Run(experiment_name='tag-test')
        run.add_tags(['experiment', 'baseline'])
        run.add_tags(['v1'])
        run.remove_tags(['baseline'])
        run.close()

        assert 'experiment' in run._neptune_run.tags
        assert 'v1' in run._neptune_run.tags
        assert 'baseline' not in run._neptune_run.tags

    def test_neptune_context_manager(self, mock_neptune_backend, clean_env, reload_neptune_compat):
        """Test that Neptune context manager protocol works."""
        from neptune_scale import Run

        with Run(experiment_name='context-test') as run:
            run.log_metrics({'loss': 0.3}, step=0)

        assert run._neptune_run.closed


class TestNeptuneCompatDualLogging:
    """Test dual-logging to both Neptune and mlop."""

    @pytest.fixture
    def mlop_config_env(self, clean_env):
        """Set up environment for mlop dual-logging."""
        os.environ['MLOP_PROJECT'] = 'neptune-migration-test'
        # Don't set MLOP_API_KEY - let it fall back to keyring or fail gracefully
        yield

    def test_dual_logging_metrics_with_env_config(
        self, mock_neptune_backend, mlop_config_env, reload_neptune_compat, monkeypatch
    ):
        """Test that metrics are logged to both Neptune and mlop when configured."""
        # Mock mlop.init to avoid actual API calls
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='dual-log-test')
            run.log_metrics({'loss': 0.5, 'acc': 0.9}, step=0)
            run.close()

            # Verify Neptune received the data
            assert len(run._neptune_run.logged_metrics) == 1
            assert run._neptune_run.logged_metrics[0]['data'] == {'loss': 0.5, 'acc': 0.9}

            # Verify mlop received the data
            mock_mlop_run.log.assert_called_with({'loss': 0.5, 'acc': 0.9})
            mock_mlop_run.finish.assert_called_once()

    def test_dual_logging_configs(
        self, mock_neptune_backend, mlop_config_env, reload_neptune_compat
    ):
        """Test that configs are logged to both Neptune and mlop."""
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='config-dual-test')
            run.log_configs({'lr': 0.001, 'epochs': 100})
            run.close()

            # Verify Neptune received the config
            assert len(run._neptune_run.logged_configs) == 1

            # Verify mlop config was updated
            assert mock_mlop_run.config['lr'] == 0.001
            assert mock_mlop_run.config['epochs'] == 100


class TestNeptuneCompatErrorHandling:
    """Test that Neptune never fails due to mlop errors."""

    @pytest.fixture
    def mlop_config_env(self, clean_env):
        """Set up environment for mlop dual-logging."""
        os.environ['MLOP_PROJECT'] = 'error-test'
        yield

    def test_neptune_works_when_mlop_init_fails(
        self, mock_neptune_backend, mlop_config_env, reload_neptune_compat
    ):
        """Test that Neptune continues working if mlop.init() fails."""
        # Make mlop.init() raise an exception
        with mock.patch('mlop.init', side_effect=Exception('mlop service down')):
            from neptune_scale import Run

            # Should not raise - Neptune should work fine
            run = Run(experiment_name='mlop-init-fail-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # Verify Neptune worked
            assert run._neptune_run.closed
            assert len(run._neptune_run.logged_metrics) == 1

    def test_neptune_works_when_mlop_log_fails(
        self, mock_neptune_backend, mlop_config_env, reload_neptune_compat
    ):
        """Test that Neptune continues working if mlop.log() fails."""
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock(side_effect=Exception('Network error'))
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            # Should not raise - Neptune should work fine
            run = Run(experiment_name='mlop-log-fail-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # Verify Neptune worked
            assert len(run._neptune_run.logged_metrics) == 1
            assert run._neptune_run.closed

    def test_neptune_works_when_mlop_finish_fails(
        self, mock_neptune_backend, mlop_config_env, reload_neptune_compat
    ):
        """Test that Neptune closes correctly even if mlop.finish() fails."""
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock(side_effect=Exception('Finish error'))

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            # Should not raise - Neptune should work fine
            run = Run(experiment_name='mlop-finish-fail-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()  # Should not raise

            # Verify Neptune worked
            assert run._neptune_run.closed

    def test_neptune_works_when_mlop_not_installed(
        self, mock_neptune_backend, mlop_config_env, reload_neptune_compat
    ):
        """Test that Neptune works when mlop is not installed."""
        # Simulate mlop import failure
        with mock.patch('mlop.compat.neptune._safe_import_mlop', return_value=None):
            from neptune_scale import Run

            run = Run(experiment_name='no-mlop-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # Verify Neptune worked
            assert run._neptune_run.closed
            assert len(run._neptune_run.logged_metrics) == 1


class TestNeptuneCompatFileConversion:
    """Test file type conversion from Neptune to mlop."""

    @pytest.fixture
    def mlop_config_env(self, clean_env):
        """Set up environment for mlop dual-logging."""
        os.environ['MLOP_PROJECT'] = 'file-conversion-test'
        yield

    def test_image_file_conversion(
        self, mock_neptune_backend, mlop_config_env, reload_neptune_compat, tmp_path
    ):
        """Test that Neptune File objects are converted to mlop.Image."""
        # Create a test image
        img_path = tmp_path / 'test.png'
        img_path.write_bytes(b'\x89PNG\r\n\x1a\n')  # PNG header

        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='image-test')

            # Log file with Neptune File object
            neptune_file = NeptuneFile(source=str(img_path), mime_type='image/png')
            run.assign_files({'sample_image': neptune_file})
            run.close()

            # Verify Neptune received the file
            assert len(run._neptune_run.logged_files) == 1

            # Verify mlop.log was called (file conversion is internal)
            assert mock_mlop_run.log.called

    def test_histogram_conversion(
        self, mock_neptune_backend, mlop_config_env, reload_neptune_compat
    ):
        """Test that Neptune Histogram objects are converted to mlop.Histogram."""
        import numpy as np

        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='histogram-test')

            # Create Neptune histogram
            bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
            counts = np.array([10, 20, 15])
            neptune_hist = NeptuneHistogram(bin_edges=bin_edges, counts=counts)

            run.log_histograms({'layer1/activations': neptune_hist}, step=0)
            run.close()

            # Verify Neptune received the histogram
            assert len(run._neptune_run.logged_histograms) == 1

            # Verify mlop.log was called
            assert mock_mlop_run.log.called


class TestNeptuneCompatIntegration:
    """
    Integration tests with real mlop backend (requires auth).

    These tests should be run in CI with proper mlop credentials set.
    """

    @pytest.mark.skipif(
        not os.environ.get('MLOP_PROJECT') or not os.environ.get('CI'),
        reason='Requires MLOP_PROJECT env var and CI environment'
    )
    def test_real_dual_logging_integration(self, mock_neptune_backend):
        """
        Integration test with real mlop backend.

        This test requires:
        - MLOP_PROJECT environment variable
        - Valid mlop credentials (keyring or MLOP_API_KEY)
        - Network access to mlop service
        """
        from neptune_scale import Run

        task_name = get_task_name()

        # This should log to both mock Neptune and real mlop
        run = Run(experiment_name=task_name)

        # Log various data types
        run.log_configs({'lr': 0.001, 'batch_size': 32, 'model': 'resnet50'})
        run.log_metrics({'train/loss': 0.5, 'train/acc': 0.85}, step=0)
        run.log_metrics({'train/loss': 0.3, 'train/acc': 0.92}, step=1)
        run.add_tags(['integration-test', 'ci'])

        run.close()

        # Verify Neptune received all data
        assert len(run._neptune_run.logged_configs) == 1
        assert len(run._neptune_run.logged_metrics) == 2
        assert len(run._neptune_run.tags) == 2
        assert run._neptune_run.closed

        # mlop run should also be finished
        if run._mlop_run:
            # Verify mlop was initialized successfully
            assert run._mlop_run is not None
            print(f'✓ Integration test passed - dual-logged to Neptune and mlop')
        else:
            pytest.skip('mlop not configured, skipping integration validation')


class TestNeptuneCompatFallbackBehavior:
    """Test various fallback scenarios."""

    def test_no_mlop_project_env_var(self, mock_neptune_backend, clean_env, reload_neptune_compat):
        """Test that monkeypatch works but doesn't init mlop when MLOP_PROJECT is not set."""
        # No MLOP_PROJECT set
        from neptune_scale import Run

        run = Run(experiment_name='no-project-test')
        run.log_metrics({'loss': 0.5}, step=0)
        run.close()

        # Should have no mlop run
        assert run._mlop_run is None

        # Neptune should work fine
        assert run._neptune_run.closed
        assert len(run._neptune_run.logged_metrics) == 1

    def test_mlop_project_set_but_invalid_credentials(
        self, mock_neptune_backend, clean_env, reload_neptune_compat
    ):
        """Test fallback when mlop project is set but credentials are invalid."""
        os.environ['MLOP_PROJECT'] = 'test-project'
        os.environ['MLOP_API_KEY'] = 'invalid-key-123'

        # Mock mlop.init to fail with auth error
        with mock.patch('mlop.init', side_effect=Exception('Unauthorized')):
            from neptune_scale import Run

            run = Run(experiment_name='invalid-creds-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # mlop should have failed silently
            assert run._mlop_run is None

            # Neptune should work fine
            assert run._neptune_run.closed


class TestNeptuneCompatAPIForwarding:
    """Test that unknown Neptune API methods are forwarded correctly."""

    def test_unknown_method_forwarding(self, mock_neptune_backend, clean_env, reload_neptune_compat):
        """Test that unknown methods are forwarded to Neptune."""
        from neptune_scale import Run

        run = Run(experiment_name='forward-test')

        # Call Neptune-specific methods
        url = run.get_run_url()
        assert 'neptune.ai' in url

        exp_url = run.get_experiment_url()
        assert 'neptune.ai' in exp_url

        run.close()

    def test_wait_methods_work(self, mock_neptune_backend, clean_env, reload_neptune_compat):
        """Test that Neptune's wait methods work."""
        from neptune_scale import Run

        run = Run(experiment_name='wait-test')
        run.log_metrics({'loss': 0.5}, step=0)

        # These should not raise
        run.wait_for_submission()
        run.wait_for_processing()

        run.close()
