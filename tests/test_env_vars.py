import logging
import os

from mlop.sets import setup


class TestMLOPDebugLevel:
    def test_string_values(self):
        """Test DEBUG, INFO, WARNING, ERROR, CRITICAL"""
        test_cases = [
            ('DEBUG', 10),
            ('INFO', 20),
            ('WARNING', 30),
            ('ERROR', 40),
            ('CRITICAL', 50),
        ]
        for env_val, expected_level in test_cases:
            os.environ['MLOP_DEBUG_LEVEL'] = env_val
            settings = setup()
            assert settings.x_log_level == expected_level
            del os.environ['MLOP_DEBUG_LEVEL']

    def test_case_insensitive(self):
        """Test lowercase and mixed case"""
        os.environ['MLOP_DEBUG_LEVEL'] = 'debug'
        settings = setup()
        assert settings.x_log_level == 10
        del os.environ['MLOP_DEBUG_LEVEL']

        os.environ['MLOP_DEBUG_LEVEL'] = 'Info'
        settings = setup()
        assert settings.x_log_level == 20
        del os.environ['MLOP_DEBUG_LEVEL']

    def test_numeric_values(self):
        """Test numeric strings"""
        os.environ['MLOP_DEBUG_LEVEL'] = '15'
        settings = setup()
        assert settings.x_log_level == 15
        del os.environ['MLOP_DEBUG_LEVEL']

        os.environ['MLOP_DEBUG_LEVEL'] = '25'
        settings = setup()
        assert settings.x_log_level == 25
        del os.environ['MLOP_DEBUG_LEVEL']

    def test_precedence(self):
        """Test that function params override env var"""
        os.environ['MLOP_DEBUG_LEVEL'] = 'DEBUG'
        settings = setup({'x_log_level': 30})
        assert settings.x_log_level == 30  # Dict wins
        del os.environ['MLOP_DEBUG_LEVEL']

    def test_default_when_not_set(self):
        """Test default value when env var not set"""
        # Make sure env var is not set
        if 'MLOP_DEBUG_LEVEL' in os.environ:
            del os.environ['MLOP_DEBUG_LEVEL']

        settings = setup()
        assert settings.x_log_level == 16  # Default value

    def test_invalid_value_warning(self, caplog):
        """Test warning on invalid value"""
        os.environ['MLOP_DEBUG_LEVEL'] = 'INVALID'
        with caplog.at_level(logging.WARNING):
            settings = setup()
            assert 'invalid MLOP_DEBUG_LEVEL' in caplog.text
            assert settings.x_log_level == 16  # Falls back to default
        del os.environ['MLOP_DEBUG_LEVEL']

    def test_empty_string(self):
        """Test empty string uses default"""
        os.environ['MLOP_DEBUG_LEVEL'] = ''
        settings = setup()
        # Empty string is falsy, so env var will be None and default is used
        assert settings.x_log_level == 16
        del os.environ['MLOP_DEBUG_LEVEL']


class TestMLOPURLEnvironmentVariables:
    def test_url_app(self):
        """Test MLOP_URL_APP environment variable"""
        os.environ['MLOP_URL_APP'] = 'https://custom-app.example.com'
        settings = setup()
        assert settings.url_app == 'https://custom-app.example.com'
        del os.environ['MLOP_URL_APP']

    def test_url_api(self):
        """Test MLOP_URL_API environment variable"""
        os.environ['MLOP_URL_API'] = 'https://custom-api.example.com'
        settings = setup()
        assert settings.url_api == 'https://custom-api.example.com'
        del os.environ['MLOP_URL_API']

    def test_url_ingest(self):
        """Test MLOP_URL_INGEST environment variable"""
        os.environ['MLOP_URL_INGEST'] = 'https://custom-ingest.example.com'
        settings = setup()
        assert settings.url_ingest == 'https://custom-ingest.example.com'
        del os.environ['MLOP_URL_INGEST']

    def test_url_py(self):
        """Test MLOP_URL_PY environment variable"""
        os.environ['MLOP_URL_PY'] = 'https://custom-py.example.com'
        settings = setup()
        assert settings.url_py == 'https://custom-py.example.com'
        del os.environ['MLOP_URL_PY']

    def test_all_urls(self):
        """Test all URL environment variables together"""
        os.environ['MLOP_URL_APP'] = 'https://app.example.com'
        os.environ['MLOP_URL_API'] = 'https://api.example.com'
        os.environ['MLOP_URL_INGEST'] = 'https://ingest.example.com'
        os.environ['MLOP_URL_PY'] = 'https://py.example.com'
        settings = setup()
        assert settings.url_app == 'https://app.example.com'
        assert settings.url_api == 'https://api.example.com'
        assert settings.url_ingest == 'https://ingest.example.com'
        assert settings.url_py == 'https://py.example.com'
        del os.environ['MLOP_URL_APP']
        del os.environ['MLOP_URL_API']
        del os.environ['MLOP_URL_INGEST']
        del os.environ['MLOP_URL_PY']

    def test_url_precedence(self):
        """Test that function params override env vars"""
        os.environ['MLOP_URL_APP'] = 'https://env-app.example.com'
        os.environ['MLOP_URL_API'] = 'https://env-api.example.com'
        os.environ['MLOP_URL_INGEST'] = 'https://env-ingest.example.com'
        os.environ['MLOP_URL_PY'] = 'https://env-py.example.com'
        settings = setup(
            {
                'url_app': 'https://param-app.example.com',
                'url_api': 'https://param-api.example.com',
                'url_ingest': 'https://param-ingest.example.com',
                'url_py': 'https://param-py.example.com',
            }
        )
        assert settings.url_app == 'https://param-app.example.com'  # Dict wins
        assert settings.url_api == 'https://param-api.example.com'
        assert settings.url_ingest == 'https://param-ingest.example.com'
        assert settings.url_py == 'https://param-py.example.com'
        del os.environ['MLOP_URL_APP']
        del os.environ['MLOP_URL_API']
        del os.environ['MLOP_URL_INGEST']
        del os.environ['MLOP_URL_PY']

    def test_default_urls_when_not_set(self):
        """Test default URLs when env vars not set"""
        # Make sure env vars are not set
        for var in ['MLOP_URL_APP', 'MLOP_URL_API', 'MLOP_URL_INGEST', 'MLOP_URL_PY']:
            if var in os.environ:
                del os.environ[var]

        settings = setup()
        # These should be the production defaults
        assert settings.url_app == 'https://trakkur.trainy.ai'
        assert settings.url_api == 'https://trakkur-api.trainy.ai'
        assert settings.url_ingest == 'https://trakkur-ingest.trainy.ai'
        assert settings.url_py == 'https://trakkur-py.trainy.ai'
