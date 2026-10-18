"""Configuration validation tests for optional OIDC authentication."""
import unittest

from src.auth import auth_configuration_error


class AuthConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "redirect_uri": "http://localhost:8501/oauth2callback",
            "cookie_secret": "a-long-random-cookie-secret",
            "auth0": {
                "client_id": "example-client-id",
                "client_secret": "example-client-secret",
                "server_metadata_url": (
                    "https://example.auth0.com/.well-known/openid-configuration"
                ),
            },
        }

    def test_complete_configuration_is_accepted(self):
        self.assertIsNone(auth_configuration_error(self.settings))

    def test_non_table_configuration_is_rejected(self):
        self.assertIsNotNone(auth_configuration_error(None))

    def test_missing_provider_is_rejected(self):
        settings = {key: value for key, value in self.settings.items() if key != "auth0"}
        self.assertIn("[auth.auth0]", auth_configuration_error(settings))

    def test_empty_cookie_secret_is_rejected(self):
        settings = {**self.settings, "cookie_secret": "  "}
        self.assertIn("cookie_secret", auth_configuration_error(settings))

    def test_missing_provider_secret_is_rejected(self):
        provider = {**self.settings["auth0"], "client_secret": ""}
        settings = {**self.settings, "auth0": provider}
        self.assertIn("provider value", auth_configuration_error(settings))


if __name__ == "__main__":
    unittest.main()
