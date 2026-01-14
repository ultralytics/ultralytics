from unittest.mock import MagicMock, patch

import pytest

from ultralytics.hub.session import HUBTrainingSession
from ultralytics.utils.errors import HUBModelError


class TestHUBTrainingSession:
    @patch("ultralytics.hub.session.HUBTrainingSession._parse_identifier")
    @patch("hub_sdk.HUBClient")
    def test_create_model_failure(self, mock_hub_client, mock_parse, caplog):
        """Test that HUBModelError is raised when model creation fails."""
        # Setup mocks
        mock_parse.return_value = ("api_key", "model_id", "filename.pt")

        # Mock the client and model
        mock_client_instance = MagicMock()
        mock_hub_client.return_value = mock_client_instance

        mock_model_instance = MagicMock()
        mock_client_instance.model.return_value = mock_model_instance

        # Simulate model creation failure (id remains None)
        mock_model_instance.id = None

        # Create session
        session = HUBTrainingSession("filename.pt")
        session.model = mock_model_instance
        session.filename = "filename.pt"

        # Test
        with pytest.raises(HUBModelError, match="Failed to create model"):
            session.create_model({"epochs": 1})

    def test_parse_identifier_invalid(self):
        """Test that HUBModelError is raised for invalid identifier formats."""
        with pytest.raises(HUBModelError, match="invalid, correct format is"):
            HUBTrainingSession._parse_identifier("invalid_format_string")
