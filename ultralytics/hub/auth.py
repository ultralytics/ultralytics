# Ultralytics YOLO 🚀, GPL-3.0 license

import requests

from ultralytics.hub.utils import HUB_API_ROOT, request_with_credentials
from ultralytics.yolo.utils import is_colab

API_KEY_PATH = "https://hub.ultralytics.com/settings?tab=api+keys"


class Auth:
    id_token = api_key = model_key = False

    def __init__(self, api_key=None):
        self.api_key = self._clean_api_key(api_key)
        self.authenticate() if self.api_key else self.auth_with_cookies()

    @staticmethod
    def _clean_api_key(key: str) -> str:
        """Strip model from key if present"""
        separator = "_"
        return key.split(separator)[0] if separator in key else key

    def authenticate(self) -> bool:
        """Attempt to authenticate with server"""
        try:
            header = self.get_auth_header()
            if header:
                r = requests.post(f"{HUB_API_ROOT}/v1/auth", headers=header)
                if not r.json().get('success', False):
                    raise ConnectionError("Unable to authenticate.")
                return True
            raise ConnectionError("User has not authenticated locally.")
        except ConnectionError:
            self.id_token = self.api_key = False  # reset invalid
            return False

    def auth_with_cookies(self) -> bool:
        """
        Attempt to fetch authentication via cookies and set id_token.
        User must be logged in to HUB and running in a supported browser.
        """
        if not is_colab():
            return False  # Currently only works with Colab
        try:
            authn = request_with_credentials(f"{HUB_API_ROOT}/v1/auth/auto")
            if authn.get("success", False):
                self.id_token = authn.get("data", {}).get("idToken", None)
                self.authenticate()
                return True
            raise ConnectionError("Unable to fetch browser authentication details.")
        except ConnectionError:
            self.id_token = False  # reset invalid
            return False

    def get_auth_header(self):
        if self.id_token:
            return {"authorization": f"Bearer {self.id_token}"}
        elif self.api_key:
            return {"x-api-key": self.api_key}
        else:
            return None

    def get_state(self) -> bool:
        """Get the authentication state"""
        return self.id_token or self.api_key

    def set_api_key(self, key: str):
        """Get the authentication state"""
        self.api_key = key
