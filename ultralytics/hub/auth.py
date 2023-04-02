# Ultralytics YOLO 🚀, GPL-3.0 license

import requests

from ultralytics.hub.utils import HUB_API_ROOT, request_with_credentials, PREFIX
from ultralytics.yolo.utils import SETTINGS, is_colab, set_settings, LOGGER

API_KEY_PATH = 'https://hub.ultralytics.com/settings?tab=api+keys'


class Auth:
    id_token = api_key = model_key = False

    def __init__(self, api_key=None):
        """
        Initialize the Auth class with an optional API key.

        Args:
            api_key (str, optional): May be an API key or a combination API key and model ID, i.e. key_id
        """
        self.api_key = self._clean_api_key(api_key)
        if SETTINGS.get('api_key') != self.api_key:
            self.authenticate() if self.api_key else self.auth_with_cookies()
            set_settings({'api_key': self.api_key})
            LOGGER.info(f'{PREFIX}New login successful ✅')
        else:
            LOGGER.info(f'{PREFIX}Logged in ✅')

    @staticmethod
    def _clean_api_key(key: str) -> str:
        """
        Strip the model from the key if present.

        Args:
            key (str): The API key string.

        Returns:
            str: The cleaned API key string.
        """
        separator = '_'
        return key.split(separator)[0] if separator in key else key

    def authenticate(self) -> bool:
        """
        Attempt to authenticate with the server using either id_token or API key.

        Returns:
            bool: True if authentication is successful, False otherwise.
        """
        try:
            header = self.get_auth_header()
            if header:
                r = requests.post(f'{HUB_API_ROOT}/v1/auth', headers=header)
                if not r.json().get('success', False):
                    raise ConnectionError('Unable to authenticate.')
                return True
            raise ConnectionError('User has not authenticated locally.')
        except ConnectionError:
            self.id_token = self.api_key = False  # reset invalid
            return False

    def auth_with_cookies(self) -> bool:
        """
        Attempt to fetch authentication via cookies and set id_token.
        User must be logged in to HUB and running in a supported browser.

        Returns:
            bool: True if authentication is successful, False otherwise.
        """
        if not is_colab():
            return False  # Currently only works with Colab
        try:
            authn = request_with_credentials(f'{HUB_API_ROOT}/v1/auth/auto')
            if authn.get('success', False):
                self.id_token = authn.get('data', {}).get('idToken', None)
                self.authenticate()
                return True
            raise ConnectionError('Unable to fetch browser authentication details.')
        except ConnectionError:
            self.id_token = False  # reset invalid
            return False

    def get_auth_header(self):
        """
        Get the authentication header for making API requests.

        Returns:
            dict: The authentication header if id_token or API key is set, None otherwise.
        """
        if self.id_token:
            return {'authorization': f'Bearer {self.id_token}'}
        elif self.api_key:
            return {'x-api-key': self.api_key}
        else:
            return None

    def get_state(self) -> bool:
        """
        Get the authentication state.

        Returns:
            bool: True if either id_token or API key is set, False otherwise.
        """
        return self.id_token or self.api_key

    def set_api_key(self, key: str):
        """
        Set the API key for authentication.

        Args:
            key (str): The API key string.
        """
        self.api_key = key
