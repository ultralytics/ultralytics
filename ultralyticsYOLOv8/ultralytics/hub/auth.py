# Ultralytics YOLO 🚀, GPL-3.0 license

import requests

from ultralytics.hub.utils import HUB_API_ROOT, PREFIX, request_with_credentials
from ultralytics.yolo.utils import LOGGER, SETTINGS, emojis, is_colab, set_settings

API_KEY_URL = 'https://hub.ultralytics.com/settings?tab=api+keys'


class Auth:
    id_token = api_key = model_key = False

    def __init__(self, api_key=''):
        """
        Initialize the Auth class with an optional API key.

        Args:
            api_key (str, optional): May be an API key or a combination API key and model ID, i.e. key_id
        """
        # Split the input API key in case it contains a combined key_model and keep only the API key part
        api_key = api_key.split('_')[0]

        # Set API key attribute as value passed or SETTINGS API key if none passed
        self.api_key = api_key or SETTINGS.get('api_key', '')

        # If an API key is provided
        if self.api_key:
            # If the provided API key matches the API key in the SETTINGS
            if self.api_key == SETTINGS.get('api_key'):
                # Log that the user is already logged in
                LOGGER.info(f'{PREFIX}Authenticated ✅')
                return
            else:
                # Attempt to authenticate with the provided API key
                success = self.authenticate()
        # If the API key is not provided and the environment is a Google Colab notebook
        elif is_colab():
            # Attempt to authenticate using browser cookies
            success = self.auth_with_cookies()
        else:
            # Request an API key
            success = self.request_api_key()

        # Update SETTINGS with the new API key after successful authentication
        if success:
            set_settings({'api_key': self.api_key})
            # Log that the new login was successful
            LOGGER.info(f'{PREFIX}New authentication successful ✅')
        else:
            LOGGER.info(f'{PREFIX}Retrieve API key from {API_KEY_URL}')

    def request_api_key(self, max_attempts=3):
        """
        Prompt the user to input their API key. Returns the model ID.
        """
        import getpass
        for attempts in range(max_attempts):
            LOGGER.info(f'{PREFIX}Login. Attempt {attempts + 1} of {max_attempts}')
            input_key = getpass.getpass(f'Enter API key from {API_KEY_URL} ')
            self.api_key = input_key.split('_')[0]  # remove model id if present
            if self.authenticate():
                return True
        raise ConnectionError(emojis(f'{PREFIX}Failed to authenticate ❌'))

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
            LOGGER.warning(f'{PREFIX}Invalid API key ⚠️')
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
