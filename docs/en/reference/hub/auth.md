---
description: Learn how to manage API key and cookie-based authentication in Ultralytics with the Auth class. Step-by-step guide for effective authentication.
keywords: Ultralytics, authentication, API key, cookies, Auth class, YOLO, API, guide
---

# Reference for `ultralytics/hub/auth.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/auth.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/auth.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`Auth`](#ultralytics.hub.auth.Auth)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`Auth.request_api_key`](#ultralytics.hub.auth.Auth.request_api_key)
        - [`Auth.authenticate`](#ultralytics.hub.auth.Auth.authenticate)
        - [`Auth.auth_with_cookies`](#ultralytics.hub.auth.Auth.auth_with_cookies)
        - [`Auth.get_auth_header`](#ultralytics.hub.auth.Auth.get_auth_header)


## Class `ultralytics.hub.auth.Auth` {#ultralytics.hub.auth.Auth}

```python
Auth(self, api_key: str = "", verbose: bool = False)
```

Manages authentication processes including API key handling, cookie-based authentication, and header generation.

The class supports different methods of authentication:
1. Directly using an API key.
2. Authenticating using browser cookies (specifically in Google Colab).
3. Prompting the user to enter an API key.

Handles API key validation, Google Colab authentication, and new key requests. Updates SETTINGS upon successful authentication.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `api_key` | `str` | API key or combined key_id format. | `""` |
| `verbose` | `bool` | Enable verbose logging. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `id_token` | `str | bool` | Token used for identity verification, initialized as False. |
| `api_key` | `str | bool` | API key for authentication, initialized as False. |
| `model_key` | `bool` | Placeholder for model key, initialized as False. |

**Methods**

| Name | Description |
| --- | --- |
| [`auth_with_cookies`](#ultralytics.hub.auth.Auth.auth_with_cookies) | Attempt to fetch authentication via cookies and set id_token. |
| [`authenticate`](#ultralytics.hub.auth.Auth.authenticate) | Attempt to authenticate with the server using either id_token or API key. |
| [`get_auth_header`](#ultralytics.hub.auth.Auth.get_auth_header) | Get the authentication header for making API requests. |
| [`request_api_key`](#ultralytics.hub.auth.Auth.request_api_key) | Prompt the user to input their API key. |

**Examples**

```python
Initialize Auth with an API key
>>> auth = Auth(api_key="your_api_key_here")

Initialize Auth without API key (will prompt for input)
>>> auth = Auth()
```

<details>
<summary>Source code in <code>ultralytics/hub/auth.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/auth.py#L9-L151"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Auth:
    """Manages authentication processes including API key handling, cookie-based authentication, and header generation.

    The class supports different methods of authentication:
    1. Directly using an API key.
    2. Authenticating using browser cookies (specifically in Google Colab).
    3. Prompting the user to enter an API key.

    Attributes:
        id_token (str | bool): Token used for identity verification, initialized as False.
        api_key (str | bool): API key for authentication, initialized as False.
        model_key (bool): Placeholder for model key, initialized as False.

    Methods:
        authenticate: Attempt to authenticate with the server using either id_token or API key.
        auth_with_cookies: Attempt to fetch authentication via cookies and set id_token.
        get_auth_header: Get the authentication header for making API requests.
        request_api_key: Prompt the user to input their API key.

    Examples:
        Initialize Auth with an API key
        >>> auth = Auth(api_key="your_api_key_here")

        Initialize Auth without API key (will prompt for input)
        >>> auth = Auth()
    """

    id_token = api_key = model_key = False

    def __init__(self, api_key: str = "", verbose: bool = False):
        """Initialize Auth class and authenticate user.

        Handles API key validation, Google Colab authentication, and new key requests. Updates SETTINGS upon successful
        authentication.

        Args:
            api_key (str): API key or combined key_id format.
            verbose (bool): Enable verbose logging.
        """
        # Split the input API key in case it contains a combined key_model and keep only the API key part
        api_key = api_key.split("_", 1)[0]

        # Set API key attribute as value passed or SETTINGS API key if none passed
        self.api_key = api_key or SETTINGS.get("api_key", "")

        # If an API key is provided
        if self.api_key:
            # If the provided API key matches the API key in the SETTINGS
            if self.api_key == SETTINGS.get("api_key"):
                # Log that the user is already logged in
                if verbose:
                    LOGGER.info(f"{PREFIX}Authenticated ✅")
                return
            else:
                # Attempt to authenticate with the provided API key
                success = self.authenticate()
        # If the API key is not provided and the environment is a Google Colab notebook
        elif IS_COLAB:
            # Attempt to authenticate using browser cookies
            success = self.auth_with_cookies()
        else:
            # Request an API key
            success = self.request_api_key()

        # Update SETTINGS with the new API key after successful authentication
        if success:
            SETTINGS.update({"api_key": self.api_key})
            # Log that the new login was successful
            if verbose:
                LOGGER.info(f"{PREFIX}New authentication successful ✅")
        elif verbose:
            LOGGER.info(f"{PREFIX}Get API key from {API_KEY_URL} and then run 'yolo login API_KEY'")
```
</details>

<br>

### Method `ultralytics.hub.auth.Auth.auth_with_cookies` {#ultralytics.hub.auth.Auth.auth\_with\_cookies}

```python
def auth_with_cookies(self) -> bool
```

Attempt to fetch authentication via cookies and set id_token.

User must be logged in to HUB and running in a supported browser.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if authentication is successful, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/hub/auth.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/auth.py#L121-L140"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def auth_with_cookies(self) -> bool:
    """Attempt to fetch authentication via cookies and set id_token.

    User must be logged in to HUB and running in a supported browser.

    Returns:
        (bool): True if authentication is successful, False otherwise.
    """
    if not IS_COLAB:
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
```
</details>

<br>

### Method `ultralytics.hub.auth.Auth.authenticate` {#ultralytics.hub.auth.Auth.authenticate}

```python
def authenticate(self) -> bool
```

Attempt to authenticate with the server using either id_token or API key.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if authentication is successful, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/hub/auth.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/auth.py#L101-L119"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def authenticate(self) -> bool:
    """Attempt to authenticate with the server using either id_token or API key.

    Returns:
        (bool): True if authentication is successful, False otherwise.
    """
    import requests  # scoped as slow import

    try:
        if header := self.get_auth_header():
            r = requests.post(f"{HUB_API_ROOT}/v1/auth", headers=header)
            if not r.json().get("success", False):
                raise ConnectionError("Unable to authenticate.")
            return True
        raise ConnectionError("User has not authenticated locally.")
    except ConnectionError:
        self.id_token = self.api_key = False  # reset invalid
        LOGGER.warning(f"{PREFIX}Invalid API key")
        return False
```
</details>

<br>

### Method `ultralytics.hub.auth.Auth.get_auth_header` {#ultralytics.hub.auth.Auth.get\_auth\_header}

```python
def get_auth_header(self)
```

Get the authentication header for making API requests.

**Returns**

| Type | Description |
| --- | --- |
| `dict | None` | The authentication header if id_token or API key is set, None otherwise. |

<details>
<summary>Source code in <code>ultralytics/hub/auth.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/auth.py#L142-L151"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_auth_header(self):
    """Get the authentication header for making API requests.

    Returns:
        (dict | None): The authentication header if id_token or API key is set, None otherwise.
    """
    if self.id_token:
        return {"authorization": f"Bearer {self.id_token}"}
    elif self.api_key:
        return {"x-api-key": self.api_key}
```
</details>

<br>

### Method `ultralytics.hub.auth.Auth.request_api_key` {#ultralytics.hub.auth.Auth.request\_api\_key}

```python
def request_api_key(self, max_attempts: int = 3) -> bool
```

Prompt the user to input their API key.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `max_attempts` | `int` | Maximum number of authentication attempts. | `3` |

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if authentication is successful, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/hub/auth.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/auth.py#L82-L99"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def request_api_key(self, max_attempts: int = 3) -> bool:
    """Prompt the user to input their API key.

    Args:
        max_attempts (int): Maximum number of authentication attempts.

    Returns:
        (bool): True if authentication is successful, False otherwise.
    """
    import getpass

    for attempts in range(max_attempts):
        LOGGER.info(f"{PREFIX}Login. Attempt {attempts + 1} of {max_attempts}")
        input_key = getpass.getpass(f"Enter API key from {API_KEY_URL} ")
        self.api_key = input_key.split("_", 1)[0]  # remove model id if present
        if self.authenticate():
            return True
    raise ConnectionError(emojis(f"{PREFIX}Failed to authenticate ❌"))
```
</details>

<br><br>
