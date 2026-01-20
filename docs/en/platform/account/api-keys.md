---
comments: true
description: Create and manage API keys for Ultralytics Platform with scoped permissions for remote training, inference, and programmatic access.
keywords: Ultralytics Platform, API keys, authentication, remote training, security, access control
---

# API Keys

[Ultralytics Platform](https://platform.ultralytics.com) API keys enable secure programmatic access for remote training, inference, and automation. Create scoped keys with specific permissions for different use cases.

<!-- Screenshot: platform-apikeys-list.avif -->

## Create API Key

Create a new API key:

1. Go to **Settings > API Keys**
2. Click **Create Key**
3. Enter a name for the key
4. Select permission scopes
5. Click **Create**

<!-- Screenshot: platform-apikeys-create.avif -->

### Key Name

Give your key a descriptive name:

- `training-server` - For remote training machines
- `ci-pipeline` - For CI/CD integration
- `mobile-app` - For mobile applications

### Permission Scopes

Select scopes to limit key permissions:

<!-- Screenshot: platform-apikeys-scopes.avif -->

| Scope        | Permissions                        |
| ------------ | ---------------------------------- |
| **training** | Start training, stream metrics     |
| **models**   | Upload, download, delete models    |
| **datasets** | Access and modify datasets         |
| **read**     | Read-only access to all resources  |
| **write**    | Full write access                  |
| **admin**    | Account management (use carefully) |

!!! tip "Least Privilege"

    Create keys with only the permissions needed. Use separate keys for different applications.

### Key Display

After creation, the key is displayed once:

<!-- Screenshot: platform-apikeys-created.avif -->

!!! warning "Copy Your Key"

    The full key is only shown once. Copy it immediately and store securely. You cannot retrieve it later.

## Key Format

API keys follow this format:

```
ul_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0
```

- **Prefix**: `ul_` identifies Ultralytics keys
- **Body**: 40 random hexadecimal characters
- **Total**: 43 characters

## Using API Keys

### Environment Variable

Set your key as an environment variable:

=== "Linux/macOS"

    ```bash
    export ULTRALYTICS_API_KEY="ul_your_key_here"
    ```

=== "Windows"

    ```powershell
    $env:ULTRALYTICS_API_KEY = "ul_your_key_here"
    ```

### In Code

Use the key in your Python scripts:

```python
import os

# From environment (recommended)
api_key = os.environ.get("ULTRALYTICS_API_KEY")

# Or directly (not recommended for production)
api_key = "ul_your_key_here"
```

### HTTP Headers

Include the key in API requests:

```bash
curl -H "Authorization: Bearer ul_your_key_here" \
  https://platform.ultralytics.com/api/...
```

### Remote Training

Enable metric streaming with your key.

!!! warning "Package Version Requirement"

    Platform integration requires **ultralytics>=8.4.0**. Lower versions will NOT work with Platform.

    ```bash
    pip install "ultralytics>=8.4.0"
    ```

```bash
export ULTRALYTICS_API_KEY="ul_your_key_here"
yolo train model=yolo26n.pt data=coco.yaml project=username/project name=exp1
```

## Manage Keys

### View Keys

All keys are listed in Settings > API Keys:

| Column        | Description          |
| ------------- | -------------------- |
| **Name**      | Key identifier       |
| **Scopes**    | Assigned permissions |
| **Created**   | Creation date        |
| **Last Used** | Most recent use      |

### Revoke Key

Revoke a key that's compromised or no longer needed:

1. Click the key's menu
2. Select **Revoke**
3. Confirm revocation

!!! warning "Immediate Effect"

    Revocation is immediate. Any applications using the key will stop working.

### Regenerate Key

If a key is compromised:

1. Create a new key with same scopes
2. Update your applications
3. Revoke the old key

## Security Best Practices

### Do

- Store keys in environment variables
- Use separate keys for different environments
- Revoke unused keys promptly
- Use minimal required scopes
- Rotate keys periodically

### Don't

- Commit keys to version control
- Share keys between applications
- Use admin scope unnecessarily
- Log keys in application output
- Embed keys in client-side code

### Key Rotation

Rotate keys periodically for security:

1. Create new key with same scopes
2. Update applications to use new key
3. Verify applications work correctly
4. Revoke old key

!!! tip "Rotation Schedule"

    Consider rotating keys every 90 days for sensitive applications.

## Troubleshooting

### Invalid Key Error

```
Error: Invalid API key
```

Solutions:

1. Verify key is copied correctly
2. Check key hasn't been revoked
3. Ensure key has required scopes
4. Confirm environment variable is set

### Permission Denied

```
Error: Permission denied for this operation
```

Solutions:

1. Check key scopes include required permission
2. Verify you're the resource owner
3. Create new key with correct scopes

### Rate Limited

```
Error: Rate limit exceeded
```

Solutions:

1. Reduce request frequency
2. Implement exponential backoff
3. Contact support for limit increase

## FAQ

### How many keys can I create?

There's no hard limit on API keys. Create as many as needed for different applications and environments.

### Do keys expire?

Keys don't expire automatically. They remain valid until revoked. Consider implementing rotation for security.

### Can I see my key after creation?

No, the full key is shown only once at creation. If lost, create a new key and revoke the old one.

### Are keys region-specific?

Keys work across regions but access data in your account's region only.

### Can I share keys with team members?

Better practice: Have each team member create their own key. This enables:

- Individual activity tracking
- Selective revocation
- Proper access control
