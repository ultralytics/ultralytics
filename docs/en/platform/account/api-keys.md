---
plans: [free, pro, enterprise]
title: API Key Management
comments: true
description: Create and manage API keys for Ultralytics Platform with secure AES-256-GCM encryption for remote training and programmatic access.
keywords: Ultralytics Platform, API keys, authentication, remote training, security, access control
---

# API Keys

[Ultralytics Platform](https://platform.ultralytics.com) API keys enable secure programmatic access for remote training, inference, and automation. Create named keys with AES-256-GCM encryption for different use cases.

![Ultralytics Platform Settings API Keys Tab Key List](https://cdn.ul.run/i/f2c74d17fe21805f988c87adbc456674.avif)<!-- screenshot -->

!!! note "Owner-Only"

    Only the workspace owner can create, view, or revoke a workspace's API keys, because a key authenticates as the
    workspace owner. Members with any other role see a note on the tab instead of the key list. API keys themselves
    cannot create or revoke other API keys. The one exception is [On Premise worker keys](#on-premise-worker-keys),
    which are revoked by disconnecting the host from the On Premise integration.

## Create API Key

Create a new API key:

1. Go to **Settings > API Keys**
2. Click **Create Key**
3. Enter a name for the key (e.g., "Training Server")
4. Click **Create Key**

![Ultralytics Platform Settings API Keys Tab Create API Key Dialog](https://cdn.ul.run/i/263a91df7402a10d57923827fe00aa0b.avif)<!-- screenshot -->

### Key Name

Give your key a descriptive name:

- `training-server` - For remote training machines
- `ci-pipeline` - For CI/CD integration
- `local-dev` - For local development

### Key Display

After creation, the key is displayed in a confirmation dialog:

![Ultralytics Platform Settings API Keys Tab API Key Created Copy Dialog](https://cdn.ul.run/i/9d54f61a64e1d9887f622d64834d7d2e.avif)<!-- screenshot -->
!!! tip "Copy Your Key"

    Copy your key after creation for easy reference. Keys are also visible in the key list — the platform decrypts and
    displays full key values so you can copy them anytime.

## Key Format

API keys follow this format:

```text
ul_a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4
```

- **Prefix**: `ul_` identifies Ultralytics keys
- **Body**: 40 random hexadecimal characters
- **Total**: 43 characters

### Key Security

- Keys are stored with **AES-256-GCM encryption**, never in plaintext
- The first 11 characters (`ul_` plus 8 hex characters) act as a display prefix, so a key can be identified without exposing it

## Using API Keys

### Environment Variable

Set your key as an environment variable:

=== "Linux/macOS"

    ```bash
    export ULTRALYTICS_API_KEY="YOUR_API_KEY"
    ```

=== "Windows"

    ```powershell
    $env:ULTRALYTICS_API_KEY = "YOUR_API_KEY"
    ```

### YOLO CLI

Validate and save the key using the YOLO CLI:

```bash
yolo login YOUR_API_KEY
```

Remove the saved key with `yolo logout`.

### HTTP Headers

Include the key in API requests:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://platform.ultralytics.com/api/...
```

See the [REST API Reference](../api/index.md) for all available endpoints.

### Remote Training

Enable metric streaming with your key.

Install or update the Ultralytics package before starting:

```bash
pip install -U ultralytics
```

```bash
export ULTRALYTICS_API_KEY="YOUR_API_KEY"
yolo train model=yolo26n.pt data=coco.yaml project=username/project name=exp1
```

See [Cloud Training](../train/cloud-training.md#remote-training) for the complete remote training guide.

## Manage Keys

### View Keys

All keys are listed on the `Settings > API Keys` tab:

Each key card shows the key name, the copyable key value, the relative creation time, and a revoke button.

### Revoke Key

Revoke a key that's compromised or no longer needed:

1. Find the key in the API Keys section
2. Click the **Revoke** (trash) button
3. Confirm revocation

!!! warning "Immediate Effect"

    Revocation is immediate and permanent — the key record is deleted, not disabled. Any applications using the key
    will stop working.

### Regenerate Key

If a key is compromised:

1. Create a new key with the same name
2. Update your applications
3. Revoke the old key

## Workspace API Keys

API keys are scoped to the currently active workspace:

- **Personal workspace**: Keys authenticate as your personal account
- **Team workspace**: Keys authenticate as the team workspace owner, with full owner permissions in that workspace

When switching workspaces in the sidebar, the API Keys section shows keys for that workspace. Because a workspace key
carries owner permissions, only the workspace owner can create, view, or revoke one. See [Teams](teams.md) for role
details.

### On Premise Worker Keys

Connecting an [On Premise](../integrations/on-premise.md) host mints a separate worker key. Worker keys are managed from
the On Premise integration rather than this tab, are never listed alongside your API keys, and are revoked by
disconnecting the host — which also cancels that host's queued and running jobs.

## Security Best Practices

### Do

- Store keys in environment variables
- Use separate keys for different environments
- Revoke unused keys promptly
- Rotate keys periodically
- Use descriptive names to identify key purposes

### Don't

- Commit keys to version control
- Share keys between applications
- Log keys in application output
- Embed keys in client-side code

### Key Rotation

Rotate keys periodically for security:

1. Create new key with same name
2. Update applications to use new key
3. Verify applications work correctly
4. Revoke old key

!!! tip "Rotation Schedule"

    Consider rotating keys every 90 days for sensitive applications.

## Troubleshooting

### Invalid Key Error

```text
Error: Invalid API key
```

Solutions:

1. Verify key is copied correctly (including the `ul_` prefix)
2. Check key hasn't been revoked
3. Confirm environment variable is set
4. Ensure you're using `ultralytics>=8.4.120`

### Permission Denied

```text
Error: Permission denied for this operation
```

Solutions:

1. Verify you're the resource owner or have appropriate workspace access
2. Check the key belongs to the correct workspace
3. If you're managing keys in a team workspace, confirm you're the workspace owner — other roles get
   `Workspace owner access required`
4. Create a new key if needed

### Rate Limited

```text
Error: Rate limit exceeded
```

Solutions:

1. Reduce request frequency — see the [rate limit table](../api/index.md#rate-limits) for per-category limits
2. Implement exponential backoff using the `Retry-After` header
3. Use a [dedicated endpoint](../deploy/endpoints.md) when you need isolated inference capacity

## FAQ

### How many keys can I create?

There's no hard limit on API keys. Create as many as needed for different applications and environments.

### Do keys expire?

Keys don't expire automatically. They remain valid until revoked. Consider implementing rotation for security.

### Can I see my key after creation?

Yes, full key values are visible in the key list on `Settings > API Keys`. The Platform decrypts and displays your keys so you can copy them anytime.

### Are keys region-specific?

Keys work across regions but access data in your account's region only.

### Can I share keys with team members?

No — a team workspace key authenticates as the workspace owner, so only the owner can create or view one, and sharing it
hands over owner permissions. Have each member create a key in their own personal workspace instead, and ask the owner
to mint a dedicated workspace key for shared automation such as CI.

### Do keys work in every workspace I belong to?

No. A key belongs to the workspace it was created in and only reaches that workspace's resources. Create a separate key
for each workspace you automate.
