---
comments: true
description: Manage your Ultralytics Platform account including API keys, billing, and user settings with security and GDPR compliance.
keywords: Ultralytics Platform, account, settings, API keys, billing, security, GDPR
---

# Account Management

[Ultralytics Platform](https://platform.ultralytics.com) provides comprehensive account management for API keys, billing, and user settings. Manage your account securely with GDPR-compliant data handling.

## Overview

The Account section helps you:

- **Create** and manage API keys for programmatic access
- **Track** credit balance and billing
- **Configure** profile and preferences
- **Export** your data for GDPR compliance

<!-- Screenshot: platform-account-overview.avif -->

## Account Features

| Feature      | Description                                    |
| ------------ | ---------------------------------------------- |
| **API Keys** | Secure keys for remote training and API access |
| **Billing**  | Credits, payments, and usage tracking          |
| **Activity** | Track events and account actions               |
| **Trash**    | Recover deleted items within 30 days           |
| **Settings** | Profile, region, and preferences               |
| **GDPR**     | Data export and account deletion               |

## Security

Ultralytics Platform implements multiple security measures:

### Authentication

- **OAuth2**: Sign in with Google, Apple, or GitHub
- **Email**: Traditional email/password authentication
- **Session management**: Secure, expiring sessions

### Data Protection

- **Encryption**: All data encrypted at rest and in transit
- **API Keys**: Securely encrypted storage
- **Region isolation**: Data stays in your selected region

### Access Control

- **Per-key scopes**: Limit API key permissions
- **Session timeout**: Automatic logout after inactivity
- **Audit logging**: Track all account activity

## Quick Links

- [**API Keys**](api-keys.md): Create and manage API keys
- [**Billing**](billing.md): Credits and payment management
- [**Activity**](activity.md): Track account events and notifications
- [**Trash**](trash.md): Recover deleted projects, datasets, and models
- [**Settings**](settings.md): Profile and preferences

## FAQ

### How do I change my email address?

Email changes are managed through your OAuth provider (Google, Apple, GitHub) or:

1. Go to Settings
2. Click **Edit Profile**
3. Update email address
4. Verify new email

### How do I delete my account?

Account deletion is available in Settings:

1. Go to Settings > Privacy
2. Click **Delete Account**
3. Confirm deletion

!!! warning "Permanent Action"

    Account deletion is permanent. All data, models, and deployments are removed. Export your data first if needed.

### Is my data secure?

Yes, Ultralytics Platform implements:

- Secure encrypted connections
- Encryption at rest
- Regional data isolation
- Regular security audits

### Can I change my data region?

No, data region is selected during signup and cannot be changed. To use a different region:

1. Export your data
2. Create a new account in desired region
3. Re-upload your data

This ensures data residency compliance.
