---
comments: true
description: Configure your Ultralytics Platform profile, preferences, and data settings with GDPR-compliant data export and deletion options.
keywords: Ultralytics Platform, settings, profile, preferences, GDPR, data export, privacy
---

# Settings

[Ultralytics Platform](https://platform.ultralytics.com) settings allow you to configure your profile, preferences, and manage your data with GDPR-compliant export and deletion options.

## Profile Settings

Update your profile information:

<!-- Screenshot: platform-settings-profile.avif -->

| Field             | Description                      |
| ----------------- | -------------------------------- |
| **Display Name**  | Your public name                 |
| **Username**      | Unique identifier (used in URLs) |
| **Bio**           | Short description                |
| **Company**       | Organization name                |
| **Use Case**      | Primary application              |
| **Profile Image** | Avatar displayed across Platform |

### Edit Profile

1. Go to **Settings > Profile**
2. Update fields
3. Click **Save**

### Username Rules

- 3-30 characters
- Lowercase letters, numbers, hyphens
- Cannot start/end with hyphen
- Must be unique

!!! warning "Username Changes"

    Changing username updates all your public URLs. Old URLs will stop working.

## Social Links

Add links to your profiles:

<!-- Screenshot: platform-settings-social.avif -->

| Platform     | URL Format               |
| ------------ | ------------------------ |
| **GitHub**   | github.com/username      |
| **Twitter**  | twitter.com/username     |
| **LinkedIn** | linkedin.com/in/username |
| **Website**  | your-website.com         |

Social links appear on your public profile page.

## Data Region

View your data region:

<!-- Screenshot: platform-settings-region.avif -->

| Region | Location             | Best For                                |
| ------ | -------------------- | --------------------------------------- |
| **US** | Iowa, USA            | Americas users, fastest for Americas    |
| **EU** | Belgium, Europe      | European users, GDPR compliance         |
| **AP** | Taiwan, Asia-Pacific | Asia-Pacific users, lowest APAC latency |

!!! note "Region is Permanent"

    Data region is selected during signup and cannot be changed. All your data stays in this region.

## Storage Usage

Monitor your storage consumption:

<!-- Screenshot: platform-settings-storage.avif -->

| Type         | Description             |
| ------------ | ----------------------- |
| **Datasets** | Image and label storage |
| **Models**   | Checkpoint storage      |
| **Exports**  | Exported model formats  |

### Storage Limits

| Plan       | Limit     |
| ---------- | --------- |
| Free       | 100 GB    |
| Pro        | 500 GB    |
| Enterprise | Unlimited |

### Reduce Storage

To free up storage:

1. Delete unused datasets
2. Remove old model checkpoints
3. Delete exported formats
4. Empty trash (Settings > Trash)

## Trash

Deleted items go to Trash for 30 days:

1. Go to **Settings > Trash**
2. View deleted projects, datasets, models
3. **Restore** to recover, or **Delete** permanently

### Auto-Cleanup

Items in Trash are permanently deleted after 30 days. This cannot be undone.

## GDPR Compliance

Ultralytics Platform supports GDPR rights:

### Data Export

Download all your data:

<!-- Screenshot: platform-settings-gdpr.avif -->

1. Go to **Settings > Privacy**
2. Click **Export Data**
3. Receive download link via email

Export includes:

- Profile information
- Dataset metadata
- Model metadata
- Training history
- API key metadata (not secrets)

### Account Deletion

Permanently delete your account:

1. Go to **Settings > Privacy**
2. Click **Delete Account**
3. Type confirmation phrase
4. Confirm deletion

!!! warning "Irreversible Action"

    Account deletion is permanent. All data is removed within 30 days per GDPR requirements.

### What's Deleted

- Profile and settings
- All datasets and images
- All models and checkpoints
- All deployments
- API keys
- Billing history

### What's Retained

- Anonymized analytics
- Server logs (90 days)
- Legal compliance records

## Notifications

Configure notification preferences:

| Type                  | Options          |
| --------------------- | ---------------- |
| **Training Complete** | Email, none      |
| **Deployment Status** | Email, none      |
| **Billing Alerts**    | Email (required) |
| **Product Updates**   | Email, none      |

## Theme

Select your preferred theme:

| Theme      | Description      |
| ---------- | ---------------- |
| **Light**  | Light background |
| **Dark**   | Dark background  |
| **System** | Match OS setting |

## Sessions

Manage active sessions:

1. Go to **Settings > Security**
2. View active sessions
3. **Revoke** suspicious sessions

Session information:

- Device type
- Browser
- Location (approximate)
- Last active

## FAQ

### How do I change my email?

Email is managed through your OAuth provider:

1. Update email in Google/Apple/GitHub
2. Sign out and sign in again
3. Platform updates automatically

### Can I have multiple accounts?

You can create accounts in different regions, but:

- Each needs a unique email
- Data doesn't transfer between accounts
- Billing is separate

### How do I change my password?

Passwords are managed by your OAuth provider:

- **Google**: accounts.google.com
- **Apple**: appleid.apple.com
- **GitHub**: github.com/settings/security

### Is two-factor authentication available?

2FA is handled by your OAuth provider. Enable 2FA in:

- Google Account settings
- Apple ID settings
- GitHub Security settings

### How long until deleted data is removed?

| Type                 | Timeline      |
| -------------------- | ------------- |
| **Trash items**      | 30 days       |
| **Account deletion** | Up to 30 days |
| **Backups**          | 90 days       |
