---
plans: [free, pro, enterprise]
comments: true
description: Configure your Ultralytics Platform profile, preferences, and data settings with GDPR-compliant data export and deletion options.
keywords: Ultralytics Platform, settings, profile, preferences, GDPR, data export, privacy
title: Account Settings
---

# Settings

[Ultralytics Platform](https://platform.ultralytics.com) settings allow you to configure your profile, social links, workspace preferences, and manage your data with GDPR-compliant export and deletion options.

Settings is organized into seven tabs (in order): `Profile`, `API Keys`, `Plans`, `Billing`, `Teams`, `Integrations`, and `Trash`.

## Profile Tab

The `Profile` tab contains your profile information, social links, data region, and account management options.

### Profile Information

Update your profile information:

![Ultralytics Platform Settings Profile Tab Display Name Bio Company Fields](https://cdn.ul.run/i/679e7eb184fe9db51391b1d3548c9aa2.avif)<!-- screenshot -->

| Field            | Description                               |
| ---------------- | ----------------------------------------- |
| **Display Name** | Your public name                          |
| **Username**     | Unique identifier (set at signup)         |
| **Company**      | Company or organization name              |
| **Use Case**     | Primary application (select from list)    |
| **Bio**          | Short description (minimum 10 characters) |
| **Profile Icon** | Avatar with color, initials, or image     |

#### Username Rules

- 4-32 characters
- Lowercase letters, numbers, hyphens
- Cannot start/end with hyphen
- Must be unique

!!! note "Username is Permanent"

    Your username is set during onboarding and cannot be changed. It appears in all your public URLs (e.g., `platform.ultralytics.com/username`).

#### Use Case Options

| Use Case                | Description                |
| ----------------------- | -------------------------- |
| Manufacturing & QC      | Quality control workflows  |
| Retail & Inventory      | Retail and inventory tasks |
| Security & Surveillance | Security monitoring        |
| Healthcare & Medical    | Medical imaging            |
| Automotive & Robotics   | Self-driving and robotics  |
| Agriculture             | Agricultural monitoring    |
| Research & Academia     | Academic research          |
| Personal Project        | Personal or hobby projects |

### Edit Profile

1. Go to **Settings > Profile**
2. Update fields (display name, company, use case, bio)
3. Wait for the **Saved** indicator next to the Profile heading. Changes save automatically after you stop typing.

### Social Links

Connect your professional profiles:

![Ultralytics Platform Settings Profile Tab Social Links Grid](https://cdn.ul.run/i/6632010d3fc5d6ffd4045df2e1db0e89.avif)<!-- screenshot -->

| Platform           | Format         |
| ------------------ | -------------- |
| **GitHub**         | username       |
| **LinkedIn**       | profile-slug   |
| **X (Twitter)**    | username       |
| **YouTube**        | channel-handle |
| **Google Scholar** | user-id        |
| **Discord**        | username       |
| **Website**        | `example.com`  |

Social links appear on your public profile page.

### Emails

Manage email addresses linked to your account in the `Profile` tab:

![Ultralytics Platform Settings Profile Tab Emails Section](https://cdn.ul.run/i/f09baef9e8e5b2ceec2fa688b597eeeb.avif)<!-- screenshot -->

| Action             | Description                                    |
| ------------------ | ---------------------------------------------- |
| **Add Email**      | Add a new email address to your account        |
| **Remove**         | Remove a non-primary email address             |
| **Verify**         | Send a verification email to confirm ownership |
| **Set as Primary** | Set a verified email as your primary address   |

!!! note "Primary Email"

    Your primary email is used for notifications and account recovery. Only verified emails can be set as primary.

### Data Region

View your data region on the `Profile` tab:

| Region | Location      | Best For                        |
| ------ | ------------- | ------------------------------- |
| **US** | United States | Americas users                  |
| **EU** | Europe        | European users, GDPR compliance |
| **AP** | Asia Pacific  | Asia-Pacific users              |

!!! note "Data Region"

    Your data region is selected during onboarding and cannot be changed yourself. It applies to datasets, models, and
    managed training data. Dedicated deployments use the region selected when each endpoint is created. Contact support
    to request an account data-region change.

### Storage Usage

Monitor your storage consumption on the `Profile` tab and the **Home** page:

![Ultralytics Platform Settings Profile Tab Storage Usage Card](https://cdn.ul.run/i/d4907e21c741a134223d33d80be6f9ed.avif)<!-- screenshot -->
The storage card shows:

- **Overall progress bar** with color-coded status (green under 70%, amber 70-89%, red 90%+)
- **Category breakdown** for datasets, models, and exports
- **Resource counts** for projects, datasets, models, images, and deployments
- **Largest items** to help identify what consumes the most space

| Category     | Description                                           |
| ------------ | ----------------------------------------------------- |
| **Datasets** | Uploaded images, videos, labels, and annotation files |
| **Models**   | Trained model checkpoints (`.pt` files)               |
| **Exports**  | Exported model formats (ONNX, TensorRT, CoreML, etc)  |

#### Storage Limits

| Plan           | Storage   | Models    | Deployments |
| -------------- | --------- | --------- | ----------- |
| **Free**       | 100 GB    | 100       | 3           |
| **Pro**        | 500 GB    | 500       | 10          |
| **Enterprise** | Unlimited | Unlimited | Unlimited   |

#### Upload Size Limits

| File Type                                           | Free  | Pro   | Enterprise |
| --------------------------------------------------- | ----- | ----- | ---------- |
| **Image**                                           | 50 MB | 50 MB | 50 MB      |
| **Video**                                           | 1 GB  | 1 GB  | 1 GB       |
| **Model (.pt)**                                     | 1 GB  | 1 GB  | 1 GB       |
| **Dataset (ZIP/TAR incl. `.tar.gz`/`.tgz`/NDJSON)** | 10 GB | 20 GB | 50 GB      |

#### Trash and Storage

Items in the trash still count toward your storage quota. To free up space, permanently delete items from the trash. Trash items are automatically removed after 30 days. See [Trash](trash.md) for details.

#### Reduce Storage

To free up storage:

1. Delete unused datasets or remove unnecessary images
2. Remove old model checkpoints
3. Delete exported model formats you no longer need
4. Empty trash in [**Settings > Trash**](trash.md)

### Security

The `Profile` tab includes a Security card at the bottom:

- **Two-Factor Authentication**: Marked **Coming Soon** in Platform settings
- **Connected Accounts**: Shows the connected OAuth account displayed by Platform

### GDPR Compliance

Ultralytics Platform supports GDPR rights:

#### Data Export

Download all your data:

1. Go to **Settings > Profile**
2. Scroll to the bottom section
3. Click **Export All Data**
4. An asynchronous export job runs in the background; a **Download Export** link appears on the same page when the job completes (download link valid for 1 hour)

Export includes:

- Profile information
- Dataset metadata
- Model metadata
- Project metadata
- Activity history (recent events)
- API key metadata (keys themselves are never exported in plaintext)

#### Account Deletion

Permanently delete your account:

1. Go to **Settings > Profile**
2. Scroll to the bottom section
3. Click **Delete My Account**
4. Type `DELETE` in the confirmation field, then confirm

!!! warning "Irreversible Action"

    Account deletion is permanent. Your sign-in account is removed immediately and a background job deletes the
    associated Platform data and stored files.

##### What's Deleted

- All projects and trained models
- All datasets and images
- All API keys and credentials
- All activity history
- Credit balance

## API Keys Tab

The `API Keys` tab lets you create and manage API keys for remote training and inference. See [API Keys](api-keys.md) for full documentation.

## Plans Tab

The `Plans` tab lets you compare available plans and upgrade or downgrade your subscription.

![Ultralytics Platform Settings Plans Tab Free Pro Enterprise Comparison](https://cdn.ul.run/i/4687f31bbcab35be3b474784751759e5.avif)<!-- screenshot -->

| Plan           | Storage   | Models    | Deployments | Concurrent Trainings | Team Seats |
| -------------- | --------- | --------- | ----------- | -------------------- | ---------- |
| **Free**       | 100 GB    | 100       | 3           | 3                    | —          |
| **Pro**        | 500 GB    | 500       | 10          | 10                   | Up to 5    |
| **Enterprise** | Unlimited | Unlimited | Unlimited   | Unlimited            | Custom     |

From this tab you can:

- **Compare features** across Free, Pro, and Enterprise tiers
- **Upgrade to Pro** to unlock more storage, models, team collaboration, and B200/B300 GPU access
- **Review Enterprise** capabilities including SSO/SAML and commercial licensing — see [Ultralytics Licensing](https://www.ultralytics.com/license)

See [Billing](billing.md) for detailed plan information, pricing, and upgrade instructions.

## Billing Tab

The `Billing` tab is where you manage credits, payment methods, and review transaction history. Credits pay for
metered cloud training.

![Ultralytics Platform Settings Billing Tab Credit Balance And Plan Card](https://cdn.ul.run/i/8deb4532660afd808780789930cfbeb6.avif)<!-- screenshot -->
From this tab you can:

- **View credit balance** and monitor remaining credits
- **Add credits** via manual top-up (presets from $10–$500 or custom amounts up to $1,000)
- **Enable auto top-up** to automatically add credits when your balance falls below a threshold, reducing the chance of
  training interruption
- **Manage payment methods** and update your billing address
- **Review transaction history** to track all credit movements including purchases, training costs, and refunds

!!! tip "Training Costs"

    Before each training run, the platform estimates the cost based on your selected GPU, dataset size, and epochs.
    The estimate is a balance check, not a credit reservation; actual GPU usage is metered and settled against your
    balance.

See [Billing](billing.md) for full documentation on credits, payment, and plan management.

## Teams Tab

The `Teams` tab lets you manage workspace members, roles, and invitations. Teams are available on [Pro and Enterprise plans](billing.md#plans).

![Ultralytics Platform Teams Member List With Roles](https://cdn.ul.run/i/b680a4b6f2db15b3a34bc19adab8515e.avif)<!-- screenshot -->

### Roles and Permissions

| Role       | Description                                                          |
| ---------- | -------------------------------------------------------------------- |
| **Owner**  | Full control including billing, member management, and team deletion |
| **Admin**  | Manage members, resources, and settings (cannot delete team)         |
| **Editor** | Create and edit projects, datasets, and models                       |
| **Viewer** | Read-only access to shared resources                                 |

### Manage Members

Owners and admins can manage the team:

- **Invite members** via email (invites expire after 14 days; pending invites count against the seat limit)
- **Change roles**: Click the role dropdown next to a member (only the owner can assign/remove the admin role)
- **Remove members**: Click the menu and select **Remove**
- **Cancel invites**: Cancel pending invitations that haven't been accepted
- **Resend invites**: Resend invitation emails
- **Transfer ownership**: Transfer workspace ownership to another member (Owner only)

### Shared Resources

All resources created in a team workspace belong to the team, not individual members. Team members share:

- **Datasets, projects, and models** — accessible by all members based on their role
- **Credit balance** — shared across team members for cloud training
- **Storage and resource limits** — counted at the team level

!!! note "Team Billing"

    On Pro plans, each team member is a paid seat. The team credit balance is shared across all members.

See [Teams](teams.md) for full documentation on team creation, switching workspaces, and enterprise features.

## Integrations Tab

The `Integrations` tab lets you import datasets and projects from external services and connect third-party tools:

- **Google Cloud Storage** — use datasets stored in GCS without uploading a copy.
- **Amazon S3** — use datasets stored in S3 without uploading a copy.
- **Azure Blob Storage** — use datasets stored in Azure without uploading a copy.
- **Ultralytics HUB** — import your existing datasets and projects from [Ultralytics HUB](../integrations/ultralytics-hub.md).
- **Roboflow** — import annotated datasets from a [Roboflow](../integrations/roboflow.md) workspace using a Roboflow API key.
- **Slack** — send selected training, export, and deployment results to a [Slack channel](../integrations/slack.md).
- **On Premise** — connect Enterprise CPU/GPU workers and keep dataset pixels on your own host. See [On Premise](../integrations/on-premise.md).

See [Integrations](../integrations/index.md) for the full list of supported services.

## Trash Tab

The `Trash` tab shows all deleted items and lets you restore or permanently remove them. Deleted items follow a 30-day soft delete policy before automatic permanent deletion.

![Ultralytics Platform Settings Trash Tab With Items And Storage Treemap](https://cdn.ul.run/i/1fda3fe06d0527f579017b71afa6a2ff.avif)<!-- screenshot -->
From this tab you can:

- **Browse deleted items** filtered by type (All, Projects, Datasets, Models)
- **View the storage treemap** to see the relative size of trashed items
- **Restore items** to their original location with all data intact
- **Permanently delete** individual items or use **Empty Trash** to remove everything at once

!!! warning "Storage Impact"

    Items in the trash still count toward your storage quota. To free up space immediately, permanently delete items you no longer need.

See [Trash](trash.md) for full documentation including cascade behavior and API access.

## Help & Feedback

The **Help** page is accessible from the sidebar footer. Use it to:

- **Rate your experience** with a 1-5 star rating
- **Choose a feedback type**: Bug, Feature, or General
- **Describe the issue** with a text message
- **Attach screenshots** for visual context

Feedback is private and sent directly to the Ultralytics team to help prioritize features and fix issues.

## FAQ

### How do I change my email?

Manage your email addresses directly on the platform:

1. Go to **Settings > Profile**
2. Scroll to the **Emails** section
3. Add a new email, verify it, and set it as primary

### How do I change my password?

If you signed up with email and password, use the password reset flow on the sign-in page. If you signed up with an OAuth provider, manage your password through that provider:

- **Google**: accounts.google.com
- **GitHub**: github.com/settings/security

### Is two-factor authentication available?

Platform currently marks two-factor authentication as **Coming Soon** in its Security card. If you sign in through
Google or GitHub, configure multi-factor authentication with that provider.

### How long until deleted data is removed?

- **Trash items** remain recoverable for 30 days before automatic permanent deletion.
- **Account deletion** removes the sign-in account immediately and queues deletion of associated Platform records and
  stored files. The action cannot be undone.
