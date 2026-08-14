---
plans: [free, pro, enterprise]
comments: true
description: Configure your Ultralytics Platform profile, plan, billing, usage, teams, integrations, and data settings with GDPR-compliant data export and deletion options.
keywords: Ultralytics Platform, settings, profile, preferences, usage, integrations, GDPR, data export, privacy
title: Account Settings
---

# Settings

[Ultralytics Platform](https://platform.ultralytics.com) settings allow you to configure your profile, social links, workspace preferences, and manage your data with GDPR-compliant export and deletion options.

Settings is organized into eight tabs (in order): `Profile`, `API Keys`, `Plans`, `Billing`, `Usage`, `Teams`, `Integrations`, and `Trash`.

Settings is workspace-aware. Switch workspaces in the sidebar and every tab — profile, keys, billing, usage, members,
integrations, and trash — shows data for the workspace you're in. On extra-wide screens a sidebar next to the tabs lists
your five most recent training charges and payments.

## Profile Tab

The `Profile` tab contains your profile information, social links, data region, security, and account management
options.

### Profile Information

Update your profile information:

![Ultralytics Platform Settings Profile Tab Display Name Bio Company Fields](https://cdn.ul.run/i/679e7eb184fe9db51391b1d3548c9aa2.avif)<!-- screenshot -->

| Field                      | Description                                              |
| -------------------------- | -------------------------------------------------------- |
| **Display Name**           | Your public name (required)                              |
| **Username**               | Unique identifier (set at signup, read-only)             |
| **Company / Organization** | Company or organization name                             |
| **Primary Use Case**       | Primary application (select from list)                   |
| **Bio**                    | Short description (minimum 10 characters when filled in) |
| **Profile Icon**           | Avatar with color, initials, or image                    |

In a team workspace the same card edits the workspace profile and icon, and requires the Admin role or higher.

#### Username Rules

- 4-32 characters
- Lowercase letters, numbers, and hyphens
- No leading, trailing, or consecutive hyphens
- Must be unique, and cannot use a name reserved for Platform routes (for example `settings`, `explore`, or `docs`)

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
3. Wait for the **Saved** indicator next to the Profile heading. Changes save automatically about a second after you
   stop typing — there is no Save button.

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

Manage email addresses linked to your account in the `Profile` tab. The Emails card appears on personal accounts only —
team workspaces have no separate email list.

![Ultralytics Platform Settings Profile Tab Emails Section](https://cdn.ul.run/i/f09baef9e8e5b2ceec2fa688b597eeeb.avif)<!-- screenshot -->

| Action             | Description                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| **Add**            | Add a new email address, which immediately sends a 6-digit code            |
| **Verify**         | Enter the 6-digit code sent to the address (use **Resend** to get another) |
| **Set as primary** | Set a verified email as your primary address                               |
| **Remove**         | Remove a non-primary email address                                         |

Each address is labelled with **Primary**, **Verified** or **Unverified**, and **Company** badges.

!!! note "Primary Email"

    Your primary email is used for notifications and account recovery. Only verified emails can be set as primary.

!!! tip "Company Email Bonus"

    Verifying a company or work email address (not gmail.com, outlook.com, and similar consumer domains) adds the
    remaining $20 of the $25 signup credit to your balance. The bonus is granted once per account. See
    [Billing](billing.md#free-plan).

### Data Region

View your data region on the `Profile` tab:

{% include "macros/platform-data-regions.md" %}

!!! note "Data Region"

    Your data region is selected during onboarding and cannot be changed yourself. It applies to datasets, models, and
    managed training data. Dedicated deployments use the region selected when each endpoint is created. Contact support
    to request an account data-region change.

### Security

The `Profile` tab includes a Security card:

- **Two-Factor Authentication**: marked **Coming Soon** in Platform settings
- **Connected Accounts**: shows the OAuth provider linked to your sign-in

### Storage Usage

Monitor your storage consumption on the [`Usage` tab](#usage-tab) and the **Home** page:

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

### GDPR Compliance

Ultralytics Platform supports GDPR rights:

#### Data Export

Download all your data:

1. Go to **Settings > Profile**
2. Scroll to the **Export Your Data** card
3. Click **Export All Data**
4. An asynchronous export job runs in the background; a **Download Export** link appears on the same card when the job
   completes (the link is valid for 60 minutes)

The export is a single JSON file of metadata — images and model weights are not included. It contains:

- Profile information
- Storage usage records
- Project metadata
- Dataset metadata
- Model metadata
- Full activity history
- API key metadata: key ID, name, and prefix only (key values are never exported)

#### Account Deletion

Permanently delete your account:

1. Go to **Settings > Profile**
2. Scroll to the **Delete My Account** card
3. Click **Delete My Account**
4. Type `DELETE` in the confirmation field, then confirm

!!! warning "Irreversible Action"

    Account deletion is permanent. Your sign-in account is removed immediately, and the associated Platform data and
    stored files are deleted.

If you own any team workspaces, deletion is refused until you delete those teams or
[transfer ownership](teams.md#roles-and-permissions) to someone else.

##### What's Deleted

- All projects and trained models
- All datasets and images
- All API keys and credentials
- All activity history
- Credit balance

#### Team Deletion

In a team workspace the same card reads **Delete Team** and is visible only to the team owner. Remove every non-owner
member first — the confirmation refuses while other members hold seats — then type `DELETE` to confirm. Team resources
are deleted automatically.

## API Keys Tab

The `API Keys` tab lets you create and manage API keys for remote training and inference. Only the workspace owner can
create, view, or revoke keys — a workspace key acts as the owner. See [API Keys](api-keys.md) for full documentation.

## Plans Tab

The `Plans` tab lets you compare available plans and upgrade or downgrade your subscription.

![Ultralytics Platform Settings Plans Tab Free Pro Enterprise Comparison](https://cdn.ul.run/i/4687f31bbcab35be3b474784751759e5.avif)<!-- screenshot -->

{% include "macros/platform-plan-comparison.md" %}

From this tab you can:

- **Compare features** across Free, Pro, and Enterprise tiers, in monthly or yearly pricing
- **Upgrade to Pro** to unlock more storage, models, team collaboration, cloud storage datasets, and B200/B300 GPU access
- **Cancel or resume** a Pro subscription — the card shows the date it cancels on
- **Request an Enterprise demo**, including SSO/SAML and commercial licensing — see [Ultralytics Licensing](https://www.ultralytics.com/license)

In a team workspace, plan changes require the Admin role or higher; other members see **Admin Required**.

See [Billing](billing.md) for detailed plan information, pricing, and upgrade instructions.

## Billing Tab

The `Billing` tab is where you manage credits, payment methods, and review transaction history. Credits pay for
metered cloud training.

![Ultralytics Platform Settings Billing Tab Credit Balance And Plan Card](https://cdn.ul.run/i/8deb4532660afd808780789930cfbeb6.avif)<!-- screenshot -->
From this tab you can:

- **View your current plan** and cancel, resume, or upgrade it from the plan card
- **View credit balance** and monitor remaining credits
- **Add credits** via manual top-up (presets from $10–$500 or custom amounts of $5–$1,000)
- **Enable auto top-up** to automatically add credits when your balance falls below a threshold, reducing the chance of
  training interruption
- **Manage payment methods** and set the default card used for top-ups and renewals
- **Set a billing address** used on invoices
- **Review transaction history** with search, a date-range filter, and CSV/JSON export

!!! tip "Training Costs"

    Before each training run, the platform estimates the cost based on your selected GPU, dataset size, and epochs.
    The estimate is a balance check, not a credit reservation; GPU usage is metered against your balance while the run
    is in progress and settled when it ends.

In a team workspace, billing actions require the Admin role or higher. Other members can view the balance but see no
top-up, card, or address controls.

See [Billing](billing.md) for full documentation on credits, payment, and plan management.

## Usage Tab

The `Usage` tab charts credit spend and storage for the active workspace.

- **Stat cards**: current balance, total spend, usage events, and average cost per event for the selected period
- **Spend over time**: a daily or monthly bar chart of settled usage spend
- **Group by**: no grouping, spend category, GPU type, user, API key, or dataset — applied to both the bar chart and
  the share-of-spend pie chart
- **Date range**: any custom range, defaulting to the last 30 days
- **Storage card**: the same storage breakdown shown on the **Home** page (see [Storage Usage](#storage-usage))

Grouping by user or API key makes it easy to see which team member or automation is consuming a shared team balance.

## Teams Tab

The `Teams` tab lets you manage workspace members, roles, and invitations. Member seats are available on
[Pro and Enterprise plans](billing.md#plans) — on the Free plan the tab shows the roles reference and an
**Upgrade to Pro** button instead of an invite control.

![Ultralytics Platform Teams Member List With Roles](https://cdn.ul.run/i/b680a4b6f2db15b3a34bc19adab8515e.avif)<!-- screenshot -->

The member card header shows the workspace name, plan badge, your own role, and a seat summary such as
`3 of 5 seats used · 2 available (includes 1 pending invite)`.

### Roles and Permissions

{% include "macros/platform-team-roles.md" %}

The tab also renders a full permission matrix covering resource access, member management, billing, and ownership
transfer. See [Teams](teams.md#roles-and-permissions) for the same matrix.

### Manage Members

Owners and admins can manage the team from the actions menu on each row:

- **Invite members** via email (invites expire after 14 days; pending invites reserve a seat)
- **Change roles**: **Set as Admin**, **Set as Editor**, or **Set as Viewer** — only the owner can assign or remove the
  Admin role, and no one can act on a member at or above their own role
- **Remove from team**: removes the member immediately
- **Leave team**: any member can leave their own team, which returns them to their personal workspace
- **Cancel invite** / **Resend invite**: cancel a pending invitation to free its seat, or rotate its token and restart
  the 14-day window
- **Transfer ownership**: transfer workspace ownership to another member (Owner only; you become an Admin)

### Shared Resources

All resources created in a team workspace belong to the team, not individual members. Team members share:

- **Datasets, projects, and models** — accessible by all members based on their role
- **Credit balance** — shared across team members for cloud training
- **Storage and resource limits** — counted at the team level

!!! note "Team Billing"

    On Pro plans, each member occupies a paid seat at $29/month or $290/year. Adding a member mid-cycle charges a
    prorated amount for the rest of the period, unless a seat you already paid for this period is vacant. The team
    credit balance is shared across all members.

See [Teams](teams.md) for full documentation on team creation, switching workspaces, and enterprise features.

## Integrations Tab

The `Integrations` tab is a searchable list, grouped into three categories, of external services you can connect to the
active workspace.

**Infrastructure**

- **On Premise** — run Platform training on your own hardware and keep dataset files on your host. See [On Premise](../integrations/on-premise.md).
- **Amazon S3** — use datasets stored in S3 without uploading a copy. See [Amazon S3](../integrations/amazon-s3.md).
- **Google Cloud Storage** — use datasets stored in GCS without uploading a copy. See [Google Cloud Storage](../integrations/google-cloud-storage.md).
- **Azure Blob Storage** — use datasets stored in Azure without uploading a copy. See [Azure Blob Storage](../integrations/azure-blob-storage.md).

**Notifications**

- **Slack** — send selected training, export, and deployment results to a [Slack channel](../integrations/slack.md).

**Imports**

- **Roboflow** — preview and import annotated datasets from a [Roboflow](../integrations/roboflow.md) workspace using a Roboflow API key.
- **Labelbox** — upload a [Labelbox](../integrations/labelbox.md) NDJSON export directly, with no key to connect.
- **LabelMe** — export offline annotations to YOLO format and upload the archive using the [LabelMe guide](../integrations/labelme.md).
- **CVAT** — marked **Coming Soon**; upload a CVAT Ultralytics YOLO export today. See [CVAT](../integrations/cvat.md).
- **Label Studio** — marked **Coming Soon**; upload a Label Studio "YOLO with Images" export today. See [Label Studio](../integrations/label-studio.md).

!!! note "Cloud Storage Requires a Paid Plan"

    Google Cloud Storage, Amazon S3, and Azure Blob Storage datasets require a Pro or Enterprise plan. On Premise is an
    Enterprise capability. Import integrations and Slack are available on every plan.

See [Integrations](../integrations/index.md) for the full list of supported services.

## Trash Tab

The `Trash` tab shows all deleted items and lets you restore or permanently remove them. Deleted items follow a 30-day soft delete policy before automatic permanent deletion.

![Ultralytics Platform Settings Trash Tab With Items And Storage Treemap](https://cdn.ul.run/i/1fda3fe06d0527f579017b71afa6a2ff.avif)<!-- screenshot -->
From this tab you can:

- **Browse deleted items** filtered by type (All, Datasets, Projects, Models) and search them by name
- **View the storage treemap** to see the relative size of trashed items
- **Restore items** to their original location with all data intact
- **Permanently delete** individual items, or use the trash icon in the header to empty the whole trash at once

Restoring and deleting require the Editor role or higher in a team workspace.

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
3. Add a new email, enter the 6-digit code sent to it, then click **Set as primary**

### How do I change my password?

If you signed up with email and password, use the password reset flow on the sign-in page. If you signed up with an OAuth provider, manage your password through that provider:

- **Google**: accounts.google.com
- **GitHub**: github.com/settings/security

### Is two-factor authentication available?

Platform currently marks two-factor authentication as **Coming Soon** in its Security card. If you sign in through
Google or GitHub, configure multi-factor authentication with that provider.

### How long until deleted data is removed?

- **Trash items** remain recoverable for 30 days before automatic permanent deletion.
- **Account deletion** removes the sign-in account immediately, along with associated Platform records and stored
  files. The action cannot be undone.
