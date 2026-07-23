---
plans: [free, pro, enterprise]
title: Billing & Credits
comments: true
description: Manage credits, payments, and subscriptions on Ultralytics Platform with transparent pricing for cloud training and deployments.
keywords: Ultralytics Platform, billing, credits, pricing, subscription, payments, training costs
---

# Billing

[Ultralytics Platform](https://platform.ultralytics.com) uses credits for metered cloud training. Add credits, track
usage, and manage your subscription from `Settings > Billing`.

![Ultralytics Platform Settings Billing Tab Credit Balance And Plan Card](https://cdn.ul.run/i/91dbcbf01b4e198db621c42f1a1dd939.avif)<!-- screenshot -->

## Plans

Choose the plan that fits your needs. Compare plans in `Settings > Plans`:

![Ultralytics Platform Settings Plans Tab Free Pro Enterprise Comparison](https://cdn.ul.run/i/6983407099b65484aa9ebbb8f5843296.avif)<!-- screenshot -->
| Feature | Free | Pro ($29/mo) | Enterprise |
| ---------------------------------------------------------- | ---------- | --------------- | ----------- |
| **Signup Credit** | $5 / $25\* | - | Custom |
| **Monthly Credit** | - | $30/seat/month | Custom |
| **Models** | 100 | 500 | Unlimited |
| **Concurrent Trainings** | 3 | 10 | Unlimited |
| **Storage** | 100 GB | 500 GB | Unlimited |
| **Dataset Upload (ZIP/TAR incl. `.tar.gz`/`.tgz`/NDJSON)** | 10 GB | 20 GB | 50 GB |
| **Deployments** | 3 | 10 | Unlimited |
| **Cloud GPU Types** | 24 | 26 | 26 |
| **Best GPUs (B200, B300)** | - | Yes | Yes |
| **Teams** | - | Up to 5 members | Custom size |
| **SSO / SAML** | - | - | Yes |
| **Enterprise License** | - | - | Yes |
| **License** | AGPL-3.0 | AGPL-3.0 | Enterprise |

\*Free plan: $5 at signup, or $25 if you verify a company/work email address.

### Free Plan

Get started at no cost:

- $5 signup credit ($25 for verified company/work emails)
- Unlimited public and private projects and datasets
- 100 models
- 3 concurrent cloud trainings
- 3 deployments
- 100 GB storage · 10 GB dataset upload limit
- Model export to all 19 formats
- Draw and Smart annotation modes
- 24 cloud GPU types including 5090, H100 & H200 ($0.24–$4.39/hr)
- Community support

!!! tip "Company Email Bonus"

    Sign up with a company email address (not gmail.com, outlook.com, etc.) to receive $25 in signup credits instead of $5.

### Pro Plan

For professionals and small teams ($29/month or $290/year):

- $30/seat/month in credits (recurring)
- 500 models
- 10 concurrent cloud trainings
- 500 GB storage · 20 GB dataset upload limit
- 10 cloud deployments
- [Team collaboration](teams.md) with 4-role RBAC (up to 5 members)
- Access to the best GPUs (B200, B300)
- Priority support

!!! tip "Save with Yearly Billing"

    Choose yearly billing ($290/year) to save 17% compared to monthly billing.

### Enterprise

For organizations with advanced needs:

- Custom credit allocation
- Unlimited models, storage, trainings, and deployments · 50 GB dataset upload limit
- Enterprise License (commercial use, non-AGPL)
- SSO / SAML authentication
- [On Premise](../integrations/on-premise.md) data and compute
- [ISO/IEC 27001:2022 and SOC 2 Type I compliance](https://www.ultralytics.com/security)
- Enterprise SLA guarantees
- Enterprise support

See [Ultralytics Licensing](https://www.ultralytics.com/license) for Enterprise plan details.

## Credits

Credits are the currency for Platform compute services.

### Credit Balance

View your balance in `Settings > Billing`:

![Ultralytics Platform Settings Billing Tab Credit Balance With Topup Button](https://cdn.ul.run/i/fc071d4f78b5eb61d16899a0b83a9075.avif)<!-- screenshot -->
| Balance Type | Description |
| ------------- | ------------------------------------ |
| **Available** | Credits available for cloud training |

### Credit Uses

Credits are consumed by:

| Service            | Rate             |
| ------------------ | ---------------- |
| **Cloud Training** | GPU rate x hours |

## Add Credits

Top up your balance:

1. Go to **Settings > Billing**
2. Click **Top Up**
3. Select or enter amount ($5 - $1,000)
4. Complete payment

![Ultralytics Platform Settings Billing Tab Topup Amount Selection Dialog](https://cdn.ul.run/i/c24e1b95dc28e3f05e4cf904bf478063.avif)<!-- screenshot -->

### Top-Up Presets

| Amount |
| ------ |
| $10    |
| $20    |
| $50    |
| $100   |
| $500   |

Custom amounts between $5 and $1,000 are also supported.

### Auto Top-Up

Enable automatic credit purchases when your balance drops below a threshold:

1. Go to **Settings > Billing**
2. Toggle **Auto Top-Up** on
3. Set **Threshold** (balance level that triggers a top-up)
4. Set **Amount** (credits to purchase when triggered)
5. Click **Save**

Default settings: threshold $20, amount $100.

!!! tip "Reduce Training Interruptions"

    Auto top-up can reduce the chance that a paid cloud training job is stopped for insufficient credits. It requires
    a valid default payment method.

### Payment Methods

Manage payment methods in `Settings > Billing`:

- **Add Card**: Click **Add Card** to add a credit or debit card
- **Set as Default**: Set a default payment method for top-ups and subscriptions
- **Remove**: Remove payment methods you no longer need

### Billing Address

Set a billing address for invoices:

1. Go to **Settings > Billing**
2. Click **Add Address** (or **Edit** if already set)
3. Enter your billing details (name, address, country)
4. Click **Save**

## Training Cost Flow

Cloud training estimates cost before start and charges for actual GPU time used.

```mermaid
flowchart LR
    A[Start Training]:::start --> B[Estimate Cost]:::proc
    B --> C[Run Training]:::proc
    C --> D[Charge Actual Usage]:::out

    classDef start fill:#4CAF50,color:#fff
    classDef proc fill:#2196F3,color:#fff
    classDef out fill:#9C27B0,color:#fff
```

### How It Works

1. **Estimate**: Platform calculates estimated cost based on model size, dataset size, epochs, and GPU
2. **Authorize Start**: Your available balance is checked before training starts
3. **Train**: Job runs on the selected GPU
4. **Charge**: On completion (or cancellation), billing uses actual runtime

!!! note "Actual Usage"

    You pay for actual compute time used, including partial runs that are cancelled.

## Training Costs

Cloud training costs depend on GPU selection:

{% include "macros/platform-gpu-table.md" %}

B200 and B300 GPUs require a [Pro or Enterprise plan](#plans). All other GPUs are available on all plans.

See [Cloud Training](../train/cloud-training.md) for complete GPU options and pricing.

### Cost Calculation

```text
Total Cost = GPU Rate x Training Time (hours)
```

Example: Training for 2.5 hours on RTX PRO 6000

```text
$2.09 x 2.5 = $5.23
```

## Upgrade to Pro

Upgrade for more features and monthly credits:

1. Go to **Settings > Plans**
2. Click **Upgrade to Pro**
3. Choose billing cycle (Monthly or Yearly)
4. Complete checkout

![Ultralytics Platform Settings Plans Tab Upgrade to Pro Dialog](https://cdn.ul.run/i/12e87f262b43116d06b9c8c4371ceb31.avif)<!-- screenshot -->

### Pro Benefits

After upgrading:

- $30/seat/month credit added immediately and each month
- Storage increased to 500 GB · 20 GB dataset upload limit
- 500 models
- 10 concurrent cloud trainings
- 10 cloud deployments
- [Team collaboration](teams.md) (up to 5 members)
- Access to best GPUs (B200, B300)
- Priority support

### Cancel Pro

Cancel anytime from the Plans tab:

1. Go to **Settings > Plans**
2. Click **Cancel Subscription** on the Pro plan card
3. Confirm in the dialog

If you cancel before the end of your billing period, a **Resume Subscription** button appears — click it to undo the cancellation before the period ends.

!!! note "Cancellation Timing"

    Pro features remain active until the end of your current billing period. Monthly credits stop being granted at cancellation.

### Downgrading to Free

When your Pro subscription ends (cancelled or expired), your account reverts to the Free plan. Here's what happens to your existing resources:

| Resource                                                   | What Happens                                                                     |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Models**                                                 | All models preserved. Cannot create new models beyond 100-model limit            |
| **Deployments**                                            | All deployments preserved. Cannot create new beyond 3-deployment limit           |
| **Storage**                                                | All data preserved. Cannot upload new data beyond 100 GB limit                   |
| **Dataset Upload (ZIP/TAR incl. `.tar.gz`/`.tgz`/NDJSON)** | Upload limit reduced from 20 GB to 10 GB per file                                |
| **Credit Balance**                                         | Existing credits preserved and usable                                            |
| **Monthly Credits**                                        | $30/seat/month grants stop immediately                                           |
| **Team Members**                                           | Members notified and lose access to team resources                               |
| **GPU Access**                                             | Standard GPUs remain available. Best GPUs (B200, B300) require Pro or Enterprise |
| **Concurrent Trainings**                                   | Limit reduced from 10 to 3                                                       |

!!! tip "No Data Loss"

    Downgrading does not automatically delete models, datasets, or deployments. The workspace owner retains access,
    while Free-plan creation limits apply and team members lose access to team resources.

## Transaction History

View all transactions in `Settings > Billing`:

![Ultralytics Platform Settings Billing Tab Transaction History Table](https://cdn.ul.run/i/21fee06d70fc825f2cb02f5cad204759.avif)<!-- screenshot -->
| Column | Description |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Date** | Transaction date |
| **Type** | Signup, Purchase, Subscription, Monthly Grant, Training, Refund, Adjustment, Promo, Auto Top-Up, Auto Top-Up Failed, Pro Credit Expiry |
| **Amount** | Transaction value (green for credits, red for charges) |
| **Balance** | Resulting balance after transaction |
| **Details** | Additional context (model link, receipt, period) |

## FAQ

### What happens when I run out of credits?

- **Running paid cloud training**: Stops when metered usage pushes the balance below zero
- **New training**: Cannot start new jobs until balance is positive
- **Deployments**: Continue running regardless of balance

Add credits to restore a positive balance before starting new training jobs. Enable [auto top-up](#auto-top-up) to
reduce the chance of an active job being stopped for insufficient funds.

### How do I get an invoice?

Transaction receipts are available in the transaction history. Click the receipt icon next to any purchase transaction.

### What if training fails?

If a cloud GPU has started, failed, cancelled, completed, and auto-terminated jobs are charged for elapsed GPU time.
Validation or launch failures before cloud compute starts have no GPU usage charge. See
[Cloud Training Billing](../train/cloud-training.md#billing-by-job-status) for the full breakdown.

### Is there a free trial?

The Free plan includes $5 in signup credit, increased to $25 after verifying a company email. No credit card is
required to start.
