---
plans: [free, pro, enterprise]
title: Billing & Credits
comments: true
description: Manage credits, payments, and subscriptions on Ultralytics Platform with transparent cloud training pricing.
keywords: Ultralytics Platform, billing, credits, pricing, subscription, payments, training costs
---

# Billing

[Ultralytics Platform](https://platform.ultralytics.com) uses credits for metered cloud training. Add credits, track
usage, and manage your subscription from `Settings > Billing`.

![Ultralytics Platform Settings Billing Tab Credit Balance And Plan Card](https://cdn.ul.run/i/8deb4532660afd808780789930cfbeb6.avif)<!-- screenshot -->

## Plans

Choose the plan that fits your needs. Compare plans in `Settings > Plans`:

![Ultralytics Platform Settings Plans Tab Free Pro Enterprise Comparison](https://cdn.ul.run/i/4687f31bbcab35be3b474784751759e5.avif)<!-- screenshot -->

{% include "macros/platform-plan-comparison.md" %}

### Free Plan

Get started at no cost:

- $5 signup credit ($25 total after verifying a company/work email)
- Unlimited public and private projects and datasets
- 100 models
- 3 concurrent cloud training jobs
- 3 cloud deployments
- 100 GB storage · 10 GB dataset upload limit
- Model export to all 20 formats
- Manual, SAM3, and YOLO Smart annotation
- 24 cloud GPU types including 5090, H100 & H200 ($0.24–$4.39/hr)
- Community support

!!! tip "Company Email Bonus"

    Sign up with, or later add and verify, a company email address (not gmail.com, outlook.com, and similar consumer
    domains) to top your signup credit up from $5 to $25. The bonus is granted once per account — add the address in
    [`Settings > Profile`](settings.md#emails).

### Pro Plan

For professionals and small teams ($29 per seat/month or $290 per seat/year):

- $30/seat/month in credits, granted every calendar month
- 500 models
- 10 concurrent cloud training jobs
- 500 GB storage · 20 GB dataset upload limit
- 10 cloud deployments
- [Google Cloud Storage, Amazon S3, and Azure Blob Storage datasets](../integrations/index.md)
- [Team collaboration](teams.md) with 4-role RBAC (up to 5 members)
- Access to the best GPUs (B200, B300) — 26 GPU types in total
- Full monitoring dashboard
- Priority support

!!! tip "Save with Yearly Billing"

    Choose yearly billing ($290 per seat/year) to save about 17% compared to monthly billing.

### Enterprise

For organizations with advanced needs:

- Custom credit allocation
- Unlimited models, storage, training jobs, and deployments · 50 GB dataset upload limit
- Custom team capacity (50 seats by default)
- Enterprise License (commercial use, non-AGPL)
- SSO / SAML authentication and custom role-based access controls
- [On Premise](../integrations/on-premise.md) data and compute
- [ISO/IEC 27001:2022 and SOC 2 Type I compliance](https://www.ultralytics.com/security)
- Enterprise SLA guarantees
- Dedicated onboarding and support

Enterprise plans are provisioned by the Ultralytics team. Click **Request Enterprise Demo** on the Enterprise card in
`Settings > Plans` to get in touch, or see [Ultralytics Licensing](https://www.ultralytics.com/license) for plan
details.

## Credits

Credits are the currency for Platform compute services.

### Credit Balance

View your balance in `Settings > Billing`:

![Ultralytics Platform Settings Billing Tab Credit Balance With Topup Button](https://cdn.ul.run/i/e7db27e18b14d2a8d2672966455c965f.avif)<!-- screenshot -->

| Balance Type  | Description                          |
| ------------- | ------------------------------------ |
| **Available** | Credits available for cloud training |

The balance is a single wallet per workspace. It can go negative if a training run's final settlement exceeds the
credits on hand, and it is shown in red when it does.

### Credit Uses

Credits are consumed by:

| Service            | Rate             |
| ------------------ | ---------------- |
| **Cloud Training** | GPU rate x hours |

!!! note "Monthly Credits Don't Roll Over"

    Pro monthly grants are use-it-or-lose-it. At each billing cycle boundary — and when a Pro plan ends — any unused
    portion of the grants issued that cycle is removed with a **Credits Expired** transaction. Credits you purchased
    yourself are never expired.

## Add Credits

Top up your balance:

1. Go to **Settings > Billing**
2. Click **Top Up**
3. Pick a preset amount, or choose **Custom amount…** and enter $5 – $1,000
4. Complete payment in the secure checkout tab that opens

Your balance updates automatically once the payment succeeds — you don't need to reload the page.

![Ultralytics Platform Settings Billing Tab Topup Amount Selection Dialog](https://cdn.ul.run/i/41dea87cf64f1a2c6366f0707b7ab3fa.avif)<!-- screenshot -->

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
4. Set **Amount** (credits to purchase when triggered, $5 – $5,000)
5. Click **Save**

Default settings: threshold $20, amount $100.

Auto top-up is evaluated whenever a charge lowers your balance, including mid-run training charges, and charges your
default payment method. Top-ups are briefly rate-limited to prevent duplicate charges. If the card is declined, a
**Auto Top-Up Failed** row is added to your transaction history with the reason, and no credits are added.

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

Cloud training estimates cost before start, meters GPU time while the run is in progress, and settles the remainder
when the run ends.

```mermaid
flowchart LR
    A[Start Training]:::start --> B[Estimate Cost]:::proc
    B --> C[Meter GPU Usage]:::proc
    C --> D[Settle at Terminal State]:::out

    classDef start fill:#4CAF50,color:#fff
    classDef proc fill:#2196F3,color:#fff
    classDef out fill:#9C27B0,color:#fff
```

### How It Works

1. **Estimate**: Platform calculates estimated cost based on model size, dataset size, epochs, and GPU
2. **Authorize Start**: Your available balance is checked before training starts
3. **Meter**: As the job runs, accrued GPU time is debited from your balance in steps — so a long run's cost appears
   in your balance while it is still training
4. **Settle**: When the job reaches a terminal state (completed, failed, cancelled, or auto-terminated), the remaining
   tail is debited and a single **Training** transaction is written for the whole run

!!! note "Actual Usage"

    You pay for actual compute time used, including partial runs that are cancelled. If metering pushes your balance
    below zero, your running cloud jobs are shut down — after any [auto top-up](#auto-top-up) has had a chance to
    land.

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
3. Choose **Personal** to upgrade your own workspace, or **Team** to create a new team workspace at the same time
4. Choose billing cycle (Monthly or Yearly)
5. Complete checkout

![Ultralytics Platform Settings Plans Tab Upgrade to Pro Dialog](https://cdn.ul.run/i/c5c4e48ad1cb59d059bc5112c1c6ed2f.avif)<!-- screenshot -->

### Pro Benefits

After upgrading:

- $30/seat/month credit added immediately and every calendar month after
- Storage increased to 500 GB · 20 GB dataset upload limit
- 500 models
- 10 concurrent cloud training jobs
- 10 cloud deployments
- [Google Cloud Storage, Amazon S3, and Azure Blob Storage datasets](../integrations/index.md)
- [Team collaboration](teams.md) (up to 5 members, including you)
- Access to best GPUs (B200, B300)
- Full monitoring dashboard
- Priority support

### Renewals

Ultralytics bills the Pro plan directly from your saved payment method. At the end of each billing period,
your default payment method is charged for the seats currently in use. If the charge fails you're emailed, and the
platform retries on the following days — after three failed attempts the workspace is downgraded to Free.

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
| **Purchased Credits**                                      | Preserved and usable                                                             |
| **Unused Monthly Credits**                                 | Expired at the downgrade, recorded as a **Credits Expired** transaction          |
| **Monthly Credits**                                        | $30/seat/month grants stop                                                       |
| **Team Members**                                           | All non-owner members removed and notified by email; pending invites cancelled   |
| **Cloud Storage Datasets**                                 | GCS, S3, and Azure Blob Storage datasets require Pro or Enterprise               |
| **GPU Access**                                             | Standard GPUs remain available. Best GPUs (B200, B300) require Pro or Enterprise |
| **Concurrent Trainings**                                   | Limit reduced from 10 to 3                                                       |

!!! tip "No Data Loss"

    Downgrading does not delete models, datasets, or deployments. The workspace owner retains access, while Free-plan
    creation limits apply and team members are removed from the workspace.

## Transaction History

View all transactions in `Settings > Billing`. The table covers the selected date range (last 30 days by default),
supports free-text search across every field, and exports to CSV or JSON from the menu in the card header.

![Ultralytics Platform Settings Billing Tab Transaction History Table](https://cdn.ul.run/i/ecd72fd02c557801a298593d0f8ad2bb.avif)<!-- screenshot -->

| Column          | Description                                                                           |
| --------------- | ------------------------------------------------------------------------------------- |
| **Transaction** | Transaction type (see below)                                                          |
| **Details**     | Additional context — a link to the charged model, a receipt link, or a billing period |
| **Time**        | When the transaction was recorded                                                     |
| **User**        | Which member incurred the charge (team workspaces with more than one actor)           |
| **GPU Type**    | GPU used, on rows where one was recorded                                              |
| **Amount**      | Transaction value (green for credits, red for charges)                                |
| **Balance**     | Resulting balance after transaction                                                   |

### Transaction Types

| Type                   | Meaning                                                       |
| ---------------------- | ------------------------------------------------------------- |
| **Signup Bonus**       | Signup credit, including the company-email top-up             |
| **Credit Purchase**    | Manual top-up                                                 |
| **Auto Top-Up**        | Automatic top-up triggered by your threshold                  |
| **Auto Top-Up Failed** | An automatic top-up was declined; no credits were added       |
| **Subscription**       | Pro subscription or seat charge                               |
| **Monthly Grant**      | $30/seat monthly Pro credit                                   |
| **Credits Expired**    | Unused monthly grant removed at a cycle boundary or downgrade |
| **Training**           | Settled cost of one cloud training run                        |
| **Refund**             | Refunded charge                                               |
| **Promo Bonus**        | Credit from a promotional code                                |
| **Adjustment**         | Manual correction applied by Ultralytics                      |

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

### Do my monthly Pro credits roll over?

No. Each billing cycle boundary expires whatever is left of that cycle's $30/seat grants before issuing the next one,
recorded as a **Credits Expired** transaction. Credits you bought with a top-up are never expired, so a balance made up
of purchased credits carries over indefinitely.

### What happens when I add a team member mid-cycle?

The invite dialog shows the prorated amount due today and the new recurring total before you confirm. If a seat you
already paid for this period is vacant — because a member left or an invite was cancelled — the new member reuses it at
no extra charge for the rest of the period. See [Teams](teams.md#inviting-members).

### Is there a free trial?

The Free plan includes $5 in signup credit, increased to $25 after verifying a company email. No credit card is
required to start.
