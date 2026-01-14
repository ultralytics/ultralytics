---
comments: true
description: Manage credits, payments, and subscriptions on Ultralytics Platform with transparent pricing for cloud training and deployments.
keywords: Ultralytics Platform, billing, credits, pricing, subscription, payments, training costs
---

# Billing

[Ultralytics Platform](https://platform.ultralytics.com) uses a credit-based billing system for cloud training and dedicated endpoints. Add credits, track usage, and manage your subscription.

<!-- Screenshot: platform-billing-overview.avif -->

## Plans

Choose the plan that fits your needs:

<!-- Screenshot: platform-billing-plans.avif -->

| Feature | Free | Pro ($29/mo) | Enterprise |
|---------|------|--------------|------------|
| **Signup Credit** | $5 | $20/month | Custom |
| **Storage** | 100 GB | 500 GB | Unlimited |
| **Private Projects** | 3 | Unlimited | Unlimited |
| **Deployments** | 1 | 5 | Unlimited |
| **Support** | Community | Email | Dedicated |
| **SSO** | - | - | Yes |
| **Audit Logs** | - | - | Yes |

### Free Plan

Get started at no cost:

- $5 signup credit (expires in 30 days)
- 100 GB storage
- 3 private projects
- 1 deployment
- Community support

### Pro Plan

For serious users and small teams:

- $20 monthly credit (expires in 30 days)
- 500 GB storage
- Unlimited private projects
- 5 deployments
- Email support

### Enterprise

For organizations with advanced needs:

- Custom credit allocation
- Unlimited storage
- SSO/SAML integration
- Audit logging
- Dedicated support

Contact [sales@ultralytics.com](mailto:sales@ultralytics.com) for Enterprise pricing.

## Credits

Credits are the currency for Platform compute services.

### Credit Balance

View your balance in Settings > Billing:

<!-- Screenshot: platform-billing-credits.avif -->

| Balance Type | Description |
|--------------|-------------|
| **Available** | Credits ready to use |
| **Reserved** | Held for active training |
| **Expiring** | Credits expiring soon |

### Credit Uses

Credits are consumed by:

| Service | Rate |
|---------|------|
| **Cloud Training** | GPU rate × hours |
| **Dedicated Endpoints** | Compute rate × hours |
| **Model Export** | Fixed per export |

### Credit Expiration

Credits have expiration dates:

- **Signup credits**: 30 days
- **Monthly credits**: 30 days from issue
- **Purchased credits**: Never expire

!!! tip "Use Expiring Credits First"

    Expiring credits are used before non-expiring credits.

## Add Credits

Top up your balance:

1. Go to **Settings > Billing**
2. Click **Add Credits**
3. Select amount ($5 - $1000)
4. Complete payment

<!-- Screenshot: platform-billing-topup.avif -->

### Payment Methods

- Credit/debit cards
- Major payment providers

### Purchase Options

| Amount | Bonus | Total |
|--------|-------|-------|
| $5 | - | $5 |
| $25 | - | $25 |
| $50 | - | $50 |
| $100 | - | $100 |
| $500 | - | $500 |
| $1000 | - | $1000 |

## Training Costs

Cloud training costs depend on GPU selection:

| GPU | Rate/Hour | Typical Job (1h) |
|-----|-----------|------------------|
| RTX 3090 | $0.44 | $0.44 |
| RTX 4090 | $0.74 | $0.74 |
| L40S | $1.14 | $1.14 |
| A100 40GB | $1.29 | $1.29 |
| A100 80GB | $1.99 | $1.99 |
| H100 80GB | $3.99 | $3.99 |

### Cost Calculation

```
Total Cost = GPU Rate × Training Time (hours)
```

Example: Training for 2.5 hours on RTX 4090
```
$0.74 × 2.5 = $1.85
```

### Billing Timing

- **Epochs mode**: Charged after each epoch
- **Timed mode**: Charged at completion
- **Cancelled**: Charged for completed time only

## Upgrade to Pro

Upgrade for more features and monthly credits:

1. Go to **Settings > Billing**
2. Click **Upgrade to Pro**
3. Complete checkout

<!-- Screenshot: platform-billing-upgrade.avif -->

### Pro Benefits

After upgrading:

- $20 credit added immediately
- $20 credit added each month
- Storage increased to 500 GB
- Unlimited private projects
- 5 deployments

### Cancel Pro

Cancel anytime from the billing portal:

1. Click **Manage Subscription**
2. Select **Cancel**
3. Confirm cancellation

!!! note "Cancellation Timing"

    Pro features remain active until the end of your billing period. Monthly credits stop at cancellation.

## Payment History

View all transactions:

<!-- Screenshot: platform-billing-history.avif -->

| Column | Description |
|--------|-------------|
| **Date** | Transaction date |
| **Description** | Credit purchase, training, etc. |
| **Amount** | Transaction value |
| **Balance** | Resulting balance |

### Download Invoice

1. Click transaction in history
2. Select **Download Invoice**
3. PDF invoice downloads

## Billing Portal

Access the billing portal for:

- Update payment method
- Download invoices
- Manage subscription
- View billing history

## FAQ

### What happens when I run out of credits?

- **Active training**: Pauses at epoch end
- **Deployments**: Continue running
- **New training**: Cannot start

Add credits to continue training.

### Are unused credits refundable?

- **Purchased credits**: No refunds
- **Signup/monthly credits**: No refunds (use it or lose it)

### Can I transfer credits?

Credits are not transferable between accounts.

### How do I get an invoice?

1. Go to **Settings > Billing**
2. Click **Billing Portal**
3. Download invoices

### What if training fails?

You're only charged for completed compute time. Failed jobs don't charge for unused time.

### Is there a free trial?

The Free plan includes $5 signup credit - essentially a free trial. No credit card required to start.
