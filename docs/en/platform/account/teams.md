---
plans: [pro, enterprise]
title: Team Management & Roles
comments: true
description: Create and manage teams on Ultralytics Platform with role-based access control, shared resources, and enterprise features for collaborative computer vision workflows.
keywords: Ultralytics Platform, teams, collaboration, enterprise, roles, permissions, RBAC, workspace, team management
---

# Teams

[Ultralytics Platform](https://platform.ultralytics.com) team features enable collaborative computer vision workflows. Create a team workspace to share datasets, projects, models, and deployments with your colleagues using role-based access control.

![Ultralytics Platform Teams Member List With Roles](https://cdn.ul.run/i/b680a4b6f2db15b3a34bc19adab8515e.avif)<!-- screenshot -->

## Overview

Teams allow multiple users to work together under a shared workspace:

- **Shared Resources**: Datasets, projects, models, and deployments are accessible to all team members
- **Role-Based Access**: Four roles (Owner, Admin, Editor, Viewer) control what each member can do
- **Shared Billing**: Team members share the workspace credit balance and resource limits
- **Seat Management**: Pro supports up to 5 members, Enterprise defaults to 50 seats with custom sizes available

!!! note "Plan Requirement"

    Member seats require a [Pro or Enterprise plan](billing.md#plans). On the Free plan the Teams tab shows the roles
    reference and an **Upgrade to Pro** button instead of an invite control.

## Personal Workspaces and Team Workspaces

There are two ways to collaborate, and the Teams tab handles both:

- **Invite members into your personal workspace.** Upgrading your own account to Pro turns your personal workspace into
  a team of up to 5 people, listed under `<Your Name>'s Team` on the Teams tab. Resources stay under your username.
- **Create a separate team workspace.** A team workspace has its own username, its own URL namespace, and its own
  credit balance and storage quota, kept entirely separate from your personal account.

Pick a separate team workspace when the work should live under a shared identity, or when you want more than one group
with different members.

## Creating a Team

Create a new team workspace:

1. Click on the workspace switcher in the sidebar
2. Click **Create Team** to open the Teams tab in Settings
3. Click **Upgrade to Pro**
4. Select **Team** in the upgrade dialog
5. Enter the team name and unique team URL, choose monthly or yearly billing, and click
   **Create Team & Continue to Checkout**
6. Complete checkout

The workspace switcher shows **Create Team** before you have a team workspace. New team workspaces inherit your
[data region](settings.md#data-region) and start with the same sample projects and datasets as a new account. Once the
team is created and upgraded, you can [invite members](#inviting-members).

The team URL follows the same rules as a username: 4-32 characters, lowercase letters, numbers, and non-repeating
hyphens, and it must not already be taken.

!!! note "Team Creation Limit"

    You can own up to 5 teams. To create another, you must first delete or transfer ownership of an existing team.

![Ultralytics Platform Teams Create Team Landing](https://cdn.ul.run/i/f70ce48d7c555aaff12423337446d3ae.avif)<!-- screenshot -->

## Switching Workspaces

Switch between your personal account and team workspaces using the workspace switcher in the sidebar. All teams you belong to appear in the list.

![Ultralytics Platform Teams Workspace Switcher Dropdown](https://cdn.ul.run/i/b5d3298767cc96743a74133b7c92fe6b.avif)<!-- screenshot -->
When you switch to a team workspace, all resources you see and create belong to that team. Your personal workspace resources remain separate.

## Roles and Permissions

Teams use a four-role hierarchy for access control. Each role inherits all permissions from the roles below it.

{% include "macros/platform-team-roles.md" %}

The Teams tab in Settings shows the same permission matrix rendered below:

| Feature                    | Owner | Admin | Editor | Viewer |
| -------------------------- | ----- | ----- | ------ | ------ |
| View public resources      | Yes   | Yes   | Yes    | Yes    |
| View private resources     | Yes   | Yes   | Yes    | Yes    |
| Create & edit resources    | Yes   | Yes   | Yes    | No     |
| Delete resources           | Yes   | Yes   | Yes    | No     |
| Upload data & train models | Yes   | Yes   | Yes    | No     |
| Export & download models   | Yes   | Yes   | Yes    | No     |
| Manage deployments         | Yes   | Yes   | Yes    | No     |
| Manage viewers & editors   | Yes   | Yes   | No     | No     |
| Change/remove admins       | Yes   | No    | No     | No     |
| Billing & plan changes     | Yes   | Yes   | No     | No     |
| Transfer ownership         | Yes   | No    | No     | No     |

No one can change or remove a member at or above their own role, so an admin cannot demote another admin.

!!! note "Single Owner"

    Each team has exactly one owner. To change the owner, transfer ownership from the Teams tab in Settings — you are
    demoted to Admin and the change cannot be undone by you. Only the owner can assign or remove the Admin role.

!!! warning "API Keys Are Owner-Only"

    Workspace [API keys](api-keys.md) authenticate as the workspace owner, so only the owner can create, view, or
    revoke them — including admins.

## Shared Resources

Resources created in a team workspace belong to the team, not the individual. All team members can view projects, datasets, models, and deployments. Editors and above can create and modify resources.

!!! tip "Personal vs Team Resources"

    Resources in your personal workspace are separate from team workspaces. To share a resource, create it while in the team workspace.

## Shared Billing and Limits

Team members share the workspace credit balance and resource limits. All members draw from the same wallet when running cloud training. See [Billing](billing.md#plans) for detailed plan limits.

The [Usage tab](settings.md#usage-tab) can group spend by user or API key, which is how you see where a shared balance
is going.

!!! note "Pro Plan Seat Billing"

    On the Pro plan, each member occupies a paid seat at $29/month (or $290/year, a ~17% saving). Monthly credits of
    $30/seat are added to the team's shared wallet every calendar month, and any unused portion expires at the next
    cycle boundary.

## Inviting Members

Admins and Owners can invite new members to the team:

1. Go to **Settings > Teams**
2. Click **Invite**
3. Enter the invitee's email address
4. Select a role (Admin, Editor, or Viewer)
5. Click **Continue**, review the seat cost, then click **Confirm & invite**

![Ultralytics Platform Teams Invite Member Dialog](https://cdn.ul.run/i/4f3fbc7dc21172bf12cd404a5ca0b863.avif)<!-- screenshot -->

What happens next depends on whether the invitee already uses the Platform:

- **Existing Platform account**: they're added to the team as soon as the seat charge succeeds, with no invitation to
  accept. The team appears in their workspace switcher immediately.
- **New to the Platform**: they receive an email invitation with a signup link. The seat is reserved as a **Pending**
  row until they accept.

Invitations expire after 14 days, and the owner is emailed when one lapses. Use **Resend invite** from the member
actions menu to rotate the token and restart the 14-day window, or **Cancel invite** to free the reserved seat.

### Seat Charges

On a Pro plan the seat is charged before the member is added. The confirmation step shows both the prorated amount due
today and the new recurring total. If a seat you already paid for this period is vacant, the new member reuses it for
the rest of the period at no extra cost. A declined card leaves the team unchanged.

!!! note "Admin Invites"

    Only the team Owner can invite members with the Admin role. Admins can invite Editors and Viewers.

The seat limit includes both active members and pending invitations. If you've reached the limit, remove a member or cancel a pending invite before sending a new one.

## Leaving and Removing Members

Any member can leave a team from their own row in the member table (**Leave team**), which returns them to their
personal workspace. Owners and admins can remove members below their own role with **Remove from team**. Both take
effect immediately, and the freed seat stays paid for the remainder of the billing period.

An owner cannot leave or be removed. Transfer ownership first, then leave.

## Enterprise

Enterprise plans include additional capabilities for organizations with advanced needs, including unlimited resources, commercial licensing, SSO/SAML, and dedicated support. See [Billing > Enterprise](billing.md#enterprise) for the full feature comparison.

!!! warning "License Expiration"

    If an Enterprise license expires, team members lose access to the workspace. The owner can still open the workspace
    to manage renewal and sees an **Enterprise license expired** banner on the Billing and Teams tabs with a renewal
    contact. See [Ultralytics Licensing](https://www.ultralytics.com/license) for details.

### Getting Started with Enterprise

Enterprise plans are provisioned by the Ultralytics team. See [Ultralytics Licensing](https://www.ultralytics.com/license) for plan details. Once your enterprise configuration is set up, you'll receive a provisioning invite to accept as the team Owner, after which you can invite your team members.

## FAQ

### Can I be a member of multiple teams?

Yes, you can belong to multiple teams simultaneously. Use the workspace switcher to move between them. Your role may differ in each team.

### What happens to team resources if I leave?

Resources you created in the team workspace stay with the team. They are not deleted or transferred to your personal account.

### How are credits shared in a team?

All team members share a single credit balance. The Owner and Admins can top up credits and manage billing from [Settings > Billing](billing.md).

### How do I upgrade from Pro to Enterprise?

Enterprise pricing and provisioning are handled directly by the Ultralytics team. Click **Request Enterprise Demo** on
the Enterprise card in `Settings > Plans`, or see [Ultralytics Licensing](https://www.ultralytics.com/license) for plan
details.

### What happens to my team if the plan downgrades to Free?

All non-owner members are removed and notified by email, and pending invitations are cancelled. The owner keeps access
to the workspace and its resources, so re-upgrading and re-inviting restores the team. See
[Downgrading to Free](billing.md#downgrading-to-free).

### Can I delete a team workspace?

Yes. The owner can delete a team from the **Delete Team** card in `Settings > Profile` while that workspace is active.
Remove every other member first, then type `DELETE` to confirm. All team resources are deleted permanently.
