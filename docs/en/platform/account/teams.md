---
plans: [pro, enterprise]
title: Team Management & Roles
comments: true
description: Create and manage teams on Ultralytics Platform with role-based access control, shared resources, and enterprise features for collaborative computer vision workflows.
keywords: Ultralytics Platform, teams, collaboration, enterprise, roles, permissions, RBAC, workspace, team management
---

# Teams

[Ultralytics Platform](https://platform.ultralytics.com) team features enable collaborative computer vision workflows. Create a team workspace to share datasets, projects, models, and deployments with your colleagues using role-based access control.

![Ultralytics Platform Teams Member List With Roles](https://cdn.ul.run/i/e4ad3b29827e62599a84b2b9b19f29b0.avif)<!-- screenshot -->

## Overview

Teams allow multiple users to work together under a shared workspace:

- **Shared Resources**: Datasets, projects, models, and deployments are accessible to all team members
- **Role-Based Access**: Four roles (Owner, Admin, Editor, Viewer) control what each member can do
- **Shared Billing**: Team members share the workspace credit balance and resource limits
- **Seat Management**: Pro teams support up to 5 members, Enterprise teams support custom team sizes

!!! note "Plan Requirement"

    Team workspaces require a [Pro or Enterprise plan](billing.md#plans). You can start team setup before upgrading, but the workspace must be on a Pro or Enterprise plan to use team features.

## Creating a Team

Create a new team workspace:

1. Click on the workspace switcher in the sidebar
2. Click **Create Team** to open the Teams tab in Settings
3. Click **Upgrade to Pro**
4. Select **Team** in the upgrade dialog
5. Enter the team name and unique team URL, choose monthly or yearly billing, and click
   **Create Team & Continue to Checkout**
6. Complete checkout

The workspace switcher shows **Create Team** before you have a team workspace. Once the team is created and upgraded,
you can [invite members](#inviting-members).

!!! note "Team Creation Limit"

    You can create up to 5 teams. To create another, you must first delete or transfer ownership of an existing team.

![Ultralytics Platform Teams Create Team Landing](https://cdn.ul.run/i/c6c47cb9c3ef6e4eeaebf093f3af03bd.avif)<!-- screenshot -->

## Switching Workspaces

Switch between your personal account and team workspaces using the workspace switcher in the sidebar. All teams you belong to appear in the list.

![Ultralytics Platform Teams Workspace Switcher Dropdown](https://cdn.ul.run/i/3e315251784e1246e294515f2ccd984e.avif)<!-- screenshot -->
When you switch to a team workspace, all resources you see and create belong to that team. Your personal workspace resources remain separate.

## Roles and Permissions

Teams use a four-role hierarchy for access control. Each role inherits all permissions from the roles below it.

| Role       | Description                                                            |
| ---------- | ---------------------------------------------------------------------- |
| **Owner**  | Full control, transfer ownership, assign admin role, remove any member |
| **Admin**  | Invite and remove members, manage billing, create and edit all content |
| **Editor** | Create and edit projects, datasets, models, start training, deploy     |
| **Viewer** | Read-only access to all team resources                                 |

!!! note "Single Owner"

    Each team has exactly one owner. To change the owner, transfer ownership from the Teams tab in Settings. Only the owner can assign or remove the admin role.

## Shared Resources

Resources created in a team workspace belong to the team, not the individual. All team members can view projects, datasets, models, and deployments. Editors and above can create and modify resources.

!!! tip "Personal vs Team Resources"

    Resources in your personal workspace are separate from team workspaces. To share a resource, create it while in the team workspace.

## Shared Billing and Limits

Team members share the workspace credit balance and resource limits. All members draw from the same wallet when running cloud training. See [Billing](billing.md#plans) for detailed plan limits.

!!! note "Pro Plan Seat Billing"

    On the Pro plan, each team member is a paid seat at $29/month (or $290/year, a ~17% saving). Monthly credits of $30/seat are added to the team's shared wallet at the start of every billing cycle.

## Inviting Members

Admins and Owners can invite new members to the team:

1. Go to **Settings > Teams**
2. Click **Invite**
3. Enter the invitee's email address
4. Select a role (Admin, Editor, or Viewer)
5. Click **Send Invitation**

![Ultralytics Platform Teams Invite Member Dialog](https://cdn.ul.run/i/ae4da6bdaeb83134b5866800d88bc650.avif)<!-- screenshot -->
The invitee receives an email invitation with a link to accept and join the team. Invitations expire after 14 days.
Once accepted, the team workspace appears in the invitee's workspace switcher. If an invite expires or is lost,
**Resend invite** from the member actions menu to rotate the token and restart the 14-day window, or **Cancel invite**
to free the reserved seat.

!!! note "Admin Invites"

    Only the team Owner can invite members with the Admin role. Admins can invite Editors and Viewers.

The seat limit includes both active members and pending invitations. If you've reached the limit, remove a member or cancel a pending invite before sending a new one.

## Enterprise

Enterprise plans include additional capabilities for organizations with advanced needs, including unlimited resources, commercial licensing, SSO/SAML, and dedicated support. See [Billing > Enterprise](billing.md#enterprise) for the full feature comparison.

!!! warning "License Expiration"

    If an Enterprise license expires, team members lose access to the workspace. The owner can still open the workspace
    to manage renewal. See [Ultralytics Licensing](https://www.ultralytics.com/license) for details.

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

Enterprise pricing and provisioning are handled directly by the Ultralytics team. See [Ultralytics Licensing](https://www.ultralytics.com/license) for plan details.
