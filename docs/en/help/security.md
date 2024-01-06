---
description: Explore Ultralytics' comprehensive security strategies safeguarding user data and systems. Learn about our diverse security tools, including Snyk, GitHub CodeQL, and Dependabot Alerts.
keywords: Ultralytics, Comprehensive Security, user data protection, Snyk, GitHub CodeQL, Dependabot, vulnerability management, coding security practices
---

# Ultralytics Security Policy

At [Ultralytics](https://ultralytics.com), the security of our users' data and systems is of utmost importance. To ensure the safety and security of our [open-source projects](https://github.com/ultralytics), we have implemented several measures to detect and prevent security vulnerabilities.

## Snyk Scanning

We utilize [Snyk](https://snyk.io/advisor/python/ultralytics) to conduct comprehensive security scans on Ultralytics repositories. Snyk's robust scanning capabilities extend beyond dependency checks; it also examines our code and Dockerfiles for various vulnerabilities. By identifying and addressing these issues proactively, we ensure a higher level of security and reliability for our users.

[![ultralytics](https://snyk.io/advisor/python/ultralytics/badge.svg)](https://snyk.io/advisor/python/ultralytics)

## GitHub CodeQL Scanning

Our security strategy includes GitHub's [CodeQL](https://docs.github.com/en/code-security/code-scanning/automatically-scanning-your-code-for-vulnerabilities-and-errors/about-code-scanning-with-codeql) scanning. CodeQL delves deep into our codebase, identifying complex vulnerabilities like SQL injection and XSS by analyzing the code's semantic structure. This advanced level of analysis ensures early detection and resolution of potential security risks.

[![CodeQL](https://github.com/ultralytics/ultralytics/actions/workflows/codeql.yaml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/codeql.yaml)

## GitHub Dependabot Alerts

[Dependabot](https://docs.github.com/en/code-security/dependabot) is integrated into our workflow to monitor dependencies for known vulnerabilities. When a vulnerability is identified in one of our dependencies, Dependabot alerts us, allowing for swift and informed remediation actions.

## GitHub Secret Scanning Alerts

We employ GitHub [secret scanning](https://docs.github.com/en/code-security/secret-scanning/managing-alerts-from-secret-scanning) alerts to detect sensitive data, such as credentials and private keys, accidentally pushed to our repositories. This early detection mechanism helps prevent potential security breaches and data exposures.

## Private Vulnerability Reporting

We enable private vulnerability reporting, allowing users to discreetly report potential security issues. This approach facilitates responsible disclosure, ensuring vulnerabilities are handled securely and efficiently.

If you suspect or discover a security vulnerability in any of our repositories, please let us know immediately. You can reach out to us directly via our [contact form](https://ultralytics.com/contact) or via [security@ultralytics.com](mailto:security@ultralytics.com). Our security team will investigate and respond as soon as possible.

We appreciate your help in keeping all Ultralytics open-source projects secure and safe for everyone 🙏.
