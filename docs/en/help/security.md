---
description: Learn about the security measures and tools used by Ultralytics to protect user data and systems. Discover how we address vulnerabilities with Snyk, CodeQL, Dependabot, and more.
keywords: Ultralytics security policy, Snyk scanning, CodeQL scanning, Dependabot alerts, secret scanning, vulnerability reporting, GitHub security, open-source security
---

# Ultralytics Security Policy

At [Ultralytics](https://www.ultralytics.com/), the security of our users' data and systems is of utmost importance. To ensure the safety and security of our [open-source projects](https://github.com/ultralytics), we have implemented several measures to detect and prevent security vulnerabilities.

## Snyk Scanning

We utilize [Snyk](https://snyk.io/advisor/python/ultralytics) to conduct comprehensive security scans on Ultralytics repositories. Snyk's robust scanning capabilities extend beyond dependency checks; it also examines our code and Dockerfiles for various vulnerabilities. By identifying and addressing these issues proactively, we ensure a higher level of security and reliability for our users.

[![ultralytics](https://snyk.io/advisor/python/ultralytics/badge.svg)](https://snyk.io/advisor/python/ultralytics)

## GitHub CodeQL Scanning

Our security strategy includes GitHub's [CodeQL](https://docs.github.com/en/code-security/code-scanning/introduction-to-code-scanning/about-code-scanning-with-codeql) scanning. CodeQL delves deep into our codebase, identifying complex vulnerabilities like SQL injection and XSS by analyzing the code's semantic structure. This advanced level of analysis ensures early detection and resolution of potential security risks.

[![CodeQL](https://github.com/ultralytics/ultralytics/actions/workflows/codeql.yaml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/codeql.yaml)

## GitHub Dependabot Alerts

[Dependabot](https://docs.github.com/en/code-security/dependabot) is integrated into our workflow to monitor dependencies for known vulnerabilities. When a vulnerability is identified in one of our dependencies, Dependabot alerts us, allowing for swift and informed remediation actions.

## GitHub Secret Scanning Alerts

We employ GitHub [secret scanning](https://docs.github.com/en/code-security/secret-scanning/managing-alerts-from-secret-scanning) alerts to detect sensitive data, such as credentials and private keys, accidentally pushed to our repositories. This early detection mechanism helps prevent potential security breaches and data exposures.

## Private Vulnerability Reporting

We enable private vulnerability reporting, allowing users to discreetly report potential security issues. This approach facilitates responsible disclosure, ensuring vulnerabilities are handled securely and efficiently.

If you suspect or discover a security vulnerability in any of our repositories, please let us know immediately. You can reach out to us directly via our [contact form](https://www.ultralytics.com/contact) or via [security@ultralytics.com](mailto:security@ultralytics.com). Our security team will investigate and respond as soon as possible.

We appreciate your help in keeping all Ultralytics open-source projects secure and safe for everyone 🙏.

## FAQ

### What are the security measures implemented by Ultralytics to protect user data?

Ultralytics employs a comprehensive security strategy to protect user data and systems. Key measures include:

- **Snyk Scanning**: Conducts security scans to detect vulnerabilities in code and Dockerfiles.
- **GitHub CodeQL**: Analyzes code semantics to detect complex vulnerabilities such as SQL injection.
- **Dependabot Alerts**: Monitors dependencies for known vulnerabilities and sends alerts for swift remediation.
- **Secret Scanning**: Detects sensitive data like credentials or private keys in code repositories to prevent data breaches.
- **Private Vulnerability Reporting**: Offers a secure channel for users to report potential security issues discreetly.

These tools ensure proactive identification and resolution of security issues, enhancing overall system security. For more details, visit our [export documentation](../modes/export.md).

### How does Ultralytics use Snyk for security scanning?

Ultralytics utilizes [Snyk](https://snyk.io/advisor/python/ultralytics) to conduct thorough security scans on its repositories. Snyk extends beyond basic dependency checks, examining the code and Dockerfiles for various vulnerabilities. By proactively identifying and resolving potential security issues, Snyk helps ensure that Ultralytics' open-source projects remain secure and reliable.

To see the Snyk badge and learn more about its deployment, check the [Snyk Scanning section](#snyk-scanning).

### What is CodeQL and how does it enhance security for Ultralytics?

[CodeQL](https://docs.github.com/en/code-security/code-scanning/introduction-to-code-scanning/about-code-scanning-with-codeql) is a security analysis tool integrated into Ultralytics' workflow via GitHub. It delves deep into the codebase to identify complex vulnerabilities such as SQL injection and Cross-Site Scripting (XSS). CodeQL analyzes the semantic structure of the code to provide an advanced level of security, ensuring early detection and mitigation of potential risks.

For more information on how CodeQL is used, visit the [GitHub CodeQL Scanning section](#github-codeql-scanning).

### How does Dependabot help maintain Ultralytics' code security?

[Dependabot](https://docs.github.com/en/code-security/dependabot) is an automated tool that monitors and manages dependencies for known vulnerabilities. When Dependabot detects a vulnerability in an Ultralytics project dependency, it sends an alert, allowing the team to quickly address and mitigate the issue. This ensures that dependencies are kept secure and up-to-date, minimizing potential security risks.

For more details, explore the [GitHub Dependabot Alerts section](#github-dependabot-alerts).

### How does Ultralytics handle private vulnerability reporting?

Ultralytics encourages users to report potential security issues through private channels. Users can report vulnerabilities discreetly via the [contact form](https://www.ultralytics.com/contact) or by emailing [security@ultralytics.com](mailto:security@ultralytics.com). This ensures responsible disclosure and allows the security team to investigate and address vulnerabilities securely and efficiently.

For more information on private vulnerability reporting, refer to the [Private Vulnerability Reporting section](#private-vulnerability-reporting).
