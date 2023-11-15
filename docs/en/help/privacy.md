---
description: Learn about how Ultralytics collects and uses data to improve user experience, ensure software stability, and address privacy concerns, with options to opt-out.
keywords: Ultralytics, Data Collection, User Privacy, Google Analytics, Sentry, Crash Reporting, Anonymized Data, Privacy Settings, Opt-Out
---

# Data Collection for Ultralytics Python Package

## Overview

[Ultralytics](https://ultralytics.com) is dedicated to the continuous enhancement of the user experience and the capabilities of our Python package, including the advanced YOLO models we develop. Our approach involves the gathering of anonymized usage statistics and crash reports, helping us identify opportunities for improvement and ensuring the reliability of our software. This transparency document outlines what data we collect, its purpose, and the choice you have regarding this data collection.

## Anonymized Google Analytics

[Google Analytics](https://developers.google.com/analytics) is a web analytics service offered by Google that tracks and reports website traffic. It allows us to collect data about how our Python package is used, which is crucial for making informed decisions about design and functionality.

### What We Collect

- **Usage Metrics**: These metrics help us understand how frequently and in what ways the package is utilized, what features are favored, and the typical command-line arguments that are used.
- **System Information**: We collect general non-identifiable information about your computing environment to ensure our package performs well across various systems.
- **Performance Data**: Understanding the performance of our models during training, validation, and inference helps us in identifying optimization opportunities.

For more information about Google Analytics and data privacy, visit [Google Analytics Privacy](https://support.google.com/analytics/answer/6004245).

### How We Use This Data

- **Feature Improvement**: Insights from usage metrics guide us in enhancing user satisfaction and interface design.
- **Optimization**: Performance data assist us in fine-tuning our models for better efficiency and speed across diverse hardware and software configurations.
- **Trend Analysis**: By studying usage trends, we can predict and respond to the evolving needs of our community.

### Privacy Considerations

We take several measures to ensure the privacy and security of the data you entrust to us:

- **Anonymization**: We configure Google Analytics to anonymize the data collected, which means no personally identifiable information (PII) is gathered. You can use our services with the assurance that your personal details remain private.
- **Aggregation**: Data is analyzed only in aggregate form. This practice ensures that patterns can be observed without revealing any individual user's activity.
- **No Image Data Collection**: Ultralytics does not collect, process, or view any training or inference images.

## Sentry Crash Reporting

[Sentry](https://sentry.io/) is a developer-centric error tracking software that aids in identifying, diagnosing, and resolving issues in real-time, ensuring the robustness and reliability of applications. Within our package, it plays a crucial role by providing insights through crash reporting, significantly contributing to the stability and ongoing refinement of our software.

!!! note

    Crash reporting via Sentry is activated only if the `sentry-sdk` Python package is pre-installed on your system. This package isn't included in the `ultralytics` prerequisites and won't be installed automatically by Ultralytics.

### What We Collect

If the `sentry-sdk` Python package is pre-installed on your system a crash event may send the following information:

- **Crash Logs**: Detailed reports on the application's condition at the time of a crash, which are vital for our debugging efforts.
- **Error Messages**: We record error messages generated during the operation of our package to understand and resolve potential issues quickly.

To learn more about how Sentry handles data, please visit [Sentry's Privacy Policy](https://sentry.io/privacy/).

### How We Use This Data

- **Debugging**: Analyzing crash logs and error messages enables us to swiftly identify and correct software bugs.
- **Stability Metrics**: By constantly monitoring for crashes, we aim to improve the stability and reliability of our package.

### Privacy Considerations

- **Sensitive Information**: We ensure that crash logs are scrubbed of any personally identifiable or sensitive user data, safeguarding the confidentiality of your information.
- **Controlled Collection**: Our crash reporting mechanism is meticulously calibrated to gather only what is essential for troubleshooting while respecting user privacy.

By detailing the tools used for data collection and offering additional background information with URLs to their respective privacy pages, users are provided with a comprehensive view of our practices, emphasizing transparency and respect for user privacy.

## Disabling Data Collection

We believe in providing our users with full control over their data. By default, our package is configured to collect analytics and crash reports to help improve the experience for all users. However, we respect that some users may prefer to opt out of this data collection.

To opt out of sending analytics and crash reports, you can simply set `sync=False` in your YOLO settings. This ensures that no data is transmitted from your machine to our analytics tools.

### Inspecting Settings

To gain insight into the current configuration of your settings, you can view them directly:

!!! example "View settings"

    === "Python"
        You can use Python to view your settings. Start by importing the `settings` object from the `ultralytics` module. Print and return settings using the following commands:
        ```python
        from ultralytics import settings

        # View all settings
        print(settings)

        # Return analytics and crash reporting setting
        value = settings['sync']
        ```

    === "CLI"
        Alternatively, the command-line interface allows you to check your settings with a simple command:
        ```bash
        yolo settings
        ```

### Modifying Settings

Ultralytics allows users to easily modify their settings. Changes can be performed in the following ways:

!!! example "Update settings"

    === "Python"
        Within the Python environment, call the `update` method on the `settings` object to change your settings:
        ```python
        from ultralytics import settings

        # Disable analytics and crash reporting
        settings.update({'sync': False})

        # Reset settings to default values
        settings.reset()
        ```

    === "CLI"
        If you prefer using the command-line interface, the following commands will allow you to modify your settings:
        ```bash
        # Disable analytics and crash reporting
        yolo settings sync=False

        # Reset settings to default values
        yolo settings reset
        ```

The `sync=False` setting will prevent any data from being sent to Google Analytics or Sentry. Your settings will be respected across all sessions using the Ultralytics package and saved to disk for future sessions.

## Commitment to Privacy

Ultralytics takes user privacy seriously. We design our data collection practices with the following principles:

- **Transparency**: We are open about the data we collect and how it is used.
- **Control**: We give users full control over their data.
- **Security**: We employ industry-standard security measures to protect the data we collect.

## Questions or Concerns

If you have any questions or concerns about our data collection practices, please reach out to us via our [contact form](https://ultralytics.com/contact) or via [support@ultralytics.com](mailto:support@ultralytics.com). We are dedicated to ensuring our users feel informed and confident in their privacy when using our package.
