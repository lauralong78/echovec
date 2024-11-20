# Security Policy

## Supported Versions

echovec is pre-1.0 research software. Security fixes are applied to the latest
released version on the `main` branch.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |

## Reporting a Vulnerability

If you believe you have found a security issue, please **do not open a public
issue**. Instead, use GitHub's private vulnerability reporting
("Security" tab → "Report a vulnerability") so it can be triaged privately.

Please include:

- a description of the issue and its impact,
- steps to reproduce, and
- any suggested mitigation.

You can expect an initial acknowledgement within a few days. Because this is a
pure-NumPy library with no network or filesystem privileges beyond reading audio
files you pass to it, the attack surface is small, but reports are still
appreciated.
