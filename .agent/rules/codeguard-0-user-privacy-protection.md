---
trigger: glob
globs: **/*.c,**/*.go,**/*.h,**/*.java,**/*.js,**/*.jsx,**/*.kt,**/*.kts,**/*.mjs,**/*.py,**/*.pyi,**/*.pyx,**/*.rb,**/*.swift,**/*.ts,**/*.tsx
description: User Privacy Protection Rule
version: 1.3.0
---

rule_id: codeguard-0-user-privacy-protection

- Implement strong cryptography, enforce HTTPS with HSTS, enable certificate pinning,
and provide user privacy features to protect data and anonymity.
- Use strong, up-to-date cryptographic algorithms for data in transit and at rest; securely hash passwords with established libraries.
- Enforce HTTPS exclusively and implement HTTP Strict Transport Security (HSTS).
- Implement certificate pinning to prevent man-in-the-middle attacks even if CAs are compromised.
- Minimize IP address leakage by blocking third-party external content loading where feasible.
- Maintain transparency by informing users about privacy limitations and data handling policies.
- Implement privacy-focused audit trails and access logging.
- Return "Invalid username or password" to prevent account enumeration
- Use Argon2 or bcrypt with unique salts per user
- Store sessions server-side with cryptographically random IDs
