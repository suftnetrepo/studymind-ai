# Transactional email (welcome + password reset)

Provider: Brevo transactional API (same as Learnify), with the older SMTP settings as a fallback. A failed or unconfigured email never blocks registration or recovery.

| Variable | Purpose |
|---|---|
| `BREVO_API_KEY` | Brevo API key (secret). When set together with a sender, Brevo is used |
| `EMAIL_FROM_ADDRESS` | Verified sender in Brevo, e.g. `noreply@your-domain` (falls back to `RECOVERY_FROM_EMAIL`) |
| `EMAIL_FROM_NAME` | Display name, default `StudyMind` |
| `EMAIL_APP_URL` | Public Web address for the Dashboard link in the welcome email, e.g. `https://studymind-web.onrender.com` |
| `EMAIL_WELCOME_ENABLED` | `true` (default) / `false` to stop welcome emails |
| `PASSWORD_RECOVERY_ENABLED` | `true` turns the reset flow on (default off) |
| `RECOVERY_PUBLIC_URL` | HTTPS address of the Web reset page, e.g. `https://studymind-web.onrender.com/my-learning/reset-password` (the token is appended as `?token=`) |
| `RECOVERY_SMTP_*` | Optional SMTP fallback (host, port, username, password) |

Endpoints: `POST /api/auth/password-recovery/request` (always answers 202, never reveals whether the email exists; rate limited) and `POST /api/auth/password-recovery/confirm` (single-use hashed token, 30 minutes, signs out all devices).
