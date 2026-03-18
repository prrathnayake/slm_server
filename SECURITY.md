# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Yes                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please report vulnerabilities via:

1. **Email**: Send details to the repository maintainer
2. **GitHub Security Advisory**: Use the "Report a vulnerability" button on the repository's Security tab

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix or Mitigation**: Depends on severity, typically within 2 weeks for critical issues

## Security Best Practices

When deploying this platform:

### Authentication
- Set a strong `API_KEY` in your `.env` file
- Never expose the API without authentication in production
- Rotate API keys regularly

### Network Security
- Use TLS/HTTPS for remote access
- Restrict network access to trusted clients
- Consider VPN or SSH tunneling for remote backends

### Model Security
- Validate imported models before loading
- Use checksum verification for model files
- Keep models in isolated directories

### Configuration
- Keep `.env` files out of version control
- Use environment-specific configurations
- Limit file system permissions on sensitive directories

### Dependencies
- Keep dependencies updated
- Use `pip audit` to check for known vulnerabilities
- Pin dependency versions in production

## Known Security Considerations

- **Local API Exposure**: The API server binds to `0.0.0.0` by default. Restrict access using firewall rules.
- **Model Loading**: Loading untrusted models can execute arbitrary code. Only import models from trusted sources.
- **File Uploads**: Model imports extract ZIP files. Validate contents before processing.
- **Job Queue**: Redis is used without authentication by default. Secure Redis in production.
