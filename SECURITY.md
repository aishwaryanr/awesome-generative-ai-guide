# Security Policy

## Supported Versions

We prioritize the security of our users by providing regular patches and updates. Our support policy is as follows: 

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | :white_check_mark: |
| 5.0.x   | :x:                |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## How to Stay Protected

**1.Upgrade Promptly:** We strongly recommend running the latest version  to ensure you have the newest features and security protections.

**2.Monitor Advisories:** Subscribe to our security mailing list or check our GitHub security advisories page.

**3.Check Version:** Use command --version to verify your current version. 

If you are running a version listed as End of Life, you should upgrade immediately to avoid security risks. 

*For more information on how we handle security vulnerabilities, please refer to our Security Policy.*

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

## Reporting Security Issues

If you believe you have found a security vulnerability in any GitHub-owned repository, please report it to us through coordinated disclosure.

Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

Instead, please send an email to opensource-security[@]github.com.

Please include as much of the information listed below as you can to help us better understand and resolve the issue:

- The type of issue (e.g., buffer overflow, SQL injection, or cross-site scripting)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- What version(s) you've tested on and what other versions you think may be affected
- The environment in which you tested the exploit, including but not limited to OS family/version and Python version
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue
- This information will help us triage your report more quickly.

> [!IMPORTANT] 
> IMPORTANT! No information should be made public about the vulnerability until it is formally announced at the end of this process. That means, for example that a GitHub Issue must NOT be created to track the issue since that will make the issue public. Also the messages associated with any commits should not make ANY reference to the security nature of the commit.

**If the issue is confirmed as a vulnerability, the team will proceed to create a private GitHub security advisory within the affected package's GitHub repo**

**Once the fix is ready, it will be merged back into the original repository and a release will be generated. The private security advisory will also be published (i.e. made public) so that package users can be notified in a timely manner.**
