# Privacy Policy

**Last Updated: December 11, 2025**

## Introduction

This Privacy Policy describes how Socials Automator ("we", "our", or "the app") collects, uses, and protects information when you use our AI-powered Instagram carousel content generation and posting tool.

## Information We Collect

### Information You Provide

- **Profile Configuration**: Instagram handle, content preferences, branding settings, and content strategy configurations stored locally in profile folders.
- **API Credentials**: Instagram/Meta API tokens, Cloudinary credentials, and AI provider API keys stored in your local `.env` file.
- **Generated Content**: Topics, captions, hashtags, and carousel images you create using the app.

### Information from Third-Party Services

When you connect your Instagram account through the Meta Graph API, we access:
- Your Instagram Business/Creator account ID
- Permission to publish content to your account
- Basic account information required for posting

We do **not** access:
- Your Instagram password
- Your followers' personal information
- Direct messages
- Historical posts or analytics

### Automatically Collected Information

- **Generation Logs**: Local logs of content generation sessions for debugging purposes.
- **Post History**: Records of generated posts stored locally in your profile's knowledge base to avoid topic repetition.

## How We Use Your Information

We use the information collected to:

1. **Generate Content**: Create AI-powered carousel posts based on your profile settings and topic preferences.
2. **Publish to Instagram**: Upload images and publish carousels to your connected Instagram account.
3. **Improve Generation Quality**: Use local post history to avoid repeating topics and maintain content variety.
4. **Store Your Preferences**: Save your branding, design templates, and content strategy locally.

## Third-Party Services

Socials Automator integrates with the following third-party services:

### Meta/Instagram Graph API
- Used to publish content to your Instagram account
- Subject to [Meta's Privacy Policy](https://www.facebook.com/privacy/policy/)
- We only request permissions necessary for content publishing

### Cloudinary
- Used to temporarily host images for Instagram API upload
- Images are deleted after successful posting
- Subject to [Cloudinary's Privacy Policy](https://cloudinary.com/privacy)

### AI Providers
The app may use various AI providers for text and image generation:
- **OpenAI** (ChatGPT, DALL-E) - [Privacy Policy](https://openai.com/privacy/)
- **Anthropic** (Claude) - [Privacy Policy](https://www.anthropic.com/privacy)
- **Google** (Gemini) - [Privacy Policy](https://policies.google.com/privacy)
- **Replicate** (Flux) - [Privacy Policy](https://replicate.com/privacy)
- **Other providers** as configured

Content prompts are sent to these services to generate text and images. Please review each provider's privacy policy for details on how they handle data.

## Data Storage and Security

### Local Storage
- All profile data, credentials, and generated content are stored **locally on your machine**.
- We do **not** operate servers that store your data.
- API credentials are stored in your local `.env` file and should be kept secure.

### Data Transmission
- Data is transmitted securely (HTTPS) to third-party APIs when generating content or posting.
- We do not transmit your data to any servers owned or operated by us.

### Security Recommendations
- Keep your `.env` file secure and never commit it to public repositories.
- Use environment-specific API tokens with minimal required permissions.
- Regularly rotate your API credentials.

## Data Retention

- **Local Data**: Remains on your machine until you delete it.
- **Cloudinary Uploads**: Temporarily stored during posting, then deleted.
- **AI Provider Data**: Subject to each provider's retention policies.
- **Instagram Posts**: Remain on Instagram until you delete them.

## Your Rights

You have the right to:

1. **Access**: View all data stored in your local profile folders.
2. **Delete**: Remove your profile folders and `.env` file to delete all local data.
3. **Disconnect**: Revoke Instagram API access through your Meta Business Settings.
4. **Control**: Choose which AI providers to use and configure their settings.

## Children's Privacy

Socials Automator is not intended for use by children under 13 years of age. We do not knowingly collect information from children under 13.

## Changes to This Policy

We may update this Privacy Policy from time to time. Changes will be reflected in the "Last Updated" date above. Continued use of the app after changes constitutes acceptance of the updated policy.

## Contact Us

If you have questions about this Privacy Policy, please contact us by opening an issue on our GitHub repository.

## Meta Platform Terms Compliance

This app complies with:
- [Meta Platform Terms](https://developers.facebook.com/terms/)
- [Instagram Platform Policy](https://developers.facebook.com/docs/instagram-policy/)
- [Meta Developer Policies](https://developers.facebook.com/devpolicy/)

We only use Instagram data for the purposes explicitly authorized by the user and required for app functionality.

---

## Summary

| Data Type | Stored Where | Shared With |
|-----------|--------------|-------------|
| Profile settings | Local machine | Not shared |
| API credentials | Local `.env` file | Respective APIs only |
| Generated images | Local + temporary Cloudinary | Instagram when posting |
| Captions/hashtags | Local machine | Instagram when posting |
| Content prompts | Not stored remotely | AI providers during generation |
| Post history | Local knowledge base | Not shared |
