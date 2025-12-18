# Instagram API Setup Cheat Sheet

Quick reference for setting up Instagram Graph API to automatically upload reels and posts.

## Prerequisites

- Instagram Business or Creator account (NOT personal)
- Facebook Page linked to the Instagram account
- Facebook Developer account

---

## Step 1: Create Facebook App

1. Go to [developers.facebook.com](https://developers.facebook.com)
2. Click **My Apps** -> **Create App**
3. Select **Other** -> **Business**
4. Enter app name (e.g., "My Instagram Automator")
5. Click **Create App**

---

## Step 2: Add Instagram API Product

1. In your app dashboard, click **Add Product**
2. Find **Instagram API** and click **Set up**
3. Select **API setup with Facebook login** (NOT "Instagram login")
   - **IMPORTANT**: "Instagram login" tokens (IG...) do NOT work for publishing
   - You need "Facebook login" tokens (EAA...) for content publishing

---

## Step 3: Get Instagram Business Account ID

### Method 1: Quick Lookup (Recommended)
1. Go to [commentpicker.com/instagram-user-id.php](https://commentpicker.com/instagram-user-id.php)
2. Enter your Instagram username (without @)
3. Copy the numeric ID

**Note**: This gives you the regular user ID. For Business Account ID, use Method 2.

### Method 2: Graph API Explorer (Accurate)
1. Go to [Graph API Explorer](https://developers.facebook.com/tools/explorer/)
2. Select your app
3. Generate a User Token with permissions:
   - `instagram_basic`
   - `instagram_content_publish`
   - `business_management`
4. Run query: `me/accounts?fields=instagram_business_account{id,username}`
5. Copy the `id` from `instagram_business_account` (format: `17841...`)

---

## Step 4: Add Instagram Tester (Development Mode)

While your app is in Development mode:

1. Go to your app -> **App Roles** -> **Roles**
2. Click **Add Instagram Testers**
3. Enter your Instagram username
4. Go to Instagram -> **Settings** -> **Apps and Websites** -> **Tester invites**
5. Accept the invitation

---

## Step 5: Generate Access Token

### Option A: API Setup with Facebook Login (Recommended)

1. In your app, go to **Use Cases** -> **Customize**
2. Select **Instagram API**
3. Click **API setup with Facebook login**
4. Under "Generate access tokens", click **Generate token** next to your Instagram account
5. Authorize all requested permissions
6. Copy the token (starts with `EAA...`)

### Option B: Graph API Explorer

1. Go to [Graph API Explorer](https://developers.facebook.com/tools/explorer/)
2. Select your app
3. Click **Generate Access Token**
4. Select permissions:
   - `instagram_basic`
   - `instagram_content_publish`
   - `business_management`
   - `pages_read_engagement`
5. Click **Generate Access Token**
6. **IMPORTANT**: Select ALL Facebook Pages you want to post to in the popup
7. Copy the token

---

## Step 6: Extend Token Lifespan

Short-lived tokens expire in 1 hour. Extend to 60 days:

1. Go to [Access Token Debugger](https://developers.facebook.com/tools/debug/accesstoken/)
2. Paste your token and click **Debug**
3. Click **Extend Access Token** at the bottom
4. Copy the new long-lived token

**Or via API**:
```
GET https://graph.facebook.com/v21.0/oauth/access_token
  ?grant_type=fb_exchange_token
  &client_id={app-id}
  &client_secret={app-secret}
  &fb_exchange_token={short-lived-token}
```

---

## Step 7: Configure Environment

Add to your `.env` file:

```bash
# For default profile
INSTAGRAM_USER_ID=17841478809122623
INSTAGRAM_ACCESS_TOKEN=EAAxxxxxxx...

# For additional profiles (use unique names)
INSTAGRAM_USER_ID_NEWS_BUT_QUICK=17841479986691055
INSTAGRAM_ACCESS_TOKEN_NEWS_BUT_QUICK=EAAxxxxxxx...
```

---

## Step 8: Configure Profile

Add to `profiles/<name>/metadata.json`:

```json
{
  "platforms": {
    "instagram": {
      "enabled": true,
      "user_id": "ENV:INSTAGRAM_USER_ID_YOUR_PROFILE",
      "access_token": "ENV:INSTAGRAM_ACCESS_TOKEN_YOUR_PROFILE"
    }
  }
}
```

---

## Step 9: Test Upload

```bash
# Always test with --dry-run first
python -m socials_automator.cli upload-reel your-profile --dry-run

# Upload single reel
python -m socials_automator.cli upload-reel your-profile --one
```

---

## Token Types Reference

| Prefix | Type | Can Publish? | Notes |
|--------|------|--------------|-------|
| `EAA...` | Facebook Login | YES | Required for publishing |
| `IG...` | Instagram Login | NO | Only for reading data |

**Always use EAA... tokens for uploading content!**

---

## Required Permissions

For publishing reels and posts:

| Permission | Purpose |
|------------|---------|
| `instagram_basic` | Read profile info |
| `instagram_content_publish` | Publish reels/posts |
| `business_management` | Manage business accounts |
| `pages_read_engagement` | Read Page insights |

---

## Rate Limits

| Limit Type | Amount | Reset |
|------------|--------|-------|
| Daily API calls | ~200/hour | Rolling |
| Content Publishing | ~25 posts/day | Midnight UTC |
| Per-account | Varies | - |

**Important**: Rate limits are shared at the **APP level**, not per account. All profiles using the same Facebook App share the same quota.

---

## Error Codes Reference

| Code | Name | Meaning | Action |
|------|------|---------|--------|
| 4 | RATE_LIMIT | Too many requests | Wait 1 hour |
| 9 | APP_RATE_LIMIT | App limit reached | Wait 5+ minutes |
| 10 | PERMISSION_DENIED | Missing permission | Check token scopes |
| 17 | USER_RATE_LIMIT | User limit reached | Wait 2 minutes |
| 190 | ACCESS_TOKEN_EXPIRED | Token expired | Refresh token |
| 2207032 | MEDIA_UPLOAD_FAILED | Upload failed | Retry |
| 2207069 | DAILY_POSTING_LIMIT | Daily limit hit | Wait until midnight UTC |

---

## Troubleshooting

### "Object does not exist" error
- Token doesn't have permission for that Page
- Regenerate token and select the Page in authorization

### "Invalid Scopes" error
- Use only: `instagram_basic`, `instagram_content_publish`, `business_management`

### Token expired
- Run: `python -m socials_automator.cli token --refresh`

### Wrong User ID format
- Regular user ID: `79928442223` (from commentpicker)
- Business Account ID: `17841479986691055` (from Graph API)
- Use the Business Account ID (starts with `17841...`)

### Rate limit hit but post went through (Ghost Publish)
- Meta API can return error AFTER successfully publishing
- The uploader automatically checks for this
- Don't retry manually - check Instagram first

---

## Quick Setup Checklist

- [ ] Facebook Developer account created
- [ ] Facebook App created (Business type)
- [ ] Instagram API product added
- [ ] Selected "API setup with Facebook login" (NOT Instagram login)
- [ ] Instagram account added as Tester (if in Development mode)
- [ ] Access token generated (starts with `EAA...`)
- [ ] Token extended to 60 days
- [ ] User ID obtained (Business Account ID format: `17841...`)
- [ ] `.env` file configured
- [ ] Profile `metadata.json` configured with `platforms` section
- [ ] Test upload successful with `--dry-run`

---

## Separate Rate Limits

To have independent rate limits for different accounts:

1. Create a **separate Facebook App** for each account
2. Each app gets its own rate limit quota
3. Configure separate tokens in `.env`

This is useful if you're posting frequently to multiple accounts.
