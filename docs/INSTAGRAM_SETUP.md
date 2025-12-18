# Instagram Profile Setup Guide

Quick guide to set up a new Instagram profile for posting with Socials Automator.

## Prerequisites

- Instagram Business or Creator account (not Personal)
- Facebook Page linked to the Instagram account
- Access to the existing "Socials Automator" Facebook App (or create a new one)

## Step 1: Create Instagram Business Account

1. Create a new Instagram account (or use existing)
2. In Instagram app: **Settings** -> **Account** -> **Switch to Professional Account**
3. Choose **Creator** or **Business**

## Step 2: Link to Facebook Page

Choose one method:

### Option A: Via Instagram App (Easiest)
1. Go to your Instagram profile
2. Tap **Edit Profile**
3. Scroll to **Public business information** -> **Page**
4. Tap **Continue** and log into Facebook
5. Select an existing Page or create a new one

### Option B: Via Meta Business Suite
1. Go to [business.facebook.com](https://business.facebook.com)
2. **Settings** -> **Business assets** -> **Accounts** -> **Instagram accounts**
3. Click **Add** and log in with Instagram credentials
4. Choose which Facebook Page to link

## Step 3: Get Instagram User ID (Quick Method)

**Do NOT use Graph API Explorer** - it requires complex permissions.

Instead, use this free tool:
1. Go to [commentpicker.com/instagram-user-id.php](https://commentpicker.com/instagram-user-id.php)
2. Enter the Instagram username (without @)
3. Copy the numeric **Instagram ID**

Example:
```
Username: news.but.quick
Instagram ID: 79928442223
```

## Step 4: Configure Profile

### 4.1 Add to .env

```bash
# Use a unique env var name for each profile
INSTAGRAM_USER_ID_YOUR_PROFILE=<instagram-id-from-step-3>
```

### 4.2 Add platforms section to profile metadata.json

```json
{
  "platforms": {
    "instagram": {
      "enabled": true,
      "user_id": "ENV:INSTAGRAM_USER_ID_YOUR_PROFILE",
      "access_token": "ENV:INSTAGRAM_ACCESS_TOKEN"
    }
  }
}
```

**Note:** The `access_token` can be shared across profiles if they use the same Facebook App. The token just needs permission for all connected Pages.

## Step 5: Verify Setup

```bash
# Test with dry run first
python -m socials_automator.cli upload-reel your-profile --dry-run

# If successful, test actual upload
python -m socials_automator.cli upload-reel your-profile --one
```

## Access Token Setup (First Time Only)

If you don't have an access token yet:

1. Go to [developers.facebook.com](https://developers.facebook.com)
2. Create an app or use existing one
3. Add **Instagram Graph API** product
4. Go to [Graph API Explorer](https://developers.facebook.com/tools/explorer/)
5. Generate token with permissions:
   - `instagram_basic`
   - `instagram_content_publish`
   - `business_management`
6. **Important:** Select ALL Facebook Pages you want to post to in the authorization popup
7. [Extend the token](https://developers.facebook.com/tools/debug/accesstoken/) to 60 days
8. Add to `.env`:
   ```bash
   INSTAGRAM_ACCESS_TOKEN=<your-long-lived-token>
   ```

## Rate Limits

**Important:** Rate limits are shared at the **app level**, not per account.

- All Instagram accounts using the same Facebook App share the same quota
- If you hit a rate limit on one account, it affects all accounts
- Error codes:
  - `4` = Application daily limit
  - `9` = Temporary throttle (wait 5-15 minutes)
  - `17` = User-level rate limit

**Workaround:** If you need higher throughput, create separate Facebook Apps for different accounts.

## Troubleshooting

### "Object does not exist" error in Graph API
Your token doesn't have permission for that Page. Regenerate the token and select the Page in the authorization flow.

### "Invalid Scopes" error
Some permissions have been deprecated. Use only:
- `instagram_basic`
- `instagram_content_publish`
- `business_management`

### Rate limit hit
Wait 5-15 minutes and try again. The system has automatic retry with backoff.
