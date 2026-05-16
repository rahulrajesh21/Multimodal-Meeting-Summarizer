# Clerk Authentication Setup Guide

## Overview
Clerk authentication has been integrated into the VelaAI application. This provides secure user authentication, session management, and user profiles.

## What Was Added

### 1. **Environment Variables** (`.env.local`)
```env
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_publishable_key_here
CLERK_SECRET_KEY=your_secret_key_here
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/
```

### 2. **Root Layout** (`frontend/src/app/layout.tsx`)
- Wrapped the entire app with `ClerkProvider`
- Enables authentication context throughout the app

### 3. **Middleware** (`frontend/src/middleware.ts`)
- Protects all routes except:
  - `/sign-in` - Sign in page
  - `/sign-up` - Sign up page
  - `/landing-preview` - Public landing page
  - `/api/webhook` - Webhook endpoints
- Automatically redirects unauthenticated users to sign-in

### 4. **Authentication Pages**
- **Sign In**: `/sign-in/[[...sign-in]]/page.tsx`
- **Sign Up**: `/sign-up/[[...sign-up]]/page.tsx`
- Both pages feature:
  - VelaAI branding
  - Dark background matching the app theme
  - Clerk's pre-built authentication UI

### 5. **Header Component** (`frontend/src/components/Header.tsx`)
- Added `UserButton` component showing user avatar
- Displays user profile menu with sign-out option
- Integrated with existing header design

### 6. **Dashboard** (`frontend/src/app/(app)/page.tsx`)
- Personalized greeting using user's first name
- Falls back to username or "there" if name not available
- Uses `useUser()` hook to access user data

### 7. **Landing Page** (`frontend/src/components/StandaloneLandingPage.tsx`)
- Shows different buttons based on auth state:
  - **Not signed in**: "Sign In" and "Get Started" buttons
  - **Signed in**: "Dashboard" button
- "Get Started" redirects to sign-up or dashboard based on auth state

## Setup Instructions

### Step 1: Create a Clerk Account
1. Go to [https://dashboard.clerk.com](https://dashboard.clerk.com)
2. Sign up for a free account
3. Create a new application

### Step 2: Get Your API Keys
1. In the Clerk Dashboard, go to **API Keys**
2. Copy your **Publishable Key**
3. Copy your **Secret Key**

### Step 3: Configure Environment Variables
1. Open `frontend/.env.local`
2. Replace the placeholder values:
   ```env
   NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_your_actual_key_here
   CLERK_SECRET_KEY=sk_test_your_actual_key_here
   ```

### Step 4: Configure Clerk Dashboard
In your Clerk Dashboard, configure the following:

#### Paths
- **Sign-in URL**: `/sign-in`
- **Sign-up URL**: `/sign-up`
- **After sign-in URL**: `/`
- **After sign-up URL**: `/`

#### Allowed Redirect URLs
Add these URLs to your allowed redirects:
- `http://localhost:3000`
- `http://localhost:3000/sign-in`
- `http://localhost:3000/sign-up`
- Your production domain (when deploying)

### Step 5: Start the Development Server
```bash
cd frontend
npm run dev
```

### Step 6: Test Authentication
1. Visit `http://localhost:3000`
2. You should be redirected to `/sign-in`
3. Click "Sign up" to create an account
4. After signing up, you'll be redirected to the dashboard
5. Your name should appear in the welcome message

## Features

### Protected Routes
All routes are protected by default except:
- Landing preview page
- Sign-in page
- Sign-up page
- API webhooks

### User Profile
- Click the user avatar in the header to:
  - View profile
  - Manage account
  - Sign out

### Personalization
- Dashboard shows personalized greeting with user's name
- User avatar displayed in header

### Session Management
- Automatic session refresh
- Secure token handling
- Cross-tab synchronization

## Customization

### Styling Clerk Components
Clerk components can be customized using the `appearance` prop:

```tsx
<SignIn 
  appearance={{
    elements: {
      rootBox: "mx-auto",
      card: "bg-white shadow-xl",
      // Add more custom styles
    }
  }}
/>
```

### Adding More User Data
Access user data in any component:

```tsx
import { useUser } from '@clerk/nextjs';

function MyComponent() {
  const { user, isLoaded, isSignedIn } = useUser();
  
  if (!isLoaded) return <div>Loading...</div>;
  if (!isSignedIn) return <div>Not signed in</div>;
  
  return (
    <div>
      <p>Email: {user.primaryEmailAddress?.emailAddress}</p>
      <p>Name: {user.firstName} {user.lastName}</p>
      <p>Username: {user.username}</p>
    </div>
  );
}
```

### Server-Side Authentication
For API routes or server components:

```tsx
import { auth } from '@clerk/nextjs/server';

export async function GET() {
  const { userId } = await auth();
  
  if (!userId) {
    return new Response('Unauthorized', { status: 401 });
  }
  
  // Your protected logic here
}
```

## Security Best Practices

1. **Never commit `.env.local`** - It's already in `.gitignore`
2. **Use different keys for development and production**
3. **Rotate keys regularly** in production
4. **Enable MFA** in Clerk Dashboard for production apps
5. **Configure allowed domains** in Clerk Dashboard

## Troubleshooting

### "Clerk: Missing publishableKey"
- Ensure `.env.local` has the correct keys
- Restart the dev server after adding keys

### Redirect loops
- Check middleware configuration
- Verify Clerk Dashboard URLs match your routes

### User data not showing
- Ensure `ClerkProvider` wraps your app in root layout
- Check that you're using `useUser()` in client components

### Styling issues
- Clerk components use their own styles
- Use the `appearance` prop to customize
- Check for CSS conflicts with Tailwind

## Next Steps

### Optional Enhancements
1. **Add user roles** - Implement role-based access control
2. **Customize user profile** - Add custom fields in Clerk Dashboard
3. **Add webhooks** - Sync user data to your database
4. **Social login** - Enable Google, GitHub, etc. in Clerk Dashboard
5. **Email templates** - Customize verification and welcome emails

### Production Deployment
1. Create a production Clerk application
2. Add production domain to allowed redirects
3. Update environment variables in your hosting platform
4. Test authentication flow in production

## Resources

- [Clerk Documentation](https://clerk.com/docs)
- [Next.js Integration Guide](https://clerk.com/docs/quickstarts/nextjs)
- [Clerk Dashboard](https://dashboard.clerk.com)
- [Clerk Community](https://clerk.com/discord)

## Support

If you encounter issues:
1. Check the [Clerk Documentation](https://clerk.com/docs)
2. Search [Clerk's GitHub Issues](https://github.com/clerk/javascript/issues)
3. Join [Clerk's Discord](https://clerk.com/discord)
