# Clerk Authentication - Quick Start

## 🚀 Get Started in 3 Steps

### 1️⃣ Get Your Clerk Keys
1. Go to [https://dashboard.clerk.com](https://dashboard.clerk.com)
2. Create a free account and new application
3. Copy your **Publishable Key** and **Secret Key** from the API Keys section

### 2️⃣ Add Keys to Environment
Open `frontend/.env.local` and replace the placeholder values:

```env
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_your_key_here
CLERK_SECRET_KEY=sk_test_your_key_here
```

### 3️⃣ Start the App
```bash
cd frontend
npm run dev
```

Visit `http://localhost:3000` - you'll be redirected to sign in!

## ✅ What's Already Configured

- ✅ ClerkProvider wrapping the entire app
- ✅ Protected routes (all except landing page)
- ✅ Sign-in page at `/sign-in`
- ✅ Sign-up page at `/sign-up`
- ✅ User button in header with profile menu
- ✅ Personalized dashboard greeting
- ✅ Automatic redirects for auth flow

## 🎨 Features

### For Users
- **Sign up** with email or social providers
- **Sign in** to access the dashboard
- **Profile management** via user button in header
- **Secure sessions** with automatic refresh
- **Sign out** from the profile menu

### For Developers
- **Protected routes** by default
- **User data** accessible via `useUser()` hook
- **Server-side auth** with `auth()` helper
- **Customizable UI** with appearance props
- **Webhook support** for user events

## 📍 Key Routes

| Route | Description | Access |
|-------|-------------|--------|
| `/` | Dashboard | Protected |
| `/sign-in` | Sign in page | Public |
| `/sign-up` | Sign up page | Public |
| `/landing-preview` | Marketing page | Public |
| All others | App pages | Protected |

## 🔧 Common Tasks

### Get User Info in a Component
```tsx
import { useUser } from '@clerk/nextjs';

function MyComponent() {
  const { user, isLoaded, isSignedIn } = useUser();
  
  return <div>Hello {user?.firstName}!</div>;
}
```

### Protect an API Route
```tsx
import { auth } from '@clerk/nextjs/server';

export async function GET() {
  const { userId } = await auth();
  if (!userId) return new Response('Unauthorized', { 401 });
  // Your logic here
}
```

### Customize Sign-In Appearance
Edit `frontend/src/app/sign-in/[[...sign-in]]/page.tsx`:
```tsx
<SignIn 
  appearance={{
    elements: {
      card: "bg-gray-100",
      // Add your styles
    }
  }}
/>
```

## 🎯 Next Steps

1. **Test the flow**: Sign up → Sign in → View dashboard → Sign out
2. **Customize branding**: Update Clerk Dashboard with your logo/colors
3. **Enable social login**: Add Google, GitHub, etc. in Clerk Dashboard
4. **Add user roles**: Implement RBAC for different user types
5. **Set up webhooks**: Sync user data to your database

## 📚 Full Documentation

See `CLERK_AUTH_SETUP.md` for complete setup guide and advanced features.

## 🆘 Troubleshooting

**Can't see sign-in page?**
- Check that keys are in `.env.local`
- Restart dev server: `npm run dev`

**Redirect loops?**
- Verify middleware config in `frontend/src/middleware.ts`
- Check Clerk Dashboard URLs match your routes

**User data not showing?**
- Ensure you're using `useUser()` in client components
- Add `'use client'` directive at top of file

## 🔗 Resources

- [Clerk Docs](https://clerk.com/docs)
- [Next.js Guide](https://clerk.com/docs/quickstarts/nextjs)
- [Dashboard](https://dashboard.clerk.com)
