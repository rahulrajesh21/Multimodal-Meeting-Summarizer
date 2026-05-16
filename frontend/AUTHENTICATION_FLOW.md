# VelaAI Authentication Flow

## 🔐 Authentication Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Root Layout                              │
│                    (ClerkProvider wraps all)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Middleware                               │
│              (Checks auth on every request)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              Authenticated?            │
                    │                   │
            ┌───────┴───────┐           │
            │               │           │
           YES             NO           │
            │               │           │
            ▼               ▼           │
    ┌──────────────┐  ┌──────────────┐ │
    │  Dashboard   │  │  Redirect to │ │
    │  & App Pages │  │   /sign-in   │ │
    └──────────────┘  └──────────────┘ │
                              │         │
                              ▼         │
                    ┌──────────────────┐│
                    │   Public Routes  ││
                    │  - /sign-in      ││
                    │  - /sign-up      ││
                    │  - /landing-     ││
                    │    preview       ││
                    └──────────────────┘│
                              │         │
                              └─────────┘
```

## 📊 User Journey

### New User Flow
```
Landing Page (/landing-preview)
    │
    │ Click "Get Started"
    ▼
Sign Up Page (/sign-up)
    │
    │ Complete registration
    ▼
Dashboard (/)
    │
    │ Personalized with user name
    ▼
Full App Access
```

### Returning User Flow
```
Any Protected Route
    │
    │ Not authenticated?
    ▼
Sign In Page (/sign-in)
    │
    │ Enter credentials
    ▼
Original Route or Dashboard
    │
    │ Session active
    ▼
Full App Access
```

### Sign Out Flow
```
Dashboard or Any Page
    │
    │ Click user avatar
    ▼
Profile Menu
    │
    │ Click "Sign Out"
    ▼
Sign In Page (/sign-in)
```

## 🛡️ Route Protection

### Protected Routes (Require Auth)
- `/` - Dashboard
- `/meetings` - Meetings list
- `/meetings/[id]` - Meeting details
- `/meetings/teams/[id]` - Team meetings
- `/roles` - Roles page
- `/settings` - Settings page
- `/ai` - AI features
- `/graph` - Knowledge graph
- `/mcp` - MCP page

### Public Routes (No Auth Required)
- `/sign-in` - Sign in page
- `/sign-up` - Sign up page
- `/landing-preview` - Marketing page
- `/api/webhook/*` - Webhook endpoints

## 🔄 Session Management

```
┌─────────────────────────────────────────────────────────┐
│                    User Session                          │
├─────────────────────────────────────────────────────────┤
│  • Automatic token refresh                              │
│  • Cross-tab synchronization                            │
│  • Secure cookie storage                                │
│  • 7-day default session length                         │
│  • "Remember me" extends to 30 days                     │
└─────────────────────────────────────────────────────────┘
```

## 🎨 UI Components

### Header Component
```
┌────────────────────────────────────────────────────────┐
│  [Logo] [Search] [Upload]     [Preview] [🔔] [👤]     │
│                                              ▲          │
│                                              │          │
│                                         User Button     │
│                                         (Clerk)         │
└────────────────────────────────────────────────────────┘
```

### User Button Menu
```
┌─────────────────────┐
│  👤 John Doe        │
│  john@example.com   │
├─────────────────────┤
│  Manage account     │
│  Sign out           │
└─────────────────────┘
```

### Landing Page (Not Signed In)
```
┌────────────────────────────────────────────────────────┐
│  [Logo]  [Home] [Features] [About]  [Sign In] [Start] │
└────────────────────────────────────────────────────────┘
```

### Landing Page (Signed In)
```
┌────────────────────────────────────────────────────────┐
│  [Logo]  [Home] [Features] [About]      [Dashboard]   │
└────────────────────────────────────────────────────────┘
```

## 🔑 Authentication States

### Component Level
```tsx
import { useUser } from '@clerk/nextjs';

function MyComponent() {
  const { user, isLoaded, isSignedIn } = useUser();
  
  // Loading state
  if (!isLoaded) return <Spinner />;
  
  // Not authenticated
  if (!isSignedIn) return <SignInPrompt />;
  
  // Authenticated - show content
  return <ProtectedContent user={user} />;
}
```

### Server Level
```tsx
import { auth } from '@clerk/nextjs/server';

export async function GET() {
  const { userId } = await auth();
  
  if (!userId) {
    return new Response('Unauthorized', { status: 401 });
  }
  
  // Protected logic
  return Response.json({ data: 'secret' });
}
```

## 📱 Multi-Device Support

```
Device A                    Clerk Cloud                Device B
   │                            │                          │
   │  Sign In                   │                          │
   ├──────────────────────────► │                          │
   │                            │                          │
   │  ◄─────────────────────────┤                          │
   │  Session Token             │                          │
   │                            │                          │
   │                            │  Sign In                 │
   │                            │ ◄────────────────────────┤
   │                            │                          │
   │                            ├─────────────────────────►│
   │                            │  Session Token           │
   │                            │                          │
   │  Sign Out                  │                          │
   ├──────────────────────────► │                          │
   │                            │                          │
   │                            │  Session Invalidated     │
   │                            ├─────────────────────────►│
   │                            │                          │
```

## 🔒 Security Features

### Built-in Protection
- ✅ CSRF protection
- ✅ XSS prevention
- ✅ SQL injection protection
- ✅ Rate limiting
- ✅ Brute force protection
- ✅ Session hijacking prevention
- ✅ Secure password hashing (bcrypt)
- ✅ Email verification
- ✅ Two-factor authentication (optional)

### Data Privacy
- ✅ GDPR compliant
- ✅ SOC 2 Type II certified
- ✅ HIPAA compliant (Enterprise)
- ✅ Data encryption at rest
- ✅ Data encryption in transit (TLS 1.3)

## 🚀 Performance

### Optimizations
- Client-side session caching
- Automatic token refresh
- Minimal bundle size impact
- Edge-compatible middleware
- CDN-distributed assets

### Metrics
- Sign-in: ~200ms average
- Token refresh: ~100ms average
- Session check: ~10ms average

## 🔧 Configuration Files

```
frontend/
├── .env.local                    # API keys (not committed)
├── .env.example                  # Template for keys
├── src/
│   ├── app/
│   │   ├── layout.tsx           # ClerkProvider wrapper
│   │   ├── sign-in/
│   │   │   └── [[...sign-in]]/
│   │   │       └── page.tsx     # Sign-in UI
│   │   └── sign-up/
│   │       └── [[...sign-up]]/
│   │           └── page.tsx     # Sign-up UI
│   ├── middleware.ts            # Route protection
│   └── components/
│       └── Header.tsx           # UserButton integration
```

## 📊 User Data Flow

```
User Signs Up
    │
    ▼
Clerk Creates User
    │
    ▼
User Object Available
    │
    ├─► Frontend: useUser() hook
    │   └─► user.firstName
    │   └─► user.email
    │   └─► user.imageUrl
    │
    └─► Backend: auth() helper
        └─► userId
        └─► sessionId
```

## 🎯 Best Practices

### ✅ Do
- Use `useUser()` for client components
- Use `auth()` for server components/API routes
- Check `isLoaded` before accessing user data
- Handle loading and error states
- Use environment variables for keys
- Enable MFA for production

### ❌ Don't
- Don't store sensitive data in client state
- Don't bypass middleware protection
- Don't commit API keys to git
- Don't use same keys for dev/prod
- Don't skip email verification
- Don't ignore security warnings

## 📈 Monitoring

### Available Metrics (Clerk Dashboard)
- Active users
- Sign-up rate
- Sign-in success rate
- Failed authentication attempts
- Session duration
- Device breakdown
- Geographic distribution

## 🔗 Integration Points

### Current Integrations
- ✅ Next.js App Router
- ✅ React components
- ✅ API routes
- ✅ Middleware

### Potential Integrations
- 🔄 Database sync (webhooks)
- 🔄 Analytics (PostHog, Mixpanel)
- 🔄 Email service (SendGrid, Resend)
- 🔄 Payment (Stripe)
- 🔄 CRM (HubSpot, Salesforce)
