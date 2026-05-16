import { SignIn } from '@clerk/nextjs';

export default function SignInPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#080808]">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2.5 mb-4">
            <img src="/vela_logo.svg" alt="VelaAI" className="w-10 h-10" />
            <h1 className="text-2xl font-bold text-white font-sans">VelaAI</h1>
          </div>
          <p className="text-white/60 text-sm font-mono">
            Sign in to access your meeting intelligence
          </p>
        </div>
        <SignIn 
          appearance={{
            elements: {
              rootBox: "mx-auto",
              card: "bg-white shadow-xl",
            }
          }}
        />
      </div>
    </div>
  );
}
