"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Eye, EyeOff, Loader2, ArrowRight } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    // Demo mode: bypass authentication
    if (email === "demo@arbor.ai" && password === "demo") {
      setTimeout(() => {
        localStorage.setItem("arbor_token", "demo_token");
        router.push("/dashboard");
      }, 500);
      return;
    }

    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/v1/auth/login`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        }
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: "Login failed" }));
        throw new Error(body.detail ?? "Invalid credentials");
      }
      const { access_token } = await res.json();
      localStorage.setItem("arbor_token", access_token);
      router.push("/dashboard");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4 relative overflow-hidden">
      {/* Grid background */}
      <div className="absolute inset-0 grid-pattern opacity-30 dark:opacity-20" />

      {/* Animated orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-primary/5 blur-3xl animate-pulse" style={{animationDelay: "1s"}} />

      <div className="w-full max-w-[400px] space-y-6 relative z-10">
        {/* Logo */}
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="flex h-10 w-10 items-center justify-center bg-[#4361FF] text-white text-lg font-bold font-mono">
            A
          </div>
          <div className="space-y-1">
            <h1 className="text-xl font-semibold text-foreground font-mono">
              Sign in to ARBOR
            </h1>
            <p className="text-sm text-muted-foreground font-mono">
              Enter your credentials to continue
            </p>
          </div>
        </div>

        {/* Demo credentials hint */}
        <div className="border-2 border-primary/20 bg-primary/5 p-4 text-center">
          <p className="text-xs font-mono text-primary">
            <strong>Demo Access:</strong> demo@arbor.ai / demo
          </p>
        </div>

        <div className="border-2 border-border bg-card p-8">
          {error && (
            <div className="mb-6 bg-destructive/10 border-2 border-destructive/20 p-3 text-sm text-destructive font-mono">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-2">
              <label
                htmlFor="email"
                className="text-sm font-medium text-foreground font-mono"
              >
                Email
              </label>
              <input
                id="email"
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="user@arbor.ai"
                className="flex h-10 w-full border-2 border-border bg-background px-3 text-sm text-foreground font-mono placeholder:text-muted-foreground focus:border-primary focus:outline-none transition-all"
              />
            </div>

            <div className="space-y-2">
              <label
                htmlFor="password"
                className="text-sm font-medium text-foreground font-mono"
              >
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  required
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="flex h-10 w-full border-2 border-border bg-background px-3 pr-10 text-sm text-foreground font-mono placeholder:text-muted-foreground focus:border-primary focus:outline-none transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="group flex h-10 w-full items-center justify-center gap-2 bg-primary font-medium font-mono text-white transition-all hover:bg-primary/90 disabled:opacity-50 relative overflow-hidden"
            >
              {/* Scan line effect */}
              <div className="absolute inset-0 scan-line opacity-0 group-hover:opacity-100" />
              <span className="relative z-10">
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin inline mr-2" />
                    Signing in...
                  </>
                ) : (
                  <>
                    Sign in
                    <ArrowRight className="h-4 w-4 inline ml-2 transition-transform group-hover:translate-x-0.5" />
                  </>
                )}
              </span>
            </button>
          </form>
        </div>

        <p className="text-center text-sm text-muted-foreground font-mono">
          No account?{" "}
          <Link
            href="/register"
            className="font-medium text-primary hover:text-primary/80 transition-colors underline"
          >
            Request access
          </Link>
        </p>
      </div>
    </div>
  );
}
export const dynamic = 'force-dynamic';
