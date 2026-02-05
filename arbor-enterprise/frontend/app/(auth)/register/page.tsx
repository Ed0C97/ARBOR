"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Eye, EyeOff, Loader2, ArrowRight } from "lucide-react";

export default function RegisterPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/v1/auth/register`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, email, password }),
        },
      );

      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: "Registration failed" }));
        throw new Error(body.detail ?? "Could not create account");
      }

      // Auto-login after registration
      const loginRes = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/v1/auth/login`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        },
      );

      if (loginRes.ok) {
        const { access_token } = await loginRes.json();
        localStorage.setItem("arbor_token", access_token);
      }

      router.push("/discover");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-jetbrains-ink p-4">
      <div className="w-full max-w-[400px] space-y-6">

        {/* Header Logo */}
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded bg-gradient-to-br from-jetbrains-blue to-jetbrains-purple text-white shadow-glow">
            <span className="font-mono text-2xl font-bold">A</span>
          </div>
          <div className="space-y-1">
            <h1 className="text-xl font-bold tracking-tight text-jetbrains-contrast">
              Initialize Account
            </h1>
            <p className="text-sm text-jetbrains-gray-light font-mono">
              Create your curator identity
            </p>
          </div>
        </div>

        <div className="rounded-sm border border-jetbrains-gray/30 bg-jetbrains-dark p-8 shadow-lg">
          {/* Error */}
          {error && (
            <div className="mb-6 border-l-2 border-jetbrains-berry bg-jetbrains-berry/10 p-3 text-sm text-jetbrains-berry">
              <span className="font-bold">Error:</span> {error}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="name" className="text-xs font-mono font-medium text-jetbrains-contrast uppercase tracking-wider">
                Full Name
              </label>
              <input
                id="name"
                type="text"
                required
                autoComplete="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="PROMETHEUS"
                className="flex h-10 w-full rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink px-3 text-sm text-jetbrains-contrast placeholder:text-jetbrains-gray font-mono focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue transition-all"
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="email" className="text-xs font-mono font-medium text-jetbrains-contrast uppercase tracking-wider">
                Email
              </label>
              <input
                id="email"
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="user@arbor.sys"
                className="flex h-10 w-full rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink px-3 text-sm text-jetbrains-contrast placeholder:text-jetbrains-gray font-mono focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue transition-all"
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="password" className="text-xs font-mono font-medium text-jetbrains-contrast uppercase tracking-wider">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  required
                  autoComplete="new-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Min. 8 characters"
                  className="flex h-10 w-full rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink px-3 pr-10 text-sm text-jetbrains-contrast placeholder:text-jetbrains-gray font-mono focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-jetbrains-gray-light hover:text-jetbrains-contrast transition-colors"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            <div className="space-y-2">
              <label htmlFor="confirmPassword" className="text-xs font-mono font-medium text-jetbrains-contrast uppercase tracking-wider">
                Confirm Password
              </label>
              <input
                id="confirmPassword"
                type="password"
                required
                autoComplete="new-password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Repeat password"
                className="flex h-10 w-full rounded-sm border border-jetbrains-gray/50 bg-jetbrains-ink px-3 text-sm text-jetbrains-contrast placeholder:text-jetbrains-gray font-mono focus:border-jetbrains-blue focus:outline-none focus:ring-1 focus:ring-jetbrains-blue transition-all"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="group flex h-10 w-full items-center justify-center gap-2 rounded-sm bg-jetbrains-blue font-medium text-white transition-all hover:bg-jetbrains-blue/90 hover:shadow-glow disabled:opacity-50 disabled:hover:shadow-none mt-6"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Initializing...
                </>
              ) : (
                <>
                  Create Account
                  <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                </>
              )}
            </button>
          </form>
        </div>

        {/* Footer */}
        <p className="text-center text-sm text-jetbrains-gray-light">
          Already verified?{" "}
          <Link href="/login" className="font-medium text-jetbrains-blue hover:underline hover:text-jetbrains-blue/80 decoration-jetbrains-blue/30 underline-offset-4">
            Access System
          </Link>
        </p>
      </div>
    </div>
  );
}
