import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Cloud,
  Crosshair,
  Loader2,
  LogIn,
  Mail,
  Shield,
  UserRound,
} from 'lucide-react';

/**
 * App entry — email-only sign-in (no password) or local guest mode.
 */
export default function WelcomeScreen({ onLogin, onGuestContinue }) {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    const trimmed = email.trim();
    if (!trimmed) {
      setError('Enter an email address to continue.');
      return;
    }
    setError('');
    setSubmitting(true);
    try {
      await onLogin(trimmed);
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="relative flex min-h-screen overflow-hidden bg-slate-950 text-base text-slate-200">
      <div className="pointer-events-none fixed inset-0 gl-cyber-grid" aria-hidden />
      <div
        className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_90%_70%_at_15%_20%,rgba(34,211,238,0.12),transparent_55%),radial-gradient(ellipse_70%_60%_at_85%_80%,rgba(59,130,246,0.1),transparent_50%)]"
        aria-hidden
      />
      <div className="gl-app-scanlines pointer-events-none fixed inset-0" aria-hidden />

      <div className="relative z-10 mx-auto flex w-full max-w-6xl flex-1 flex-col lg:flex-row lg:items-stretch">
        {/* Brand panel */}
        <motion.section
          initial={{ opacity: 0, x: -24 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="flex flex-1 flex-col justify-center px-6 py-12 lg:px-10 lg:py-16"
        >
          <div className="mb-8 flex h-14 w-14 items-center justify-center rounded-2xl border border-cyan-400/35 bg-slate-900/90 shadow-[0_0_40px_rgba(34,211,238,0.22)]">
            <Crosshair className="h-8 w-8 text-cyan-400" strokeWidth={1.25} aria-hidden />
          </div>
          <p className="font-display text-sm font-bold uppercase tracking-[0.4em] text-cyan-500/80">
            Tactical analytics platform
          </p>
          <h1 className="mt-4 font-display text-4xl font-extrabold leading-tight tracking-tight text-transparent bg-linear-to-br from-cyan-100 via-cyan-300 to-blue-600 bg-clip-text md:text-5xl lg:text-6xl">
            Welcome to GameLens
          </h1>
          <p className="mt-5 max-w-md font-data text-base leading-relaxed text-slate-300 md:text-base">
            Analyze runs, tune AI detectors, and compare game versions — from your desktop,
            with optional cloud sync when you&apos;re ready.
          </p>
          <ul className="mt-10 space-y-4">
            {[
              {
                icon: Cloud,
                title: 'Cloud Sync',
                text: 'Any email registers an account — sync data across machines.',
              },
              {
                icon: Shield,
                title: 'Local-first',
                text: 'Run fully offline as a guest; nothing is sent until you sign in.',
              },
            ].map(({ icon: Icon, title, text }) => (
              <li key={title} className="flex gap-3">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-800 bg-slate-900/60 text-cyan-400/90">
                  <Icon className="h-4 w-4" strokeWidth={1.5} aria-hidden />
                </span>
                <div>
                  <p className="font-display text-sm font-bold uppercase tracking-[0.18em] text-slate-300">
                    {title}
                  </p>
                  <p className="font-data mt-0.5 text-base text-slate-300 font-medium">{text}</p>
                </div>
              </li>
            ))}
          </ul>
        </motion.section>

        {/* Auth card */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.08 }}
          className="flex flex-1 items-center justify-center px-6 py-10 lg:px-10"
        >
          <div className="w-full max-w-md rounded-2xl border border-cyan-500/20 bg-slate-900/55 p-8 shadow-[0_0_60px_rgba(34,211,238,0.08),inset_0_1px_0_rgba(255,255,255,0.04)] backdrop-blur-xl ring-1 ring-slate-800/80">
            <div className="mb-6 flex items-center gap-2">
              <LogIn className="h-5 w-5 text-cyan-400" strokeWidth={1.5} aria-hidden />
              <h2 className="font-display text-sm font-bold uppercase tracking-[0.2em] text-cyan-200/90">
                Get started
              </h2>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
              <div>
                <label
                  htmlFor="welcome-email"
                  className="font-display text-sm font-bold uppercase tracking-[0.2em] text-slate-300"
                >
                  Email address
                </label>
                <div className="relative mt-2">
                  <Mail
                    className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-300"
                    aria-hidden
                  />
                  <input
                    id="welcome-email"
                    type="email"
                    autoComplete="email"
                    autoFocus
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    disabled={submitting}
                    className="font-data w-full rounded-xl border border-slate-700 bg-slate-950/80 py-3 pl-10 pr-4 text-base text-slate-100 placeholder:text-slate-400 outline-none transition focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 disabled:opacity-60"
                  />
                </div>
                <p className="font-data mt-2.5 text-base leading-relaxed text-slate-300">
                  Sign in to enable Cloud Sync and access your analytics from anywhere. No
                  password required — any email creates or restores your account.
                </p>
              </div>

              {error ? (
                <p
                  className="rounded-lg border border-red-500/30 bg-red-950/40 px-3 py-2 font-data text-sm text-red-300"
                  role="alert"
                >
                  {error}
                </p>
              ) : null}

              <button
                type="submit"
                disabled={submitting}
                className="flex w-full items-center justify-center gap-2 rounded-xl border border-cyan-500/40 bg-linear-to-b from-cyan-600 to-cyan-800 py-3.5 font-display text-sm font-bold uppercase tracking-[0.16em] text-white shadow-[0_0_28px_rgba(34,211,238,0.25)] transition hover:shadow-[0_0_36px_rgba(34,211,238,0.35)] disabled:cursor-not-allowed disabled:opacity-50"
              >
                {submitting ? (
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                ) : (
                  <LogIn className="h-4 w-4" aria-hidden />
                )}
                Sign In / Register
              </button>
            </form>

            <div className="relative my-7">
              <div className="absolute inset-0 flex items-center" aria-hidden>
                <div className="w-full border-t border-slate-800" />
              </div>
              <div className="relative flex justify-center">
                <span className="bg-slate-900/80 px-3 font-data text-sm font-semibold uppercase tracking-widest text-slate-300">
                  or
                </span>
              </div>
            </div>

            <button
              type="button"
              onClick={onGuestContinue}
              disabled={submitting}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-700 bg-transparent py-3 font-display text-sm font-bold uppercase tracking-[0.14em] text-slate-300 transition hover:border-slate-600 hover:bg-slate-800/40 hover:text-slate-200 disabled:opacity-50"
            >
              <UserRound className="h-4 w-4 shrink-0" strokeWidth={1.5} aria-hidden />
              Continue as Guest (Local Mode)
            </button>
            <p className="font-data mt-3 text-center text-base text-slate-300">
              Skips cloud sync · data stays on this device
            </p>
          </div>
        </motion.section>
      </div>
    </div>
  );
}
