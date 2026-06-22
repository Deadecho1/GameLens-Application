import { useState } from 'react';
import { motion } from 'framer-motion';
import { Loader2, LogIn, Mail, UserRound } from 'lucide-react';

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
    <div className="relative min-h-screen overflow-hidden text-slate-200">
      {/* Full-screen background — drop in <video> above the mesh when ready */}
      <div className="absolute inset-0 z-0" aria-hidden>
        {/*
          <video
            className="absolute inset-0 h-full w-full object-cover"
            autoPlay
            muted
            loop
            playsInline
            poster="/welcome-poster.jpg"
          >
            <source src="/welcome-bg.mp4" type="video/mp4" />
          </video>
        */}
        <div className="gl-welcome-mesh absolute inset-0" />
        <div className="absolute inset-0 bg-slate-950/25" />
      </div>

      <div className="gl-app-scanlines pointer-events-none absolute inset-0 z-1 opacity-40" aria-hidden />

      <div className="relative z-10 flex min-h-screen items-center justify-center px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
          className="flex w-full max-w-md flex-col items-center text-center"
        >
          <h1 className="gl-welcome-title font-display text-4xl font-bold tracking-tight text-white md:text-5xl">
            Welcome to GameLens
          </h1>
          <p className="font-data mt-4 max-w-sm text-base leading-relaxed text-slate-300 md:text-lg">
            Automated gameplay analysis through computer vision.
          </p>

          <div className="gl-welcome-glass mt-10 w-full rounded-2xl p-8 text-left">
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
                    className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
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
                    className="font-data w-full rounded-xl border border-white/10 bg-black/25 py-3 pl-10 pr-4 text-base text-slate-100 placeholder:text-slate-500 outline-none backdrop-blur-sm transition focus:border-cyan-400/40 focus:ring-2 focus:ring-cyan-400/15 disabled:opacity-60"
                  />
                </div>
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
                className="gl-welcome-submit flex w-full items-center justify-center gap-2 rounded-xl border border-cyan-400/30 py-3.5 font-display text-sm font-bold uppercase tracking-[0.16em] text-white disabled:cursor-not-allowed disabled:opacity-50"
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
                <div className="w-full border-t border-white/10" />
              </div>
              <div className="relative flex justify-center">
                <span className="bg-transparent px-3 font-data text-sm font-semibold uppercase tracking-widest text-slate-400">
                  or
                </span>
              </div>
            </div>

            <button
              type="button"
              onClick={onGuestContinue}
              disabled={submitting}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 py-3 font-display text-sm font-bold uppercase tracking-[0.14em] text-slate-300 backdrop-blur-sm transition hover:border-cyan-400/25 hover:bg-white/10 hover:text-slate-100 disabled:opacity-50"
            >
              <UserRound className="h-4 w-4 shrink-0" strokeWidth={1.5} aria-hidden />
              Continue as Guest (Local Mode)
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
