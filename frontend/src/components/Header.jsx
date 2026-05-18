import { useState } from "react";
import { createPortal } from "react-dom";
import {
  Activity,
  Crosshair,
  LogIn,
  LogOut,
  RefreshCw,
  Settings,
} from "lucide-react";

export default function Header({ data, onLogin, onLogout, onOpenSettings }) {
  const status = data.processing.status;
  const auth = data.auth ?? {
    loggedIn: false,
    email: null,
    syncStatus: "idle",
    syncMessage: "",
  };
  const [loginOpen, setLoginOpen] = useState(false);
  const [emailDraft, setEmailDraft] = useState("");
  const [loginError, setLoginError] = useState("");

  const online =
    status === "running"
      ? "busy"
      : status === "stopped"
        ? "degraded"
        : "online";

  const label =
    status === "running"
      ? "Pipeline active"
      : status === "stopped"
        ? "Halted"
        : status === "completed"
          ? "Last run OK"
          : "System online";

  const dotClass =
    online === "busy"
      ? "bg-cyan-400 shadow-[0_0_12px_rgba(34,211,238,0.9)] animate-pulse"
      : online === "degraded"
        ? "bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.7)]"
        : "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.7)]";

  async function submitLogin(e) {
    e.preventDefault();
    const email = emailDraft.trim();
    if (!email) return;
    setLoginError("");
    try {
      await onLogin(email);
      setLoginOpen(false);
      setEmailDraft("");
    } catch (err) {
      setLoginError(String(err?.message || err));
    }
  }

  return (
    <header className="relative z-20 border-b border-cyan-500/10 bg-slate-950/65 px-4 py-4 backdrop-blur-2xl">
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-linear-to-r from-transparent via-cyan-400/25 to-transparent" />
      <div className="mx-auto flex max-w-[1800px] items-center justify-between gap-6">
        {/* Logo */}
        <div className="flex items-center gap-4">
          <div className="relative flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-400/35 bg-slate-900/90 shadow-[0_0_28px_rgba(34,211,238,0.2),inset_0_0_20px_rgba(59,130,246,0.08)]">
            <Crosshair
              className="h-6 w-6 text-cyan-400"
              strokeWidth={1.25}
              aria-hidden
            />
          </div>
          <div>
            <h1 className="font-display text-xl font-extrabold tracking-[0.12em] text-transparent [text-shadow:0_0_24px_rgba(34,211,238,0.35)] bg-linear-to-b from-cyan-200 via-cyan-400 to-blue-600 bg-clip-text md:text-2xl">
              GAMELENS
            </h1>
            <p className="font-data text-[10px] font-medium uppercase tracking-[0.35em] text-blue-500/60">
              Dev build console
            </p>
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-3">
          {/* Sync badge — only when logged in */}
          {auth.loggedIn && auth.syncStatus === "syncing" && (
            <div className="flex items-center gap-1.5 rounded-lg border border-cyan-500/20 bg-slate-900/60 px-3 py-1.5">
              <RefreshCw
                className="h-3 w-3 animate-spin text-cyan-400"
                aria-hidden
              />
              <span className="font-data text-[10px] text-cyan-300/80 uppercase tracking-wider">
                Syncing…
              </span>
            </div>
          )}

          {/* Auth */}
          {auth.loggedIn ? (
            <div className="flex items-center gap-2 rounded-2xl border border-slate-800/90 bg-slate-900/50 px-4 py-2.5 backdrop-blur-md">
              <div className="text-right">
                <p className="font-data text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                  Signed in
                </p>
                <p className="font-data text-xs font-semibold text-cyan-100/80 truncate max-w-[160px]">
                  {auth.email}
                </p>
              </div>
              <button
                onClick={onLogout}
                className="ml-1 rounded-lg border border-slate-700 bg-slate-800/60 p-1.5 text-slate-400 hover:text-red-400 hover:border-red-500/40 transition-colors"
                title="Sign out"
              >
                <LogOut className="h-3.5 w-3.5" aria-hidden />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setLoginOpen(true)}
              className="flex items-center gap-2 rounded-2xl border border-slate-700/80 bg-slate-900/50 px-4 py-2.5 text-slate-300 hover:text-cyan-300 hover:border-cyan-500/40 backdrop-blur-md transition-colors"
            >
              <LogIn className="h-4 w-4" aria-hidden />
              <span className="font-data text-sm font-semibold tracking-wide">
                Sign in
              </span>
            </button>
          )}

          {/* Pipeline status */}
          <div className="flex items-center gap-3 rounded-2xl border border-slate-800/90 bg-slate-900/50 px-4 py-2.5 backdrop-blur-md">
            <div
              className={`h-2.5 w-2.5 shrink-0 rounded-full ${dotClass}`}
              aria-hidden
            />
            <div className="text-right">
              <p className="font-data text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                Status
              </p>
              <p className="font-data text-sm font-semibold text-cyan-100/90">
                {label}
              </p>
            </div>
            <Activity
              className="hidden h-4 w-4 text-cyan-500/50 sm:block"
              aria-hidden
            />
          </div>

          {/* Settings */}
          <button
            type="button"
            onClick={onOpenSettings}
            className="group flex h-11 w-11 items-center justify-center rounded-xl border border-cyan-500/25 bg-slate-900/60 text-cyan-300/85 shadow-[0_0_24px_rgba(34,211,238,0.08)] backdrop-blur-md transition hover:border-cyan-400/45 hover:bg-cyan-500/10 hover:text-cyan-200"
            aria-label="Open settings"
          >
            <Settings
              className="h-4.5 w-4.5 transition group-hover:rotate-12"
              aria-hidden
            />
          </button>
        </div>
      </div>

      {/* Login modal */}
      {loginOpen &&
        createPortal(
          <div className="fixed inset-0 z-120 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm">
            <form
              onSubmit={submitLogin}
              className="w-full max-w-sm rounded-2xl border border-cyan-500/20 bg-slate-900 p-6 shadow-2xl"
            >
              <h2 className="font-display text-lg font-bold tracking-widest text-cyan-300 mb-4 uppercase">
                Sign in
              </h2>
              <label className="block font-data text-xs text-slate-400 mb-1 uppercase tracking-widest">
                Email
              </label>
              <input
                type="email"
                autoFocus
                value={emailDraft}
                onChange={(e) => setEmailDraft(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 font-data text-sm text-slate-100 placeholder:text-slate-500 focus:border-cyan-500/60 focus:outline-none"
                placeholder="you@example.com"
              />
              {loginError && (
                <p className="mt-2 font-data text-xs text-red-400">
                  {loginError}
                </p>
              )}
              <div className="mt-4 flex gap-2 justify-end">
                <button
                  type="button"
                  onClick={() => {
                    setLoginOpen(false);
                    setLoginError("");
                  }}
                  className="rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 font-data text-sm text-slate-400 hover:text-slate-200 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded-lg bg-cyan-600 px-4 py-2 font-data text-sm font-bold text-slate-950 hover:bg-cyan-500 transition-colors"
                >
                  Sign in
                </button>
              </div>
            </form>
          </div>,
          document.body,
        )}
    </header>
  );
}
