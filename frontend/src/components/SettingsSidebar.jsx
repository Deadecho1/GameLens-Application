import { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Eye, EyeOff, KeyRound, Settings, UserRound, X } from 'lucide-react';

function SecureField({
  id,
  label,
  value,
  onChange,
  visible,
  onToggleVisible,
  placeholder,
}) {
  return (
    <label htmlFor={id} className="block space-y-1.5">
      <span className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
        {label}
      </span>
      <div className="relative">
        <input
          id={id}
          type={visible ? 'text' : 'password'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full rounded-xl border border-slate-700/90 bg-slate-950/80 px-3 py-2.5 pr-11 font-data text-sm text-slate-100 outline-none transition focus:border-cyan-500/55 focus:ring-1 focus:ring-cyan-500/35"
        />
        <button
          type="button"
          onClick={onToggleVisible}
          className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-slate-500 transition hover:bg-slate-800 hover:text-cyan-300"
          aria-label={visible ? 'Hide value' : 'Show value'}
        >
          {visible ? <EyeOff className="h-4 w-4" aria-hidden /> : <Eye className="h-4 w-4" aria-hidden />}
        </button>
      </div>
    </label>
  );
}

function ModalShell({ open, title, icon: Icon, onClose, children }) {
  return (
    <AnimatePresence>
      {open ? (
        <>
          <motion.button
            type="button"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-82 bg-slate-950/82 backdrop-blur-xl"
            onClick={onClose}
            aria-label="Close modal backdrop"
          />
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.985 }}
            transition={{ type: 'spring', stiffness: 320, damping: 30 }}
            className="fixed inset-0 z-83 flex items-center justify-center p-4"
          >
            <div className="w-full max-w-md overflow-hidden rounded-2xl border border-cyan-500/25 bg-slate-950/92 shadow-[0_0_80px_rgba(34,211,238,0.18)] backdrop-blur-2xl">
              <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4 text-cyan-400/85" aria-hidden />
                  <p className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-200/90">
                    {title}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={onClose}
                  className="rounded-lg p-2 text-slate-500 transition hover:bg-slate-800 hover:text-cyan-300"
                  aria-label="Close modal"
                >
                  <X className="h-4 w-4" aria-hidden />
                </button>
              </div>
              {children}
            </div>
          </motion.div>
        </>
      ) : null}
    </AnimatePresence>
  );
}

export default function SettingsSidebar({ data, onPatch, open, onClose }) {
  const setup = data?.setup ?? {};
  const selectedGame = setup.selectedGame ?? '';
  const selectedVersion = setup.selectedVersion ?? '';
  const games = Array.isArray(setup.games) ? setup.games : [];
  const versions = Array.isArray(setup.versions) ? setup.versions : [];
  const user = setup.user ?? {};
  const initials = useMemo(() => {
    const f = String(user.firstName ?? '').trim().charAt(0);
    const l = String(user.lastName ?? '').trim().charAt(0);
    return `${f}${l}`.trim() || 'A';
  }, [user.firstName, user.lastName]);
  const displayName = [user.firstName, user.lastName].filter(Boolean).join(' ').trim() || 'Admin';

  const [editModalOpen, setEditModalOpen] = useState(false);
  const [keyModalOpen, setKeyModalOpen] = useState(false);
  const [profileDraft, setProfileDraft] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
  });
  const [keyDraft, setKeyDraft] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    if (!editModalOpen) return;
    setProfileDraft({
      firstName: user.firstName ?? '',
      lastName: user.lastName ?? '',
      email: user.email ?? '',
      password: user.password ?? '',
    });
    setShowPassword(false);
  }, [editModalOpen, user.email, user.firstName, user.lastName, user.password]);

  useEffect(() => {
    if (!keyModalOpen) return;
    setKeyDraft(user.openAiKey ?? '');
    setShowKey(false);
  }, [keyModalOpen, user.openAiKey]);

  const saveProfile = () => {
    onPatch({
      setup: {
        ...setup,
        user: {
          ...user,
          firstName: profileDraft.firstName,
          lastName: profileDraft.lastName,
          email: profileDraft.email,
          password: profileDraft.password,
        },
      },
    });
    setEditModalOpen(false);
  };

  const saveKey = () => {
    onPatch({
      setup: {
        ...setup,
        user: {
          ...user,
          openAiKey: keyDraft,
        },
      },
    });
    setKeyModalOpen(false);
  };

  return (
    <>
      <AnimatePresence>
        {open ? (
          <>
            <motion.button
              type="button"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-75 bg-slate-950/70 backdrop-blur-sm"
              onClick={onClose}
              aria-label="Close settings sidebar backdrop"
            />
            <motion.aside
              initial={{ x: '100%', opacity: 0.98 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: '100%', opacity: 0.98 }}
              transition={{ type: 'spring', stiffness: 340, damping: 36 }}
              className="fixed right-0 top-0 z-76 flex h-screen w-full max-w-md flex-col border-l border-cyan-500/20 bg-slate-950/90 shadow-[-22px_0_90px_rgba(34,211,238,0.14)] backdrop-blur-2xl"
            >
              <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
                <div className="flex items-center gap-2">
                  <Settings className="h-4 w-4 text-cyan-400/85" aria-hidden />
                  <p className="font-display text-[10px] font-bold uppercase tracking-[0.22em] text-cyan-100/90">
                    Settings
                  </p>
                </div>
                <button
                  type="button"
                  onClick={onClose}
                  className="rounded-lg p-2 text-slate-500 transition hover:bg-slate-800 hover:text-cyan-300"
                  aria-label="Close settings sidebar"
                >
                  <X className="h-4 w-4" aria-hidden />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto px-5 py-5">
                <section className="rounded-2xl border border-cyan-500/20 bg-slate-900/35 p-4 ring-1 ring-cyan-500/10">
                  <div className="flex items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-500/35 bg-slate-950/75 font-display text-sm font-bold uppercase tracking-[0.14em] text-cyan-200">
                      {initials}
                    </div>
                    <div className="min-w-0">
                      <p className="truncate font-data text-sm font-semibold text-slate-100">{displayName}</p>
                      <p className="truncate font-data text-[11px] text-slate-500">{user.email ?? 'admin@gamelens.io'}</p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setEditModalOpen(true)}
                    className="mt-4 w-full rounded-xl border border-cyan-500/35 bg-cyan-500/10 px-3 py-2 font-display text-[9px] font-bold uppercase tracking-[0.2em] text-cyan-100 transition hover:border-cyan-400/60 hover:bg-cyan-500/15"
                  >
                    Edit
                  </button>
                </section>

                <section className="mt-5 space-y-4 rounded-2xl border border-slate-800 bg-slate-900/30 p-4">
                  <p className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                    Analytics targeting
                  </p>

                  <label className="block space-y-1.5">
                    <span className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                      Game
                    </span>
                    <select
                      value={selectedGame}
                      onChange={(e) => onPatch({ setup: { ...setup, selectedGame: e.target.value } })}
                      className="w-full rounded-xl border border-slate-700/90 bg-slate-950/80 px-3 py-2.5 font-data text-sm text-slate-100 outline-none transition focus:border-cyan-500/55 focus:ring-1 focus:ring-cyan-500/35"
                    >
                      {games.map((game) => (
                        <option key={game} value={game}>
                          {game}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="block space-y-1.5">
                    <span className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                      Version
                    </span>
                    <select
                      value={selectedVersion}
                      onChange={(e) => onPatch({ setup: { ...setup, selectedVersion: e.target.value } })}
                      className="w-full rounded-xl border border-slate-700/90 bg-slate-950/80 px-3 py-2.5 font-data text-sm text-slate-100 outline-none transition focus:border-cyan-500/55 focus:ring-1 focus:ring-cyan-500/35"
                    >
                      {versions.map((version) => (
                        <option key={version} value={version}>
                          {version}
                        </option>
                      ))}
                    </select>
                  </label>
                </section>
              </div>

              <div className="border-t border-slate-800 px-5 py-4">
                <button
                  type="button"
                  onClick={() => setKeyModalOpen(true)}
                  className="flex w-full items-center justify-center gap-2 rounded-xl border border-violet-500/35 bg-violet-500/10 px-3 py-2.5 font-display text-[9px] font-bold uppercase tracking-[0.2em] text-violet-100 transition hover:border-violet-400/55 hover:bg-violet-500/15"
                >
                  <KeyRound className="h-4 w-4" aria-hidden />
                  OpenAI API Configuration
                </button>
              </div>
            </motion.aside>
          </>
        ) : null}
      </AnimatePresence>

      <ModalShell
        open={editModalOpen}
        title="Edit profile"
        icon={UserRound}
        onClose={() => setEditModalOpen(false)}
      >
        <div className="space-y-4 px-5 py-5">
          <label htmlFor="settings-first-name" className="block space-y-1.5">
            <span className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
              First name
            </span>
            <input
              id="settings-first-name"
              value={profileDraft.firstName}
              onChange={(e) => setProfileDraft((p) => ({ ...p, firstName: e.target.value }))}
              className="w-full rounded-xl border border-slate-700/90 bg-slate-950/80 px-3 py-2.5 font-data text-sm text-slate-100 outline-none transition focus:border-cyan-500/55 focus:ring-1 focus:ring-cyan-500/35"
            />
          </label>
          <label htmlFor="settings-last-name" className="block space-y-1.5">
            <span className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
              Last name
            </span>
            <input
              id="settings-last-name"
              value={profileDraft.lastName}
              onChange={(e) => setProfileDraft((p) => ({ ...p, lastName: e.target.value }))}
              className="w-full rounded-xl border border-slate-700/90 bg-slate-950/80 px-3 py-2.5 font-data text-sm text-slate-100 outline-none transition focus:border-cyan-500/55 focus:ring-1 focus:ring-cyan-500/35"
            />
          </label>
          <label htmlFor="settings-email" className="block space-y-1.5">
            <span className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
              Email
            </span>
            <input
              id="settings-email"
              type="email"
              value={profileDraft.email}
              onChange={(e) => setProfileDraft((p) => ({ ...p, email: e.target.value }))}
              className="w-full rounded-xl border border-slate-700/90 bg-slate-950/80 px-3 py-2.5 font-data text-sm text-slate-100 outline-none transition focus:border-cyan-500/55 focus:ring-1 focus:ring-cyan-500/35"
            />
          </label>

          <SecureField
            id="settings-password"
            label="Password"
            value={profileDraft.password}
            onChange={(value) => setProfileDraft((p) => ({ ...p, password: value }))}
            visible={showPassword}
            onToggleVisible={() => setShowPassword((s) => !s)}
          />

          <div className="flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={() => setEditModalOpen(false)}
              className="rounded-xl border border-slate-700 px-3 py-2 font-display text-[9px] font-bold uppercase tracking-[0.16em] text-slate-300 transition hover:border-slate-500 hover:text-slate-100"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={saveProfile}
              className="rounded-xl border border-cyan-500/45 bg-cyan-500/12 px-3 py-2 font-display text-[9px] font-bold uppercase tracking-[0.16em] text-cyan-100 transition hover:border-cyan-400/65 hover:bg-cyan-500/18"
            >
              Save
            </button>
          </div>
        </div>
      </ModalShell>

      <ModalShell
        open={keyModalOpen}
        title="OpenAI API key"
        icon={KeyRound}
        onClose={() => setKeyModalOpen(false)}
      >
        <div className="space-y-4 px-5 py-5">
          <SecureField
            id="settings-openai-key"
            label="API key"
            value={keyDraft}
            onChange={setKeyDraft}
            visible={showKey}
            onToggleVisible={() => setShowKey((s) => !s)}
            placeholder="sk-..."
          />

          <div className="flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={() => setKeyModalOpen(false)}
              className="rounded-xl border border-slate-700 px-3 py-2 font-display text-[9px] font-bold uppercase tracking-[0.16em] text-slate-300 transition hover:border-slate-500 hover:text-slate-100"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={saveKey}
              className="rounded-xl border border-violet-500/45 bg-violet-500/12 px-3 py-2 font-display text-[9px] font-bold uppercase tracking-[0.16em] text-violet-100 transition hover:border-violet-400/65 hover:bg-violet-500/18"
            >
              Save key
            </button>
          </div>
        </div>
      </ModalShell>
    </>
  );
}
