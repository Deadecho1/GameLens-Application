import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { BarChart3, ChevronDown, Columns2, Gamepad2, GitCompare } from "lucide-react";
import GeneralMissionStats from "../analytics/GeneralMissionStats";
import BossesAnalytics from "../analytics/BossesAnalytics";
import ItemsPowerLab from "../analytics/ItemsPowerLab";
import { sliceAnalyticsDataByVersion } from "../../utils/analyticsVersionSlice";

const SUB_TABS = [
  { id: "general", label: "GENERAL" },
  { id: "bosses", label: "BOSSES" },
  { id: "items", label: "ITEMS" },
];

function renderActiveSubView(
  sub,
  data,
  {
    compact = false,
    compareBaseline = null,
    itemsSubTab = "build",
    onItemsSubTabChange,
  } = {},
) {
  if (sub === "general")
    return (
      <GeneralMissionStats data={data} compareBaseline={compareBaseline} />
    );
  if (sub === "bosses")
    return <BossesAnalytics data={data} compareBaseline={compareBaseline} />;
  if (sub === "items")
    return (
      <ItemsPowerLab
        data={data}
        compact={compact}
        compareBaseline={compareBaseline}
        itemsSubTab={itemsSubTab}
        onItemsSubTabChange={onItemsSubTabChange}
      />
    );
  return null;
}

function applyGameLibrarySlice(data, game, version) {
  if (!data || typeof data !== "object") return data;
  const dashboard = data.dashboard ?? {};
  const gameLibrary = dashboard.gameLibrary ?? {};
  const gameNode = game ? gameLibrary[game] : null;
  const versionNode = version ? gameNode?.[version] : null;
  if (!versionNode) return data;
  return {
    ...data,
    dashboard: {
      ...dashboard,
      items: versionNode.items ?? dashboard.items ?? [],
      bosses: versionNode.bosses ?? dashboard.bosses ?? [],
      generalStats: versionNode.generalStats ?? dashboard.generalStats ?? [],
    },
  };
}

function VersionDropdown({
  id,
  listboxId,
  label,
  value,
  versions,
  onChange,
  disabled,
  isOpen,
  onOpenChange,
}) {
  const rootRef = useRef(null);

  const close = useCallback(() => onOpenChange(false), [onOpenChange]);

  useEffect(() => {
    if (!isOpen) return;
    const onDocDown = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) close();
    };
    const onKey = (e) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("mousedown", onDocDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [isOpen, close]);

  const toggle = () => {
    if (disabled) return;
    onOpenChange(!isOpen);
  };

  const selectVersion = (v) => {
    onChange(v);
    close();
  };

  return (
    <div
      ref={rootRef}
      className="relative inline-flex max-w-full shrink-0 flex-col items-start gap-1"
    >
      <span
        id={`${id}-label`}
        className="shrink-0 font-display text-[9px] font-bold uppercase tracking-[0.18em] text-slate-500"
      >
        {label}
      </span>
      <button
        type="button"
        id={id}
        aria-labelledby={`${id}-label`}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={listboxId}
        disabled={disabled}
        onClick={toggle}
        className={`font-data relative box-border flex h-8 min-w-[5ch] max-w-[16rem] w-max cursor-pointer items-center justify-center gap-1 rounded-lg border bg-slate-900 px-3 pr-8 text-xs leading-none text-slate-200 outline-none transition ring-0 focus:outline-none focus-visible:outline-none focus-visible:ring-0 disabled:cursor-not-allowed disabled:opacity-50 ${
          isOpen ? "border-cyan-500/50" : "border-slate-800"
        }`}
      >
        <span className="min-w-0 truncate text-center">{value || "—"}</span>
        <ChevronDown
          className={`pointer-events-none absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-500 transition-transform ${isOpen ? "rotate-180" : ""}`}
          strokeWidth={2}
          aria-hidden
        />
      </button>
      {isOpen && versions.length > 0 ? (
        <ul
          id={listboxId}
          role="listbox"
          aria-labelledby={`${id}-label`}
          className="absolute left-0 top-[calc(100%+4px)] z-[45] min-w-full max-w-[min(18rem,calc(100vw-2rem))] overflow-hidden rounded-lg border border-slate-800 bg-slate-900/95 py-1 shadow-xl shadow-black/40 backdrop-blur-md"
        >
          {versions.map((v) => {
            const selected = v === value;
            return (
              <li key={v} role="presentation">
                <button
                  type="button"
                  role="option"
                  aria-selected={selected}
                  className={`font-data w-full px-3 py-2 text-left text-xs leading-snug outline-none ring-0 transition focus:outline-none ${
                    selected
                      ? "bg-slate-800/55 text-cyan-100 hover:bg-cyan-500/20 hover:text-cyan-200 focus:bg-cyan-500/20 focus:text-cyan-200"
                      : "text-slate-200 hover:bg-cyan-500/20 hover:text-cyan-200 focus:bg-cyan-500/20 focus:text-cyan-200"
                  }`}
                  onClick={() => selectVersion(v)}
                >
                  {v}
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}

/**
 * ANALYTICS — secondary nav + sub-views. Writes ui.analyticsSubTab.
 * Optional split-view compares two game versions (filtered `runsHistory` only).
 */
export default function AnalyticsTab({ data, onPatch }) {
  const sub = data.ui.analyticsSubTab;
  const games = useMemo(() => {
    const list = data?.setup?.games;
    return Array.isArray(list) ? list.filter(Boolean) : [];
  }, [data?.setup?.games]);
  const selectedGame = (data?.setup?.selectedGame ?? "").trim();

  const versions = useMemo(() => {
    const list = data?.setup?.versions;
    return Array.isArray(list) ? list.filter(Boolean) : [];
  }, [data?.setup?.versions]);

  const [versionA, setVersionA] = useState(() => versions[0] ?? "");
  const [versionB, setVersionB] = useState(
    () => versions[1] ?? versions[0] ?? "",
  );
  const [splitView, setSplitView] = useState(false);
  const [itemsSubTab, setItemsSubTab] = useState("build");
  const [isOpenA, setIsOpenA] = useState(false);
  const [isOpenB, setIsOpenB] = useState(false);

  const setDropdownAOpen = useCallback((next) => {
    setIsOpenA(next);
    if (next) setIsOpenB(false);
  }, []);

  const setDropdownBOpen = useCallback((next) => {
    setIsOpenB(next);
    if (next) setIsOpenA(false);
  }, []);

  useEffect(() => {
    const a = versions[0] ?? "";
    const b = versions[1] ?? a;
    setVersionA((prev) => (versions.includes(prev) ? prev : a));
    setVersionB((prev) => (versions.includes(prev) ? prev : b));
  }, [versions]);

  useEffect(() => {
    setIsOpenA(false);
    setIsOpenB(false);
  }, [versions]);

  const dataA = useMemo(() => {
    const source = splitView
      ? sliceAnalyticsDataByVersion(data, versionA)
      : data;
    const activeGame = (source?.setup?.selectedGame ?? "").trim();
    const activeVersion = splitView ? versionA : source?.setup?.selectedVersion;
    return applyGameLibrarySlice(source, activeGame, activeVersion);
  }, [data, splitView, versionA]);

  const dataB = useMemo(() => {
    const source = splitView
      ? sliceAnalyticsDataByVersion(data, versionB)
      : data;
    const activeGame = (source?.setup?.selectedGame ?? "").trim();
    const activeVersion = splitView ? versionB : source?.setup?.selectedVersion;
    return applyGameLibrarySlice(source, activeGame, activeVersion);
  }, [data, splitView, versionB]);

  const leftScrollRef = useRef(null);
  const rightScrollRef = useRef(null);
  const syncLock = useRef(false);

  const handleLeftScroll = useCallback((e) => {
    const right = rightScrollRef.current;
    if (!right || syncLock.current) return;
    syncLock.current = true;
    right.scrollTop = e.target.scrollTop;
    requestAnimationFrame(() => {
      syncLock.current = false;
    });
  }, []);

  const handleRightScroll = useCallback((e) => {
    const left = leftScrollRef.current;
    if (!left || syncLock.current) return;
    syncLock.current = true;
    left.scrollTop = e.target.scrollTop;
    requestAnimationFrame(() => {
      syncLock.current = false;
    });
  }, []);

  const handleSelectGame = useCallback(
    (gameName) => {
      const next = (gameName ?? "").trim();
      if (!next) return;
      onPatch({ setup: { selectedGame: next } });
    },
    [onPatch],
  );

  const goToMissionSetup = useCallback(() => {
    onPatch({
      ui: {
        ...data.ui,
        activeMainTab: "workflow",
        workflowStep: 1,
      },
    });
  }, [data.ui, onPatch]);

  if (!selectedGame) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{ duration: 0.28 }}
        className="mx-auto max-w-[1800px] px-4 py-8 md:py-10"
      >
        <header className="mb-8">
          <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
            Intelligence
          </p>
          <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
            Analytics deck
          </h2>
        </header>

        <div className="flex min-h-[min(420px,55vh)] flex-col items-center justify-center rounded-2xl border border-dashed border-slate-800 bg-slate-950/40 px-6 py-16 text-center">
          <BarChart3
            className="h-14 w-14 text-slate-700"
            strokeWidth={1.25}
            aria-hidden
          />
          <p className="mt-5 font-display text-sm font-bold uppercase tracking-[0.2em] text-slate-400">
            No game selected
          </p>
          <p className="mt-2 max-w-md font-data text-sm leading-relaxed text-slate-500">
            Please select a game to view analytics. Choose an existing game below
            or configure one under Mission Start.
          </p>

          {games.length > 0 ? (
            <label className="mt-8 flex w-full max-w-xs flex-col items-start gap-2 text-left">
              <span className="font-display text-[9px] font-bold uppercase tracking-[0.18em] text-slate-500">
                Select game
              </span>
              <select
                value=""
                onChange={(e) => handleSelectGame(e.target.value)}
                className="font-data w-full cursor-pointer rounded-xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-slate-200 outline-none transition focus:border-cyan-500/50"
              >
                <option value="" disabled>
                  Choose a game…
                </option>
                {games.map((game) => (
                  <option key={game} value={game}>
                    {game}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <button
              type="button"
              onClick={goToMissionSetup}
              className="mt-8 inline-flex items-center gap-2 rounded-xl border border-cyan-500/35 bg-cyan-500/10 px-5 py-2.5 font-display text-[10px] font-bold uppercase tracking-[0.14em] text-cyan-200 transition hover:border-cyan-400/55 hover:bg-cyan-500/15"
            >
              <Gamepad2 className="h-4 w-4 shrink-0" aria-hidden />
              Go to Mission Start
            </button>
          )}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.28 }}
      className="mx-auto max-w-[1800px] px-4 py-8 md:py-10"
    >
      <header className="mb-6">
        <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
          Intelligence
        </p>
        <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
          Analytics deck
        </h2>
      </header>

      <div
        className="sticky top-0 z-30 mb-6 overflow-visible border-b border-slate-800/80 bg-slate-950/92 px-1 py-3 shadow-[0_8px_24px_-8px_rgba(0,0,0,0.45)] backdrop-blur-md md:px-2"
        aria-label="Version comparison"
      >
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex shrink-0 items-center gap-2 text-slate-400">
            <GitCompare
              className="h-4 w-4 shrink-0 text-cyan-500/80"
              aria-hidden
            />
            <p className="font-display text-[10px] font-bold uppercase tracking-[0.22em] text-slate-300">
              Version comparison
            </p>
          </div>
          <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
            <VersionDropdown
              id="analytics-compare-version-a"
              listboxId="analytics-compare-version-a-listbox"
              label="Compare version A"
              value={versionA}
              versions={versions}
              onChange={setVersionA}
              disabled={versions.length === 0}
              isOpen={isOpenA}
              onOpenChange={setDropdownAOpen}
            />
            <VersionDropdown
              id="analytics-compare-version-b"
              listboxId="analytics-compare-version-b-listbox"
              label="Compare version B"
              value={versionB}
              versions={versions}
              onChange={setVersionB}
              disabled={versions.length === 0}
              isOpen={isOpenB}
              onOpenChange={setDropdownBOpen}
            />
            <div className="flex shrink-0 items-center gap-2 rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 transition hover:border-slate-600">
              <Columns2 className="h-4 w-4 text-slate-500" aria-hidden />
              <span
                id="analytics-split-view-label"
                className="font-display text-[9px] font-bold uppercase tracking-[0.18em] text-slate-400"
              >
                Enable split view
              </span>
              <button
                type="button"
                role="switch"
                id="analytics-split-view"
                aria-labelledby="analytics-split-view-label"
                aria-checked={splitView}
                onClick={() => setSplitView((s) => !s)}
                className={`relative h-6 w-11 shrink-0 rounded-full border transition ${
                  splitView
                    ? "border-cyan-500/50 bg-cyan-500/20"
                    : "border-slate-600 bg-slate-800/80"
                }`}
              >
                <span
                  className={`absolute top-0.5 h-5 w-5 rounded-full bg-slate-200 shadow transition-transform ${
                    splitView ? "left-5 translate-x-0.5" : "left-0.5"
                  }`}
                />
              </button>
            </div>
          </div>
        </div>
        {versions.length === 0 ? (
          <p className="font-data mt-2 text-[11px] text-amber-200/80">
            No versions in setup — add versions under Mission setup to enable
            comparison.
          </p>
        ) : null}
      </div>

      <nav
        className="mb-8 flex flex-wrap gap-2 rounded-xl border border-slate-800 bg-slate-950/40 p-2 backdrop-blur-md"
        aria-label="Analytics sections"
      >
        {SUB_TABS.map((t) => {
          const active = sub === t.id;
          const itemsTab = t.id === "items";
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={active}
              className={`relative flex-1 rounded-lg px-4 py-2.5 font-display text-[10px] font-bold tracking-[0.2em] transition sm:flex-none sm:min-w-[100px] ${
                active
                  ? itemsTab
                    ? "bg-violet-500/15 text-violet-200 ring-1 ring-violet-500/40"
                    : "bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/35"
                  : "text-slate-500 hover:bg-slate-900/60 hover:text-slate-300"
              }`}
              onClick={() =>
                onPatch({
                  ui: { ...data.ui, analyticsSubTab: t.id },
                })
              }
            >
              {t.label}
            </button>
          );
        })}
      </nav>

      <motion.div
        key={splitView ? `${sub}-split` : sub}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.22 }}
        className={splitView ? "min-h-0" : ""}
      >
        {!splitView ? (
          renderActiveSubView(sub, data, {
            itemsSubTab,
            onItemsSubTabChange: setItemsSubTab,
          })
        ) : (
          <div className="relative flex min-h-0 flex-col gap-4 lg:flex-row lg:items-stretch lg:gap-0">
            <div
              ref={leftScrollRef}
              onScroll={handleLeftScroll}
              className="analytics-split-scroll min-h-0 min-w-0 flex-1 overflow-x-hidden overflow-y-auto sm:min-h-[min(420px,50vh)] lg:max-h-[calc(100vh-11rem)] lg:min-h-0 lg:w-1/2 lg:border-r lg:border-slate-800/80 lg:pr-4"
            >
              <div className="mb-2 border-b border-slate-800/60 pb-2">
                <p className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-cyan-500/90">
                  {versionA || "Version A"}
                </p>
              </div>
              <div className="min-w-0">
                {renderActiveSubView(sub, dataA, {
                  compact: true,
                  itemsSubTab,
                  onItemsSubTabChange: setItemsSubTab,
                })}
              </div>
            </div>

            <div className="relative flex shrink-0 items-center justify-center py-2 lg:hidden">
              <div className="h-px w-full bg-gradient-to-r from-transparent via-slate-600/80 to-transparent" />
              <span className="absolute left-1/2 inline-flex -translate-x-1/2 rounded-full border border-slate-600/90 bg-slate-900/95 px-2.5 py-1 font-display text-[9px] font-bold tracking-[0.25em] text-slate-300 shadow-lg shadow-black/40 ring-1 ring-slate-800/90">
                VS
              </span>
            </div>

            <div
              ref={rightScrollRef}
              onScroll={handleRightScroll}
              className="analytics-split-scroll min-h-0 min-w-0 flex-1 overflow-x-hidden overflow-y-auto sm:min-h-[min(420px,50vh)] lg:max-h-[calc(100vh-11rem)] lg:min-h-0 lg:w-1/2 lg:pl-4"
            >
              <div className="mb-2 border-b border-slate-800/60 pb-2">
                <p className="font-display text-[9px] font-bold uppercase tracking-[0.2em] text-indigo-300/90">
                  {versionB || "Version B"}
                </p>
              </div>
              <div className="min-w-0">
                {renderActiveSubView(sub, dataB, {
                  compact: true,
                  compareBaseline: dataA,
                  itemsSubTab,
                  onItemsSubTabChange: setItemsSubTab,
                })}
              </div>
            </div>

            <div
              className="pointer-events-none absolute inset-y-0 left-1/2 z-20 hidden w-8 -translate-x-1/2 lg:flex lg:flex-col lg:items-center lg:justify-center"
              aria-hidden
            >
              <div className="absolute inset-y-6 left-1/2 w-px -translate-x-1/2 bg-gradient-to-b from-transparent via-slate-600/85 to-transparent" />
              <span className="relative z-10 inline-flex shrink-0 rounded-full border border-slate-600/90 bg-slate-950 px-2.5 py-1 font-display text-[9px] font-bold tracking-[0.25em] text-slate-300 shadow-lg shadow-black/40 ring-1 ring-slate-800/90">
                VS
              </span>
            </div>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}
