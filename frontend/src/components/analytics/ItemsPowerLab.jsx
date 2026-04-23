import { useCallback, useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { motion } from 'framer-motion';
import {
  Bomb,
  FlaskConical,
  Gem,
  Hammer,
  Heart,
  Hexagon,
  LayoutGrid,
  Package,
  RotateCcw,
  Search,
  Shield,
  Sparkles,
  Sword,
  Wand2,
  Wrench,
  Zap,
} from 'lucide-react';

const DRAG_MIME = 'application/gamelens-item-id';

const CATEGORY_TABS = [
  { id: 'all', label: 'All', Icon: LayoutGrid },
  { id: 'offensive', label: 'Offense', Icon: Sword },
  { id: 'defensive', label: 'Defense', Icon: Shield },
  { id: 'utility', label: 'Utility', Icon: Wrench },
];

function pickItemIcon(name) {
  const n = String(name ?? '').toLowerCase();
  if (n.includes('mallet') || n.includes('hammer')) return Hammer;
  if (n.includes('dagger') || n.includes('sword')) return Sword;
  if (n.includes('seed') || n.includes('explosive')) return Bomb;
  if (n.includes('charm') || n.includes('void')) return Gem;
  if (n.includes('healing') || n.includes('draught')) return Heart;
  if (n.includes('focus') || n.includes('arcane')) return Wand2;
  if (n.includes('venom') || n.includes('flask') || n.includes('potion')) return FlaskConical;
  if (n.includes('buckler') || n.includes('shield') || n.includes('plate')) return Shield;
  if (n.includes('power')) return Zap;
  return Package;
}

function inferCategory(item) {
  if (item.category) return item.category;
  const n = String(item.name ?? '').toLowerCase();
  if (/(shield|plate|buckler)/.test(n)) return 'defensive';
  if (/(sword|dagger|mallet|venom|explosive|thunder|fire|frost)/.test(n)) return 'offensive';
  return 'utility';
}

/** Power level 1–100 from popularity (mock curve). */
function itemPowerLevel(item) {
  return Math.min(100, Math.max(1, Math.round(28 + item.popularity * 0.72)));
}

/** Mock pentagon stats from popularity + category (Item Power Lab). */
function deriveItemRadar(item) {
  const p = item.popularity / 100;
  const cat = inferCategory(item);
  const core = 22 + p * 58;
  let offense = cat === 'offensive' ? core + 20 : core * 0.7;
  let defense = cat === 'defensive' ? core + 20 : core * 0.66;
  let speed = cat === 'offensive' ? core * 0.9 + 12 : cat === 'utility' ? core * 0.85 + 6 : core * 0.72;
  let utility = cat === 'utility' ? core + 20 : core * 0.7;
  let versatility = ((offense + defense + speed + utility) / 4) * (0.78 + p * 0.22);
  const clamp = (v) => Math.min(100, Math.max(4, Math.round(v)));
  return {
    offense: clamp(offense),
    defense: clamp(defense),
    speed: clamp(speed),
    utility: clamp(utility),
    versatility: clamp(versatility),
  };
}

function aggregateBuildRadar(equippedItems) {
  if (!equippedItems.length) {
    return { offense: 0, defense: 0, speed: 0, utility: 0, versatility: 0 };
  }
  const sum = { offense: 0, defense: 0, speed: 0, utility: 0, versatility: 0 };
  for (const it of equippedItems) {
    const r = deriveItemRadar(it);
    sum.offense += r.offense;
    sum.defense += r.defense;
    sum.speed += r.speed;
    sum.utility += r.utility;
    sum.versatility += r.versatility;
  }
  const n = equippedItems.length;
  return {
    offense: Math.round(sum.offense / n),
    defense: Math.round(sum.defense / n),
    speed: Math.round(sum.speed / n),
    utility: Math.round(sum.utility / n),
    versatility: Math.round(sum.versatility / n),
  };
}

function combinedBuildScore(radar) {
  const v = radar.offense + radar.defense + radar.speed + radar.utility + radar.versatility;
  return Math.round(v / 5);
}

function deriveRarity(item) {
  if (item.rarity) return item.rarity;
  if (item.popularity >= 75) return 'Legendary';
  if (item.popularity >= 45) return 'Rare';
  return 'Common';
}

function rarityStyles(rarity) {
  const r = String(rarity);
  if (r === 'Legendary')
    return 'border-amber-400/50 bg-amber-500/10 text-amber-200 shadow-[0_0_16px_rgba(251,191,36,0.2)]';
  if (r === 'Rare')
    return 'border-violet-400/45 bg-violet-500/10 text-violet-200 shadow-[0_0_14px_rgba(167,139,250,0.18)]';
  return 'border-slate-600 bg-slate-800/50 text-slate-300';
}

function mockPossessionMinutes(item) {
  if (typeof item.avgPossessionMinutes === 'number') return item.avgPossessionMinutes;
  return Math.max(8, Math.round(12 + (100 - item.popularity) * 0.42));
}

/** Popularity rank: #k of N; "top" tier % (lower is more elite among sorted desc). */
function popularityRankMeta(itemId, catalog) {
  const sorted = [...catalog].sort((a, b) => b.popularity - a.popularity);
  const rank = sorted.findIndex((i) => i.id === itemId) + 1;
  const n = sorted.length;
  const topPct = n > 0 ? Math.max(1, Math.ceil((rank / n) * 100)) : 0;
  return { rank, total: n, topPct };
}

function placeItemInSlots(slots, slotIndex, itemId) {
  const next = [...slots];
  const dup = next.findIndex((id, i) => id === itemId && i !== slotIndex);
  if (dup >= 0) next[dup] = null;
  next[slotIndex] = itemId;
  return next;
}

/**
 * ITEMS analytics — Item Power Lab sandbox (dashboard.items).
 */
export default function ItemsPowerLab({ data }) {
  const catalog = data.dashboard.items ?? [];
  const [buildSlots, setBuildSlots] = useState([null, null, null, null, null]);
  const [activeSlot, setActiveSlot] = useState(0);
  const [lastSelectedId, setLastSelectedId] = useState(null);
  const [search, setSearch] = useState('');
  const [catFilter, setCatFilter] = useState('all');

  const filteredItems = useMemo(() => {
    const q = search.trim().toLowerCase();
    return catalog.filter((item) => {
      if (catFilter !== 'all' && inferCategory(item) !== catFilter) return false;
      if (q && !String(item.name).toLowerCase().includes(q)) return false;
      return true;
    });
  }, [catalog, search, catFilter]);

  const equippedItems = useMemo(
    () => buildSlots.map((id) => (id != null ? catalog.find((i) => i.id === id) : null)).filter(Boolean),
    [buildSlots, catalog]
  );

  const buildRadar = useMemo(() => aggregateBuildRadar(equippedItems), [equippedItems]);
  const efficiencyScore = useMemo(() => combinedBuildScore(buildRadar), [buildRadar]);

  const radarChartData = useMemo(
    () => [
      { attribute: 'Offense', value: buildRadar.offense },
      { attribute: 'Defense', value: buildRadar.defense },
      { attribute: 'Speed', value: buildRadar.speed },
      { attribute: 'Utility', value: buildRadar.utility },
      { attribute: 'Versatility', value: buildRadar.versatility },
    ],
    [buildRadar]
  );

  const lastItem = useMemo(() => {
    if (lastSelectedId == null) return null;
    return catalog.find((i) => i.id === lastSelectedId) ?? null;
  }, [lastSelectedId, catalog]);

  const lastItemRank = useMemo(
    () => (lastItem ? popularityRankMeta(lastItem.id, catalog) : null),
    [lastItem, catalog]
  );

  const assignItem = useCallback(
    (itemId, slotIndex) => {
      setBuildSlots((s) => placeItemInSlots(s, slotIndex, itemId));
      setLastSelectedId(itemId);
    },
    []
  );

  const onBrowserItemActivate = (item) => {
    setLastSelectedId(item.id);
    const empty = buildSlots.findIndex((id) => id == null);
    if (empty >= 0) {
      setBuildSlots((s) => placeItemInSlots(s, empty, item.id));
      setActiveSlot(empty);
    } else {
      setBuildSlots((s) => placeItemInSlots(s, activeSlot, item.id));
    }
  };

  const onDragStartItem = (e, itemId) => {
    e.dataTransfer.setData(DRAG_MIME, String(itemId));
    e.dataTransfer.effectAllowed = 'copy';
  };

  const onSlotDrop = (e, slotIndex) => {
    e.preventDefault();
    const raw = e.dataTransfer.getData(DRAG_MIME);
    const id = parseInt(raw, 10);
    if (!Number.isFinite(id) || !catalog.some((i) => i.id === id)) return;
    assignItem(id, slotIndex);
  };

  const clearBuild = () => {
    setBuildSlots([null, null, null, null, null]);
  };

  return (
    <div className="space-y-6">
      <header>
        <div className="flex items-center gap-2">
          <Hexagon className="h-5 w-5 text-violet-400" strokeWidth={1.25} aria-hidden />
          <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-violet-400/80">
            Item power lab
          </p>
        </div>
        <h3 className="mt-2 font-display text-xl font-bold text-slate-100 md:text-2xl">
          Build sandbox · attribute synthesis
        </h3>
        <p className="font-data mt-2 text-sm text-slate-500">
          Source: <code className="text-violet-400/80">dashboard.items</code> · radar mocked from popularity + role
        </p>
      </header>

      <div className="flex min-h-[min(720px,78vh)] flex-col gap-4 lg:flex-row lg:gap-0 lg:rounded-2xl lg:border lg:border-violet-500/15 lg:bg-slate-950/45 lg:shadow-[inset_0_1px_0_rgba(167,139,250,0.08)]">
        <aside className="flex w-full flex-col border-slate-800/80 lg:w-[min(100%,320px)] lg:shrink-0 lg:border-r lg:border-violet-500/10 lg:bg-slate-950/55">
          <div className="border-b border-slate-800/80 px-4 py-3">
            <p className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-violet-300/70">
              Item browser
            </p>
            <label className="relative mt-3 block">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-violet-400/55" />
              <input
                type="search"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search items…"
                className="w-full rounded-lg border border-violet-500/35 bg-slate-950/85 py-2.5 pl-10 pr-3 font-data text-sm text-slate-100 placeholder:text-slate-600 outline-none focus:border-violet-400/65 focus:ring-1 focus:ring-violet-500/30"
              />
            </label>
            <div className="mt-3 flex flex-wrap gap-1.5" role="tablist" aria-label="Filter by role">
              {CATEGORY_TABS.map((tab) => {
                const active = catFilter === tab.id;
                const Icon = tab.Icon;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    onClick={() => setCatFilter(tab.id)}
                    className={`flex items-center gap-1 rounded-md border px-2 py-1 font-display text-[8px] font-bold uppercase tracking-[0.12em] transition ${
                      active
                        ? 'border-violet-400/50 bg-violet-500/15 text-violet-200'
                        : 'border-slate-700 text-slate-500 hover:border-slate-600 hover:text-slate-300'
                    } `}
                  >
                    <Icon className="h-3 w-3" strokeWidth={1.5} />
                    {tab.label}
                  </button>
                );
              })}
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-2 [scrollbar-color:rgba(76,29,149,0.5)_transparent]">
            {filteredItems.length === 0 ? (
              <p className="font-data py-8 text-center text-sm text-slate-500">No matches.</p>
            ) : (
              <ul className="space-y-1">
                {filteredItems.map((item) => {
                  const Icon = pickItemIcon(item.name);
                  const inBuild = buildSlots.includes(item.id);
                  return (
                    <li key={item.id}>
                      <button
                        type="button"
                        draggable
                        onDragStart={(e) => onDragStartItem(e, item.id)}
                        onClick={() => onBrowserItemActivate(item)}
                        className={`flex w-full items-center gap-3 rounded-lg border px-2 py-2.5 text-left transition ${
                          inBuild
                            ? 'border-violet-400/40 bg-violet-500/10'
                            : 'border-slate-800 bg-slate-900/40 hover:border-violet-500/25 hover:bg-slate-900/70'
                        } `}
                      >
                        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-violet-500/20 bg-slate-950/80">
                          <Icon className="h-5 w-5 text-violet-300" strokeWidth={1.15} />
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="font-display block truncate text-[11px] font-bold uppercase tracking-wide text-slate-200">
                            {item.name}
                          </span>
                          <span className="font-data text-[10px] tabular-nums text-violet-300/70">
                            PWR {itemPowerLevel(item)}
                          </span>
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </aside>

        <main className="relative min-w-0 flex-1 p-4 backdrop-blur-sm md:p-6 lg:p-8">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-10%,rgba(109,40,217,0.12),transparent)]" />

          <div className="relative space-y-8">
            <section>
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-violet-400" />
                  <h4 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-violet-200/90">
                    Full build · 5 slots
                  </h4>
                </div>
                <button
                  type="button"
                  onClick={clearBuild}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-600 px-2.5 py-1.5 font-display text-[8px] font-bold uppercase tracking-[0.15em] text-slate-400 hover:border-violet-500/40 hover:text-violet-200"
                >
                  <RotateCcw className="h-3 w-3" />
                  Reset
                </button>
              </div>
              <p className="font-data mb-4 text-[10px] text-slate-500">
                Drag onto a slot or click an item to auto-fill. Click a slot to target it for the next click.
              </p>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-5">
                {buildSlots.map((slotId, i) => {
                  const item = slotId != null ? catalog.find((x) => x.id === slotId) : null;
                  const Icon = item ? pickItemIcon(item.name) : Package;
                  const isActive = activeSlot === i;
                  return (
                    <div
                      key={i}
                      role="button"
                      tabIndex={0}
                      onClick={() => setActiveSlot(i)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          setActiveSlot(i);
                        }
                      }}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={(e) => onSlotDrop(e, i)}
                      className={`relative flex min-h-[120px] flex-col items-center justify-center rounded-xl border-2 border-dashed p-3 transition ${
                        isActive
                          ? 'border-violet-400/55 bg-violet-500/10 shadow-[0_0_20px_rgba(167,139,250,0.2)]'
                          : 'border-slate-700 bg-slate-950/50 hover:border-slate-600'
                      } `}
                    >
                      {item ? (
                        <>
                          <Icon className="h-8 w-8 text-violet-200" strokeWidth={1.1} />
                          <p className="font-data mt-2 max-w-full truncate text-center text-[9px] text-slate-400">
                            {item.name}
                          </p>
                          <p className="font-data mt-1 text-sm font-bold tabular-nums text-violet-200">
                            {itemPowerLevel(item)}
                          </p>
                          <button
                            type="button"
                            className="font-data absolute right-1 top-1 rounded px-1.5 text-[9px] text-slate-500 hover:text-slate-300"
                            onClick={(e) => {
                              e.stopPropagation();
                              setBuildSlots((s) => {
                                const n = [...s];
                                n[i] = null;
                                return n;
                              });
                            }}
                          >
                            ×
                          </button>
                        </>
                      ) : (
                        <span className="font-data text-[10px] uppercase tracking-wider text-slate-600">
                          Slot {i + 1}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>

            <div className="grid gap-6 xl:grid-cols-[1fr_280px]">
              <motion.section
                key={efficiencyScore}
                initial={{ opacity: 0.85 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.2 }}
                className="rounded-2xl border border-violet-500/20 bg-slate-900/40 p-4 ring-1 ring-violet-500/10 backdrop-blur-md md:p-5"
              >
                <h4 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-violet-300/90">
                  Build attribute map
                </h4>
                <div className="h-[300px] w-full min-w-0 md:h-[340px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarChartData} cx="50%" cy="50%" outerRadius="72%">
                      <PolarGrid stroke="#5b21b6" strokeOpacity={0.35} />
                      <PolarAngleAxis
                        dataKey="attribute"
                        tick={{ fill: '#c4b5fd', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
                      />
                      <PolarRadiusAxis
                        angle={30}
                        domain={[0, 100]}
                        tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
                      />
                      <Radar
                        name="Build"
                        dataKey="value"
                        stroke="#a78bfa"
                        fill="#7c3aed"
                        fillOpacity={0.35}
                        strokeWidth={2}
                        dot={{ r: 3, fill: '#e9d5ff', stroke: '#a78bfa' }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </motion.section>

              <div className="flex flex-col justify-center rounded-2xl border border-violet-400/25 bg-linear-to-br from-violet-950/80 to-slate-950/90 p-6 shadow-[0_0_40px_rgba(109,40,217,0.15)] backdrop-blur-md">
                <p className="font-display text-[9px] font-bold uppercase tracking-[0.25em] text-violet-300/70">
                  Combined build score
                </p>
                <p className="font-data mt-3 text-5xl font-bold tabular-nums text-white [text-shadow:0_0_32px_rgba(167,139,250,0.35)] md:text-6xl">
                  {equippedItems.length ? efficiencyScore : '—'}
                </p>
                <p className="font-data mt-2 text-[10px] text-slate-500">
                  Mean of five axes · add gear to synthesize profile
                </p>
              </div>
            </div>

            <section className="rounded-2xl border border-slate-800/90 bg-slate-950/50 p-5 backdrop-blur-md ring-1 ring-violet-500/10">
              <h4 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-violet-300/80">
                Global stats · last selected item
              </h4>
              {!lastItem ? (
                <p className="font-data mt-4 text-sm text-slate-500">Select an item from the browser to inspect.</p>
              ) : (
                <div className="mt-5 grid gap-6 sm:grid-cols-3">
                  <div>
                    <p className="font-data text-[9px] uppercase tracking-wider text-slate-500">Popularity rank</p>
                    <p className="font-data mt-2 text-lg font-bold tabular-nums text-slate-100">
                      #{lastItemRank?.rank} / {lastItemRank?.total}
                    </p>
                    <p className="font-data mt-1 text-[10px] text-violet-300/80">
                      Top {lastItemRank?.topPct}% of catalog by popularity
                    </p>
                  </div>
                  <div>
                    <p className="font-data text-[9px] uppercase tracking-wider text-slate-500">
                      Avg. possession time
                    </p>
                    <p className="font-data mt-2 text-lg font-bold tabular-nums text-slate-100">
                      {mockPossessionMinutes(lastItem)} min
                    </p>
                    <p className="font-data mt-1 text-[10px] text-slate-600">Session-weighted mock</p>
                  </div>
                  <div>
                    <p className="font-data text-[9px] uppercase tracking-wider text-slate-500">Rarity level</p>
                    <p
                      className={`font-display mt-2 inline-block rounded-lg border px-3 py-2 text-xs font-bold uppercase tracking-widest ${rarityStyles(deriveRarity(lastItem))}`}
                    >
                      {deriveRarity(lastItem)}
                    </p>
                  </div>
                </div>
              )}
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
