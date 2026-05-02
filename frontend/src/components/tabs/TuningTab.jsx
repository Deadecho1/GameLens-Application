import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Film,
  Plus,
  Play,
  Pause,
  Square,
  Trash2,
  Cpu,
  CheckSquare,
  Square as SquareIcon,
  ChevronRight,
  X,
} from 'lucide-react';
import { TUNING_MODEL_CONFIGS, TUNING_MODEL_CONFIG_BY_ID, toLocalUrl } from '../tuning/tuningConfig';

const MIN_CLIPS = 6;

function generateId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function TuningTab({ data, ipcRequest }) {
  // ── Video annotation state (local only) ──────────────────────────────────
  const [videos, setVideos] = useState([]);
  const [activeVideoId, setActiveVideoId] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeModelId, setActiveModelId] = useState(TUNING_MODEL_CONFIGS[0].id);
  const [activeCategoryId, setActiveCategoryId] = useState(
    TUNING_MODEL_CONFIGS[0].categories[0].id,
  );
  const [selectedMarkId, setSelectedMarkId] = useState(null);
  const [enabledModelIds, setEnabledModelIds] = useState([TUNING_MODEL_CONFIGS[0].id]);
  const [showNameModal, setShowNameModal] = useState(false);
  const [modelNameDraft, setModelNameDraft] = useState('');
  const [error, setError] = useState('');

  const videoRef = useRef(null);
  const timelineRef = useRef(null);
  const logRef = useRef(null);
  const nameInputRef = useRef(null);

  const tuning = data.tuning || { status: 'idle', logs: [] };
  const isTraining = tuning.status === 'running';

  // ── Derived ──────────────────────────────────────────────────────────────
  const activeVideo = useMemo(
    () => videos.find((v) => v.id === activeVideoId) || null,
    [videos, activeVideoId],
  );

  const activeModelConfig = TUNING_MODEL_CONFIG_BY_ID[activeModelId];

  const clipCounts = useMemo(() => {
    const counts = {};
    for (const video of videos) {
      for (const mark of video.marks) {
        const key = `${mark.modelId}:${mark.categoryId}`;
        counts[key] = (counts[key] || 0) + 1;
      }
    }
    return counts;
  }, [videos]);

  const getClipCount = useCallback(
    (modelId, categoryId) => clipCounts[`${modelId}:${categoryId}`] || 0,
    [clipCounts],
  );

  const isModelReady = useCallback(
    (modelId) => {
      const cfg = TUNING_MODEL_CONFIG_BY_ID[modelId];
      if (!cfg) return false;
      return cfg.categories.every((cat) => getClipCount(modelId, cat.id) >= MIN_CLIPS);
    },
    [getClipCount],
  );

  const canStartTraining = useMemo(
    () => enabledModelIds.some((id) => isModelReady(id)),
    [enabledModelIds, isModelReady],
  );

  // ── Video marks on active video ───────────────────────────────────────────
  const activeVideoMarks = useMemo(
    () => activeVideo?.marks || [],
    [activeVideo],
  );

  // ── Mark / delete helpers ─────────────────────────────────────────────────
  const markCurrentFrame = useCallback(
    (catId = activeCategoryId, modId = activeModelId) => {
      if (!activeVideo) return;
      const newMark = {
        id: generateId(),
        modelId: modId,
        categoryId: catId,
        time: currentTime,
      };
      setVideos((prev) =>
        prev.map((v) =>
          v.id === activeVideoId ? { ...v, marks: [...v.marks, newMark] } : v,
        ),
      );
      setSelectedMarkId(newMark.id);
    },
    [activeVideo, activeVideoId, activeCategoryId, activeModelId, currentTime],
  );

  const deleteSelectedMark = useCallback(() => {
    if (!selectedMarkId) return;
    setVideos((prev) =>
      prev.map((v) =>
        v.id === activeVideoId
          ? { ...v, marks: v.marks.filter((m) => m.id !== selectedMarkId) }
          : v,
      ),
    );
    setSelectedMarkId(null);
  }, [selectedMarkId, activeVideoId]);

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      // Number keys: select category + mark
      const num = parseInt(e.key, 10);
      if (!isNaN(num) && num >= 1 && activeModelConfig) {
        const cat = activeModelConfig.categories[num - 1];
        if (cat) {
          setActiveCategoryId(cat.id);
          markCurrentFrame(cat.id, activeModelId);
        }
        return;
      }

      if (e.key === 'm' || e.key === 'M') {
        markCurrentFrame();
        return;
      }

      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedMarkId) {
        deleteSelectedMark();
        return;
      }

      if (e.key === ' ') {
        e.preventDefault();
        if (videoRef.current) {
          if (isPlaying) {
            videoRef.current.pause();
          } else {
            videoRef.current.play();
          }
        }
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [activeModelConfig, activeModelId, activeCategoryId, selectedMarkId, isPlaying, markCurrentFrame, deleteSelectedMark]);

  // ── Auto-scroll terminal ──────────────────────────────────────────────────
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [tuning.logs]);

  // ── Focus name input when modal opens ────────────────────────────────────
  useEffect(() => {
    if (showNameModal) {
      setTimeout(() => nameInputRef.current?.focus(), 50);
    }
  }, [showNameModal]);

  // ── Seek helper ───────────────────────────────────────────────────────────
  const seekTo = useCallback((time) => {
    if (videoRef.current) videoRef.current.currentTime = time;
    setCurrentTime(time);
  }, []);

  // ── Timeline drag ─────────────────────────────────────────────────────────
  const handleTimelinePointerDown = useCallback(
    (e) => {
      if (!activeVideo || activeVideo.duration <= 0) return;
      e.currentTarget.setPointerCapture(e.pointerId);
      const rect = timelineRef.current.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      seekTo(pct * activeVideo.duration);
    },
    [activeVideo, seekTo],
  );

  const handleTimelinePointerMove = useCallback(
    (e) => {
      if (!activeVideo || activeVideo.duration <= 0) return;
      if (e.buttons === 0) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      seekTo(pct * activeVideo.duration);
    },
    [activeVideo, seekTo],
  );

  // ── Add video ─────────────────────────────────────────────────────────────
  const handleAddVideo = useCallback(async () => {
    try {
      const filePath = await window.gamelens?.chooseFile?.({
        filters: [{ name: 'Video', extensions: ['mp4'] }],
      });
      if (!filePath) return;
      const name = filePath.replace(/\\/g, '/').split('/').pop();
      const newVideo = { id: generateId(), name, path: filePath, duration: 0, marks: [] };
      setVideos((prev) => [...prev, newVideo]);
      setActiveVideoId(newVideo.id);
      setCurrentTime(0);
      setIsPlaying(false);
      setSelectedMarkId(null);
    } catch (err) {
      setError(String(err?.message || err));
    }
  }, []);

  // ── Start fine-tuning ─────────────────────────────────────────────────────
  const handleStartTraining = useCallback(async () => {
    const name = modelNameDraft.trim();
    if (!name) return;
    try {
      const readyModelIds = enabledModelIds.filter((id) => isModelReady(id));
      const tuningVideos = videos.map((v) => ({
        path: v.path,
        duration: v.duration,
        marks: v.marks,
      }));
      await ipcRequest('tuning:start', {
        videos: tuningVideos,
        enabledModelIds: readyModelIds,
        modelName: name,
      });
      setShowNameModal(false);
      setModelNameDraft('');
      setError('');
    } catch (err) {
      setError(String(err?.message || err));
    }
  }, [modelNameDraft, enabledModelIds, isModelReady, videos, ipcRequest]);

  const handleStopTraining = useCallback(async () => {
    try {
      await ipcRequest('tuning:stop');
    } catch (err) {
      setError(String(err?.message || err));
    }
  }, [ipcRequest]);

  const handleClearLogs = useCallback(async () => {
    try {
      await ipcRequest('tuning:clear_logs');
    } catch (err) {
      setError(String(err?.message || err));
    }
  }, [ipcRequest]);

  // ── Render ────────────────────────────────────────────────────────────────
  const duration = activeVideo?.duration || 0;
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  const formatTime = (s) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.3 }}
      className="mx-auto max-w-[1800px] px-4 py-8"
    >
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <p className="font-display text-[10px] font-bold uppercase tracking-[0.35em] text-blue-500/70">
            Fine-Tuning
          </p>
          <h2 className="mt-1 font-display text-2xl font-bold text-slate-100">
            Game Tuning
          </h2>
        </div>
        <div className="flex items-center gap-3">
          {error && (
            <span className="font-data max-w-xs truncate rounded-lg border border-red-500/40 bg-red-950/40 px-3 py-1.5 text-xs text-red-300">
              {error}
            </span>
          )}
          <button
            type="button"
            disabled={isTraining || !canStartTraining}
            onClick={() => setShowNameModal(true)}
            className="inline-flex items-center gap-2 rounded-xl border border-emerald-500/40 bg-gradient-to-b from-emerald-600 to-emerald-800 px-5 py-2.5 font-display text-xs font-bold uppercase tracking-[0.15em] text-white shadow-[0_0_24px_rgba(16,185,129,0.25)] transition hover:shadow-[0_0_32px_rgba(16,185,129,0.4)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Cpu className="h-4 w-4" />
            Start Fine-Tuning
          </button>
          {isTraining && (
            <button
              type="button"
              onClick={handleStopTraining}
              className="inline-flex items-center gap-2 rounded-xl border border-red-500/40 bg-gradient-to-b from-red-600 to-red-800 px-4 py-2.5 font-display text-xs font-bold uppercase tracking-[0.15em] text-white"
            >
              <Square className="h-4 w-4 fill-current" />
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Main 3-column layout */}
      <div className="grid gap-4" style={{ gridTemplateColumns: '220px 1fr 260px' }}>
        {/* ── Left: Video sidebar ─────────────────────────────────────────── */}
        <aside className="flex flex-col gap-2">
          <p className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-500/80">
            Videos
          </p>
          <div className="flex flex-col gap-1.5 rounded-2xl border border-slate-800/90 bg-slate-950/50 p-2 backdrop-blur-xl">
            {videos.length === 0 && (
              <p className="py-6 text-center font-data text-xs text-slate-600">
                No videos loaded
              </p>
            )}
            {videos.map((v) => {
              const totalMarks = v.marks.length;
              const isActive = v.id === activeVideoId;
              return (
                <button
                  key={v.id}
                  type="button"
                  onClick={() => {
                    setActiveVideoId(v.id);
                    setCurrentTime(0);
                    setIsPlaying(false);
                    setSelectedMarkId(null);
                  }}
                  className={`flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left transition ${
                    isActive
                      ? 'border border-cyan-500/30 bg-cyan-500/10 text-cyan-200'
                      : 'border border-transparent text-slate-400 hover:border-slate-700 hover:bg-slate-900/50 hover:text-slate-200'
                  }`}
                >
                  <Film className="h-3.5 w-3.5 shrink-0 text-blue-400/70" />
                  <span className="min-w-0 flex-1 truncate font-data text-xs">{v.name}</span>
                  {totalMarks > 0 && (
                    <span className="shrink-0 rounded-full bg-cyan-500/20 px-1.5 py-0.5 font-data text-[10px] text-cyan-400">
                      {totalMarks}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          <button
            type="button"
            onClick={handleAddVideo}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-blue-500/35 bg-slate-950/40 py-3 font-display text-[10px] font-bold uppercase tracking-wider text-slate-500 transition hover:border-cyan-400/45 hover:text-cyan-400"
          >
            <Plus className="h-3.5 w-3.5" />
            Add Video
          </button>
        </aside>

        {/* ── Center: Video player + timeline ─────────────────────────────── */}
        <div className="flex min-w-0 flex-col gap-3">
          {activeVideo ? (
            <>
              {/* Video element */}
              <div className="relative overflow-hidden rounded-2xl border border-slate-800/90 bg-black shadow-[inset_0_0_48px_rgba(0,0,0,0.6)]">
                <video
                  ref={videoRef}
                  src={toLocalUrl(activeVideo.path)}
                  className="aspect-video w-full"
                  onTimeUpdate={() => {
                    if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
                  }}
                  onLoadedMetadata={() => {
                    if (videoRef.current) {
                      const dur = videoRef.current.duration;
                      setVideos((prev) =>
                        prev.map((v) =>
                          v.id === activeVideoId ? { ...v, duration: dur } : v,
                        ),
                      );
                      setCurrentTime(0);
                    }
                  }}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
                />
              </div>

              {/* Transport controls */}
              <div className="flex items-center gap-3 rounded-xl border border-slate-800/90 bg-slate-900/50 px-4 py-2 backdrop-blur-sm">
                <button
                  type="button"
                  onClick={() => {
                    if (videoRef.current) {
                      if (isPlaying) videoRef.current.pause();
                      else videoRef.current.play();
                    }
                  }}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 transition hover:border-cyan-400/60 hover:text-cyan-100"
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </button>
                <span className="font-data shrink-0 text-xs tabular-nums text-slate-400">
                  {formatTime(currentTime)} / {formatTime(duration)}
                </span>
                <div className="flex-1" />
                {selectedMarkId && (
                  <button
                    type="button"
                    onClick={deleteSelectedMark}
                    className="flex items-center gap-1.5 rounded-lg border border-red-500/30 bg-red-500/10 px-2.5 py-1.5 font-data text-xs text-red-300 transition hover:border-red-400/50 hover:text-red-200"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Delete mark
                  </button>
                )}
              </div>

              {/* Timeline */}
              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/50 p-4 backdrop-blur-sm">
                <div
                  ref={timelineRef}
                  className="relative h-8 cursor-pointer select-none rounded-full bg-slate-800/80"
                  onPointerDown={handleTimelinePointerDown}
                  onPointerMove={handleTimelinePointerMove}
                >
                  {/* Progress fill */}
                  <div
                    className="pointer-events-none absolute inset-y-0 left-0 rounded-full bg-slate-700/60"
                    style={{ width: `${progress}%` }}
                  />

                  {/* Category marks */}
                  {activeVideoMarks.map((mark) => {
                    const markPct = duration > 0 ? (mark.time / duration) * 100 : 0;
                    const cat = TUNING_MODEL_CONFIG_BY_ID[mark.modelId]?.categories?.find(
                      (c) => c.id === mark.categoryId,
                    );
                    const color = cat?.color || '#94a3b8';
                    const isSelected = mark.id === selectedMarkId;
                    return (
                      <div
                        key={mark.id}
                        className="absolute top-0 bottom-0 w-[3px] cursor-pointer rounded-full transition-all"
                        style={{
                          left: `${markPct}%`,
                          backgroundColor: color,
                          opacity: isSelected ? 1 : 0.75,
                          boxShadow: isSelected ? `0 0 8px ${color}` : 'none',
                          transform: 'translateX(-1.5px)',
                          zIndex: isSelected ? 10 : 5,
                        }}
                        onPointerDown={(e) => {
                          e.stopPropagation();
                          setSelectedMarkId(mark.id);
                        }}
                      />
                    );
                  })}

                  {/* Playhead knob */}
                  <div
                    className="pointer-events-none absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-white shadow-[0_0_12px_rgba(255,255,255,0.4)]"
                    style={{ left: `${progress}%` }}
                  />
                </div>

                {/* Mark controls row */}
                <div className="mt-3 flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => markCurrentFrame()}
                    className="flex items-center gap-1.5 rounded-lg border border-cyan-500/35 bg-cyan-500/10 px-3 py-1.5 font-display text-[10px] font-bold uppercase tracking-wider text-cyan-300 transition hover:border-cyan-400/60 hover:text-cyan-100"
                  >
                    Mark frame
                    <span className="rounded border border-cyan-500/30 px-1 font-data text-[10px]">
                      M
                    </span>
                  </button>
                  <span className="font-data text-xs text-slate-600">
                    as
                  </span>
                  {/* Active category selector */}
                  <div className="flex gap-1.5">
                    {activeModelConfig?.categories.map((cat) => (
                      <button
                        key={cat.id}
                        type="button"
                        onClick={() => setActiveCategoryId(cat.id)}
                        className={`flex items-center gap-1 rounded-lg border px-2 py-1 font-display text-[10px] font-bold uppercase tracking-wider transition ${
                          activeCategoryId === cat.id
                            ? 'border-transparent text-slate-900'
                            : 'border-slate-700 text-slate-500 hover:text-slate-300'
                        }`}
                        style={
                          activeCategoryId === cat.id
                            ? { backgroundColor: cat.color, borderColor: cat.color }
                            : {}
                        }
                      >
                        <span className="rounded border border-current/40 px-0.5 font-data text-[9px] opacity-70">
                          {cat.key}
                        </span>
                        {cat.label}
                      </button>
                    ))}
                  </div>
                </div>
                <p className="mt-2 font-data text-[10px] text-slate-600">
                  Shortcuts: 1–4 select &amp; mark · M mark · Del remove selected · Space play/pause
                </p>
              </div>
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center rounded-2xl border border-dashed border-slate-800 py-24">
              <div className="text-center">
                <Film className="mx-auto h-12 w-12 text-slate-700" strokeWidth={1} />
                <p className="mt-3 font-display text-sm font-bold uppercase tracking-wider text-slate-600">
                  Add a video to begin
                </p>
                <button
                  type="button"
                  onClick={handleAddVideo}
                  className="mt-4 inline-flex items-center gap-2 rounded-xl border border-blue-500/35 bg-slate-900/60 px-4 py-2 font-display text-xs font-bold uppercase tracking-wider text-cyan-300 transition hover:border-cyan-400/50"
                >
                  <Plus className="h-4 w-4" />
                  Add Video
                </button>
              </div>
            </div>
          )}
        </div>

        {/* ── Right: Category panel ────────────────────────────────────────── */}
        <aside className="flex flex-col gap-3">
          <p className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-500/80">
            Categories
          </p>

          {/* Model type tabs */}
          <div className="flex flex-wrap gap-1.5">
            {TUNING_MODEL_CONFIGS.map((cfg) => (
              <button
                key={cfg.id}
                type="button"
                onClick={() => {
                  setActiveModelId(cfg.id);
                  setActiveCategoryId(cfg.categories[0].id);
                }}
                className={`rounded-lg border px-3 py-1.5 font-display text-[10px] font-bold uppercase tracking-wider transition ${
                  activeModelId === cfg.id
                    ? 'border-cyan-500/40 bg-slate-900/80 text-cyan-300'
                    : 'border-slate-800 bg-slate-900/35 text-slate-500 hover:border-slate-700 hover:text-slate-300'
                }`}
              >
                {cfg.label}
              </button>
            ))}
          </div>

          {/* Enable checkbox for active model */}
          {activeModelConfig && (
            <label className="flex cursor-pointer items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/40 px-3 py-2.5 transition hover:border-slate-700">
              <input
                type="checkbox"
                checked={enabledModelIds.includes(activeModelId)}
                onChange={(e) => {
                  setEnabledModelIds((prev) =>
                    e.target.checked
                      ? [...prev, activeModelId]
                      : prev.filter((id) => id !== activeModelId),
                  );
                }}
                className="accent-cyan-500"
              />
              <span className="font-data text-xs text-slate-400">
                Train {activeModelConfig.label}
              </span>
              {isModelReady(activeModelId) && (
                <span className="ml-auto rounded-full bg-emerald-500/20 px-2 py-0.5 font-data text-[10px] text-emerald-400">
                  READY
                </span>
              )}
            </label>
          )}

          {/* Category list */}
          <div className="flex flex-col gap-2 rounded-2xl border border-slate-800/90 bg-slate-950/50 p-3 backdrop-blur-xl">
            {activeModelConfig?.categories.map((cat) => {
              const count = getClipCount(activeModelId, cat.id);
              const pct = Math.min(1, count / MIN_CLIPS);
              const isActive = activeCategoryId === cat.id;
              return (
                <button
                  key={cat.id}
                  type="button"
                  onClick={() => setActiveCategoryId(cat.id)}
                  className={`w-full rounded-xl border p-3 text-left transition ${
                    isActive
                      ? 'border-transparent'
                      : 'border-slate-800 hover:border-slate-700'
                  }`}
                  style={isActive ? { borderColor: cat.color + '40', backgroundColor: cat.color + '12' } : {}}
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="h-3 w-3 shrink-0 rounded-full"
                      style={{ backgroundColor: cat.color }}
                    />
                    <span className="flex-1 font-display text-[10px] font-bold uppercase tracking-wider text-slate-300">
                      {cat.label}
                    </span>
                    <span
                      className="shrink-0 rounded border px-1 font-data text-[9px] text-slate-600"
                      style={{ borderColor: cat.color + '40' }}
                    >
                      {cat.key}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center gap-2">
                    <div className="flex-1 overflow-hidden rounded-full bg-slate-800 h-1.5">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ backgroundColor: cat.color }}
                        initial={false}
                        animate={{ width: `${pct * 100}%` }}
                        transition={{ type: 'spring', stiffness: 200, damping: 28 }}
                      />
                    </div>
                    <span
                      className={`shrink-0 font-data text-xs tabular-nums ${
                        count >= MIN_CLIPS ? 'text-emerald-400' : 'text-slate-500'
                      }`}
                    >
                      {count}/{MIN_CLIPS}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Total clips summary */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 px-3 py-2">
            <p className="font-data text-[10px] text-slate-500">
              Total marks:{' '}
              <span className="text-slate-300">
                {videos.reduce((s, v) => s + v.marks.length, 0)}
              </span>{' '}
              across{' '}
              <span className="text-slate-300">{videos.length}</span> video
              {videos.length !== 1 ? 's' : ''}
            </p>
          </div>
        </aside>
      </div>

      {/* Training terminal — shown when status != idle */}
      <AnimatePresence>
        {tuning.status !== 'idle' && (
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
            transition={{ duration: 0.3 }}
            className="mt-6"
          >
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-display text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500/80">
                Training Terminal
              </h3>
              <div className="flex items-center gap-2">
                <span
                  className={`font-data text-xs font-bold tracking-wider ${
                    tuning.status === 'running'
                      ? 'text-cyan-400'
                      : tuning.status === 'completed'
                        ? 'text-emerald-400'
                        : 'text-amber-400'
                  }`}
                >
                  {tuning.status === 'running'
                    ? 'TRAINING'
                    : tuning.status === 'completed'
                      ? 'COMPLETE'
                      : 'STOPPED'}
                </span>
                <button
                  type="button"
                  onClick={handleClearLogs}
                  className="font-data inline-flex items-center gap-1 rounded-lg border border-slate-800 px-2 py-1 text-[11px] font-semibold text-slate-500 transition hover:border-cyan-500/30 hover:text-cyan-400"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Clear
                </button>
              </div>
            </div>

            {/* Progress bar */}
            <div className="mb-3 h-1.5 overflow-hidden rounded-full bg-black/50 ring-1 ring-slate-800">
              {tuning.status === 'running' ? (
                <motion.div
                  className="gl-neon-bar-fill h-full rounded-full"
                  initial={{ left: '0%', width: '38%' }}
                  animate={{
                    left: ['0%', '52%', '8%', '44%'],
                    width: ['36%', '44%', '40%', '38%'],
                  }}
                  style={{ position: 'relative' }}
                  transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
                />
              ) : (
                <div
                  className={`h-full rounded-full ${
                    tuning.status === 'completed'
                      ? 'gl-neon-bar-fill w-full'
                      : 'w-1/3 bg-gradient-to-r from-amber-600 to-red-600'
                  }`}
                />
              )}
            </div>

            <div
              ref={logRef}
              className="gl-terminal-scanlines max-h-64 min-h-[160px] overflow-y-auto rounded-2xl border border-cyan-500/15 bg-black/80 p-4 font-data text-xs leading-relaxed text-emerald-400/95 shadow-[inset_0_0_48px_rgba(34,211,238,0.04)] backdrop-blur-sm"
            >
              {tuning.logs.length === 0 ? (
                <span className="text-slate-600">&gt; awaiting output…</span>
              ) : (
                tuning.logs.map((line, i) => (
                  <div key={i} className="whitespace-pre-wrap pl-2">
                    <span className="text-cyan-700/90">&gt; </span>
                    {line}
                  </div>
                ))
              )}
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      {/* Model name modal */}
      <AnimatePresence>
        {showNameModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 340, damping: 30 }}
              className="relative w-full max-w-sm rounded-2xl border border-slate-700/80 bg-slate-900/95 p-6 shadow-2xl backdrop-blur-xl"
            >
              <button
                type="button"
                onClick={() => setShowNameModal(false)}
                className="absolute right-4 top-4 text-slate-500 transition hover:text-slate-300"
              >
                <X className="h-4 w-4" />
              </button>
              <h3 className="font-display text-sm font-bold uppercase tracking-[0.2em] text-slate-100">
                Name Your Model
              </h3>
              <p className="mt-1 font-data text-xs text-slate-500">
                This name will appear in the pipeline model dropdown.
              </p>
              <input
                ref={nameInputRef}
                type="text"
                value={modelNameDraft}
                onChange={(e) => setModelNameDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && modelNameDraft.trim()) handleStartTraining();
                  if (e.key === 'Escape') setShowNameModal(false);
                }}
                placeholder="e.g. hades-v1"
                className="mt-4 w-full rounded-xl border border-slate-700 bg-black/50 px-4 py-3 font-data text-sm text-slate-100 placeholder-slate-600 outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30"
              />
              <div className="mt-4 flex gap-3">
                <button
                  type="button"
                  onClick={() => setShowNameModal(false)}
                  className="flex-1 rounded-xl border border-slate-700 py-3 font-display text-xs font-bold uppercase tracking-wider text-slate-400 transition hover:text-slate-200"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  disabled={!modelNameDraft.trim()}
                  onClick={handleStartTraining}
                  className="flex-1 rounded-xl border border-emerald-500/40 bg-gradient-to-b from-emerald-600 to-emerald-800 py-3 font-display text-xs font-bold uppercase tracking-wider text-white shadow-[0_0_20px_rgba(16,185,129,0.25)] disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Begin Training
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
