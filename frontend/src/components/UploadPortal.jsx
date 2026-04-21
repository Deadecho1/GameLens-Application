import { useCallback, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { UploadCloud, Cpu, Film } from 'lucide-react';

/**
 * Central upload hub — drag/drop and file pick write to processing.videoFiles (+ optional path hint).
 * BACKEND: replace with presigned upload / chunk transfer; push server file ids into the same store shape.
 */
export default function UploadPortal({ data, onPatch, onOpenEngine }) {
  const { processing, setup } = data;
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const ingestFiles = useCallback(
    (fileList) => {
      const files = Array.from(fileList || []).filter((f) =>
        f.type.startsWith('video/') || /\.(mp4|webm|mov|mkv|avi)$/i.test(f.name)
      );
      if (files.length === 0) return;
      const names = files.map((f) => f.name);
      onPatch({
        processing: {
          ...processing,
          videoFiles: [...new Set([...processing.videoFiles, ...names])],
          pipelinePath: `LOCAL_STAGING://${names[0]}`,
        },
      });
    },
    [onPatch, processing]
  );

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    ingestFiles(e.dataTransfer.files);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-3xl px-4 py-12 md:py-20"
    >
      <div className="mb-8 text-center">
        <p className="font-display text-xs font-bold uppercase tracking-[0.35em] text-blue-500/80">
          Upload hub
        </p>
        <h2 className="mt-2 font-display text-2xl font-bold text-slate-100 md:text-3xl">
          Ingest clip
        </h2>
        <p className="font-data mt-2 text-sm text-slate-500">
          Mission:{' '}
          <span className="text-cyan-400/90">{setup.selectedGame}</span>
          <span className="mx-2 text-slate-700">|</span>
          Build:{' '}
          <span className="text-blue-400/90">{setup.selectedVersion}</span>
        </p>
      </div>

      <div
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
        }}
        onDragEnter={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          if (!e.currentTarget.contains(e.relatedTarget)) setDragOver(false);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`group relative cursor-pointer rounded-2xl border-2 border-dashed bg-slate-950/40 px-8 py-16 text-center transition-colors md:py-24 ${
          dragOver
            ? 'border-cyan-400 bg-blue-500/10'
            : 'border-blue-500/40 gl-upload-pulse hover:border-cyan-400/50'
        }`}
      >
        <div className="pointer-events-none absolute inset-0 rounded-2xl bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.08),transparent_70%)]" />
        <input
          ref={inputRef}
          type="file"
          accept="video/*,.mp4,.webm,.mov,.mkv,.avi"
          multiple
          className="hidden"
          onChange={(e) => {
            ingestFiles(e.target.files);
            e.target.value = '';
          }}
        />
        <motion.div
          animate={{ y: dragOver ? -4 : 0 }}
          className="relative flex flex-col items-center gap-4"
        >
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-cyan-500/30 bg-slate-900/80 text-cyan-400 shadow-[0_0_32px_rgba(34,211,238,0.15)]">
            <UploadCloud className="h-8 w-8" strokeWidth={1.25} />
          </div>
          <div>
            <p className="font-display text-lg font-semibold text-slate-200">
              Drag &amp; drop video clip
            </p>
            <p className="font-data mt-2 text-sm text-slate-500">
              or click to browse — files merge into{' '}
              <code className="rounded bg-slate-900 px-1.5 py-0.5 text-cyan-500/80">
                processing.videoFiles
              </code>
            </p>
          </div>
          {processing.videoFiles.length > 0 && (
            <div className="font-data mt-2 flex flex-wrap items-center justify-center gap-2 text-xs text-slate-500">
              <Film className="h-4 w-4 text-blue-400" />
              <span>
                {processing.videoFiles.length} file
                {processing.videoFiles.length !== 1 ? 's' : ''} staged
              </span>
            </div>
          )}
        </motion.div>
      </div>

      <div className="mt-8 flex flex-wrap justify-center gap-4">
        <div className="gl-radial-glow relative inline-block rounded-2xl p-[1px]">
          <button
            type="button"
            className="relative z-[1] inline-flex items-center gap-2 rounded-2xl border border-blue-500/40 bg-slate-950 px-8 py-3 font-display text-sm font-bold uppercase tracking-wider text-cyan-300 shadow-[0_0_28px_rgba(59,130,246,0.25)] transition hover:border-cyan-400/60 hover:text-white"
            onClick={(e) => {
              e.stopPropagation();
              onOpenEngine();
            }}
          >
            <Cpu className="h-4 w-4" />
            Open engine console
          </button>
        </div>
      </div>

      <p className="font-data mt-10 text-center text-[11px] text-slate-600">
        When <span className="text-emerald-500/90">processing.status</span> is{' '}
        <span className="text-cyan-500/80">completed</span>, the tactical dashboard replaces this hub.
      </p>
    </motion.div>
  );
}
