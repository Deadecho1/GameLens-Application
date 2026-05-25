/** Interactive onboarding tour for Game Tuning tab (react-joyride). */

export const TUNING_TOUR_STEPS = [
  {
    target: '.tuning-tour-add-video',
    title: 'Step 1: Load your Gameplay',
    content:
      'Start by adding MP4 videos. You can load multiple videos here. Note: Refreshing the app will clear your loaded videos, so try to finish your session!',
    disableBeacon: true,
    placement: 'right',
  },
  {
    target: '.tuning-tour-model-tabs',
    title: 'Step 2: Choose What to Teach',
    content:
      'Select which AI model you want to improve. Each model requires a different type of annotation (Points vs. Time Segments).',
    disableBeacon: true,
    placement: 'left',
  },
  {
    target: '.tuning-tour-annotate',
    title: 'Step 3: Annotate the Footage',
    content:
      'For Events: Scrub the video and mark exact frames (Start, End, Choice).\nFor Bosses: Click and drag on the timeline to mark segments of fighting.',
    disableBeacon: true,
    placement: 'bottom',
  },
  {
    target: '.tuning-tour-train-checkbox',
    title: 'Step 4: Enable for Training',
    content:
      'Once you have enough marks (e.g., 6 points or 5 segments), check this box to queue this model for the training run.',
    disableBeacon: true,
    placement: 'left',
  },
  {
    target: '.tuning-tour-start-training',
    title: 'Step 5: Launch the Training',
    content:
      "When you're ready, click here! You'll be asked to name your model, and then a terminal will appear showing the real-time AI training process.",
    disableBeacon: true,
    placement: 'bottom',
  },
];

export const TUNING_JOYRIDE_STYLES = {
  options: {
    arrowColor: '#0f172a',
    backgroundColor: '#0f172a',
    overlayColor: 'rgba(2, 6, 23, 0.72)',
    primaryColor: '#22d3ee',
    textColor: '#e2e8f0',
    zIndex: 10050,
  },
  tooltip: {
    borderRadius: 12,
    border: '1px solid rgba(34, 211, 238, 0.35)',
    boxShadow: '0 0 32px rgba(34, 211, 238, 0.12)',
    fontFamily: 'JetBrains Mono, ui-monospace, monospace',
    fontSize: 13,
    padding: 16,
  },
  tooltipTitle: {
    fontFamily: 'Orbitron, ui-sans-serif, system-ui, sans-serif',
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    color: '#67e8f9',
    marginBottom: 8,
  },
  tooltipContent: {
    color: '#cbd5e1',
    lineHeight: 1.55,
    padding: 0,
  },
  buttonNext: {
    backgroundColor: '#0891b2',
    borderRadius: 8,
    color: '#f8fafc',
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
  },
  buttonBack: {
    color: '#94a3b8',
    fontSize: 11,
    marginRight: 8,
  },
  buttonSkip: {
    color: '#64748b',
    fontSize: 11,
  },
  buttonClose: {
    color: '#94a3b8',
  },
  spotlight: {
    borderRadius: 12,
  },
};

export const TUNING_JOYRIDE_LOCALE = {
  back: 'Back',
  close: 'Close',
  last: 'Done',
  next: 'Next',
  skip: 'Skip tour',
};
