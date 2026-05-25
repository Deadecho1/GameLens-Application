/** Interactive onboarding tour for Game Tuning tab (react-joyride). */

export const TUNING_TOUR_STEPS = [
  {
    target: '.tuning-tour-add-video',
    title: 'Step 1: Load Footage',
    content:
      'Add one or more gameplay videos. Note: If you refresh the app, your un-trained progress will be lost, so complete your tagging in one session!',
    disableBeacon: true,
    placement: 'right',
  },
  {
    target: '.tuning-tour-model-tabs',
    title: 'Step 2: Pick a Model',
    content:
      "Choose what to train. 'Event Detector' requires marking specific frames. 'Boss Detector' requires dragging to highlight time segments.",
    disableBeacon: true,
    placement: 'left',
  },
  {
    target: '.tuning-tour-timeline',
    title: 'Step 3: Navigate & Drag',
    content:
      'Scrub to find the right moment. If you are training the BOSS DETECTOR: Click and drag across this timeline to highlight a fight segment (must be at least 3 seconds long).',
    disableBeacon: true,
    placement: 'top',
  },
  {
    target: '.tuning-tour-annotation-controls',
    title: 'Step 4: Mark specific Frames',
    content:
      "If you are training the EVENT DETECTOR: Pause at the exact moment an event happens, select the category (Start/End/Choice), and click 'MARK FRAME' (or press M). You need at least 6 marks per category.",
    disableBeacon: true,
    placement: 'top',
  },
  {
    target: '.tuning-tour-start-training',
    title: 'Step 5: Train!',
    content:
      "Once you've gathered enough marks (watch the progress bars!), check the 'Train' box and click here to start the AI fine-tuning process.",
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
