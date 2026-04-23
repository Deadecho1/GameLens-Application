import { useEffect, useState } from 'react';

/**
 * Animate integer from 0 to endValue over duration (ms). Re-runs when endValue changes.
 */
export function useCountUp(endValue, duration = 1400) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    const target = Math.max(0, Math.floor(Number(endValue) || 0));
    let frame;
    const t0 = performance.now();

    const step = (now) => {
      const t = Math.min((now - t0) / duration, 1);
      const eased = 1 - (1 - t) ** 3;
      setValue(Math.round(target * eased));
      if (t < 1) frame = requestAnimationFrame(step);
    };

    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [endValue, duration]);

  return value;
}
