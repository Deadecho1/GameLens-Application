/** Deterministic accent for list rows / slots (no image assets). */
const ACCENT_HUES = [187, 262, 142, 38, 280, 200];

export function itemAccentHue(id) {
  return ACCENT_HUES[Math.abs(Number(id) || 0) % ACCENT_HUES.length];
}

export function itemAccentDotStyle(id, active = false) {
  const hue = itemAccentHue(id);
  return active
    ? {
        backgroundColor: `hsl(${hue} 70% 50%)`,
        boxShadow: `0 0 10px hsl(${hue} 70% 50% / 0.45)`,
      }
    : { backgroundColor: `hsl(${hue} 55% 42%)` };
}

/** Boss rail accents (offset hue set from item palette). */
export function bossAccentDotStyle(id, active = false) {
  const hue = (itemAccentHue(id) + 24) % 360;
  return active
    ? {
        backgroundColor: `hsl(${hue} 75% 52%)`,
        boxShadow: `0 0 12px hsl(${hue} 75% 52% / 0.5)`,
      }
    : { backgroundColor: `hsl(${hue} 45% 38%)` };
}
