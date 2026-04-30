import { describe, it, expect } from 'vitest';
import { durationToSeconds, secondsToMinutes, formatSecondsAsHMS } from './duration.js';

describe('durationToSeconds', () => {
  it('parses HH:MM:SS', () => {
    expect(durationToSeconds('01:30:00')).toBe(5400);
    expect(durationToSeconds('00:05:20')).toBe(320);
    expect(durationToSeconds('1:00:01')).toBe(3601);
  });

  it('parses MM:SS', () => {
    expect(durationToSeconds('05:20')).toBe(320);
    expect(durationToSeconds('00:00')).toBe(0);
    expect(durationToSeconds('2:15')).toBe(135);
  });

  it('parses bare seconds', () => {
    expect(durationToSeconds('30')).toBe(30);
    expect(durationToSeconds('0')).toBe(0);
  });

  it('returns 0 for empty / null / non-string', () => {
    expect(durationToSeconds('')).toBe(0);
    expect(durationToSeconds(null)).toBe(0);
    expect(durationToSeconds(undefined)).toBe(0);
    expect(durationToSeconds(42)).toBe(0);
  });

  it('returns 0 when parts contain NaN', () => {
    expect(durationToSeconds('01:ab:00')).toBe(0);
    expect(durationToSeconds('xx:yy')).toBe(0);
  });
});

describe('secondsToMinutes', () => {
  it('converts whole minutes', () => {
    expect(secondsToMinutes(60)).toBe(1);
    expect(secondsToMinutes(120)).toBe(2);
  });

  it('rounds to 2 decimal places', () => {
    expect(secondsToMinutes(90)).toBe(1.5);
    expect(secondsToMinutes(100)).toBe(1.67);
  });

  it('handles zero', () => {
    expect(secondsToMinutes(0)).toBe(0);
  });
});

describe('formatSecondsAsHMS', () => {
  it('formats with hours when >= 3600', () => {
    expect(formatSecondsAsHMS(3661)).toBe('1:01:01');
    expect(formatSecondsAsHMS(3600)).toBe('1:00:00');
    expect(formatSecondsAsHMS(7322)).toBe('2:02:02');
  });

  it('formats as M:SS when < 3600', () => {
    expect(formatSecondsAsHMS(90)).toBe('1:30');
    expect(formatSecondsAsHMS(60)).toBe('1:00');
    expect(formatSecondsAsHMS(5)).toBe('0:05');
  });

  it('handles zero', () => {
    expect(formatSecondsAsHMS(0)).toBe('0:00');
  });

  it('clamps negative to zero', () => {
    expect(formatSecondsAsHMS(-10)).toBe('0:00');
  });

  it('floors fractional seconds', () => {
    expect(formatSecondsAsHMS(90.9)).toBe('1:30');
  });
});
