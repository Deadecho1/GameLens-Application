import {
  Bomb,
  FlaskConical,
  Gem,
  Hammer,
  Heart,
  Package,
  Shield,
  Sword,
  Wand2,
  Zap,
} from 'lucide-react';

export function pickItemIcon(name) {
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

export function inferCategory(item) {
  if (item?.category) return item.category;
  const n = String(item?.name ?? '').toLowerCase();
  if (/(shield|plate|buckler)/.test(n)) return 'defensive';
  if (/(sword|dagger|mallet|venom|explosive|thunder|fire|frost)/.test(n)) return 'offensive';
  return 'utility';
}
