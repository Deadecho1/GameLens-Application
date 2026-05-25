/**
 * Resolves header/sidebar profile label from auth state (guest vs signed-in).
 */
export function resolveUserProfileDisplay(auth) {
  const loggedIn = Boolean(auth?.loggedIn);
  const email = String(auth?.email ?? '').trim();

  if (loggedIn && email) {
    const prefix = email.includes('@') ? email.split('@')[0] : email;
    const displayName = prefix || email;
    const initials = displayName.slice(0, 2).toUpperCase() || 'U';
    return {
      displayName,
      subtitle: email,
      initials,
      badge: { label: 'Online', variant: 'online' },
      isGuest: false,
    };
  }

  return {
    displayName: 'Local Guest',
    subtitle: 'Local Holocure demo · SQLite on this device',
    initials: 'LG',
    badge: { label: 'Offline (SQLite)', variant: 'offline' },
    isGuest: true,
  };
}
