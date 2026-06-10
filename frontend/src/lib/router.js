import { writable } from 'svelte/store';

/**
 * Reactive store holding the current hash-based route (e.g. "/", "/scenario").
 *
 * Components subscribe to it (typically via the `$currentRoute` auto-subscription
 * syntax in Svelte) to react to navigation changes. The store is updated by
 * `Router.svelte` on `hashchange` events.
 */
export const currentRoute = writable('/');

/**
 * Read the current hash-based path from `window.location`.
 *
 * Strips the leading `#` so that `#/scenario` becomes `/scenario`. When no
 * hash is present, falls back to `/` so the root route is always defined.
 *
 * @returns {string} The current path without the leading `#`.
 */
export function getHashPath() {
    const hash = window.location.hash || '#/';
    return hash.startsWith('#') ? hash.slice(1) : hash;
}
