import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))

/**
 * Resolve the dev config from the single source of truth: the repo-root `.env`,
 * falling back to `.env.example` (logged) — the same files Docker Compose and
 * the backend read. An explicit process env var (set by Docker's build arg or
 * by `make dev-frontend`) always wins. This keeps `npm run dev` and the IDE
 * preview consistent with the rest of the stack without a second config file.
 */
function loadRootEnv() {
  const root = resolve(__dirname, '..')
  let file = null
  if (existsSync(resolve(root, '.env'))) file = '.env'
  else if (existsSync(resolve(root, '.env.example'))) {
    file = '.env.example'
    console.warn("[config] .env non trovato — uso il fallback .env.example")
  }
  const out = {}
  if (file) {
    for (const line of readFileSync(resolve(root, file), 'utf-8').split('\n')) {
      const s = line.trim()
      if (!s || s.startsWith('#') || !s.includes('=')) continue
      const idx = s.indexOf('=')
      out[s.slice(0, idx).trim()] = s.slice(idx + 1).trim().replace(/^['"]|['"]$/g, '')
    }
  }
  return out
}

const env = loadRootEnv()
const apiBase = process.env.VITE_API_BASE || env.VITE_API_BASE || 'http://localhost:8000/api'
const port = Number(process.env.FRONTEND_PORT || env.FRONTEND_PORT || 5173)

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  // Bake the resolved API base into the client (api.js reads it with a fallback).
  define: {
    'import.meta.env.VITE_API_BASE': JSON.stringify(apiBase),
  },
  server: {
    host: '127.0.0.1',
    port,
  },
})
