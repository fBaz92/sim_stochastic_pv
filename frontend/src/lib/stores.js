import { writable } from 'svelte/store';

/**
 * Currently running backend job, or ``null`` when no job is active.
 *
 * Shape (when non-null), as produced by the backend job-status endpoint and
 * consumed by `JobProgress.svelte`:
 *   {
 *     id: string,
 *     kind: "analysis" | "optimization" | string,
 *     status: "pending" | "running" | "done" | "failed",
 *     progress_done: number,
 *     progress_total: number,
 *     progress_fraction: number,   // 0..1
 *     message?: string,
 *     error?: string,
 *     run_id?: number | null
 *   }
 *
 * Pages set it after kicking off a Monte Carlo run; `JobProgress` polls the
 * backend and updates it; once `status === "done"`, the resulting `run_id`
 * is written to `pendingRunId` so the Dashboard can auto-load the results.
 */
export const activeJob = writable(null);

/**
 * ID of a run the Dashboard should load on its next mount, or ``null``.
 *
 * Set by `JobProgress.svelte` when a job completes, consumed (and cleared)
 * by `Dashboard.svelte` so the user lands directly on the fresh results
 * without having to pick them from the list.
 */
export const pendingRunId = writable(null);

/**
 * ID of a saved configuration that the Scenario/Campaign builder should
 * rehydrate on mount, or ``null``.
 *
 * Set by the Database managers when the user clicks "Modifica nel wizard",
 * then cleared by the builder page once the configuration is loaded.
 */
export const pendingConfigurationId = writable(null);
