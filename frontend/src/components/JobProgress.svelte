<script>
    /**
     * Phase 12 — Floating bottom-left progress widget.
     *
     * Watches the ``activeJob`` store. When non-null, polls the backend
     * job status every 500 ms, updates the visible bar, and on
     * ``status='done'`` redirects to the Dashboard with the newly
     * created run pre-selected. On ``status='failed'`` shows the error
     * and lets the user dismiss the widget.
     */
    import { onDestroy } from "svelte";
    import { api } from "../api";
    import { activeJob, pendingRunId } from "../lib/stores";

    let poll = null;
    let dismissed = false;

    /** Stop polling and clear the active-job store. */
    function stopPolling() {
        if (poll != null) {
            clearInterval(poll);
            poll = null;
        }
    }

    /** Build a human-readable label for the job kind. */
    function kindLabel(kind) {
        if (kind === "analysis") return "Analisi scenario";
        if (kind === "optimization") return "Esecuzione design";
        return "Esecuzione";
    }

    /** Round a 0–1 fraction to one decimal, capped to 100%. */
    function fmtPct(f) {
        if (f == null || !isFinite(f)) return "0.0";
        return (Math.min(1, Math.max(0, f)) * 100).toFixed(1);
    }

    /**
     * React to changes in the activeJob store: (re)start polling when a
     * new job arrives, stop polling when cleared.
     */
    $: if ($activeJob && $activeJob.status !== "done" && $activeJob.status !== "failed") {
        dismissed = false;
        if (poll == null) {
            const id = $activeJob.id;
            poll = setInterval(async () => {
                try {
                    const snap = await api.getJob(id);
                    activeJob.set(snap);
                    if (snap.status === "done") {
                        stopPolling();
                        // Auto-redirect to the Dashboard. For analyses the
                        // run_id is set; for design sweeps it may be null
                        // (multi-run): we still redirect, the Dashboard
                        // will show the most recent runs.
                        if (snap.run_id != null) pendingRunId.set(snap.run_id);
                        window.location.hash = "/";
                    } else if (snap.status === "failed") {
                        stopPolling();
                    }
                } catch (e) {
                    console.error("Job polling failed:", e);
                }
            }, 500);
        }
    } else if (!$activeJob) {
        stopPolling();
    }

    onDestroy(stopPolling);

    function dismiss() {
        dismissed = true;
        activeJob.set(null);
    }
</script>

{#if $activeJob && !dismissed}
    <div class="job-progress" class:done={$activeJob.status === "done"} class:failed={$activeJob.status === "failed"}>
        <div class="row top">
            <span class="kind">{kindLabel($activeJob.kind)}</span>
            {#if $activeJob.status === "failed"}
                <button class="dismiss" type="button" on:click={dismiss} title="Chiudi">×</button>
            {/if}
        </div>
        {#if $activeJob.status === "failed"}
            <div class="message error-msg">Errore: {$activeJob.error || "esecuzione fallita"}</div>
        {:else if $activeJob.status === "done"}
            <div class="message">Completato — apertura risultati...</div>
            <div class="bar"><div class="bar-fill" style="width: 100%"></div></div>
        {:else}
            <div class="message">
                {#if $activeJob.progress_total > 0}
                    {$activeJob.progress_done} / {$activeJob.progress_total}
                    ({fmtPct($activeJob.progress_fraction)} %)
                {:else}
                    {$activeJob.message || "Inizializzazione..."}
                {/if}
            </div>
            <div class="bar">
                <div
                    class="bar-fill"
                    style="width: {fmtPct($activeJob.progress_fraction)}%"
                ></div>
            </div>
            {#if $activeJob.message}
                <div class="sub">{$activeJob.message}</div>
            {/if}
        {/if}
    </div>
{/if}

<style>
    .job-progress {
        position: fixed;
        bottom: 1rem;
        left: 1rem;
        width: 320px;
        z-index: 1000;
        background: var(--color-bg-primary, #fff);
        border: 1px solid var(--color-border, #e2e8f0);
        border-left: 4px solid var(--color-accent, #0d6efd);
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
    }
    .job-progress.done {
        border-left-color: var(--color-success, #198754);
    }
    .job-progress.failed {
        border-left-color: var(--color-danger, #dc3545);
    }
    .row.top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .kind {
        font-weight: 600;
        color: var(--color-text, #1f2937);
    }
    .dismiss {
        background: transparent;
        border: none;
        font-size: 1.1rem;
        cursor: pointer;
        color: var(--color-text-secondary, #6c757d);
    }
    .dismiss:hover {
        color: var(--color-danger, #dc3545);
    }
    .message {
        margin-bottom: 0.4rem;
        color: var(--color-text-secondary, #6c757d);
    }
    .error-msg {
        color: var(--color-danger, #dc3545);
    }
    .bar {
        width: 100%;
        height: 6px;
        background: var(--color-bg-tertiary, #e9ecef);
        border-radius: 3px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        background: var(--color-accent, #0d6efd);
        transition: width 0.3s ease;
    }
    .job-progress.done .bar-fill {
        background: var(--color-success, #198754);
    }
    .sub {
        margin-top: 0.35rem;
        font-size: 0.75rem;
        color: var(--color-text-muted, #adb5bd);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
