<script>
    import { onMount } from "svelte";
    import { currentRoute, getHashPath } from "../lib/router";

    export let routes = {};
    export let fallback = null;

    let path = "/";

    function updateRoute() {
        path = getHashPath();
        currentRoute.set(path);
    }

    onMount(() => {
        updateRoute();
        window.addEventListener("hashchange", updateRoute);
        return () => window.removeEventListener("hashchange", updateRoute);
    });

    $: Component = routes[path] ?? fallback ?? routes["/"] ?? null;
</script>

{#if Component}
    <svelte:component this={Component} />
{:else}
    <p>Page not found.</p>
{/if}
