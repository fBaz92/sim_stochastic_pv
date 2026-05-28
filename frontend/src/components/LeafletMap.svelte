<!--
    LeafletMap — Phase 14.

    Thin wrapper around Leaflet for the wizard "Luogo" step. Renders an
    OpenStreetMap-tiled slippy map with a single draggable marker. The
    parent binds ``lat`` and ``lon`` and listens to ``on:change`` to react
    to user interactions (marker drag or map click).

    OpenStreetMap tiles are free as long as the consuming app stays
    well within their tile-usage policy (a single user typing a few queries
    per minute is fine). No API key required.

    Props:
        lat, lon      — current marker position, two-way bound.
        height        — CSS height for the map container. Default 320px.
        zoom          — initial zoom level. Default 8 (regional view).

    Events:
        change        — fired when the user drags the marker or clicks the
                        map, with detail = { lat, lon }.

    Notes:
        - Leaflet's CSS is imported here so callers don't have to wire it
          in their global stylesheet.
        - The component is reactive to external prop changes: if the parent
          mutates ``lat``/``lon`` (e.g. after picking a Nominatim result),
          the marker and view follow.
-->
<script>
    import { createEventDispatcher, onMount, onDestroy, tick } from "svelte";
    import L from "leaflet";
    import "leaflet/dist/leaflet.css";

    // Leaflet's default marker icons rely on relative URLs that break when
    // bundled by Vite. Patch them with the bundled assets to avoid the
    // common "broken marker image" issue.
    import markerIcon from "leaflet/dist/images/marker-icon.png";
    import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
    import markerShadow from "leaflet/dist/images/marker-shadow.png";

    delete L.Icon.Default.prototype._getIconUrl;
    L.Icon.Default.mergeOptions({
        iconRetinaUrl: markerIcon2x,
        iconUrl: markerIcon,
        shadowUrl: markerShadow,
    });

    export let lat = 44.336;
    export let lon = 10.831;
    export let height = "320px";
    export let zoom = 8;

    const dispatch = createEventDispatcher();

    let container;
    let map = null;
    let marker = null;
    /** True while the component is mutating lat/lon itself, so we don't
     *  recursively trigger the reactive block. */
    let suppressReactive = false;

    onMount(async () => {
        await tick();
        map = L.map(container, { zoomControl: true }).setView([lat, lon], zoom);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            maxZoom: 19,
            attribution: "© OpenStreetMap contributors",
        }).addTo(map);

        marker = L.marker([lat, lon], { draggable: true }).addTo(map);

        marker.on("dragend", () => {
            const ll = marker.getLatLng();
            updateFromUser(ll.lat, ll.lng);
        });

        map.on("click", (e) => {
            marker.setLatLng(e.latlng);
            updateFromUser(e.latlng.lat, e.latlng.lng);
        });
    });

    onDestroy(() => {
        if (map) {
            map.remove();
            map = null;
        }
    });

    function updateFromUser(newLat, newLon) {
        suppressReactive = true;
        lat = newLat;
        lon = newLon;
        // Allow the reactive block to run once with the new values without
        // moving the marker (we already did that locally).
        Promise.resolve().then(() => {
            suppressReactive = false;
        });
        dispatch("change", { lat: newLat, lon: newLon });
    }

    // Sync marker + view when parent mutates lat/lon (e.g. after picking a
    // Nominatim result). Guard against the reactive feedback loop.
    $: if (map && marker && !suppressReactive) {
        marker.setLatLng([lat, lon]);
        map.setView([lat, lon], Math.max(map.getZoom(), 10), { animate: true });
    }
</script>

<div bind:this={container} class="leaflet-container" style:height></div>

<style>
    .leaflet-container {
        width: 100%;
        border-radius: 8px;
        border: 1px solid var(--border, #d1d5db);
        z-index: 0; /* keep below modals */
    }
</style>
