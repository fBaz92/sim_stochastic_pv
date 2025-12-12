<script>
    import { currentRoute } from "../lib/router";
</script>

<nav class="navbar">
    <div class="container navbar-content">
        <div class="brand">
            <a href="#/" class="brand-link">PV Sim</a>
        </div>
        <ul class="nav-links">
            <li>
                <a href="#/" class:active={$currentRoute === "/"}>Dashboard</a>
            </li>
            <li>
                <a
                    href="#/scenario"
                    class:active={$currentRoute === "/scenario"}>Scenario</a
                >
            </li>
            <li>
                <a
                    href="#/campaign"
                    class:active={$currentRoute === "/campaign"}>Campaign</a
                >
            </li>
            <li>
                <a
                    href="#/database"
                    class:active={$currentRoute === "/database"}>Database</a
                >
            </li>
        </ul>
    </div>
</nav>

<style>
    .navbar {
        background-color: var(--color-bg-primary);
        border-bottom: 1px solid var(--color-border);
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .navbar-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-link {
        font-weight: 700;
        font-size: 1.25rem;
        color: var(--color-text-primary);
    }
    .nav-links {
        display: flex;
        gap: 1.5rem;
        list-style: none;
    }
    .nav-links a {
        font-weight: 500;
        color: var(--color-text-secondary);
        transition: color 0.2s;
    }
    .nav-links a:hover {
        color: var(--color-accent);
    }
    :global(.active) {
        color: var(--color-accent) !important;
    }
</style>
