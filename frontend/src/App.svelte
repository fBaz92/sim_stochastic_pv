<script>
  import Router from './components/Router.svelte';
  import routes from './routes';
  import Navbar from './components/Navbar.svelte';
  import JobProgress from './components/JobProgress.svelte';
</script>

<Navbar />

<main>
  <Router {routes} />
</main>

<!-- Phase 12 — floating progress widget for background MC jobs. -->
<JobProgress />

<style>
  main {
    padding-bottom: 2rem;
  }
</style>
