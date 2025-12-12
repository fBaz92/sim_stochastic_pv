<script>
  import Router from './components/Router.svelte';
  import routes from './routes';
  import Navbar from './components/Navbar.svelte';
</script>

<Navbar />

<main>
  <Router {routes} />
</main>

<style>
  main {
    padding-bottom: 2rem;
  }
</style>
