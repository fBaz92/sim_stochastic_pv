from __future__ import annotations

from fastapi import FastAPI

from .routes import (
    configurations_router,
    execution_router,
    hardware_router,
    profiles_router,
    simulation_router,
)


def create_app() -> FastAPI:
    """
    Instantiate the FastAPI application and register routers.

    Creates the main FastAPI application with CORS middleware and
    registers all domain-specific routers:
    - hardware: Hardware catalog management (inverters, panels, batteries)
    - simulation: Direct simulation execution (analysis, optimization)
    - profiles: Load and price profile management
    - configurations: Saved configuration management
    - execution: Database-driven execution of saved configs

    Returns:
        FastAPI: Configured FastAPI application instance ready to serve.

    Example:
        ```python
        # Direct usage
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)

        # Or use the pre-created instance
        from sim_stochastic_pv.api.app import app
        ```
    """
    app = FastAPI(
        title="Simulatore Stocastico PV API",
        version="0.1.0",
        description="API per eseguire analisi e ottimizzazioni degli scenari PV.",
    )

    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register all domain-specific routers
    app.include_router(hardware_router)
    app.include_router(simulation_router)
    app.include_router(profiles_router)
    app.include_router(configurations_router)
    app.include_router(execution_router)

    return app


app = create_app()
