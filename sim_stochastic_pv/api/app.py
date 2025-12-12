from __future__ import annotations

from fastapi import FastAPI

from .routes import router


def create_app() -> FastAPI:
    """Instantiate the FastAPI application and register routers."""
    app = FastAPI(
        title="Simulatore Stocastico PV API",
        version="0.1.0",
        description="API per eseguire analisi e ottimizzazioni degli scenari PV.",
    )
    app.include_router(router)
    return app


app = create_app()
