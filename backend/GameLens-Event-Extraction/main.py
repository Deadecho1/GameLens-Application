import contextlib

from fastapi import FastAPI

from gamelens.db import DatabaseConnection
from gamelens.extraction.boss import router as boss_router
from gamelens.extraction.choice import router as choice_router
from gamelens.util import init_db


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    yield

    print("Shutting down FastAPI application...")
    await DatabaseConnection.close()


app = FastAPI(lifespan=lifespan)
app.include_router(choice_router)
app.include_router(boss_router)
