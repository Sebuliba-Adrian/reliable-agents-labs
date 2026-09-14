"""The real data source for Project 2. No vector database yet (chapter
12), no embeddings yet (chapter 11), just the actual, current package
metadata this whole project exists to make a model answer questions
about correctly.

PyPI's JSON API is public, unauthenticated, and free: `GET
https://pypi.org/pypi/<name>/json` for any published package. Chosen
over npm for the same reason Gemini was chosen as the default model
provider: this book's own toolchain already lives on PyPI, so every
example doubles as a real check against packages this project actually
depends on.
"""

import httpx
from pydantic import BaseModel

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"


class PackageMetadata(BaseModel):
    name: str
    version: str
    summary: str
    requires_dist: list[str] = []


async def fetch_package_metadata(name: str) -> PackageMetadata:
    """A real, live call to PyPI's public API, current metadata for
    whatever `name` resolves to right now, not a snapshot frozen at
    whatever point a model was trained.
    """
    url = PYPI_JSON_URL.format(name=name)
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
    payload = response.json()
    info = payload["info"]
    return PackageMetadata(
        name=info["name"],
        version=info["version"],
        summary=info["summary"] or "",
        requires_dist=info.get("requires_dist") or [],
    )
