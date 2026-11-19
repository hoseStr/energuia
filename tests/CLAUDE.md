# Tests

- Los tests de la API usan `httpx2` (grupo `dev`), no `httpx`: Starlette 1.10 marca `httpx` como obsoleto para `TestClient`.
