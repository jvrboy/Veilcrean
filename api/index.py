"""Root Vercel entrypoint.

This wrapper lets Vercel deploy from the repository root while keeping the real
serverless API implementation in vercel_app/api/index.py.
"""
from vercel_app.api.index import app  # noqa: F401
