"""
Supabase storage/auth — OPTIONAL layer.

Per the build plan, Supabase auth/storage is the first thing to cut if
time runs short ("Cut Supabase auth and use a simple no-login flow").
So this module is written to fail SAFE: if SUPABASE_URL / SUPABASE_KEY
aren't set, storage is simply skipped and /cloak still works fully.
"""

import os
import uuid

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

_enabled = bool(SUPABASE_URL and SUPABASE_KEY)
_client = None

if _enabled:
    try:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        # If the supabase package isn't installed or the client fails to
        # init, storage silently disables itself rather than crashing
        # the whole API.
        _enabled = False
        _client = None


def is_enabled() -> bool:
    return _enabled


def save_temp_image(image_bytes: bytes, bucket: str = "cloaked-temp") -> str | None:
    """
    Uploads a cloaked image to a temp Supabase bucket, returns a storage
    path/key. Returns None if storage is disabled — callers must treat
    None as "not persisted, that's fine" rather than an error.
    """
    if not _enabled or _client is None:
        return None

    key = f"{uuid.uuid4()}.png"
    try:
        _client.storage.from_(bucket).upload(key, image_bytes, {"content-type": "image/png"})
        return key
    except Exception:
        return None