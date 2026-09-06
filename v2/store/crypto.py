"""Fernet encryption at rest (mandatory — no plaintext fallback)."""
from __future__ import annotations

import logging
import os
import base64

from cryptography.fernet import Fernet, InvalidToken
from v2.settings import get_settings

log = logging.getLogger(__name__)


def _get_fernet() -> Fernet:
    """Get Fernet instance from settings; raises if key missing/invalid."""
    settings = get_settings()
    key = settings.ENCRYPTION_KEY
    if not key:
        raise RuntimeError(
            "ENCRYPTION_KEY not set — set DEEPSEEKER_ENCRYPTION_KEY in env or .env. "
            "Generate with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    try:
        # Accept both raw 32-byte key and base64-encoded
        if len(key) == 44 and key.endswith("="):
            return Fernet(key.encode())
        # Assume 32-byte raw key, base64 encode for Fernet
        return Fernet(base64.urlsafe_b64encode(key.encode()[:32].ljust(32, b"\0")))
    except Exception as e:
        raise RuntimeError(f"ENCRYPTION_KEY invalid: {e}") from e


def fernet_encrypt(plaintext: str | bytes) -> bytes:
    """Encrypt string or bytes; returns base64-encoded ciphertext."""
    f = _get_fernet()
    data = plaintext.encode() if isinstance(plaintext, str) else plaintext
    return f.encrypt(data)


def fernet_decrypt(ciphertext: bytes) -> str:
    """Decrypt bytes; returns UTF-8 string. Raises on auth failure."""
    f = _get_fernet()
    try:
        return f.decrypt(ciphertext).decode()
    except InvalidToken as e:
        raise RuntimeError("fernet_decrypt failed — wrong key or corrupted data") from e


def generate_fernet_key() -> str:
    """Generate a new Fernet key (base64-encoded)."""
    return Fernet.generate_key().decode()


# Legacy migration helper: re-encrypt all tokens from v1 key to current key
async def reencrypt_tokens(v1_key: str, new_key: str | None = None) -> int:
    """Decrypt all tokens with v1_key, re-encrypt with current (or new) key.
    Returns number of tokens re-encrypted.
    """
    from v2.store.db import connect
    from v2.store.repo_tokens import TokenRepo

    if new_key:
        os.environ["DEEPSEEKER_ENCRYPTION_KEY"] = new_key
    # Temporarily override the key for decryption
    old_get_fernet = _get_fernet
    v1_fernet = Fernet(v1_key.encode() if len(v1_key) != 44 else v1_key.encode())

    async def _v1_decrypt(ct: bytes) -> str:
        return v1_fernet.decrypt(ct).decode()

    conn = await connect()
    repo = TokenRepo(conn)
    tokens = await repo.list_all()
    count = 0
    for t in tokens:
        try:
            plain = _v1_decrypt(t.secret_enc)
            new_enc = fernet_encrypt(plain)
            await repo.update_secret(t.id, new_enc)
            count += 1
        except InvalidToken:
            log.warning("token_reencrypt_failed id=%s", t.id)
    return count