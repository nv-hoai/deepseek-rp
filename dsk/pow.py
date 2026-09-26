"""Proof-of-work challenge solver (WebAssembly sha3).

Frontend parity: ``X-DS-PoW-Response: base64(json({algorithm, challenge, salt,
answer, signature, target_path}))``. The bundled WASM hash
(``7b9ca65ddd``) matches ``fe-static/.../static/sha3_wasm_bg.*.wasm``.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import wasmtime

from . import config
from .models import PowChallenge

WASM_PATH = Path(__file__).parent / "wasm" / config.WASM_FILENAME

_COMPILED_LOCK = threading.Lock()
_COMPILED: dict[str, tuple[Any, Any]] = {}


def _compiled_module(wasm_path: str | Path) -> tuple[Any, Any]:
    """Compile once per process; every request was recompiling the WASM."""
    key = os.fspath(wasm_path)
    try:
        return _COMPILED[key]
    except KeyError:
        pass
    with _COMPILED_LOCK:
        if key not in _COMPILED:
            engine = wasmtime.Engine()
            module = wasmtime.Module(engine, Path(key).read_bytes())
            _COMPILED[key] = (engine, module)
        return _COMPILED[key]


class DeepSeekHash:
    def __init__(self) -> None:
        self.instance: Any = None
        self.memory: Any = None
        self.store: Any = None

    def init(self, wasm_path: str | Path) -> "DeepSeekHash":
        engine, module = _compiled_module(wasm_path)
        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        self.instance = linker.instantiate(self.store, module)
        self.memory = self.instance.exports(self.store)["memory"]
        return self

    def _write_to_memory(self, text: str) -> tuple[int, int]:
        encoded = text.encode("utf-8")
        exports = self.instance.exports(self.store)
        ptr = exports["__wbindgen_export_0"](self.store, len(encoded), 1)
        view = self.memory.data_ptr(self.store)
        for i, byte in enumerate(encoded):
            view[ptr + i] = byte
        return ptr, len(encoded)

    def calculate_hash(self, challenge: str, salt: str,
                       difficulty: int, expire_at: int) -> int | None:
        prefix = f"{salt}_{expire_at}_"
        exports = self.instance.exports(self.store)
        retptr = exports["__wbindgen_add_to_stack_pointer"](self.store, -16)
        try:
            challenge_ptr, challenge_len = self._write_to_memory(challenge)
            prefix_ptr, prefix_len = self._write_to_memory(prefix)
            exports["wasm_solve"](
                self.store, retptr,
                challenge_ptr, challenge_len,
                prefix_ptr, prefix_len,
                float(difficulty),
            )
            view = self.memory.data_ptr(self.store)
            status = int.from_bytes(
                bytes(view[retptr:retptr + 4]), byteorder="little", signed=True)
            if status == 0:
                return None
            value_bytes = bytes(view[retptr + 8:retptr + 16])
            return int(np.frombuffer(value_bytes, dtype=np.float64)[0])
        finally:
            exports["__wbindgen_add_to_stack_pointer"](self.store, 16)


class DeepSeekPOW:
    def __init__(self, wasm_path: str | Path = WASM_PATH) -> None:
        self.hasher = DeepSeekHash().init(os.fspath(wasm_path))

    def solve_challenge(self, challenge: dict[str, Any] | PowChallenge) -> str:
        """Solve a PoW challenge and return the encoded ``X-DS-PoW-Response``."""
        data = (challenge if isinstance(challenge, dict)
                else {"algorithm": challenge.algorithm,
                      "challenge": challenge.challenge,
                      "salt": challenge.salt,
                      "signature": challenge.signature,
                      "difficulty": challenge.difficulty,
                      "expire_at": challenge.expire_at,
                      "target_path": challenge.target_path})
        answer = self.hasher.calculate_hash(
            data["challenge"], data["salt"],
            int(data["difficulty"]), int(data["expire_at"]),
        )
        result = {
            "algorithm": data["algorithm"],
            "challenge": data["challenge"],
            "salt": data["salt"],
            "answer": answer,
            "signature": data["signature"],
            "target_path": data["target_path"],
        }
        return base64.b64encode(json.dumps(result).encode()).decode()
