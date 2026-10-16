"""Minimal OpenAI-compatible chat client (stdlib only) for the local vLLM server (or another provider).
Images are sent as base64 JPEG data URLs. Every call is logged by the caller; keys are read from the
environment only (never from configs or logs).
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key_env: str | None = None, timeout_s: float = 600):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key = os.environ.get(api_key_env) if api_key_env else None
        self.timeout_s = timeout_s
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "seconds": 0.0}

    @staticmethod
    def local(cfg: dict) -> "ChatClient":
        return ChatClient(f"http://127.0.0.1:{cfg['port']}/v1", cfg["served_name"])

    def wait_ready(self, max_s: float = 900) -> None:
        t0 = time.time()
        while time.time() - t0 < max_s:
            try:
                with urllib.request.urlopen(self.base_url + "/models", timeout=10) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(10)
        raise TimeoutError("LLM server not ready")

    def chat(self, messages: list, tools: list | None = None, temperature: float = 0.2, max_tokens: int = 2048,
             json_mode: bool = False) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:600]!r}") from None
        u = out.get("usage") or {}
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        self.usage["calls"] += 1
        self.usage["seconds"] += time.time() - t0
        return out["choices"][0]["message"]


def image_part(rgb: np.ndarray, long_side: int = 896, quality: int = 85) -> dict:
    from PIL import Image
    im = Image.fromarray(np.asarray(rgb, np.uint8))
    s = long_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}


def parse_json_reply(text: str) -> dict:
    """First JSON object in a reply (models sometimes wrap it in ```json fences or add prose)."""
    t = text.strip()
    if "```" in t:
        seg = t.split("```")
        for s in seg:
            s = s.strip()
            if s.startswith("json"):
                s = s[4:].strip()
            if s.startswith("{"):
                t = s
                break
    i, depth = t.find("{"), 0
    if i < 0:
        raise ValueError("no JSON object in reply")
    for j in range(i, len(t)):
        depth += {"{": 1, "}": -1}.get(t[j], 0)
        if depth == 0:
            return json.loads(t[i:j + 1])
    raise ValueError("unterminated JSON in reply")
