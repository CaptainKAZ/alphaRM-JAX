#!/usr/bin/env python3
"""RM 3v3 训练片段观察窗服务（纯标准库）。

  /                 -> rm_live.html
  /stats.html       -> 跨 iter 的终局占比与得分统计
  /rm_live.json     -> outputs/live/rm_live.json
  /rm_stats.json    -> 由 /rm_live.json 累积的逐 iter 统计
  /status.json      -> 数据时间/大小/训练是否在跑

用法：python3 liveview/serve_rm_live.py --port 8768
"""
from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent / "outputs" / "live"
JSON_PATH = LIVE / "rm_live.json"
STATS_PATH = LIVE / "rm_stats_history.json"
HTML_PATH = HERE / "rm_live.html"
STATS_HTML_PATH = HERE / "rm_stats.html"
CHART_PATH = HERE / "chart.umd.min.js"
OUTPUTS_ROOT = (HERE.parent / "outputs").resolve()

# RmState / observation 当前布局
GLOBAL_OFFSET = 180
TARGET_SCORE = 90.0
_history_lock = threading.Lock()
_parse_cache = {"mtime": None, "size": None, "iter": None, "names": {}}


def training_running() -> int:
    try:
        r = subprocess.run(
            ["pgrep", "-fc", r"train_rm(_jax)?\.py"], capture_output=True, text=True, timeout=5
        )
        return int(r.stdout.strip() or 0)
    except Exception:
        return 0


def _load_history() -> list[dict]:
    try:
        raw = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def _save_history(history: list[dict]) -> None:
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, STATS_PATH)


def _clip_terminal_code(clip: dict) -> int:
    codes = clip.get("codes") or []
    # codes 是逐帧广播；终局后通常保持该码。取最后一个非 0，
    # 避免末尾 padding/未结束帧把已结束片段误记为 0。
    for code in reversed(codes):
        try:
            code = int(code)
        except Exception:
            continue
        if code != 0:
            return code
    return 0


def _clip_scores(clip: dict) -> tuple[float, float, float]:
    """返回片段内 (red_delta, blue_delta, total_delta)。"""

    state = clip.get("state") or []
    if not state:
        return 0.0, 0.0, 0.0

    def score_at(row):
        try:
            return (
                float(row[GLOBAL_OFFSET]) * TARGET_SCORE,
                float(row[GLOBAL_OFFSET + 1]) * TARGET_SCORE,
            )
        except Exception:
            return 0.0, 0.0

    r0, b0 = score_at(state[0])
    r1, b1 = score_at(state[-1])
    return r1 - r0, b1 - b0, (r1 + b1) - (r0 + b0)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _max(xs):
    return max(xs) if xs else 0.0


def _float_list(raw) -> list[float]:
    if not isinstance(raw, list):
        return []
    out = []
    for v in raw:
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def _record_for_payload(payload: dict) -> dict:
    clips = payload.get("clips") or []
    n_clips = len(clips)

    # 下一次训练起，回调会写全 batch 的轨迹统计；优先使用它。
    raw_counts = payload.get("all_outcome_counts")
    all_deltas = _float_list(payload.get("trajectory_score_deltas"))
    if isinstance(raw_counts, dict) and all_deltas:
        counts = {}
        for k, v in raw_counts.items():
            try:
                counts[str(int(k))] = int(v)
            except Exception:
                continue
        n_done = int(payload.get("all_n_done", sum(counts.values())))
        n_envs = int(payload.get("all_n_envs", len(all_deltas)))
        red_deltas = _float_list(payload.get("trajectory_score_delta_red"))
        blue_deltas = _float_list(payload.get("trajectory_score_delta_blue"))
        finals = _float_list(payload.get("trajectory_score_final"))
        return {
            "iter": int(payload.get("iter", 0)),
            "source": "all",
            "n_clips": n_clips,
            "n_trajectories": n_envs,
            "n_terminal": n_done,
            "outcome_counts": counts,
            "score_delta_mean": round(_mean(all_deltas), 3),
            "score_delta_max": round(_max(all_deltas), 3),
            "score_delta_red_mean": round(_mean(red_deltas), 3),
            "score_delta_blue_mean": round(_mean(blue_deltas), 3),
            "score_final_mean": round(_mean(finals), 3),
            "score_final_max": round(_max(finals), 3),
            # 每条轨迹一个数，供统计页画分布；500 iter 也只有约几 MB。
            "trajectory_score_deltas": [round(v, 3) for v in all_deltas],
            "trajectory_score_final": [round(v, 3) for v in finals],
        }

    # 兼容旧训练：只有抽样的 10 个片段。
    counts = {}
    deltas, finals, red_deltas, blue_deltas = [], [], [], []
    for clip in clips:
        code = str(_clip_terminal_code(clip))
        counts[code] = counts.get(code, 0) + 1
        dr, db, dt = _clip_scores(clip)
        red_deltas.append(dr)
        blue_deltas.append(db)
        deltas.append(dt)
        state = clip.get("state") or []
        if state:
            try:
                finals.append(
                    (float(state[-1][GLOBAL_OFFSET]) + float(state[-1][GLOBAL_OFFSET + 1]))
                    * TARGET_SCORE
                )
            except Exception:
                finals.append(0.0)

    return {
        "iter": int(payload.get("iter", 0)),
        "source": "clips",
        "n_clips": n_clips,
        "n_trajectories": n_clips,
        "n_terminal": sum(v for k, v in counts.items() if k != "0"),
        "outcome_counts": counts,
        "score_delta_mean": round(_mean(deltas), 3),
        "score_delta_max": round(_max(deltas), 3),
        "score_delta_red_mean": round(_mean(red_deltas), 3),
        "score_delta_blue_mean": round(_mean(blue_deltas), 3),
        "score_final_mean": round(_mean(finals), 3),
        "score_final_max": round(_max(finals), 3),
    }


def update_history() -> list[dict]:
    """读取最新 rm_live.json，若 iter 未记录则追加。返回完整历史。"""
    with _history_lock:
        try:
            st = JSON_PATH.stat()
        except OSError:
            return _load_history()
        # 文件没变化就不重复解析 20MB JSON
        if (
            _parse_cache["mtime"] == st.st_mtime
            and _parse_cache["size"] == st.st_size
            and STATS_PATH.exists()
        ):
            return _load_history()
        try:
            payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            return _load_history()
        record = _record_for_payload(payload)
        history = _load_history()
        seen = any(int(r.get("iter", -1)) == record["iter"] for r in history)
        if not seen:
            # 新训练通常从 iter 1 重新开始；不要把上一轮的历史点混进来。
            if history and record["iter"] <= int(history[-1].get("iter", 0)):
                history = []
            history.append(record)
            history.sort(key=lambda r: int(r.get("iter", 0)))
            _save_history(history)
        _parse_cache.update(
            {
                "mtime": st.st_mtime,
                "size": st.st_size,
                "iter": record["iter"],
                "names": payload.get("outcome_names") or {},
            }
        )
        return history


def stats_payload() -> dict:
    history = update_history()
    names = _parse_cache.get("names") or {}
    return {
        "history": history,
        "outcome_names": names,
        "updated": time.time(),
        "training_processes": training_running(),
    }



FILES_CSS = """
<style>
:root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#e6edf3; --muted:#8b949e;
        --blue:#58a6ff; --green:#3fb950; --yellow:#d29922; }
* { box-sizing:border-box; }
html, body { margin:0; background:var(--bg); color:var(--fg);
  font-family:system-ui,"Noto Sans CJK SC",sans-serif; }
body { min-height:100dvh; }
#app { max-width:1100px; margin:0 auto; padding:0 14px 36px; }
header { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
  padding:12px 0; border-bottom:1px solid var(--line); }
header h1 { font-size:16px; margin:0; font-weight:650; }
header .links { display:flex; gap:8px; margin-left:auto; flex-wrap:wrap; }
a { color:var(--blue); text-decoration:none; }
a:hover { text-decoration:underline; }
.pill { padding:3px 9px; border:1px solid var(--line); border-radius:999px;
  color:var(--muted); font-size:12px; white-space:nowrap; }
.crumbs { margin:14px 0 10px; font-size:13px; overflow-wrap:anywhere; }
.sep { color:var(--muted); margin:0 5px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:12px; margin-top:12px; min-width:0; }
.card h2 { font-size:13px; margin:0 0 10px; font-weight:600; }
.listing { display:block; width:100%; }
.entry { display:grid; grid-template-columns:28px minmax(0,1fr) 88px 116px;
  gap:8px; align-items:center; padding:9px 6px; border-bottom:1px solid #21262d;
  color:var(--fg); border-radius:5px; width:100%; }
.entry:hover { background:#21262d; text-decoration:none; }
.ico { font-size:17px; text-align:center; }
.name { overflow-wrap:anywhere; min-width:0; }
.size, .time { color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums;
  text-align:right; }
.meta { display:flex; gap:8px 16px; flex-wrap:wrap; color:var(--muted);
  font-size:12px; margin:0 0 12px; }
.meta b { color:var(--fg); font-weight:500; }
.note { color:var(--yellow); font-size:12px; margin:0 0 10px; overflow-wrap:anywhere; }
pre { margin:0; padding:12px; background:#0a0e14; border:1px solid var(--line);
  border-radius:7px; overflow:auto; max-height:70vh; white-space:pre-wrap;
  overflow-wrap:anywhere; font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace; }
video { display:block; width:100%; max-height:72vh; background:#000;
  border-radius:8px; }
img { display:block; max-width:100%; height:auto; border-radius:8px; }
.actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
.btn { display:inline-block; padding:8px 12px; border:1px solid var(--line);
  border-radius:7px; background:#21262d; color:var(--fg); font-size:13px; }
.btn:hover { background:#30363d; text-decoration:none; }
.empty { color:var(--muted); padding:12px 4px; font-size:13px; }
@media (max-width:700px) {
  #app { padding:0 9px 28px; }
  header h1 { width:100%; font-size:14px; }
  header .links { margin-left:0; width:100%; }
  .pill { font-size:11px; padding:2px 7px; }
  .card { padding:9px; border-radius:8px; }
  .entry { grid-template-columns:24px minmax(0,1fr) 70px; padding:11px 4px; }
  .time { display:none; }
  .size { font-size:11px; }
  .btn { min-height:40px; padding:9px 12px; flex:1 1 auto; text-align:center; }
  pre { max-height:62vh; font-size:11px; }
}
</style>
"""

TEXT_EXTS = {".log", ".txt", ".json", ".yaml", ".yml", ".md", ".py", ".csv",
             ".svg", ".cfg", ".ini", ".sh", ".html", ".xml"}
TEXT_LIMIT = 300_000


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def _mtime(st: os.stat_result) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime))


def _safe_output_path(rel: str) -> Path | None:
    """Resolve a browser path under outputs/; reject traversal/symlink escape."""
    rel = unquote(rel or "").strip().lstrip("/")
    try:
        p = (OUTPUTS_ROOT / rel).resolve()
        p.relative_to(OUTPUTS_ROOT)
    except (OSError, ValueError):
        return None
    return p


def _tail_text(p: Path) -> tuple[str, bool]:
    size = p.stat().st_size
    truncated = size > TEXT_LIMIT
    with p.open("rb") as f:
        if truncated:
            f.seek(size - TEXT_LIMIT)
        data = f.read()
    return data.decode("utf-8", errors="replace"), truncated


def _file_preview(rel: str, p: Path) -> str:
    st = p.stat()
    ext = p.suffix.lower()
    raw = "/raw?path=" + quote(rel, safe="/")
    download = raw + "&download=1"
    meta = (
        f'<div class="meta"><span>大小 <b>{_human_size(st.st_size)}</b></span>'
        f'<span>修改 <b>{_mtime(st)}</b></span>'
        f'<span>类型 <b>{html.escape(ext or "无扩展名")}</b></span></div>'
    )
    if ext in {".mp4", ".webm", ".mov", ".m4v"}:
        preview = (
            f'<video controls playsinline preload="metadata" '
            f'src="{html.escape(raw, quote=True)}">'
            '你的浏览器不支持视频播放，请使用下载链接。</video>'
        )
    elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        preview = f'<img src="{html.escape(raw, quote=True)}" alt="预览"/>'
    elif ext in TEXT_EXTS:
        text, truncated = _tail_text(p)
        note = ""
        if truncated:
            note = (
                f'<p class="note">文件较大，仅显示最后 {_human_size(TEXT_LIMIT)}；'
                f'完整内容请下载或打开原始文件。</p>'
            )
        preview = note + f"<pre>{html.escape(text)}</pre>"
    else:
        preview = (
            '<p class="note">该类型不内嵌预览，可使用下面的下载/原始文件链接。</p>'
        )
    refresh = "/files?path=" + quote(rel, safe="/")
    actions = (
        f'<div class="actions"><a class="btn" href="{html.escape(download, quote=True)}">'
        '下载文件</a>'
        f'<a class="btn" href="{html.escape(raw, quote=True)}" target="_blank">原始文件</a>'
        f'<a class="btn" href="{html.escape(refresh, quote=True)}">刷新预览</a>'
        '</div>'
    )
    return meta + preview + actions


def _files_page(rel: str, p: Path) -> str:
    rel = rel.strip("/")
    parts = [x for x in rel.split("/") if x]
    crumbs = '<a href="/files">outputs</a>'
    for i, part in enumerate(parts):
        url = "/files?path=" + quote("/".join(parts[: i + 1]), safe="/")
        crumbs += f' <span class="sep">/</span> <a href="{html.escape(url, quote=True)}">{html.escape(part)}</a>'

    if p.is_dir():
        try:
            children = list(p.iterdir())
        except OSError:
            children = []
        children.sort(key=lambda c: (not c.is_dir(), c.name.lower()))
        rows = ""
        if parts:
            parent = "/files?path=" + quote("/".join(parts[:-1]), safe="/")
            rows += (
                f'<a class="entry" href="{html.escape(parent, quote=True)}">'
                '<span class="ico">↩</span><span class="name">上一级</span>'
                '<span class="size"></span><span class="time"></span></a>'
            )
        for child in children:
            try:
                real = child.resolve()
                real.relative_to(OUTPUTS_ROOT)
                st = child.stat()
            except (OSError, ValueError):
                continue
            child_rel = str(real.relative_to(OUTPUTS_ROOT)).replace(os.sep, "/")
            is_dir = real.is_dir()
            href = "/files?path=" + quote(child_rel, safe="/")
            icon = "📁" if is_dir else "📄"
            size = "—" if is_dir else _human_size(st.st_size)
            rows += (
                f'<a class="entry" href="{html.escape(href, quote=True)}">'
                f'<span class="ico">{icon}</span>'
                f'<span class="name">{html.escape(child.name)}</span>'
                f'<span class="size">{size}</span>'
                f'<span class="time">{_mtime(st)}</span></a>'
            )
        if not rows:
            rows = '<div class="empty">目录为空。</div>'
        content = (
            '<div class="card"><h2>文件列表</h2>'
            f'<div class="listing">{rows}</div></div>'
        )
    else:
        content = _file_preview(rel, p)

    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/>'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>'
        '<title>RM · outputs 文件</title>' + FILES_CSS + '</head><body><div id="app">'
        '<header><h1>RM · outputs 文件</h1><span class="pill">只读浏览</span>'
        '<div class="links"><a class="pill" href="/">片段观察窗</a>'
        '<a class="pill" href="/stats.html">统计</a></div></header>'
        f'<div class="crumbs">{crumbs}</div>{content}</div></body></html>'
    )


def _query_path(handler: BaseHTTPRequestHandler) -> str:
    qs = parse_qs(urlparse(handler.path).query)
    return (qs.get("path") or [""])[0]


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str):
        if not path.exists():
            self.send_error(404, f"{path.name} not found")
            return
        self._send(path.read_bytes(), ctype)

    def _raw_file(self, path: Path, head: bool = False, download: bool = False):
        if not path.is_file():
            self.send_error(404, "file not found")
            return
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ctype.startswith("text/"):
            ctype += "; charset=utf-8"
        range_header = self.headers.get("Range", "")
        start, end = 0, max(size - 1, 0)
        status = 200
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
            if not m:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            left, right = m.group(1), m.group(2)
            if left == "":
                length = int(right or 0)
                start = max(size - length, 0)
            else:
                start = int(left)
            if right != "":
                end = min(int(right), size - 1)
            if size == 0 or start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206
        length = end - start + 1 if size else 0
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            fname = quote(path.name)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{fname}")
        self.end_headers()
        if head:
            return
        try:
            with path.open("rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(256 * 1024, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._file(HTML_PATH, "text/html; charset=utf-8")
        elif path in ("/stats", "/stats.html"):
            self._file(STATS_HTML_PATH, "text/html; charset=utf-8")
        elif path == "/chart.umd.min.js":
            self._file(CHART_PATH, "application/javascript; charset=utf-8")
        elif path == "/rm_live.json":
            if not JSON_PATH.exists():
                self._send(b"{}", "application/json; charset=utf-8")
                return
            self._send(JSON_PATH.read_bytes(), "application/json; charset=utf-8")
        elif path == "/rm_stats.json":
            body = json.dumps(stats_payload(), ensure_ascii=False)
            self._send(body.encode("utf-8"), "application/json; charset=utf-8")
        elif path == "/status.json":
            st = JSON_PATH.stat() if JSON_PATH.exists() else None
            # Force the stats cache to consume the latest JSON even when the
            # stats tab is not open; otherwise a new run leaves old history.
            history = update_history()
            live_iter = int(history[-1].get("iter", -1)) if history else -1
            body = {
                "training_processes": training_running(),
                "json_exists": st is not None,
                "json_size": st.st_size if st else 0,
                "json_mtime": st.st_mtime if st else 0,
                "json_age_s": round(time.time() - st.st_mtime, 1) if st else None,
                "live_iter": live_iter,
                "history_len": len(history),
                "now": time.time(),
            }
            self._send(
                json.dumps(body, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )
        elif path in ("/files", "/files/"):
            rel = _query_path(self)
            p = _safe_output_path(rel)
            if p is None:
                self.send_error(403, "path outside outputs")
                return
            if not p.exists():
                self.send_error(404, "path not found")
                return
            self._send(_files_page(rel, p).encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/raw":
            rel = _query_path(self)
            p = _safe_output_path(rel)
            if p is None:
                self.send_error(403, "path outside outputs")
                return
            qs = parse_qs(urlparse(self.path).query)
            download = (qs.get("download") or ["0"])[0] in ("1", "true", "yes")
            self._raw_file(p, head=False, download=download)
        else:
            self.send_error(404)

    def do_HEAD(self):
        path = urlparse(self.path).path
        if path == "/raw":
            p = _safe_output_path(_query_path(self))
            if p is None:
                self.send_error(403, "path outside outputs")
                return
            self._raw_file(p, head=True)
            return
        self.send_error(405)

    def log_message(self, *args):
        pass


def _poll_loop():
    while True:
        try:
            update_history()
        except Exception:
            pass
        time.sleep(2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8768)
    p.add_argument("--bind", default="0.0.0.0")
    args = p.parse_args()
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    update_history()
    threading.Thread(target=_poll_loop, daemon=True).start()
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"RM liveview: http://localhost:{args.port}/  (stats: /stats.html)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
