# backend/server.py

import os, json, uuid, tempfile, shutil, textwrap, subprocess, contextlib, re, asyncio, threading, time
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import httpx

import numpy as np
import soundfile as sf

# --- Groq Chat Completions (native SDK) ---
from groq import Groq

from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model

# ---------- config ----------
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001") 
RENDERS_DIR = os.path.abspath("./renders")
SAMPLE_RATE = 24000  # Kokoro default
MANIM_TIMEOUT_SEC = int(os.getenv("MANIM_TIMEOUT_SEC", "120"))
DEFAULT_TARGET_DURATION_S = float(os.getenv("TARGET_DURATION_S", "30"))  
DEFAULT_WHISPER_REPO = os.getenv("WHISPER_REPO", None)

os.makedirs(RENDERS_DIR, exist_ok=True)

# ---------- FastAPI & static mount ----------
app = FastAPI(title="Mythos Render Server")
app.mount("/assets", StaticFiles(directory=RENDERS_DIR), name="assets")

# ---------- schema ----------
class ScriptItem(BaseModel):
    time: str = Field(..., description="HH:MM:SS or HH:MM:SS.mmm")
    text: str
    model_config = {"extra": "forbid"}

class ManimItem(BaseModel):
    time: str = Field(..., description="HH:MM:SS")
    duration_ms: int
    manim_code: str
    output_asset_id: str
    model_config = {"extra": "forbid"}

class Plan(BaseModel):
    script: List[ScriptItem]
    manim: List[ManimItem]
    model_config = {"extra": "forbid"}

class RenderRequest(BaseModel):
    prompt: str
    voice: str = "af_heart"
    lang_code: str = "a"     # 'a' American, 'b' British, 'j' Japanese, 'z' Mandarin
    speed: float = 1.0       # 0.5–2.0
    reasoning_effort: Optional[str] = "high"  

def hhmmss_to_samples(tc: str) -> int:
    h, m, s = tc.split(":")
    sec = float(s)
    total = (int(h) * 60 + int(m)) * 60 + sec
    return int(round(total * SAMPLE_RATE))

# ---------- MCP Client Functions ----------
async def call_mcp_tool(tool_name: str, params: dict) -> dict:
    """Call MCP server tool via HTTP"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MCP_SERVER_URL}/call_tool",
                json={"tool": tool_name, "params": params},
                timeout=30.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"MCP call failed: {response.status_code} {response.text}")
                return {"ok": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        print(f"MCP call exception: {e}")
        return {"ok": False, "error": str(e)}

# ---------- Groq plan generation (NO robot tools in initial call) ----------
def _system_instructions() -> str:
    return """
You are a storytelling AI director.

Produce a ~30 second spoken script (Hook → Build → Payoff), split into short timestamped lines (≤1 sentence each).
Use humor, emotion, conflict, and resolution in your storytelling.

Also produce ONE continuous Manim scene that VISUALLY explains the concept with animated diagrams according to the script.

Timing & pacing:
- Script timestamps may include milliseconds (HH:MM:SS.mmm).
- Use self.wait(...) so the Manim scene duration matches the script (~30s).

Manim requirements (Manim Community v0.19):
- Include `from manim import *`
- Define exactly one `class AutoScene(Scene):` with `def construct(self):`
- Use at least THREE non-text visuals (NumberPlane/Axes, Arrow/Vector, Line, Dot/Circle/Rectangle, Brace)
  and real animations (Create, Transform, MoveAlongPath, Rotate, Indicate, FadeIn/FadeOut, Write).
- Minimal labels only; no paragraphs. Self-contained; no external assets.
- Do NOT use set_stroke(dash_length=...) or set_style(stroke_dasharray=...). If you need dashes, use DashedVMobject.
""".strip()

async def get_plan_from_groq(prompt: str, reasoning_effort: Optional[str] = None) -> Plan:
    """Generate content plan without robot actions"""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    client = Groq(api_key=GROQ_API_KEY)

    user_message = _system_instructions() + "\n\n---\nUSER PROMPT:\n" + prompt.strip()

    kwargs = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": user_message}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "script_and_manim",
                "schema": Plan.model_json_schema(),
            }
        },
        "temperature": 0.6,
        "top_p": 0.95,
        "max_completion_tokens": 4096,
        "stream": False,
        "include_reasoning": False,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    completion = client.chat.completions.create(**kwargs)
    
    raw_text = (completion.choices[0].message.content or "").strip()
    raw_obj = json.loads(raw_text or "{}")
    plan = Plan.model_validate(raw_obj)
    
    return plan

# ---------- Robot action synchronization ----------
async def execute_synchronized_robot_actions(timeline: list[dict]) -> list[dict]:
    """Execute robot actions synchronized with audio timeline"""
    robot_actions = []
    
    try:
        result = await call_mcp_tool("move_pose_tool", {"name": "presenting", "duration": 1.0})
        robot_actions.append({
            "tool_name": "move_pose_tool",
            "params": {"name": "presenting", "duration": 1.0},
            "result": result,
            "timestamp": 0.0
        })
        
        # Start the synchronized execution in background
        threading.Thread(
            target=_sync_robot_with_timeline, 
            args=(timeline,), 
            daemon=True
        ).start()
        
        for item in timeline:
            duration = item.get("duration_s", 0.0)
            if duration > 0:
                robot_actions.append({
                    "tool_name": "talking_tool",
                    "params": {"seconds": duration},
                    "result": {"ok": True, "scheduled": True},
                    "timestamp": item.get("start_s", 0.0)
                })
        
    except Exception as e:
        print(f"Robot sync setup failed: {e}")
        robot_actions.append({
            "tool_name": "error",
            "params": {},
            "result": {"ok": False, "error": str(e)},
            "timestamp": 0.0
        })
    
    return robot_actions

def _sync_robot_with_timeline(timeline: list[dict]):
    """Background thread to sync robot actions with audio playback"""
    start_time = time.time()
    
    for item in timeline:
        try:
            start_s = item.get("start_s", 0.0)
            duration_s = item.get("duration_s", 0.0)
            
            # Wait until it's time for this line
            elapsed = time.time() - start_time
            if start_s > elapsed:
                time.sleep(start_s - elapsed)
            
            # Execute talking action for this line's duration
            if duration_s > 0:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    call_mcp_tool("talking_tool", {"seconds": duration_s})
                )
                loop.close()
                
                print(f"Robot talked for {duration_s}s: {result.get('ok', False)}")
                
        except Exception as e:
            print(f"Robot sync error for line: {e}")
            continue

# ---------- Kokoro caching ----------
@lru_cache(maxsize=4)
def get_kokoro(lang_code: str) -> KokoroPipeline:
    model_id = "prince-canuma/Kokoro-82M"
    model = load_model(model_id)
    return KokoroPipeline(lang_code=lang_code, model=model, repo_id=model_id)

# ---------- tiny DSP helpers ----------
def fade_edges(x: np.ndarray, ms: float = 8.0) -> np.ndarray:
    n = max(1, int(SAMPLE_RATE * ms / 1000.0))
    if len(x) >= 2 * n:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x

def fade_out(x: np.ndarray, ms: float) -> np.ndarray:
    n = max(1, int(SAMPLE_RATE * ms / 1000.0))
    if n > 0 and len(x) > n:
        ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
        x[-n:] *= ramp
    return x

# ---------- TTS (mlx-audio Kokoro) ----------
def render_tts(
    plan: Plan,
    out_path: str,
    voice: str,
    lang_code: str,
    speed: float,
) -> tuple[str, list[dict]]:
    """
    Generate one mixed WAV with timeline info
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pipe = get_kokoro(lang_code)

    SAFETY_PAD_SAMPLES = int(0.02 * SAMPLE_RATE)  # 20 ms between lines
    placements: List[tuple[int, np.ndarray]] = []
    timeline: list[dict] = []

    cursor = 0
    for idx, line in enumerate(plan.script):
        chunks = []
        for _, _, audio in pipe(line.text.strip(), voice=voice, speed=speed, split_pattern=None):
            chunks.append(np.asarray(audio[0], dtype=np.float32))  # mono [0]
        audio_arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        audio_arr = fade_edges(audio_arr, ms=8.0)

        desired = hhmmss_to_samples(line.time)
        start = max(desired, cursor)
        placements.append((start, audio_arr))
        cursor = start + len(audio_arr) + SAFETY_PAD_SAMPLES
        timeline.append({
            "index": idx,
            "text": line.text,
            "start_s": float(start) / float(SAMPLE_RATE),
            "duration_s": float(len(audio_arr)) / float(SAMPLE_RATE),
        })

    final_len = cursor if placements else 0
    mix = np.zeros(final_len, dtype=np.float32)
    for start_idx, arr in placements:
        end_idx = start_idx + len(arr)
        if end_idx > len(mix):
            mix = np.pad(mix, (0, end_idx - len(mix)))
        mix[start_idx:end_idx] += arr

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0:
        target = 0.98
        if peak > target:
            mix = (mix / peak) * target

    sf.write(out_path, mix, SAMPLE_RATE)
    return out_path, timeline

# ---------- Manim prep/validation (minimal sanitizer) ----------
_DASH_ARG_RE = re.compile(r"""
    (?:,\s*)?
    (?:dash_length|stroke_dasharray)
    \s*=\s*
    (?:\[[^\]]*\]|[^,\)\s]+)
    (?=\s*(?:,|\)))
""", re.VERBOSE)

def _strip_unsupported_dashing_kwargs(code: str) -> str:
    lines = []
    for line in code.splitlines():
        if ".set_stroke(" in line or ".set_style(" in line:
            line = _DASH_ARG_RE.sub("", line)
        lines.append(line)
    return "\n".join(lines)

def _wrap_in_scene_if_needed(code: str) -> str:
    if "class AutoScene" in code:
        if "from manim import" not in code:
            code = "from manim import *\n\n" + code
        return code
    body = textwrap.indent(code.strip(), "        ")
    return (
        "from manim import *\n\n"
        "class AutoScene(Scene):\n"
        "    def construct(self):\n"
        f"{body}\n"
    )

def _compile_or_repair(src: str) -> str:
    try:
        compile(src, "<manim_src>", "exec")
        return src
    except SyntaxError as e:
        if "unexpected character after line continuation character" in str(e) or "\\n" in src:
            repaired = src.replace("\\n", "\n")
            compile(repaired, "<manim_src_repaired>", "exec")
            return repaired
        raise

def safe_prepare_manim_code(raw_code: str) -> str:
    sanitized = _strip_unsupported_dashing_kwargs(raw_code)
    src = _wrap_in_scene_if_needed(sanitized)
    src = _compile_or_repair(src)
    return src

# ---------- Manim render ----------
def write_manim_file(code: str) -> str:
    prepared = safe_prepare_manim_code(code)
    fd, path = tempfile.mkstemp(suffix=".py", prefix="manim_scene_")
    with os.fdopen(fd, "w") as f:
        f.write(prepared)
    return path

def render_manim_video(manim_code: str, output_basename: str, renders_dir: str) -> str:
    scene_file = None
    tmp_media = tempfile.mkdtemp(prefix="manim_media_")
    try:
        scene_file = write_manim_file(manim_code)

        cmd = [
            "manim", scene_file, "AutoScene",
            "-qk",
            "-o", output_basename,
            "--media_dir", tmp_media
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=MANIM_TIMEOUT_SEC
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"Manim timed out after {MANIM_TIMEOUT_SEC}s\n"
                f"partial stdout:\n{e.stdout or ''}\n\npartial stderr:\n{e.stderr or ''}"
            )

        generated = None
        for dirpath, _, filenames in os.walk(tmp_media):
            for fn in filenames:
                if fn == f"{output_basename}.mp4":
                    generated = os.path.join(dirpath, fn)
                    break
            if generated:
                break

        if not generated:
            raise RuntimeError(
                "Manim did not produce the expected mp4.\n"
                f"Command stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )

        os.makedirs(renders_dir, exist_ok=True)
        final_path = os.path.join(renders_dir, f"{output_basename}.mp4")
        shutil.move(generated, final_path)
        return final_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Manim failed with code {e.returncode}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        )
    finally:
        with contextlib.suppress(Exception):
            if scene_file:
                os.remove(scene_file)
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_media, ignore_errors=True)

# ---------- whisper helpers ----------
ALLOWED_AUDIO_EXACT = {
    "application/octet-stream",
    "video/webm",
}
ALLOWED_AUDIO_PREFIXES = ("audio/",)

def _lazy_import_whisper():
    try:
        import mlx_whisper  # type: ignore
        return mlx_whisper
    except Exception as e:
        raise RuntimeError(
            "mlx-whisper is not installed. Run `pip install mlx-whisper` "
            "and ensure ffmpeg is available (e.g., `brew install ffmpeg`)."
        ) from e

# ---------- endpoints ----------
@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    whisper_repo: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    word_timestamps: bool = Form(False),
    temperature: float = Form(0.0),
):
    job_id = str(uuid.uuid4())[:8]

    ctype = (file.content_type or "").lower()
    if not (any(ctype.startswith(p) for p in ALLOWED_AUDIO_PREFIXES) or ctype in ALLOWED_AUDIO_EXACT or ctype == ""):
        return JSONResponse({"error": "unsupported_content_type", "content_type": ctype}, status_code=400)

    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    fd, tmp_audio = tempfile.mkstemp(suffix=suffix, prefix="upload_audio_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())

        mlx_whisper = _lazy_import_whisper()

        kwargs = {"word_timestamps": bool(word_timestamps), "temperature": float(temperature)}
        repo_choice = whisper_repo or DEFAULT_WHISPER_REPO
        if repo_choice:
            kwargs["path_or_hf_repo"] = repo_choice
        if language:
            kwargs["language"] = language

        used_repo = repo_choice or "default(tiny)"
        try:
            out = mlx_whisper.transcribe(tmp_audio, **kwargs)
        except Exception as e_first:
            if repo_choice:
                kwargs.pop("path_or_hf_repo", None)
                try:
                    out = mlx_whisper.transcribe(tmp_audio, **kwargs)
                    used_repo = "default(tiny)"
                except Exception:
                    return JSONResponse({"error": f"whisper_failed: {e_first}"}, status_code=500)
            else:
                return JSONResponse({"error": f"whisper_failed: {e_first}"}, status_code=500)

        text = (out.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "empty_transcript"}, status_code=400)

        resp = {
            "job_id": job_id,
            "model_used": used_repo,
            "language": out.get("language"),
            "text": text,
        }
        if word_timestamps:
            resp["segments"] = out.get("segments", [])

        return JSONResponse(resp, status_code=200)

    finally:
        with contextlib.suppress(Exception):
            os.remove(tmp_audio)

@app.post("/render")
async def render_endpoint(req: RenderRequest):
    job_id = str(uuid.uuid4())[:8]

    # --- Step 1: Generate content plan (no robot actions yet) ---
    try:
        plan = await get_plan_from_groq(req.prompt, reasoning_effort=req.reasoning_effort)
    except Exception as e:
        return JSONResponse({"error": f"planning_failed: {e}"}, status_code=500)

    # --- Step 2: Generate TTS with timeline ---
    audio_path = os.path.join(RENDERS_DIR, f"{job_id}_voice.wav")
    try:
        audio_path, speech_timeline = render_tts(
            plan, audio_path, voice=req.voice, lang_code=req.lang_code, speed=req.speed
        )
    except Exception as e:
        return JSONResponse({"error": f"tts_failed: {e}"}, status_code=500)

    # --- Step 3: Setup synchronized robot actions ---
    robot_actions = await execute_synchronized_robot_actions(speech_timeline)

    # --- Step 4: Generate Manim video ---
    video_basename = f"{job_id}_video"
    video_url: Optional[str] = None
    video_status = "absent"
    video_error: Optional[str] = None
    try:
        if plan.manim:
            manim_code = plan.manim[0].manim_code
            video_path = render_manim_video(
                manim_code,
                output_basename=video_basename,
                renders_dir=RENDERS_DIR
            )
            video_url = f"/assets/{os.path.basename(video_path)}"
            video_status = "ok"
        else:
            video_status = "absent"
            video_error = "plan_contains_no_manim_scene"
    except Exception as e:
        video_status = "error"
        video_error = str(e)

    audio_url = f"/assets/{os.path.basename(audio_path)}"

    return JSONResponse({
        "job_id": job_id,
        "plan": plan.model_dump(),
        "robot_actions": robot_actions,
        "audio_url": audio_url,
        "video_url": video_url,
        "video_status": video_status,
        "video_error": video_error
    }, status_code=200)