"""ATDEV_STEPLOG=<path>: per-engine-step CPU wall timings (scheduler.schedule, runner.execute_model, runner.sample_tokens)
as JSON lines {t, what, dur_ms, ntok, nreq}. Diagnostics only; default off."""
import json, os, time, logging

log = logging.getLogger("atdev_serve")


def install():
    path = os.environ.get("ATDEV_STEPLOG")
    if not path:
        return
    f = open(f"{path}.{os.getpid()}", "a", buffering=1)
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    st = {"ntok": 0, "nreq": 0}

    def wrap(cls, name, pre=None):
        orig = getattr(cls, name)

        def w(self, *a, **k):
            t0 = time.perf_counter()
            r = orig(self, *a, **k)
            t1 = time.perf_counter()
            if name == "schedule":
                st["ntok"] = r.total_num_scheduled_tokens
                st["nreq"] = len(r.num_scheduled_tokens)
            f.write(json.dumps(dict(t=t1, what=name, dur_ms=(t1 - t0) * 1e3, ntok=st["ntok"], nreq=st["nreq"])) + "\n")
            return r
        setattr(cls, name, w)
    wrap(Scheduler, "schedule")
    wrap(GPUModelRunner, "execute_model")
    wrap(GPUModelRunner, "sample_tokens")
    # frontend / input-path timings (API-server and engine-core side)
    import asyncio
    def wrap_any(cls, name, label):
        orig = getattr(cls, name)
        if asyncio.iscoroutinefunction(orig):
            async def w(self, *a, **k):
                t0 = time.perf_counter()
                f.write(json.dumps(dict(t=t0, what=label + ":enter", dur_ms=0, ntok=0, nreq=0)) + "\n")
                try:
                    return await orig(self, *a, **k)
                finally:
                    t1 = time.perf_counter()
                    f.write(json.dumps(dict(t=t1, what=label, dur_ms=(t1 - t0) * 1e3, ntok=0, nreq=0)) + "\n")
        else:
            def w(self, *a, **k):
                t0 = time.perf_counter()
                try:
                    return orig(self, *a, **k)
                finally:
                    t1 = time.perf_counter()
                    f.write(json.dumps(dict(t=t1, what=label, dur_ms=(t1 - t0) * 1e3, ntok=0, nreq=0)) + "\n")
        setattr(cls, name, w)
    try:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.engine.core import EngineCore
        from vllm.renderers.base import BaseRenderer
        wrap_any(OpenAIServingChat, "_create_chat_completion", "api_create")
        wrap_any(OpenAIServingChat, "render_chat_request", "api_render")
        wrap_any(AsyncLLM, "add_request", "api_add_request")
        wrap_any(BaseRenderer, "_tokenize_prompt", "tokenize")
        wrap_any(EngineCore, "add_request", "core_add_request")
    except Exception as e:  # pragma: no cover
        log.warning("atdev-serve: steplog frontend hooks failed: %s", e)
    log.warning("atdev-serve: ATDEV_STEPLOG -> %s.<pid>", path)
