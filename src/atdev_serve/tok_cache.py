"""ATDEV_TOK_CACHE=1: per-segment tokenization cache in the API-server renderer (frontend, exact).

Chat prompts are re-tokenized in full for every request of a conversation (about 10 ms for 4-7k tokens, serial in front of
the engine at C1).  Special tokens (<|im_start|>, <|im_end|>) are atomic and split the pre-tokenizer input, so
tokenize(text) == concat(tokenize(segment)) for segments delimited by them.  Long segments are cached by text.
Self-check: the first 16 prompts and every 256th afterwards are compared with the stock full tokenization; any
mismatch disables the cache permanently (logged).  Falls back to stock for offsets / non-plain encode kwargs.
"""
import logging
import re
from collections import OrderedDict

log = logging.getLogger("atdev_serve")
_SPLIT = re.compile(r"(<\|im_start\|>|<\|im_end\|>)")
_MIN_CACHE_CHARS = 48


def install():
    from vllm.renderers.base import BaseRenderer
    if getattr(BaseRenderer, "_atdev_tok_cache", False):
        return
    BaseRenderer._atdev_tok_cache = True
    orig = BaseRenderer._tokenize_prompt
    cache: OrderedDict = OrderedDict()
    st = {"n": 0, "bad": False, "hit": 0, "miss": 0}

    def seg_ids(tok, s):
        v = cache.get(s)
        if v is not None:
            cache.move_to_end(s)
            st["hit"] += 1
            return v
        v = tok(s, add_special_tokens=False)["input_ids"]
        st["miss"] += 1
        if len(s) >= _MIN_CACHE_CHARS:
            cache[s] = v
            if len(cache) > 4096:
                cache.popitem(last=False)
        return v

    def _tokenize_prompt(self, prompt, params):
        text = prompt.get("prompt")
        kwargs = params.get_encode_kwargs()
        if (st["bad"] or not isinstance(text, str) or self._wants_offsets(prompt, params)
                or kwargs.get("add_special_tokens", False) or len(text) < 2048):
            return orig(self, prompt, params)
        tok = self.get_tokenizer()
        ids = []
        for seg in _SPLIT.split(text):
            if not seg:
                continue
            if seg[0] == "<" and _SPLIT.fullmatch(seg):
                ids.extend(tok(seg, add_special_tokens=False)["input_ids"])
            else:
                ids.extend(seg_ids(tok, seg))
        ml = kwargs.get("max_length")
        if kwargs.get("truncation") and ml is not None:
            ids = ids[:ml]
        st["n"] += 1
        if st["n"] <= 16 or st["n"] % 256 == 0:
            ref = tok(text, **kwargs)["input_ids"]
            if list(ref) != ids:
                st["bad"] = True
                log.error("atdev-serve: ATDEV_TOK_CACHE mismatch vs stock tokenization -> cache disabled")
                return orig(self, prompt, params)
        return self._build_tokens_prompt(ids, prompt, offset_mapping=None)

    BaseRenderer._tokenize_prompt = _tokenize_prompt
    # a prefix-cache reset (benchmark cold start) also drops the tokenization cache, so "cold" never benefits from earlier passes
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        orig_reset = AsyncLLM.reset_prefix_cache

        async def reset_prefix_cache(self, *a, **k):
            cache.clear()
            return await orig_reset(self, *a, **k)
        AsyncLLM.reset_prefix_cache = reset_prefix_cache
    except Exception as e:  # pragma: no cover
        log.warning("atdev-serve: tok_cache reset hook not installed (%s)", e)
    log.warning("atdev-serve: ATDEV_TOK_CACHE enabled (segment tokenization cache)")
