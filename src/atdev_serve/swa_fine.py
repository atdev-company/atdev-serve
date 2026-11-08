"""ATDEV_SWA_FINE=1: fine-grained (hash-unit, 16 token) prefix-cache hits for sliding-window KV groups.

vLLM's SlidingWindowManager only supports hits at its physical block boundary (480 tokens for the
DFlash2 drafter group in the C1 stack) and thereby disables fine-grained hits for the whole hybrid
coordinator ("Disabling fine-grained prefix-cache hits ... SlidingWindowManager").  A cached block is
append-only, so the KV of its first t < block_size tokens stays valid for any request sharing those t
tokens; FullAttentionManager already exploits that by (a) registering the prompt-end partial block under
the hash at the prompt-end boundary and (b) probing interior hash boundaries.  Here the same two things
are added to SlidingWindowManager (window blocks before the boundary must be cached as usual):
  * cache_blocks: also register the prompt-end partial tail block (same helper as full attention),
  * find_longest_cache_hit: hit length h on any hash-unit boundary such that every block covering
    [h-W+1, h) is cached (last block looked up under the partial/fine hash when h is not block aligned).
The partial-hit copy-on-write path of SingleTypeKVCacheManager is generic and already used by FA/mamba.
Model files, drafter config and attention math are untouched; only which tokens are recomputed changes.
Falls back to the stock implementation when alignment >= block size, or when the EAGLE block drop is on.
"""
import logging

log = logging.getLogger("atdev_serve")


def fine_find_longest_cache_hit(orig):
    def find(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
             drop_eagle_block, alignment_tokens, dcp_world_size=1, pcp_world_size=1):
        bs = kv_cache_spec.block_size
        hbs = block_pool.hash_block_size
        fine = (alignment_tokens < bs and bs % alignment_tokens == 0 and alignment_tokens % hbs == 0)
        if not fine:
            return orig.__func__(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
                                 drop_eagle_block, alignment_tokens, dcp_world_size, pcp_world_size)
        if drop_eagle_block:  # not supported in fine mode: coarse hit (stock behaviour on the block-size view)
            return orig.__func__(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
                                 drop_eagle_block, bs, dcp_world_size, pcp_world_size)
        W = kv_cache_spec.sliding_window
        null = block_pool.null_block
        ng = len(kv_cache_group_ids)
        h = min(max_length, len(block_hashes) * hbs)
        h -= h % alignment_tokens
        while h > 0:
            last_blk = (h - 1) // bs
            first_blk = max(0, h - W + 1) // bs
            found = []
            for blk in range(last_blk, first_blk - 1, -1):
                t = h if blk == last_blk else (blk + 1) * bs
                cb = block_pool.get_cached_block(block_hashes[t // hbs - 1], kv_cache_group_ids)
                if not cb:
                    break
                found.append(cb)
            else:
                computed = tuple([null] * first_blk + [cb[g] for cb in reversed(found)] for g in range(ng))
                return computed, h
            h -= alignment_tokens
        return tuple([] for _ in range(ng)), 0

    return find


def install():
    from vllm.v1.core import single_type_kv_cache_manager as m
    S, F = m.SlidingWindowManager, m.FullAttentionManager
    if getattr(S, "_atdev_swa_fine", False):
        return
    S._atdev_swa_fine = True
    S.supports_fine_grained_hash_lookup = True
    S._cache_partial_tail_block = F._cache_partial_tail_block
    orig_cache = S.cache_blocks
    orig_find = S.__dict__["find_longest_cache_hit"]

    def cache_blocks(self, request, num_tokens, retention_interval=None, *, replay_boundaries):
        orig_cache(self, request, num_tokens, retention_interval=retention_interval, replay_boundaries=replay_boundaries)
        if self.block_size != self.block_pool.hash_block_size and self.enable_caching:
            self._cache_partial_tail_block(request, num_tokens)

    S.cache_blocks = cache_blocks
    S.find_longest_cache_hit = classmethod(fine_find_longest_cache_hit(orig_find))
    log.warning("atdev-serve: ATDEV_SWA_FINE enabled (fine-grained sliding-window prefix hits)")
