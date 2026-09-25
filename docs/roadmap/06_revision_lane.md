# 06 · Revision lane: learning that something *changed*

## Why
Everything consolidated today is protected as if it were permanently true.
Real knowledge gets revised ("policy X was superseded by Y on date T").  A
system that only prevents forgetting becomes rigid and eventually wrong.  A
system that just overwrites loses history.  The correct behaviour is
**contextual revision**: X is valid before T, Y after, and the shared concept
stays stable.

## Status
Placeholder: design and tasks only.

## Design
1. **Time/context-indexed facts.**  Extend `FactTask` with a context token:
   `[BOS, FACT, CTX_t, e1, e2, SEP, value]`.  `CTX_t` is a reserved token for
   an era or version.  Engram keys then include the context, so the new
   value lands in different rows and the old rows stay intact.
2. **Conflict detection at wake.**  When a new fact record arrives, query
   the current model (with no context token, or with the latest era).  If
   its confident answer differs from the new value, raise a *revision
   candidate* instead of treating the fact as new.
3. **Revision policy** (in `lifecycle/revision.py`):
   - evidence insufficient → quarantine (keep in Engram only, don't consolidate);
   - old value wrong everywhere → **replace**.  The audit's retention probes
     for that fact are *rewritten* to the new value, which is what makes the
     audit accept the change instead of flagging a regression;
   - old value valid under old context → **branch**.  Both are kept, keyed
     by context, and the default (no context) answer moves to the new value.
4. **Audit integration.**  `DualHemisphereBrain.audit` currently treats any
   drop as a regression.  Add an allow-list of `expected_changes` (fact id →
   new value) supplied by the revision policy.

## Tasks
- Synthetic benchmark: 200 facts learned, then 20% revised in two waves,
  some as replacements and some as branches.  Metrics: accuracy on current
  facts, accuracy on historical queries (with the old context token),
  unintended changes to unrelated facts.
- Implement the above, with tests for each policy branch.
- Compare three variants: (a) no revision lane, where the audit blocks every
  revision; (b) overwrite, with no protection; (c) the revision lane.

## Acceptance criteria
- A benchmark JSON and table in `results/revision/`.
- The revision lane beats (a) on current-fact accuracy and (b) on historical
  accuracy and unrelated-fact retention.
