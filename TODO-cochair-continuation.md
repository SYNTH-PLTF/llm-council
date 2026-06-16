# TODO: Co-chair continuation for the Chairman (handle finish_reason == "length")

Status: planned. Logged 2026-06-14 from a live resume-review run.

## Problem
For long structured outputs (the 8-step / 20+ row claim-audit resume review), the
Chairman model exhausts its output-token budget mid-document and returns
`finish_reason == "length"`, truncating before the final sections.

Observed: Opus-4.8 as chair on the 8-step prompt produced STEP 1-3 in full
(~3,734 words) and stopped before STEP 4-8 (claim-audit table, positioning
options, final verdict). Raising `max_output_tokens` alone does not solve it
(slower, more timeout risk, and still a hard ceiling).

## Idea (from user)
When the chair truncates, hand off to a cheaper "co-chair" model that continues
from where the chair stopped and writes ONLY the remaining sections, grounded in
the chair's partial output so it stays consistent. Cheap because it does the
mechanical tail, not the hard synthesis. (Validated manually this session with
DeepSeek-V4-Pro finishing STEP 4-8 from Opus's STEP 1-3.)

## Implementation sketch
- Config: add `council.chairman.co_chair_model` (cheap tier, e.g. DeepSeek-V4-Pro)
  and `council.chairman.continuation: { enabled: bool, max_rounds: int }`.
- Gateway already captures `finish_reason` in `ProviderResult` (`_extract` in
  gateway/client.py). Make the Chairman branch on it (currently it is ignored).
- In `Chairman.synthesize` (the plain-text / long-form path):
  1. Call the chair. If `finish_reason == "length"` (or required trailing
     section markers are absent), call the co-chair with a continuation prompt:
     original instruction + resume + "PARTIAL OUTPUT SO FAR (do not repeat;
     continue from the exact cutoff):" + chair_text.
  2. Loop up to `max_rounds` (bound it, e.g. 2) until `finish_reason != length`.
  3. `final_answer = chair_text + co_chair_continuation(s)` concatenated.
- Trace each continuation as its own span/stage; record which model finished
  (so observability shows chair vs co-chair contribution).
- Keep the structured (JSON `response_format`) chairman path separate; the
  continuation pattern is for long-form plain-text synthesis.

## Related changes already made this session (keep)
- `council/proposers.py`: an empty / whitespace-only proposer response now counts
  as a FAILURE (`council.proposer_empty`) instead of passing as `ok=True`, so
  quorum and graceful degradation work. (Fixes the earlier Kimi "empty but ok"
  case where the chair confabulated agreement.)
- For long-form reviews, run a plain-text chairman (no JSON straitjacket) and call
  Opus-4.8 with NO `temperature` (the Azure-hosted Anthropic `/anthropic`
  endpoint rejects `temperature` for Opus-4.8: "temperature is deprecated for
  this model").
- Frontier endpoints are reached via litellm `azure_ai/<model>` to each model's
  own host EXCEPT claude-opus-4-8, which is `anthropic/<model>` against
  `<host>/anthropic`.
