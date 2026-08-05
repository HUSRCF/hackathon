# HipFire Qwen3.6-27B Agent and proxy validation — 2026-08-04

## Scope and provenance

This is a local functional and exploratory performance validation, not a new scientific benchmark.
No private sequence, ligand, assay, case, or knowledge-base content was sent to the model.

- Hardware: 2 × AMD Radeon Pro W7900 (`gfx1100`); the server used GPU 0 and GPU 1 remained idle.
- Server model: `qwen3.6:27b`.
- Target weight: `qwen3.6-27b.mq4`, 14,984,158,208 bytes.
- Target SHA-256: `86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc`.
- HipFire source revision: `e2f7dd1a47be23873d72090ac778e4f5197b7fe2`, clean checkout.
- ProtBind source revision: `e9f727597765c3ad29406d87aa5ce5ab78196871`, clean checkout before result logging.
- Observed server command: `hipfire serve --model qwen3.6:27b ... --kv-mode asym3 --kv-backend vmm --idle-timeout 0`.
- Direct endpoint: `http://127.0.0.1:11435/v1/chat/completions`.
- OpenAI proxy endpoint: `http://127.0.0.1:11436/v1/chat/completions`.

Raw artifacts are under `experiment-results/hipfire-qwen36-agent-20260804/`. That directory remains
ignored by the repository-wide artifact policy; the hashes below bind the local files.

## Raw result table

| Test | Repetitions / turns | Result | Principal measurements |
| --- | ---: | --- | --- |
| Direct text completion | 1 | PASS | HTTP 200; wall 0.266 s; HipFire TTFT 109.8 ms; decode 59.0 tok/s; exact requested text |
| Proxy text completion | 1 | PASS | HTTP 200; wall 0.281 s; HipFire TTFT 109.1 ms; decode 41.4 tok/s; exact requested text |
| Proxy forced tool call | 1 | PASS | HTTP 200; valid `get_status` call with `{}` arguments; `finish_reason=tool_calls`; DFlash active |
| Proxy SSE | 1 | PASS | Correct incremental chunks, `stop`, usage-only terminal chunk, and `[DONE]`; TTFT 110.6 ms |
| HipFire serve battery | 5 turns | PARTIAL | 4 natural stops; reasoning answer hit the intentionally low 192-token cap; no empty or attractor output; mean decode 66.9 tok/s |
| HipFire serve chain | 5 turns | PASS | 5 natural stops; no empty/runaway/attractor output; answers manually checked; mean decode 51.2 tok/s |
| ProtBind minimal routed `doctor` | 3 | PASS | 3/3 exact one-tool success; 2 exposed tools; 454 schema bytes; 4,695 mean prompt tokens; 9.425 ± 0.292 s |
| ProtBind full ToolSpec `doctor` | 3 | PASS | 3/3 exact one-tool success; 25 exposed tools; 9,794 schema bytes; 10,413.7 mean prompt tokens; 26.054 ± 9.052 s |

The routed/full raw comparison is bound by
`doctor-routing-ab-summary.json` SHA-256
`799bd3a8dc6dfb78c09559eaec43960bb8f93574ab4bbb7718ecd851679e5a2e`.

## Key findings

1. **Observation:** Both the direct endpoint and the OpenAI proxy produced valid text, while the
   proxy also preserved structured tool calls and complete SSE termination semantics.
   **Interpretation:** Port 11436 is usable by ordinary OpenAI-compatible clients even though it
   does not implement `GET /v1/models`; ProtBind's evidence-bound Agent should continue to use the
   direct loopback backend because its health/process checks require HipFire's native endpoints.
   **Implication:** The proxy is a compatible UI/client surface, not a substitute for direct
   benchmark provenance. **Next step:** add a bounded concurrent-request scheduler test.

2. **Observation:** Three byte-identical routed Agent runs called only `doctor`, succeeded, exposed
   two tools, and returned byte-identical final answers. The full ToolSpec also succeeded 3/3 but
   exposed 25 tools. **Interpretation:** Qwen3.6-27B can reliably emit ProtBind's structured tool
   syntax, and deterministic routing materially reduces context size. **Implication:** tool routing
   reduced exposed tools by 92%, schema bytes by 95.4%, and prompt tokens by about 54.9% for this
   workload. **Next step:** run a counterbalanced routed/full design after explicitly controlling
   the server prefix cache.

3. **Observation:** Routed elapsed time was 9.425 ± 0.292 s, whereas full ToolSpec time was
   26.054 ± 9.052 s. The full sequence dropped from 33.323 to 15.915 s as it warmed.
   **Interpretation:** the latency difference is directionally consistent with the exact token
   reduction, but native prefix/KV caching and order strongly confound the magnitude.
   **Implication:** do not claim a stable 2.76× Agent speedup from this run.
   **Next step:** alternate conditions across fresh server processes or expose cache counters in
   the client receipt.

4. **Observation:** The first broad Chinese `doctor` prompt succeeded operationally but its final
   summary incorrectly said that Vina was an experimental free energy. The fixed minimal prompt
   avoided that unsupported restatement. **Interpretation:** tool success alone does not establish
   evidence-faithful synthesis. **Implication:** a scientific-boundary semantic assertion must be
   part of the Agent acceptance suite. **Next step:** add deterministic negative assertions such as
   “must state that a Vina score is not an experimental binding free energy.”

5. **Observation:** The chain battery produced coherent manually inspected code, arithmetic,
   factual, prose, and instruction answers, with no attractor. Decode throughput varied from 23.7
   to 101.6 tok/s and DFlash acceptance `tau` from 1.05 to 8.58.
   **Interpretation:** the large genre dependence matches HipFire's documented speculative-decoding
   behavior. **Implication:** a single short prompt cannot support a general throughput claim.
   **Next step:** use a committed ProtBind prompt battery and report per-genre distributions.

## Artifact hashes

- `direct-text-smoke.json`: `e91e44cda709afda5e530eda6932be6b1bfcd9b297c6739d4a08286eeee45881`
- `proxy-text-smoke.json`: `db3098a15764a9a7a8ab4772e2df577295c2f0e47701ad69b19627e3784c4202`
- `proxy-toolcall-smoke.json`: `3f3db803900b0e120d7861c1ec9efc626d9f5eaef505da4b92e16055aafa08b0`
- `proxy-stream-smoke.sse`: `440577dc702ef7a19e3901ea698ed34572e8546c7ee9cce6da606eb223657fa1`
- `serve-battery.json`: `d6bfe25fafedf27e37d33736c1b0d2287aea861ceb6062d6da1619d93855a25f`
- `serve-chain.json`: `4d582a48362908ec8c2ef31daeed416e3eadcb64800c9a521940c688503ca57c`

## Evidence boundary

These results support local serving compatibility, proxy protocol compatibility, coherent general
generation on a small prompt battery, and reliable execution of one read-only ProtBind tool. They
do not validate the three-tool `case_status → knowledge_search → memory_write` workflow, approval
resume semantics, scientific docking quality, experimental binding, multi-user scalability, or a
formal routed/full latency speedup. A new public `REPORTED` fixture and a hash-admitted local
embedding model are still required before rerunning the formal Agent benchmark.
