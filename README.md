# context-rag-memory-skills

A framework-free, test-driven Agent Runtime that extends the Week 1 `agent-runtime-lab` with context assembly, hybrid retrieval, long-term memory, and progressive skill activation.

This repository keeps the core premise of Week 1: the model can propose text or tool calls, but deterministic runtime code owns state transitions, validation, side effects, trace events, and safety boundaries. Week 2 adds the context layer around that control loop.

## Project Status

This is an educational runtime lab and prototype, not a production agent platform.

It is designed to make agent mechanics visible and testable:

- deterministic runtime loop
- typed tool calls and validation
- traceable execution state
- context budget management
- BM25 and vector retrieval
- citation formatting
- prompt-injection trust boundaries
- SQLite-backed memory
- skill discovery and progressive disclosure

Requires Python 3.11+. Tested locally with Python 3.12.

## What Changed From Week 1

Week 1 focused on the runtime core:

- provider adapter boundary
- tool registry allowlist
- Pydantic tool validation
- retry policy
- async execution
- subprocess allowlist
- FastAPI run lifecycle

Week 2 keeps those pieces and adds:

- `ContextAssembler` for priority-based context retention and dropping
- `ContextItem` trust levels such as `system_trusted`, `user_supplied`, `retrieved_untrusted`, and `application_trusted`
- BM25 lexical retrieval
- deterministic fake vector retrieval
- Reciprocal Rank Fusion for hybrid retrieval
- citation map construction for retrieved evidence
- SQLite memory store with namespace isolation and upsert semantics
- memory write policy against persistent prompt injection
- `SkillLoader` and `SkillSelector`
- progressive skill activation that indexes only skill metadata until a skill is selected
- runtime integration that passes assembled context into the provider as a system message

## Architecture

```text
==================================================================================================
                         CONTEXT-RAG-MEMORY-SKILLS ARCHITECTURE
==================================================================================================

      [ User Input ]
            |
            v
   +--------------------------+
   |      AgentRuntime        |
   |  run_id / state / trace  |
   +------------+-------------+
                |
                | pre-model context preparation
                v
   +--------------------------+        +--------------------------+
   |      SkillSelector       |        | HybridRetrievalPipeline  |
   | metadata-only routing    |        | BM25 + vector + RRF      |
   | progressive disclosure   |        | citations + evidence     |
   +------------+-------------+        +------------+-------------+
                |                                   |
                v                                   v
   +--------------------------+        +--------------------------+
   |      SkillLoader         |        | Retrieved Evidence       |
   | load full SKILL.md only  |        | trust=retrieved_untrusted|
   | after activation         |        +--------------------------+
   +------------+-------------+
                |
                v
   +--------------------------+
   | SQLiteMemoryStore        |
   | namespace-scoped memory  |
   +------------+-------------+
                |
                v
   +--------------------------+
   |    ContextAssembler      |
   | priority budget + dedup  |
   | system instructions kept |
   +------------+-------------+
                |
                v
   +--------------------------+
   | Provider Input           |
   | system context message   |
   | + conversation history   |
   +------------+-------------+
                |
                v
   +--------------------------+
   | ToolRegistry / Tools     |
   | allowlist + validation   |
   +--------------------------+
```

## Repository Layout

```text
context-rag-memory-skills/
  pyproject.toml
  README.md
  LICENSE
  skills/
    rolling-backtest/
      SKILL.md
      references/checklist.md
    seasonal-diagnostics/
      SKILL.md
  sample_knowledge/
    time_series/
      arima_basics.md
      backtesting.md
  sample_data/
    series/monthly_sales.csv
  scripts/
    safe_describe_csv.py
    safe_series_summary.py
  src/
    agent_runtime/
      api.py
      config.py
      errors.py
      models.py
      tracing.py
      context/
        assembler.py
        budget.py
        dedup.py
        models.py
      memory/
        models.py
        policy.py
        sqlite_store.py
      providers/
        base.py
        fake_provider.py
      retrieval/
        bm25.py
        chunker.py
        citations.py
        embeddings.py
        hybrid.py
        models.py
        pipeline.py
        reranker.py
        tokenizer.py
        vector_index.py
      runtime/
        async_executor.py
        loop.py
        state.py
      skills/
        loader.py
        models.py
        selector.py
      tools/
        arithmetic.py
        base.py
        idempotency_writer.py
        registry.py
        script_runner.py
  tests/
```

## Quick Start

Install the project in editable mode:

```bash
pip install -e .
pip install pytest pytest-asyncio
```

Run the full test suite:

```bash
pytest -q
```

Expected result at the time of writing:

```text
47 passed
```

Run focused Week 2 tests:

```bash
pytest tests/test_context_integration.py tests/test_skill_loader.py tests/test_skill_activation.py -v
```

Run the FastAPI service:

```bash
uvicorn agent_runtime.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Core Concepts

### 1. Context Assembly

`ContextAssembler` converts raw context candidates into a bounded `ContextBundle`.

It applies:

- content deduplication
- priority sorting
- deterministic token estimation
- budget-based dropping
- hard retention for priority 100 system instructions

This gives the runtime a deterministic policy for what enters the model context.

### 2. Trust Levels

Context items carry explicit trust levels:

- `system_trusted`
- `application_trusted`
- `user_supplied`
- `retrieved_untrusted`

Retrieved evidence is passed to the model as data, not instructions. The runtime adds a system instruction that tells the provider to ignore any instruction-like text found in retrieved evidence.

### 3. Hybrid Retrieval

Retrieval uses two deterministic local routes:

- `BM25Retriever` for lexical matching
- `VectorIndex` with `FakeEmbeddingProvider` for stable semantic tests

`ReciprocalRankFusion` combines rankings without assuming comparable score scales. `CitationBuilder` formats evidence blocks and returns a citation map for trace auditing.

### 4. Progressive Skill Activation

Skills live under `skills/*/SKILL.md` with frontmatter metadata:

```yaml
---
name: rolling-backtest
description: Use this skill when evaluating forecasting designs...
allowed_tools: ["run_approved_script"]
---
```

Discovery reads only lightweight metadata. The full skill body is loaded only after `SkillSelector` chooses a matching skill.

This is intentionally different from ordinary RAG over file content. Skill selection is routing, not answer retrieval.

### 5. Long-Term Memory

`SQLiteMemoryStore` provides a minimal local memory layer:

- namespace isolation
- lookup by namespace/key
- upsert on `(namespace, key)`
- structured `MemoryRecord` models

`MemoryWritePolicy` rejects persistent prompt-injection-like content before it reaches storage.

### 6. Runtime Integration

Before calling the provider, `AgentRuntime` prepares:

- activated skill instructions
- retrieved evidence
- namespace memories
- core system safety instruction
- current message history

The assembled bundle is passed as a temporary system message plus the original conversation history. The temporary context message is not appended to persisted `state.messages`.

## Test Matrix

| Area | Coverage |
| --- | --- |
| Runtime loop | direct answer, tool call, max steps, validation errors |
| Tool registry | allowlist, duplicate registration, schema listing |
| Async executor | concurrency, timeout, idempotency |
| Script runner | approved scripts, path traversal, timeout |
| Retrieval | tokenizer, BM25, vector index, RRF, citations, insufficient evidence |
| Context | deduplication, priority ordering, budget dropping, system retention |
| Memory | namespace isolation, upsert, injection rejection |
| Skills | metadata discovery, activation, fuzzy selection, progressive disclosure |
| Integration | RAG + skill fusion, retrieved prompt-injection boundary |
| API | health check, run submission, polling, trace fetch, missing run boundary |

Run all tests:

```bash
pytest -v
```

Run one test file:

```bash
pytest tests/test_context_integration.py -v
```

Run one test:

```bash
pytest tests/test_skill_activation.py::test_fuzzy_intent_can_activate_without_skill_name -v
```

## Security Notice

This project demonstrates application-level safety controls:

- tool allowlists
- Pydantic validation
- structured runtime errors
- context trust levels
- retrieved-evidence instruction boundaries
- memory write filtering
- approved subprocess scripts
- path checks
- hard subprocess timeout

It is not a secure sandbox.

Do not run untrusted scripts through this project. Do not expose the FastAPI service directly to the public internet. Do not treat the subprocess runner as an isolation boundary. Production-grade execution of untrusted code requires operating-system-level isolation such as containers, cgroups, seccomp, gVisor, Firecracker, separate users, filesystem restrictions, CPU and memory quotas, and network egress controls.

## Known Limitations

- The vector index uses deterministic fake embeddings for tests, not production embeddings.
- SQLite memory is local and minimal.
- The API uses in-memory run tracking.
- Skill selection indexes metadata only; skill-local reference retrieval is future work.
- Context token estimation is approximate.
- The prompt-injection defense is an application-level boundary, not a formal security proof.
- Provider integration is represented by `FakeProvider`; real model adapters are future work.
- The subprocess runner is allowlisted but not OS-sandboxed.

## Future Work

Potential next steps:

- real embedding provider adapters
- persistent run store
- skill-scoped reference retrieval after activation
- provider adapters for OpenAI-compatible APIs and local models
- streaming responses
- richer memory ranking and decay policy
- context compaction and summarization
- MCP-compatible tool registration
- OS-level sandboxing for subprocess tools
- golden dataset evaluation for retrieval and skill activation

## Design Summary

The model is probabilistic. The runtime is deterministic.

This project demonstrates a compact control plane for wrapping model behavior in explicit software boundaries: typed messages, schema-validated tools, registry allowlists, bounded loops, structured errors, trace events, retrieval trust levels, memory policy, and progressive skill activation.

## License

MIT License. See [LICENSE](LICENSE).
