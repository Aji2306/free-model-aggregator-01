# Free Model Aggregator

An intelligent, robust AI model aggregator for the `free-model-aggregator-01` competition. It serves OpenAI Chat Completions (`POST /v1/chat/completions`) and Anthropic Messages (`POST /v1/messages`) APIs (in JSON and streaming), lists answered models (`GET /v1/models`), probes and registers with the trial environment, ranks available free-tier models by quality ratings, implements automatic multi-provider failover, serializes calls per provider, and speaks OpenAI, Anthropic, and Gemini upstream wires.

## Features
- **Quality-Based Routing:** Ranks free candidate models by quality ratings (`rating_milli`) to maximize score.
- **Robust Failover:** Automatically retries requests across alternative free models when a provider refuses (402, 429, 5xx).
- **Multi-Wire Support:** Translates client requests and upstream responses across OpenAI, Anthropic, and Gemini wire protocols.
- **Provider Serialization:** Manages concurrency per provider to respect trial constraints.
- **Deterministic & Zero-Config:** Runs on Node 22 with standard library only.
