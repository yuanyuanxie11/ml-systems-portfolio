# Recommendation Service API (v1)

Public REST contract for the recommendation and segmentation service described in
[`ARCHITECTURE.md`](../ARCHITECTURE.md). This document is the source of truth for
client integrations; the OpenAPI 3.1 spec generated from it lives at
`docs/openapi.yaml` (TODO — generated from this doc by the CI job
`docs/scripts/md_to_openapi.py`).

If anything in this document conflicts with the architecture doc, the architecture
doc wins and this file is the bug.

---

## 1. Conventions

### 1.1 Base URL

```
https://api.reco.example.com/v1
```

`v1` is encoded in the path. A new major version (`/v2`) ships only when a
breaking change is made; minor and patch versions are always backwards
compatible and surfaced in the response header `X-Reco-Schema-Version`
(semver, e.g. `1.4.2`).

### 1.2 Transport

* HTTPS only. Plain HTTP is rejected at the edge with `403`.
* TLS 1.2+ (TLS 1.3 preferred). Weak ciphers disabled at API Gateway.
* Request and response bodies are JSON (`application/json; charset=utf-8`).
* Numeric IDs are strings, not integers, so callers in JavaScript don't lose
  precision on 64-bit values.

### 1.3 Authentication

All endpoints require a bearer JWT issued by the platform's Cognito user pool:

```
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
```

The token must contain:

| Claim | Required | Notes |
|---|---|---|
| `sub` | yes | Cognito subject; not used for personalisation |
| `iss` | yes | Must equal `https://cognito-idp.us-east-1.amazonaws.com/<pool-id>` |
| `aud` | yes | Must equal the API client ID |
| `exp` | yes | Tokens older than 15 min are rejected |
| `scope` | yes | Must include `reco:read` for recommendations and segments, `reco:write` for feedback |
| `tenant_id` | yes | Tenant scoping; recommendations are namespaced by tenant |

Service-to-service callers (the in-app recommendation widget, the email
batch job) use the OAuth 2.0 client-credentials flow and do not have a real
end user `sub`. For those callers the API trusts the `user_id` field in the
request body, but only if `scope` includes `reco:impersonate`.

### 1.4 Rate limits

Enforced at the API Gateway usage-plan level:

| Caller | Rate (RPS) | Burst | Daily quota |
|---|---:|---:|---:|
| End-user JWT | 20 | 50 | 200,000 |
| Server-side client (impersonate) | 1,000 | 2,000 | unlimited |
| Internal A/B harness | 5,000 | 10,000 | unlimited |

When the limit is exceeded the API returns `429 Too Many Requests` with a
`Retry-After` header (seconds) and `X-RateLimit-Remaining: 0`.

### 1.5 Idempotency

`POST /v1/feedback` accepts an `Idempotency-Key` header. Repeated calls with
the same key within a 24-hour window are deduplicated and return the
original response — important because the in-app SDK retries on network
errors.

### 1.6 Latency SLOs

| Endpoint | p50 | p95 | p99 | Timeout |
|---|---:|---:|---:|---:|
| `POST /v1/recommendations` | 60 ms | 150 ms | 300 ms | 1,000 ms |
| `GET  /v1/users/{user_id}/segments` | 25 ms | 80 ms | 150 ms | 500 ms |
| `POST /v1/feedback` | 15 ms | 40 ms | 100 ms | 500 ms |
| `GET  /v1/health` | 5 ms | 20 ms | 50 ms | 200 ms |

p95 is the acceptance criterion that gates a production rollout. Numbers
above are end-to-end including TLS handshake reuse, measured at API Gateway,
not Lambda init.

The 150 ms p95 budget for `/recommendations` decomposes roughly as:
20 ms API Gateway + auth, 5 ms DynamoDB cache lookup (cache hit path), 80 ms
SageMaker endpoint reranking (cache miss path), 5 ms response shaping. The
cache-miss share is bounded to ~15% of traffic by the offline pre-computation
job, which keeps the average path well under p95.

### 1.7 Errors

Errors share a single envelope:

```json
{
  "error": {
    "code": "user_not_found",
    "message": "User u-12345 not found in cohort.",
    "request_id": "req-9c7e1a40-...",
    "docs_url": "https://docs.reco.example.com/errors/user_not_found"
  }
}
```

Standard codes:

| HTTP | `error.code` | When |
|---:|---|---|
| 400 | `validation_error` | Request body fails JSON-schema validation |
| 401 | `unauthorized` | Missing, expired, or invalid JWT |
| 403 | `forbidden` | Token valid but scope or tenant disallows the operation |
| 404 | `user_not_found` | `user_id` not in the cohort and `cold_start_policy` is `error` |
| 409 | `idempotency_conflict` | Same `Idempotency-Key` reused with a different body |
| 422 | `invalid_argument` | Body parses but values are out of range (e.g. `n > 200`) |
| 429 | `rate_limited` | Usage-plan limit hit |
| 500 | `internal_error` | Bug in the service; logged as a SEV-3 |
| 503 | `serving_unavailable` | SageMaker endpoint is unhealthy or DynamoDB throttled |
| 504 | `timeout` | Upstream timed out before responding |

Clients should treat `503` and `504` as retryable with exponential backoff
(starting 100 ms, jittered) and a max of 3 retries. `5xx` responses also
include the upstream component in `error.message`, which is useful for
on-call but is not part of the contract.

### 1.8 Tracing

Every request returns an `X-Request-Id` header. The same value is propagated
to the SageMaker endpoint and emitted in CloudWatch logs and X-Ray traces,
so a customer-reported bug can be reconstructed end-to-end from a single
ID.

---

## 2. Endpoints

### 2.1 `POST /v1/recommendations`

Return a ranked list of items for a user. The same endpoint serves the
home-page row, "Because you watched X", and "Continue watching" surfaces by
varying the `context` parameter.

#### Request

```http
POST /v1/recommendations HTTP/1.1
Host: api.reco.example.com
Authorization: Bearer ...
Content-Type: application/json
X-Request-Id: req-abc123

{
  "user_id": "u-12345",
  "n": 25,
  "context": {
    "surface": "home_row_for_you",
    "device": "ios",
    "session_id": "s-7a91...",
    "seed_item_id": null,
    "filters": {
      "exclude_already_rated": true,
      "min_release_year": 1990,
      "max_release_year": null,
      "genres": null,
      "exclude_item_ids": ["m-1", "m-2"]
    }
  },
  "experiment": {
    "variant_id": "hybrid_svd_itemcf_v1",
    "force_variant": false
  },
  "include_explanations": true,
  "cold_start_policy": "popularity_fallback"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | string | yes | Tenant-scoped user identifier. ASCII, length ≤ 64. |
| `n` | int (1–200) | no, default 25 | Hard cap is 200; values above are rejected with `422`. |
| `context.surface` | enum | yes | `home_row_for_you`, `because_you_watched`, `continue_watching`, `email_weekly`, `cold_start`. Drives feature-set selection. |
| `context.device` | enum | no | `ios`, `android`, `web`, `tv`, `email`. Used for downstream re-ranking, e.g. shorter titles for TV. |
| `context.session_id` | string | no | Used for de-duplication across requests in a single session. |
| `context.seed_item_id` | string | no | Required when `surface = "because_you_watched"`. |
| `context.filters` | object | no | Hard filters applied after ranking. See §2.1.1. |
| `experiment.variant_id` | string | no | If absent, the A/B router picks a variant. Forcing a variant requires `reco:internal` scope. |
| `include_explanations` | bool | no, default `false` | Adds an `explanation` field to each item. |
| `cold_start_policy` | enum | no, default `popularity_fallback` | `popularity_fallback`, `genre_popularity_fallback`, `error`. |

##### 2.1.1 Filters

`filters` is intersected with the candidate set BEFORE re-ranking, except
for `exclude_already_rated`, which is applied AFTER re-ranking so the
list size still hits `n`.

| Filter | Type | Notes |
|---|---|---|
| `exclude_already_rated` | bool | Default `true`. Excludes items the user has rated. |
| `min_release_year` | int | Inclusive. |
| `max_release_year` | int | Inclusive. |
| `genres` | string[] | Whitelist; OR semantics. |
| `exclude_item_ids` | string[] | Hard exclude (e.g. recently shown). Capped at 1,000. |

#### Response (`200 OK`)

```json
{
  "user_id": "u-12345",
  "items": [
    {
      "item_id": "m-571",
      "rank": 1,
      "score": 4.62,
      "score_calibrated": 0.91,
      "candidate_source": "svd_topn_cache",
      "explanation": {
        "reason_code": "similar_to_rated",
        "reason_text": "Because you rated The Shawshank Redemption highly.",
        "evidence_item_ids": ["m-318"]
      }
    }
  ],
  "metadata": {
    "model_version": "netflix-svd-hybrid-1.0.0",
    "experiment_variant": "hybrid_svd_itemcf_v1",
    "cache_hit": true,
    "cold_start": false,
    "served_at": "2026-04-29T15:11:42Z",
    "request_id": "req-abc123"
  }
}
```

| Field | Notes |
|---|---|
| `items[].score` | Predicted rating on the 1–5 scale, clipped. Same scale across surfaces. |
| `items[].score_calibrated` | Probability of a positive interaction (0–1). Calibrated via Platt scaling on a holdout set; preferred field for downstream ranking blends. |
| `items[].candidate_source` | One of `svd_topn_cache`, `item_cf_residual`, `vector_neighbour`, `popularity_fallback`, `editorial_pin`. Useful for offline analysis. |
| `metadata.cache_hit` | `true` means the response came from DynamoDB precomputed top-N; `false` means SageMaker endpoint reranked at request time. |
| `metadata.cold_start` | `true` means the user had < 5 ratings and the cold-start path fired. |

If `cold_start = true` and `cold_start_policy = "error"`, the API returns
`404 user_not_found` instead of a list.

#### Status codes

| Code | Notes |
|---|---|
| 200 | OK |
| 400 | Body fails schema validation |
| 401 | Missing / invalid token |
| 403 | Wrong scope or tenant mismatch |
| 404 | `cold_start_policy = error` and user is cold-start |
| 422 | `n > 200`, unknown `surface`, etc. |
| 429 | Rate limited |
| 503 | Endpoint unhealthy; clients should retry |

---

### 2.2 `GET /v1/users/{user_id}/segments`

Return the user's behavioural cluster (from `04_clustering`) and RFM segment
(from `05_rfm_analysis`). Used by lifecycle marketing and the admin tool;
not used by the recommender itself.

#### Request

```http
GET /v1/users/u-12345/segments HTTP/1.1
Host: api.reco.example.com
Authorization: Bearer ...
```

Query parameters:

| Param | Type | Notes |
|---|---|---|
| `as_of` | ISO-8601 date | Optional. Default = latest snapshot. |
| `include_history` | bool | Default `false`. Returns segment-membership history for the last 12 snapshots. |

#### Response (`200 OK`)

```json
{
  "user_id": "u-12345",
  "as_of": "2026-04-28",
  "behavioural_cluster": {
    "cluster_id": 2,
    "name": "Power Users",
    "confidence": 0.81
  },
  "rfm": {
    "segment_id": "loyal_customers",
    "segment_name": "Loyal Customers",
    "recency_score": 5,
    "frequency_score": 4,
    "monetary_score": 4
  },
  "history": null,
  "metadata": {
    "model_version": "kmeans-user-1.2.0",
    "rfm_version": "rfm-quintile-1.0.0",
    "request_id": "req-..."
  }
}
```

If the user does not exist the API returns `404 user_not_found` with a
machine-readable error envelope. Newly-created users without a snapshot yet
return `200` with `behavioural_cluster.cluster_id = null` and
`rfm.segment_id = "new_user"`.

#### Status codes

| Code | Notes |
|---|---|
| 200 | OK |
| 401 / 403 | Auth |
| 404 | User not found |
| 410 | `as_of` requested is older than the 12-month retention window |

---

### 2.3 `POST /v1/feedback`

Log an interaction event. Implemented as a thin wrapper that forwards the
payload to the `interactions-stream` Kinesis stream and returns `202`
immediately; the event becomes visible to feature-store consumers within
~60 seconds.

#### Request

```http
POST /v1/feedback HTTP/1.1
Host: api.reco.example.com
Authorization: Bearer ...
Content-Type: application/json
Idempotency-Key: 4a8c2f0e-...

{
  "user_id": "u-12345",
  "item_id": "m-571",
  "event_type": "play_complete",
  "occurred_at": "2026-04-29T15:13:08Z",
  "session_id": "s-7a91...",
  "context": {
    "surface": "home_row_for_you",
    "rank_shown": 1,
    "request_id": "req-abc123"
  },
  "value": {
    "rating": null,
    "watch_seconds": 5612,
    "completion_pct": 0.94
  }
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `event_type` | enum | yes | `impression`, `click`, `play_start`, `play_complete`, `rating`, `add_to_list`, `dismiss`. |
| `occurred_at` | ISO-8601 timestamp | yes | Must be within ±15 min of server time, else `422`. |
| `context.request_id` | string | no | The original `request_id` from the recommendation that produced the impression. Critical for offline attribution. |
| `value.rating` | number 1–5 | only when `event_type = "rating"` | |
| `value.watch_seconds` | int ≥ 0 | only when `event_type` ∈ `play_*` | |
| `value.completion_pct` | float 0–1 | only when `event_type = "play_complete"` | |

#### Response (`202 Accepted`)

```json
{
  "accepted": true,
  "event_id": "evt-...",
  "request_id": "req-..."
}
```

The `202` is intentional: the event has been durably enqueued in Kinesis but
not yet processed. Clients should not treat the response as proof that the
event has reached the warehouse.

#### Status codes

| Code | Notes |
|---|---|
| 202 | Accepted (default success) |
| 400 / 422 | Schema validation |
| 401 / 403 | Auth |
| 409 | Idempotency conflict |
| 429 | Rate limited |
| 503 | Kinesis unavailable; client should retry |

---

### 2.4 `GET /v1/health`

Liveness + readiness probe. Public, unauthenticated. Used by the load
balancer and the on-call dashboard.

```json
{
  "status": "ok",
  "version": "1.4.2",
  "uptime_seconds": 73411,
  "dependencies": {
    "dynamodb": "ok",
    "sagemaker_endpoint": "ok",
    "feature_store_online": "ok",
    "kinesis": "ok"
  }
}
```

A dependency in `degraded` state returns `200` with an overall
`status: "degraded"`; only a fully unhealthy service returns `503`.

---

### 2.5 `GET /v1/models`

Internal-only (`reco:internal` scope). Lists the active model and
experiment variants for debugging and the admin console.

```json
{
  "active": {
    "model_id": "netflix-svd-hybrid",
    "version": "1.0.0",
    "trained_at": "2026-04-22T03:14:11Z",
    "probe_rmse": 0.9491
  },
  "variants": [
    {"variant_id": "hybrid_svd_itemcf_v1", "traffic_pct": 50, "status": "live"},
    {"variant_id": "two_tower_implicit_v0", "traffic_pct": 5,  "status": "shadow"}
  ]
}
```

---

## 3. Schemas (JSON Schema, abbreviated)

The full machine-readable schemas live next to the OpenAPI file. The most
important ones are inlined below for review.

### 3.1 `RecommendationItem`

```json
{
  "type": "object",
  "required": ["item_id", "rank", "score"],
  "properties": {
    "item_id": {"type": "string", "maxLength": 64},
    "rank": {"type": "integer", "minimum": 1},
    "score": {"type": "number", "minimum": 1.0, "maximum": 5.0},
    "score_calibrated": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "candidate_source": {
      "type": "string",
      "enum": ["svd_topn_cache", "item_cf_residual", "vector_neighbour",
               "popularity_fallback", "editorial_pin"]
    },
    "explanation": {"$ref": "#/components/schemas/Explanation"}
  }
}
```

### 3.2 `Explanation`

```json
{
  "type": "object",
  "required": ["reason_code"],
  "properties": {
    "reason_code": {
      "type": "string",
      "enum": ["similar_to_rated", "popular_in_cohort", "trending",
              "matches_genre_taste", "editorial", "cold_start_popularity"]
    },
    "reason_text": {"type": "string", "maxLength": 240},
    "evidence_item_ids": {
      "type": "array",
      "items": {"type": "string"},
      "maxItems": 5
    }
  }
}
```

### 3.3 `FeedbackEvent`

See §2.3 for the wire shape. The schema enforces:

* `value.rating` is required iff `event_type = "rating"`;
* `value.watch_seconds` is required iff `event_type ∈ {play_start, play_complete}`;
* `value.completion_pct` is required iff `event_type = play_complete`.

These constraints are encoded as `oneOf` branches in the JSON Schema.

---

## 4. SDK behaviour expectations

The official SDKs (TypeScript, Swift, Kotlin, Python) wrap the API and
hide the boilerplate. To keep server and client behaviour aligned, SDKs
must:

1. **Retry idempotent calls** (`GET`, `POST /v1/feedback` with key) with
   capped exponential backoff (`100ms → 200ms → 400ms`, max 3) on `502`,
   `503`, `504`, network errors.
2. **Never retry `POST /v1/recommendations` automatically.** A retry storm
   on a degraded endpoint is the worst case for cache pressure; clients
   should fall back to a stale cached row instead.
3. **Stamp `Idempotency-Key`** as a UUIDv4 on every `POST /v1/feedback`.
4. **Surface `X-Request-Id`** in client logs.
5. **Honour `Retry-After`** on `429`.
6. **Cap concurrency** at 4 in-flight requests per device.

---

## 5. Versioning and deprecation

* **Backwards-compatible** changes (new optional request fields, new
  response fields, new enum values that map to a default) ship as a minor
  bump and require no client action.
* **Backwards-incompatible** changes ship under a new major path (`/v2`).
  The previous major is supported for 12 months from the new major's GA
  date, with a 90-day deprecation warning emitted via the
  `Deprecation` and `Sunset` HTTP headers.
* **Field removal** is announced at least 90 days in advance via the
  `Deprecation` header on every response that includes the field.

---

## 6. Observability

Every request emits:

* a CloudWatch metric in the `Reco/API` namespace
  (dimensions: `endpoint`, `status_code`, `experiment_variant`,
  `cache_hit`);
* an X-Ray segment with the `request_id` as the trace key;
* a structured JSON log line in CloudWatch Logs with PII redacted
  (the `user_id` is hashed with HMAC-SHA-256 + tenant salt before logging
  unless the caller has the `pii:read` scope).

The on-call dashboard surfaces, in order: `5xx` rate, p95 latency, cache
hit rate, cold-start rate, model-version mix, and fallback rate. SEV
thresholds are defined in [`MLOPS.md`](MLOPS.md#5-on-call-runbook).

---

## 7. Open items

* OpenAPI 3.1 spec generation from this doc (`docs/scripts/md_to_openapi.py`).
* gRPC mirror for internal callers (lower latency, schema-first).
* Streaming endpoint (`POST /v1/recommendations/stream`) returning
  Server-Sent Events for "type-ahead" reco surfaces; not on the v1 roadmap.

---

*Last reviewed: 2026-04-29. Owner: Yuanyuan Xie. Review cadence: quarterly,
or any time `ARCHITECTURE.md` §4.5 or §5 changes.*
