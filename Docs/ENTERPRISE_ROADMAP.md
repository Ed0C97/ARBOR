# ARBOR Enterprise Implementation Manual (v2.0)

> **DOCUMENTO OPERATIVO PER AI DEVELOPER**
> Questo file non è una semplice lista di cose da fare. È una **specifica tecnica dettagliata** per l'agente che eseguirà il refactoring.
> **ISTRUZIONI PER L'AGENTE:** Implementa ogni punto seguendo ESATTAMENTE lo stack tecnologico, i pattern e i criteri di accettazione descritti. Non improvvisare sull'architettura.

---

## TIER 1 - SECURITY & CORRECTNESS (Priority: IMMEDIATE)

### 1. Centralized Secrets Management (HashiCorp Vault / GCP Secret Manager)
- **Problem:** Credenziali hardcoded in `config.py` o `.env` committati. Rischio critico di leak.
- **Implementation Spec:**
    - **Library:** `google-cloud-secret-manager` (prod) + `python-dotenv` (local).
    - **Pattern:** Factory pattern per caricare i secret.
    - **Actions:**
        1. Creare `backend/app/core/secrets_manager.py`.
        2. Definire classe `SecretManager`.
        3. Metodo `get_secret(secret_id: str, version_id: str = "latest") -> str`.
        4. In `local`: leggere da `os.environ`.
        5. In `prod`: chiamare API GCP.
    - **Code Snippet:**
      ```python
      from google.cloud import secretmanager
      
      def get_secret(secret_id):
          if os.getenv("ENV") == "local":
              return os.getenv(secret_id)
          client = secretmanager.SecretManagerServiceClient()
          name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
          response = client.access_secret_version(request={"name": name})
          return response.payload.data.decode("UTF-8")
      ```
    - **Acceptance Criteria:** `config.py` non deve contenere stringhe di default per nessuna chiave sensibile. Il server deve fallire l'avvio se manca un secret.

### 2. Async Event Loop Integrity (Non-blocking IO)
- **Problem:** `cohere.Client` e `QdrantClient` sincroni bloccano il main thread di FastAPI.
- **Implementation Spec:**
    - **Target:** 100% Async Coverage sui driver I/O bound.
    - **Actions:**
        1. **Cohere:** Migrare a `cohere.AsyncClientV2`.
           ```python
           import cohere
           async with cohere.AsyncClientV2(api_key=...) as co:
               response = await co.embed(...)
           ```
        2. **Qdrant:** Migrare a `qdrant_client.AsyncQdrantClient`.
           ```python
           from qdrant_client import AsyncQdrantClient
           client = AsyncQdrantClient(url=..., grpc_port=6334, prefer_grpc=True)
           await client.search(...)
           ```
        3. **Legacy:** Wrappare eventuali chiamate CPU-bound residue con `asyncio.to_thread`.
    - **Acceptance Criteria:** Nessun warning `BlockingIOError` nei log. `py-spy` non deve mostrare il main thread bloccato su socket read.

### 3. PostgreSQL Keyset Pagination
- **Problem:** `OFFSET 10000` è lento (scansione lineare). `repository.py` carica tutto in RAM.
- **Implementation Spec:**
    - **Pattern:** "Seek Method" / Cursor Pagination.
    - **Query:** Invece di `OFFSET N`, usare `WHERE (created_at, id) < (last_created_at, last_id) ORDER BY created_at DESC, id DESC LIMIT N`.
    - **Actions:**
        1. Modificare `backend/app/api/v1/endpoints/entities.py` per accettare parametro query `cursor` (base64 encoded).
        2. Decodificare il cursore in `(timestamp, id)`.
        3. Applicare il filtro in SQLAlchemy.
    - **Acceptance Criteria:** Endpoint `/entities` risponde in <50ms costanti anche per la pagina 1000.

---

## TIER 2 - CRITICAL PERFORMANCE

### 4. Vector Ingestion Batch Pipeline
- **Problem:** `gateway.py` esegue 1 chiamata API per ogni documento. Latency = N * 200ms.
- **Implementation Spec:**
    - **Constraint:** Cohere max batch size = 96.
    - **Pattern:** Generator-based chunking.
    - **Actions:**
        1. Implementare `chunker(iterable, n)` in `utils.py`.
        2. In `gateway.py`:
           ```python
           async def get_embeddings_batch(texts: List[str]):
               batches = list(chunker(texts, 96))
               tasks = [co.embed(texts=b, model="embed-multilingual-v3.0") for b in batches]
               results = await asyncio.gather(*tasks)
               return [emb for res in results for emb in res.embeddings]
           ```
    - **Acceptance Criteria:** Ingestion di 1000 entità deve impiegare < 10 secondi (vs 200s attuali).

### 5. Advanced Database Indexing Strategy
- **Problem:** Full table scan su filtri JSONB e geospaziali.
- **Implementation Spec:**
    - **PostgreSQL:**
        1. `CREATE INDEX CONCURRENTLY idx_enrichment_vibe_gin ON arbor_enrichment USING GIN (vibe_dna jsonb_path_ops);`
        2. `CREATE INDEX CONCURRENTLY idx_venue_geo ON venue USING GIST (location);`
    - **Neo4j:**
        1. `CREATE CONSTRAINT FOR (e:Entity) REQUIRE e.id IS UNIQUE;`
        2. `CREATE INDEX FOR (e:Entity) ON (e.category);`
        3. `CREATE FULLTEXT INDEX entity_text_search FOR (n:Entity) ON EACH [n.name, n.description];`
    - **Acceptance Criteria:** `EXPLAIN ANALYZE` mostra `Bitmap Heap Scan` o `Index Scan` per tutte le query core.

### 6. Dynamic Connection Pool Tuning
- **Problem:** Errori `TimeoutError: QueuePool limit of size 5 overflow 10 reached`.
- **Implementation Spec:**
    - **Configuration:**
        - `POOL_SIZE = 40` (Active connections held)
        - `MAX_OVERFLOW = 20` (Burst connections)
        - `POOL_RECYCLE = 1800` (Reset connection every 30m to avoid stale firewall states)
        - `POOL_PRE_PING = True` (Check liveness before checkout)
    - **Files:** `backend/app/db/session.py`.
    - **Acceptance Criteria:** Zero errori di connection pool sotto load test di 200 RPS.

### 7. Compiled Graph Singleton Pattern
- **Problem:** LangGraph ricompila il grafo stateful a ogni richiesta API. Overhead ~200ms.
- **Implementation Spec:**
    - **Pattern:** Application State Singleton.
    - **Actions:**
        1. In `graph.py`, esporre una variabile globale (o singleton class) `COMPILED_GRAPH`.
        2. Inizializzarla solo se `None`.
        3. `app.state.graph = create_agent_graph().compile()` in `main.py` `@app.on_event("startup")`.
    - **Acceptance Criteria:** Tempo di "Time to First Byte" per `/discover` ridotto di 200ms.

### 8. Semantic Cache "Hit-and-Return" Optimization
- **Problem:** Cache Miss paga il costo dell'embedding, ma poi non lo riutilizza.
- **Implementation Spec:**
    - **Logic Change:**
        1. `check_cache(query_text)` -> restituisce `(hit: bool, payload: dict, query_embedding: vector)`.
        2. Se `hit` è False, usa `query_embedding` (già calcolato per cercare in Qdrant) per passarlo al resto della pipeline.
        3. Evita la seconda chiamata a `co.embed(query_text)` successiva.
    - **Acceptance Criteria:** 1 sola chiamata embedding per singola richiesta utente, SEMPRE.

### 9. Asyncio.gather Service Initialization
- **Problem:** Startup sequenziale (DB -> Qdrant -> Neo4j -> Redis) richiede 15s+.
- **Implementation Spec:**
    - **Pattern:** Concurrent Future Execution.
    - **Code Snippet:**
      ```python
      @asynccontextmanager
      async def lifespan(app: FastAPI):
          await asyncio.gather(
              postgres.connect(),
              qdrant.connect(),
              neo4j.verify_connectivity(),
              redis.ping()
          )
          yield
          await asyncio.gather(...) 
      ```
    - **Acceptance Criteria:** Startup time < 5 secondi.

---

## TIER 3 - RESILIENCE & RELIABILITY

### 10. External Service Circuit Breakers
- **Problem:** Quando Cohere è down, i thread si accumulano in attesa fino al timeout.
- **Implementation Spec:**
    - **Library:** `pybreaker` o custom implementation stateful.
    - **Config:**
        - `fail_max = 5` (dopo 5 errori consecutivi apre il circuito)
        - `reset_timeout = 60s` (aspetta 1 min prima di riprovare)
    - **Fallback:** Se Cohere Circuit è OPEN -> return "Service Degraded" status o cached results.
    - **Files:** `backend/app/core/circuit.py`.
    - **Acceptance Criteria:** Il sistema risponde immediatamente con errore 503 (o fallback) invece di hangare per 30s.

### 11. Granular Node Timeouts (LangGraph)
- **Problem:** Un agente "zombie" può bloccare l'intera chain indefinitamente.
- **Implementation Spec:**
    - **Wrapper:** Decoratore `@timeout(seconds=N)` sui nodi del grafo.
    - **Budget:**
        - `IntentNode`: 2.0s
        - `SearchNode`: 5.0s
        - `SynthesisNode`: 15.0s
    - **Handling:** Se timeout → raise `NodeTimeoutError` → instrada a `ErrorNode` per risposta gentile.
    - **Acceptance Criteria:** Nessuna richiesta HTTP supera mai i 25s totali (timeout del load balancer).

### 12. Tenacity Retry Policies
- **Problem:** Glitch di rete temporanei causano 500 Internal Server Error.
- **Implementation Spec:**
    - **Library:** `tenacity`.
    - **Strategy:** Jittered Exponential Backoff.
    - **Code:**
      ```python
      @retry(
          stop=stop_after_attempt(3),
          wait=wait_exponential_jitter(initial=0.5, max=5.0),
          retry=retry_if_exception_type((httpx.NetworkError, cohere.ServiceUnavailableError))
      )
      async def call_external_api(...):
      ```
    - **Acceptance Criteria:** I test di integrazione con "flaky" network mock passano al 100%.

### 13. Kafka Dead Letter Queue (DLQ)
- **Problem:** Messaggio malformato blocca il consumer loop in crash continuo.
- **Implementation Spec:**
    - **Architecture:** Consumer `MainTopic` -> Exception? -> Retry 3x -> Exception? -> Publish to `DLQTopic`.
    - **Monitor:** Alert se `DLQTopic` message count > 0.
    - **Files:** `backend/app/worker/kafka_consumer.py`.
    - **Acceptance Criteria:** Un messaggio "avvelenato" viene scartato dopo 3 tentativi e il consumer prosegue col messaggio successivo.

### 14. Blocking Guardrails
- **Problem:** Guardrails logga ma permette output potenzialmente dannosi.
- **Implementation Spec:**
    - **Action:** Enforce strict checking.
    - **Logic:**
        ```python
        score = validate(response)
        if score.hallucination_risk > 0.8:
            logger.warning("Hallucination intercepted")
            return "I apologize, verify details..." # Override response
        ```
    - **Acceptance Criteria:** Test case con prompt "malevolo" non deve MATE ritornare una risposta unsafe.

---

## TIER 4 - SEARCH & AI QUALITY

### 15. Reciprocal Rank Fusion (RRF) Hybrid Search
- **Problem:** Vector search fallisce su ricerche esatte (es. "Bar Basso").
- **Implementation Spec:**
    - **Algorithm:** $Score = \frac{1}{k + rank_{vector}} + \frac{1}{k + rank_{keyword}}$
    - **Vectors:** Named vectors in Qdrant (`text-dense`, `keyword-sparse`).
    - **Orchestration:** Eseguire search parallela, normalizzare score, applicare formula RRF.
    - **Files:** `backend/app/agents/hybrid_search.py`.
    - **Acceptance Criteria:** Query specifica "Bar Basso" deve ritornare il venue specifico al result #1, non cose "simili".

### 16. Entity Resolution & Merging Strategy
- **Problem:** Stessa entità trovata sia per nome che per vibe appare due volte.
- **Implementation Spec:**
    - **Logic:**
        1. Raccogliere tutti i risultati candidati.
        2. Raggruppare per `entity_uuid`.
        3. Se `entity_uuid` manca, raggruppare per fuzzy match su `normalized_name` + `address`.
        4. Merge dei metadati (preferire Neo4j > Postgres > Vector).
    - **Acceptance Criteria:** La lista finale dei risultati contiene solo ID unici.

### 17. Cache Threshold Calibration (0.88 - 0.92)
- **Problem:** Soglia 0.95 richiede query quasi identiche.
- **Implementation Spec:**
    - **Action:** Impostare `SCORING_THRESHOLD = 0.90` in `config.py`.
    - **Feature:** Implementare "Fuzzy matching" sulla cache text search prima del vector check (per catch rapidi di typo).
    - **Acceptance Criteria:** "Ristorante romantico" e "posto romantico per cena" dovrebbero hittare la stessa cache entry.

### 18. Qdrant Cache Invalidation Hooks
- **Problem:** Dati in cache diventano stale (prezzi vecchi, venue chiusi).
- **Implementation Spec:**
    - **Trigger:** Su update/delete entità in PostgreSQL -> Invia event `ENTITY_UPDATED`.
    - **Worker:** Ascolta evento -> Esegue `qdrant_client.delete(collection="cache", filter=...)`.
    - **Acceptance Criteria:** Modifica dell'orario di un locale si riflette nella risposta discovery entro 5 secondi.

### 19. Cohere Rerank v3 Integration
- **Problem:** I top 10 risultati vettoriali non sono sempre i più rilevanti semanticamente.
- **Implementation Spec:**
    - **Pipeline:** Retrieve 50 -> Rerank (Cohere API) -> Keep Top 10 -> LLM Synthesis.
    - **Model:** `rerank-multilingual-v3.0`.
    - **Code:**
      ```python
      rerank_results = await co.rerank(model="rerank-multi...", query=q, documents=docs, top_n=10)
      sorted_docs = [docs[hit.index] for hit in rerank_results.results]
      ```
    - **Acceptance Criteria:** Precision@5 aumenta dal 60% al 85% sui test set.

---

## TIER 5 - OBSERVABILITY

### 20. OpenTelemetry Tracing (Full Stack)
- **Problem:** "Perché questa richiesta ci ha messo 8 secondi?" -> Nessuna risposta.
- **Implementation Spec:**
    - **SDK:** `opentelemetry-distro`, `opentelemetry-exporter-otlp`.
    - **Instrumentation:** `FastAPIInstrumentor`, `SQLAlchemyInstrumentor`, `RequestsInstrumentor`.
    - **Propagation:** Assicurare header `traceparent` nelle chiamate inter-servizio.
    - **Acceptance Criteria:** Trace visualizzabile in Jaeger/Grafana Tempo che copre HTTP -> DB -> Ext API.

### 21. Business Metrics Implementation (Prometheus)
- **Problem:** Monitoriamo CPU/RAM ma non il business value.
- **Implementation Spec:**
    - **Library:** `prometheus_client`.
    - **Metrics:**
        - `counter("arbor_discover_requests_total", labels=["status"])`
        - `histogram("arbor_llm_latency_seconds", buckets=[...])`
        - `gauge("arbor_active_curators")`
    - **Acceptance Criteria:** Endpoint `/metrics` espone dati validi scrappabili da Prometheus.

### 22. Deep Health Checks
- **Problem:** K8s riavvia i pod solo se HTTP fallisce, ma se DB è down il pod resta "Healthy" ma inutile.
- **Implementation Spec:**
    - **Endpoint:** `/health/readiness`.
    - **Logic:** Check connettività reale (ping) verso Redis, PG, Qdrant. Return 503 se uno critico è down.
    - **Acceptance Criteria:** Se fermo il container Postgres, il pod backend deve diventare `NotReady` entro 10s.

### 23. Prometheus Alert Rules
- **Problem:** Scopriamo i disservizi dagli utenti.
- **Implementation Spec:**
    - **Rules (YAML):**
        - `HighErrorRate`: rate(500s) > 1% per 5m.
        - `SlowResponses`: p99 > 5s per 5m.
        - `LowCacheHit`: rate(hits) / rate(total) < 0.1 per 1h.
    - **Acceptance Criteria:** Simulando errori, scatta notifica su Slack/PagerDuty.

---

## TIER 6 - INFRASTRUCTURE HARDENING

### 24. Redis Sliding Window Rate Limiter
- **Problem:** Rate limit fisso resetta allo scoccare del minuto, permettendo burst doppi.
- **Implementation Spec:**
    - **Algorithm:** Sliding Window Log (o Approximation con Sorted Set).
    - **Key:** `rate_limit:{user_id}`.
    - **Logic:** `ZREMRANGEBYSCORE` (remove old timestamps) -> `ZCARD` (count) -> `ZADD` (new request).
    - **Acceptance Criteria:** Impossibile superare il limit distribuyendo le richieste a cavallo del minuto.

### 25. Cloudflare CDN Integration
- **Problem:** Latency alta per utenti fuori regione.
- **Implementation Spec:**
    - **DNS:** Proxying mode (orange cloud).
    - **Caching:** Page Rules -> `*.arbor.app/assets/*` -> Cache Level: Everything.
    - **WAF:** Block challenges per User-Agent sospetti.
    - **Acceptance Criteria:** TTFB per assets statici < 50ms globale.

### 26. Protobuf for Kafka Events
- **Problem:** Payload JSON enormi saturano la banda Kafka.
- **Implementation Spec:**
    - **Schema:** Definire `events.proto` (EntityCreated, SearchExecuted).
    - **Compile:** `protoc` generate python classes.
    - **Usage:** Producer serializza in bytes, Consumer deserializza.
    - **Acceptance Criteria:** Dimensione messaggi ridotta del 40-60%.

### 27. Hardened CI Pipeline
- **Problem:** Codice rotto o non sicuro arriva in main.
- **Implementation Spec:**
    - **Jobs:**
        1. `lint`: flake8 + black.
        2. `types`: mypy --strict.
        3. `security`: gitleaks detect --source . + bandit -r agent.
        4. `test`: pytest --cov=app --cov-fail-under=80.
    - **Acceptance Criteria:** PR bloccata se coverage scende sotto 80% o se trovate vulnerabilità.

### 28. Argo Rollouts (Canary)
- **Problem:** Deploy rischiosi "big bang".
- **Implementation Spec:**
    - **Manifest:** `Rollout` resource invece di `Deployment`.
    - **Strategy:**
        ```yaml
        strategy:
          canary:
            steps:
            - setWeight: 5
            - pause: {duration: 10m}
            - setWeight: 50
            - pause: {duration: 10m}
        ```
    - **Acceptance Criteria:** Deploy graduale automatico visibile in dashboard Argo.

---

## TIER 7 - ADVANCED ARCHITECTURE

### 29. gRPC over REST for Qdrant
- **Problem:** Conversione JSON-Obj costosa per grandi vettori.
- **Implementation Spec:**
    - **Client:** `QdrantClient(..., prefer_grpc=True)`.
    - **Protocol:** HTTP/2 transport.
    - **Acceptance Criteria:** Throughput insert vettori +30%.

### 30. CQRS (Command Query Responsibility Segregation)
- **Problem:** Scritture bloccano letture su DB unico.
- **Implementation Spec:**
    - **DB:** Configurare Master (RW) e Replica (RO).
    - **Code:** `get_db_read()` vs `get_db_write()` dependencies in FastAPI.
    - **Routing:** GET requests -> Replica. POST/PUT -> Master.
    - **Acceptance Criteria:** Load test letture scala indipendentemente dalle scritture.

### 31. HNSW Index Fine-Tuning
- **Problem:** Default settings Qdrant non ottimali.
- **Implementation Spec:**
    - **Parameters:** `m=16` (connections), `ef_construct=128` (build accuracy).
    - **Quantization:** `Scalar (Int8)` per ridurre RAM 4x con perdita precisione minima.
    - **Acceptance Criteria:** Recall > 0.98 con riduzione RAM 30%.

### 32. API Idempotency (Redis Keys)
- **Problem:** Doppio click utente crea doppia entità.
- **Implementation Spec:**
    - **Header:** Client invia `Idempotency-Key: UUID`.
    - **Middleware:**
        1. Check Redis key.
        2. Se esiste -> return cached response (409 conflict o 200 replay).
        3. Se no -> process -> save response to Redis (TTL 24h).
    - **Acceptance Criteria:** Replay dello stesso payload cURL 2 volte non crea 2 record DB.

### 33. Server-Sent Events (SSE) Streaming
- **Problem:** UX bloccata.
- **Implementation Spec:**
    - **Protocol:** `text/event-stream`.
    - **Pattern:** Generator function che `yield` chunk di JSON o plain text.
    - **Frontend:** `EventSource` API o `fetch` con reader.
    - **Acceptance Criteria:** Primo token visibile in < 1s, resto dello stream fluido.

### 34. Pre-computation Background Jobs
- **Problem:** "Cold Start" per nuove entità.
- **Implementation Spec:**
    - **Orchestrator:** Temporal.io o Celery.
    - **Workflow:** `DailyEmbeddingUpdate`.
    - **Logic:** Trova entità con `last_indexed < now - 24h` -> Re-embed -> Re-index.
    - **Acceptance Criteria:** Nessuna entità attiva ha embeddings più vecchi di 24 ore.

---

## TIER 8 - FRONTEND ENTERPRISE-GRADE

### 35. Secure HttpOnly Cookie Auth
- **Problem:** XSS ruba token da LocalStorage.
- **Implementation Spec:**
    - **Backend:** `response.set_cookie(key="access_token", value=jwt, httponly=True, secure=True, samesite='Lax')`.
    - **Frontend:** API calls non inviano header manuale, browser lo allega automaticamente.
    - **Acceptance Criteria:** `document.cookie` vuoto in console browser.

### 36. Error Boundary Isolation
- **Problem:** Bug in un widget rompe tutto.
- **Implementation Spec:**
    - **Wrapper:** `react-error-boundary`.
    - **Fallback:** Componente visuale "Module temporarily unavailable" con pulsante retry.
    - **Acceptance Criteria:** Eccezione lanciata manualmente in `VibeRadar` non nasconde la navbar.

### 37. WCAG 2.1 AA Compliance
- **Problem:** Esclusione utenti disabili.
- **Implementation Spec:**
    - **Audit:** Axe DevTools.
    - **Fixes:**
        - `aria-label` su tutti i button icon-only.
        - Navigazione tastiera logica (tabindex).
        - Contrasto colori 4.5:1.
    - **Acceptance Criteria:** Axe report pulito (zero violazioni critiche).

### 38. Bundle Optimization (Dynamic Imports)
- **Problem:** `_app.js` scarica librerie grafiche (D3.js) anche nella home.
- **Implementation Spec:**
    - **Next.js:**
      ```javascript
      const HeavyChart = dynamic(() => import('./HeavyChart'), {
        loading: () => <Skeleton />,
        ssr: false
      })
      ```
    - **Acceptance Criteria:** Lighthouse Performance score > 90.

### 39. JSON-LD Structured Data
- **Problem:** Google non capisce che le pagine sono Luoghi/Ristoranti.
- **Implementation Spec:**
    - **Schema:** Inserire `<script type="application/ld+json">` nel layout.
    - **Type:** `LocalBusiness`, `Restaurant`, `TouristAttraction`.
    - **Acceptance Criteria:** Google Rich Results Test valida lo snippet.

### 40. Skeleton Loading States
- **Problem:** Layout Shift (CLS) durante il caricamento dati.
- **Implementation Spec:**
    - **UI:** Creare versioni `gray-pulse` di EntityCard.
    - **Usage:** Visualizzare durante `isLoading` hook di SWR/React Query.
    - **Acceptance Criteria:** Cumulative Layout Shift (CLS) score < 0.1.

### 41. React Query (TanStack Query) Integration
- **Problem:** Stato server gestito manualmente (useEffect, useState). Race conditions.
- **Implementation Spec:**
    - **Features:** Deduplication, Stale-while-revalidate, Window focus refetching.
    - **Code:** `const { data, isLoading } = useQuery(['entity', id], fetchEntity)`.
    - **Acceptance Criteria:** Navigare avanti/indietro tra pagine entità è istantaneo (cache).

### 42. Strict Image Security Policy
- **Problem:** SSRF tramite image optimization endpoint.
- **Implementation Spec:**
    - **Config:**
      ```javascript
      images: {
        remotePatterns: [
          { protocol: 'https', hostname: '**.cdninstagram.com' },
          { protocol: 'https', hostname: 'storage.googleapis.com' }
        ]
      }
      ```
    - **Acceptance Criteria:** Richiesta immagine da dominio non whitelistato ritorna 400.

### 43. CSP & Security Headers
- **Problem:** Rischio injection e clickjacking.
- **Implementation Spec:**
    - **Middleware:** `next-secure-headers` o config manuale.
    - **Directives:** `default-src 'self'`; `script-src 'self' 'unsafe-eval'` (per dev env).
    - **Acceptance Criteria:** Mozilla Observatory score: A.

---

## TIER 9 - TESTING & QA

### 44. Unit Testing Analyzers
- **Problem:** Logica complessa non testata.
- **Implementation Spec:**
    - **Framework:** Pytest.
    - **Scope:** Testare isolatamente `PriceAnalyzer`, `VibeAnalyzer`.
    - **Cases:** Input validi, input edge case (testo vuoto), input malformati.
    - **Acceptance Criteria:** Coverage > 90% per directory `analyzers/`.

### 45. Mocking LLMs (VCR.py / Pytest-Mock)
- **Problem:** Test suite costa soldi (API calls) e flava.
- **Implementation Spec:**
    - **Tool:** `pytest-recording` (VCR).
    - **Flow:** Prima run registra cassette YAML. Run successive usano cassetta.
    - **Acceptance Criteria:** Test suite gira offline in < 10 secondi.

### 46. Integration Tests (Multi-DB)
- **Problem:** Componenti funzionano da soli ma non insieme.
- **Implementation Spec:**
    - **Environment:** `testcontainers` per spawnare PG, Redis, Qdrant usa-e-getta.
    - **Scenario:** Create Entity -> Verify in PG -> Verify in Vector Search -> Verify Graph Node.
    - **Acceptance Criteria:** Consistency check passa su stack effimero.

### 47. Load Testing (K6)
- **Problem:** Performance degradation non rilevata.
- **Implementation Spec:**
    - **Script:**
        - Ramp-up 0-50 VUs in 1m.
        - Hold 50 VUs for 5m.
        - Checks: status 200 > 99%.
    - **Acceptance Criteria:** Confermare capacità di throughput per il lancio.

### 48. Chaos Engineering (Toxiproxy)
- **Problem:** Resilienza teorica, mai provata.
- **Implementation Spec:**
    - **Simulations:** Aggiungere latenza 2000ms a connessione Redis. Tagliare connessione Neo4j.
    - **Expectation:** Errori gestiti (fallback o messaggi errore puliti), no crash stacktrace.
    - **Acceptance Criteria:** Sistema sopravvive a perdita di 1 dipendenza non critica.

### 49. E2E Testing (Playwright)
- **Problem:** Regressioni UI non rilevate.
- **Implementation Spec:**
    - **Suites:** `LoginFlow`, `DiscoveryFlow`, `CurationFlow`.
    - **Browsers:** Chromium, Firefox, WebKit.
    - **Acceptance Criteria:** Video recording delle sessioni di test disponibile in CI.

---

## TIER 10 - DATA & ML OPS

### 50. Transactional Outbox Pattern
- **Problem:** Dual write problem (DB commit ok, Kafka publish fail).
- **Implementation Spec:**
    - **Table:** `outbox_events (id, topic, payload, processed_at)`.
    - **Transaction:** Insert entity + Insert outbox event (Atomic).
    - **Relay:** Processo background legge outbox -> Invia Kafka -> Mark processed.
    - **Acceptance Criteria:** Even se Kafka è down per 1 ora, nessun evento viene perso.

### 51. Drift Detection Pipeline
- **Problem:** Modello "invecchia" (nuovi slang, nuovi trend).
- **Implementation Spec:**
    - **Metric:** Embedding distribution distance (KL Divergence).
    - **Tool:** Evidently AI o custom script.
    - **Alert:** Se drift score > threshold -> Trigger retrain/re-index advice.
    - **Acceptance Criteria:** Report drift generato settimanalmente.

### 52. Keyword/Ranking A/B Framework
- **Problem:** Dibattiti soggettivi su "quale prompt è meglio".
- **Implementation Spec:**
    - **Assignment:** Hash(User-ID) % 2 == 0 ? VariantA : VariantB.
    - **Tracking:** Loggare `experiment_group` in ogni evento analytics.
    - **Acceptance Criteria:** Dashboard comparativa KPI (Click-through rate).

### 53. Feast Feature Store
- **Problem:** Feature engineering duplicata tra training e serving.
- **Implementation Spec:**
    - **Store:** Redis (Online), BigQuery/Parquet (Offline).
    - **Features:** `user_avg_spend`, `user_preferred_vibe`.
    - **Usage:** In `discovery.py`, fetch features -> `get_online_features()`.
    - **Acceptance Criteria:** Condivisione definizioni feature team-wide.

### 54. Prompt Version Management (YAML)
- **Problem:** Prompt sparsi nel codice ("Spaghetti Prompts").
- **Implementation Spec:**
    - **Structure:** `prompts/v1/synthesis.yaml`, `prompts/v2/synthesis.yaml`.
    - **Loader:** Libreria per caricare template jinja2 da file.
    - **Config:** `CURRENT_PROMPT_VERSION = "v2"`.
    - **Acceptance Criteria:** Hot-swap dei prompt senza redeploy codice.

### 55. CDC (Debezium/Custom)
- **Problem:** Polling database è inefficiente.
- **Implementation Spec:**
    - **Source:** Postgres WAL (Write Ahead Log).
    - **Sink:** Kafka Connect -> Qdrant Sink.
    - **Acceptance Criteria:** Latency end-to-end modifica->search < 2 secondi.

### 56. RAG Evaluation Suite (Ragas)
- **Problem:** Qualità output aneddotica.
- **Implementation Spec:**
    - **Metrics:** Faithfulness, Answer Relevance, Context Recall.
    - **Dataset:** Golden Set di 100 Q&A verificate umanamente.
    - **CI Integration:** Run eval suite pre-release.
    - **Acceptance Criteria:** Score medio > 0.8.

---

## TIER 11 - INFRASTRUCTURE (ADVANCED)

### 57. Service Mesh (Linkerd)
- **Problem:** Zero visibility traffico est-ovest, no mTLS.
- **Implementation Spec:**
    - **Deploy:** Inject proxy sidecars in K8s pods.
    - **Features:** mTLS auto, retries network-level, latency metrics.
    - **Acceptance Criteria:** Dashboard Linkerd mostra topologia traffico live.

### 58. PgBouncer Deployment
- **Problem:** Connessioni SSL dirette costose.
- **Implementation Spec:**
    - **Mode:** Transaction pooling.
    - **Resources:** Sidecar o deployment separato.
    - **Acceptance Criteria:** Riduzione CPU Postgres del 30% per gestione connessioni.

### 59. API Gateway Caching (Nginx/Kong)
- **Problem:** Richiesta identica ricalcolata N volte.
- **Implementation Spec:**
    - **Policy:** Cache GET `/entities/*` per 60s.
    - **Invalidation:** Purge API chiamabile da backend.
    - **Acceptance Criteria:** HIT ratio > 80% su entity details.

### 60. Multi-Zone Availability
- **Problem:** Single point of failure (Zone outage).
- **Implementation Spec:**
    - **GCP:** Region `us-central1`, Zones `a, b, c`.
    - **K8s:** Topology Spread Constraints per distribuire i pod.
    - **Acceptance Criteria:** Simulazione spegnimento 1 zona -> servizio up.

### 61. Automated Secrets Rotation
- **Problem:** Chiavi compromesse valide per sempre.
- **Implementation Spec:**
    - **Schedule:** Lambda function / Cloud Run job ogni 30gg.
    - **Flow:** Genera nuova key -> Aggiorna Secret Manager -> Riavvia pod (o reload dinamico).
    - **Acceptance Criteria:** Nessuna chiave più vecchia di 30gg.

### 62. K8s Network Policies
- **Problem:** Flat network (tutti parlano con tutti).
- **Implementation Spec:**
    - **Rule:** `Postgres` accetta traffico SOLO da `Backend`.
    - **Default:** Deny All Ingress.
    - **Acceptance Criteria:** `kubectl exec` da frontend pod non riesce a contattare porta Postgres.

### 63. Pod Security Standard (Restricted)
- **Problem:** Root access nei container.
- **Implementation Spec:**
    - **Context:** `runAsNonRoot: true`, `readOnlyRootFilesystem: true`, `allowPrivilegeEscalation: false`.
    - **Acceptance Criteria:** Container fallisce avvio se tenta di scrivere su `/`.

---

## TIER 12 - UX & PRODUCT

### 64. Typewriter Streaming Effect
- **Problem:** Testo appare a blocchi o tutto alla fine.
- **Implementation Spec:**
    - **Frontend:** Parsare chunk SSE e appendere allo stato.
    - **Visual:** Cursore lampeggiante, smooth scroll automatica.
    - **Acceptance Criteria:** Percezione fluidità "ChatGPT-like".

### 65. PWA (Progressive Web App)
- **Problem:** Discovery lenta su mobile network.
- **Implementation Spec:**
    - **File:** `manifest.json` (Icone, Colori).
    - **Service Worker:** Cache strategy `StaleWhileRevalidate` per API GET.
    - **Acceptance Criteria:** App installabile su Home Screen iOS/Android.

### 66. Strong i18n Architecture
- **Problem:** Traduzioni hardcoded o chiavi mancanti.
- **Implementation Spec:**
    - **Library:** `next-intl`.
    - **Structure:** Type-safe message keys.
    - **Acceptance Criteria:** Cambio lingua (EN/IT) istantaneo senza reload.

### 67. System-Sync Dark Mode
- **Problem:** UI accecante di notte.
- **Implementation Spec:**
    - **CSS:** Tailwind `dark:` classes.
    - **Context:** `prefers-color-scheme` media query listener.
    - **Acceptance Criteria:** App segue lo switch tema del sistema operativo autom.

### 68. Real-time Presence (Admin)
- **Problem:** Collisione lavoro curator.
- **Implementation Spec:**
    - **Tech:** WebSocket / Pusher.
    - **UI:** Avatar "Mario is editing..." sulla card.
    - **Acceptance Criteria:** Blocco editing concorrente su stesso ID.

---

## TIER 13 - COMPLIANCE & GOVERNANCE

### 69. Immutable Audit Log (Append-Only)
- **Problem:** impossibile provare "chi ha fatto cosa".
- **Implementation Spec:**
    - **Table:** `audit_log` (NO UPDATE/DELETE permission a livello DB user).
    - **Data:** Timestamp, Actor, Action, PreviousState, NewState.
    - **Acceptance Criteria:** Storico completo modifiche entità.

### 70. GDPR "Right to be Forgotten" Automation
- **Problem:** Processo manuale rischioso per le sanzioni.
- **Implementation Spec:**
    - **Job:** `DeleteUserJob(user_id)`.
    - **Scope:** Soft-delete DB, hard-delete PII, anonymize logs.
    - **Acceptance Criteria:** Esecuzione job rimuove ogni traccia e-mail/nome.

### 71. Data Retention Policy Enforcer
- **Problem:** Storage costoso inutile.
- **Implementation Spec:**
    - **Cron:** `CleanupLogs` (Delete > 90d), `CleanupCache` (Delete LRU).
    - **Acceptance Criteria:** DB size stabile nel tempo.

### 72. Interactive API Documentation
- **Problem:** Developer esterni persi.
- **Implementation Spec:**
    - **Tool:** Swagger UI / ReDoc.
    - **Content:** Esempi Request/Response per ogni codice stato (200, 400, 404, 500).
    - **Acceptance Criteria:** Possibile testare API "Try it out" direttamente da browser.
