# 🤖 A.R.B.O.R. ENTERPRISE — AI EXECUTION PIPELINE

> **Versione**: 1.0  
> **Target**: AI Agent che deve generare il progetto completo  
> **Formato**: Istruzioni sequenziali con checkpoint di validazione

---

## ⚙️ CONFIGURAZIONE INIZIALE

```yaml
# L'AI deve leggere questi parametri prima di iniziare
project:
  name: "arbor-enterprise"
  base_path: "{USER_SPECIFIED_PATH}"  # Chiedi all'utente
  domain: "{USER_SPECIFIED_DOMAIN}"   # lifestyle | realestate | hr | custom

tech_stack:
  backend: "Python 3.12 + FastAPI + Poetry"
  frontend: "Next.js 14 + TypeScript + shadcn/ui"
  mobile: "Flutter 3.x"
  databases:
    sql: "PostgreSQL 16 + PostGIS"
    vector: "Qdrant"
    graph: "Neo4j"
    cache: "Redis"
  ai:
    gateway: "LiteLLM"
    primary: "OpenAI GPT-4o"
    embeddings: "text-embedding-3-small"
  orchestration: "LangGraph + Temporal.io"
  infrastructure: "Docker + Kubernetes + Terraform"

conventions:
  python:
    formatter: "black"
    linter: "ruff"
    type_checker: "mypy"
  typescript:
    formatter: "prettier"
    linter: "eslint"
  git:
    branch_prefix: "feat/ | fix/ | docs/"
    commit_format: "conventional commits"
```

---

## 📋 PIPELINE DI ESECUZIONE

### FASE 0: SETUP AMBIENTE
**Prerequisiti**: Nessuno  
**Output**: Directory progetto + dipendenze

```bash
# STEP 0.1: Crea struttura base
mkdir -p {base_path}/arbor-enterprise
cd {base_path}/arbor-enterprise

# STEP 0.2: Inizializza Git
git init
git checkout -b main

# STEP 0.3: Crea struttura directory
mkdir -p infrastructure/{terraform,kubernetes,docker}
mkdir -p backend/app/{api/v1,core,agents,llm,db/{postgres,qdrant,neo4j,redis},ingestion/{scrapers,analyzers},workflows,events,ml,observability}
mkdir -p backend/tests/{unit,integration,e2e}
mkdir -p frontend/{app,components,lib,styles}
mkdir -p mobile/lib/{screens,widgets,services,models}
mkdir -p config/{domains,guardrails,prompts,ontologies}
mkdir -p scripts docs .github/workflows

# STEP 0.4: File base
touch README.md Makefile .env.example .gitignore
```

**✅ CHECKPOINT 0**: Verifica che tutte le directory esistano con `find . -type d | head -50`

---

### FASE 1: CONFIGURAZIONE PROGETTO
**Prerequisiti**: Fase 0 completata

```bash
# STEP 1.1: Backend Python (Poetry)
cd backend
poetry init --name arbor-backend --python "^3.12" -n
poetry add fastapi uvicorn[standard] pydantic pydantic-settings
poetry add sqlalchemy[asyncio] asyncpg alembic
poetry add qdrant-client pinecone-client neo4j
poetry add redis aioredis
poetry add langchain langchain-openai langgraph
poetry add litellm openai anthropic
poetry add httpx tenacity python-jose[cryptography]
poetry add opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi
poetry add --group dev pytest pytest-asyncio black ruff mypy

# STEP 1.2: Frontend Next.js
cd ../frontend
npx -y create-next-app@latest . --typescript --tailwind --eslint --app --src-dir=false --import-alias="@/*" --use-npm
npm install @radix-ui/react-slot class-variance-authority clsx tailwind-merge
npx shadcn-ui@latest init -y
npx shadcn-ui@latest add button card input dialog

# STEP 1.3: Docker Compose dev
cd ..
# Crea docker-compose.dev.yml (vedi contenuto sotto)
```

**✅ CHECKPOINT 1**: `poetry check` e `npm run build` senza errori

---

### FASE 2: DATABASE SCHEMAS
**Prerequisiti**: Fase 1 completata

**FILE DA CREARE**:

1. `backend/app/db/postgres/models.py` - SQLAlchemy models
2. `backend/alembic/versions/001_initial.py` - Migration iniziale
3. `backend/app/db/qdrant/collections.py` - Setup collezioni
4. `backend/app/db/neo4j/schema.py` - Cypher constraints

**VALIDAZIONE**:
```bash
# PostgreSQL
docker-compose -f docker-compose.dev.yml up -d postgres
cd backend && alembic upgrade head

# Qdrant
curl http://localhost:6333/collections

# Neo4j
docker exec neo4j cypher-shell -u neo4j -p password "SHOW CONSTRAINTS"
```

**✅ CHECKPOINT 2**: Tutti i database accessibili e schema applicato

---

### FASE 3: CONFIGURAZIONE DOMINIO
**Prerequisiti**: Fase 2 completata

**FILE DA CREARE**:
1. `config/domains/{domain}.yaml` - Configurazione completa dominio
2. `config/ontologies/vibe_ontology.yaml` - Ontologia stili
3. `config/prompts/*.txt` - System prompts

**CONTENUTO OBBLIGATORIO config/domains/{domain}.yaml**:
```yaml
domain:
  name: "{domain}"
categories: [...]        # Almeno 5 categorie
dimensions: [...]        # Almeno 5 dimensioni con weight
price_tiers: [...]       # 5 livelli
target_audiences: [...]  # Almeno 4 tipi
graph_relationships: [...] 
styles: [...]            # Almeno 6 stili
curator_persona: {...}
example_queries: [...]   # Almeno 5 esempi
```

**✅ CHECKPOINT 3**: YAML valido, tutti i campi presenti

---

### FASE 4: LLM GATEWAY
**Prerequisiti**: Fase 3 completata

**FILE DA CREARE**:
1. `backend/app/llm/gateway.py` - LiteLLM router multi-provider
2. `backend/app/llm/cache.py` - Semantic cache GPTCache
3. `backend/app/llm/guardrails.py` - NeMo Guardrails
4. `backend/app/llm/prompts/` - Prompt loader

**TEST OBBLIGATORIO**:
```python
# tests/integration/test_llm_gateway.py
async def test_fallback():
    """Testa che fallback funzioni se provider primario fallisce."""
    response = await gateway.complete(messages=[...], model="gpt-4o")
    assert response is not None

async def test_cache_hit():
    """Testa semantic cache."""
    r1 = await gateway.complete(messages=[{"role": "user", "content": "test"}])
    r2 = await gateway.complete(messages=[{"role": "user", "content": "test"}])
    # r2 deve venire da cache
```

**✅ CHECKPOINT 4**: Test gateway passano, cache funziona

---

### FASE 5: INGESTION PIPELINE
**Prerequisiti**: Fase 4 completata

**FILE DA CREARE**:
1. `backend/app/ingestion/scrapers/base.py` - Abstract BaseScraper
2. `backend/app/ingestion/scrapers/google_maps.py` - Google Places API
3. `backend/app/ingestion/analyzers/vision.py` - GPT-4o Vision
4. `backend/app/ingestion/analyzers/vibe_extractor.py` - Review analysis
5. `backend/app/ingestion/analyzers/embedding.py` - Vector generation
6. `backend/app/ingestion/orchestrator.py` - MasterIngestor
7. `backend/app/workflows/ingestion_workflow.py` - Temporal workflow

**TEST OBBLIGATORIO**:
```python
async def test_full_ingestion():
    """Testa pipeline completa per 1 entità."""
    result = await ingestor.ingest_single(url="https://maps.google.com/...")
    assert result.entity_id is not None
    assert result.embedding is not None
    assert result.vibe_dna is not None
```

**✅ CHECKPOINT 5**: Ingestion di 10 entità test completata

---

### FASE 6: AGENTIC SWARM
**Prerequisiti**: Fase 5 completata

**FILE DA CREARE**:
1. `backend/app/agents/state.py` - AgentState TypedDict
2. `backend/app/agents/router.py` - IntentRouter
3. `backend/app/agents/vector_agent.py` - Qdrant search
4. `backend/app/agents/metadata_agent.py` - PostgreSQL queries
5. `backend/app/agents/historian_agent.py` - Neo4j + GraphRAG
6. `backend/app/agents/curator.py` - Synthesis
7. `backend/app/agents/graph.py` - LangGraph orchestration

**STRUTTURA LANGGRAPH OBBLIGATORIA**:
```
Entry → IntentRouter → [VectorAgent, MetadataAgent, HistorianAgent] → Curator → End
```

**TEST OBBLIGATORIO**:
```python
async def test_discovery_flow():
    result = await agent_graph.ainvoke({
        "user_query": "Cerco una sartoria con taglio napoletano",
        "user_location": "Roma"
    })
    assert result["final_response"] is not None
    assert len(result["recommendations"]) > 0
```

**✅ CHECKPOINT 6**: Query test < 3 secondi, risposta coerente

---

### FASE 7: API LAYER
**Prerequisiti**: Fase 6 completata

**FILE DA CREARE**:
1. `backend/app/main.py` - FastAPI app
2. `backend/app/api/v1/discover.py` - POST /discover
3. `backend/app/api/v1/entities.py` - CRUD entities
4. `backend/app/api/v1/search.py` - Search endpoints
5. `backend/app/api/v1/graph.py` - Graph queries
6. `backend/app/api/v1/admin.py` - Admin endpoints
7. `backend/app/core/security.py` - Auth + RBAC
8. `backend/app/core/rate_limiter.py` - Tiered limits

**ENDPOINTS OBBLIGATORI**:
```
POST   /api/v1/discover          # Main discovery
GET    /api/v1/entities          # List entities
GET    /api/v1/entities/{id}     # Get entity
POST   /api/v1/entities          # Create (admin)
PUT    /api/v1/entities/{id}     # Update (admin)
DELETE /api/v1/entities/{id}     # Delete (admin)
GET    /api/v1/search/vector     # Vector search
GET    /api/v1/graph/related     # Graph queries
GET    /health                   # Health check
```

**TEST OBBLIGATORIO**:
```bash
# API deve rispondere
curl -X POST http://localhost:8000/api/v1/discover \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "limit": 5}'
```

**✅ CHECKPOINT 7**: Tutti gli endpoint rispondono, OpenAPI generato

---

### FASE 8: FRONTEND WEB
**Prerequisiti**: Fase 7 completata

**FILE DA CREARE**:
1. `frontend/app/layout.tsx` - Root layout
2. `frontend/app/page.tsx` - Landing page
3. `frontend/app/(dashboard)/discover/page.tsx` - Chat interface
4. `frontend/app/(dashboard)/entity/[id]/page.tsx` - Entity detail
5. `frontend/components/chat/ChatInterface.tsx`
6. `frontend/components/chat/RecommendationCard.tsx`
7. `frontend/components/entity/VibeRadar.tsx` - Radar chart
8. `frontend/lib/api.ts` - API client

**COMPONENTI UI OBBLIGATORI**:
- Chat con input e messaggi
- Card risultati con Vibe score
- Radar chart dimensioni
- Dark mode support

**✅ CHECKPOINT 8**: `npm run build` passa, UI funziona in browser

---

### FASE 9: CURATOR DASHBOARD
**Prerequisiti**: Fase 8 completata

**FILE DA CREARE**:
1. `frontend/app/(admin)/layout.tsx` - Admin layout
2. `frontend/app/(admin)/curator/page.tsx` - Entity management
3. `frontend/app/(admin)/analytics/page.tsx` - Dashboard
4. `frontend/components/admin/EntityEditor.tsx`
5. `frontend/components/admin/GraphViewer.tsx`

**FUNZIONALITÀ OBBLIGATORIE**:
- Lista entità paginata con filtri
- Form edit entità con Vibe sliders
- Visualizzazione Knowledge Graph
- Metriche base (entità, query, utenti)

**✅ CHECKPOINT 9**: CRUD entità funziona da dashboard

---

### FASE 10: OBSERVABILITY
**Prerequisiti**: Fase 9 completata

**FILE DA CREARE**:
1. `backend/app/observability/telemetry.py` - OpenTelemetry
2. `backend/app/observability/langfuse.py` - LLM tracing
3. `backend/app/observability/metrics.py` - Prometheus metrics
4. `infrastructure/kubernetes/monitoring/` - Grafana + Prometheus

**METRICHE OBBLIGATORIE**:
```python
# Da tracciare
- arbor_query_latency_seconds (histogram)
- arbor_cache_hits_total (counter)
- arbor_llm_tokens_used (counter)
- arbor_active_users (gauge)
```

**✅ CHECKPOINT 10**: Traces visibili in Langfuse, metriche in Grafana

---

### FASE 11: EVENTS & ML
**Prerequisiti**: Fase 10 completata

**FILE DA CREARE**:
1. `backend/app/events/producer.py` - Kafka producer
2. `backend/app/events/consumers/analytics.py`
3. `backend/app/events/consumers/ml_feedback.py`
4. `backend/app/ml/reranker.py` - Cohere + custom
5. `backend/app/ml/feedback_loop.py`

**EVENTI DA EMETTERE**:
```python
- entity.created
- entity.updated
- search.performed (query, results, latency)
- user.clicked (query, result_id, position)
- user.converted (query, result_id)
```

**✅ CHECKPOINT 11**: Eventi fluiscono in Kafka, consumer processa

---

### FASE 12: DEPLOY & SECURITY
**Prerequisiti**: Fase 11 completata

**FILE DA CREARE**:
1. `infrastructure/terraform/main.tf`
2. `infrastructure/kubernetes/deployments/*.yaml`
3. `infrastructure/docker/Dockerfile.api`
4. `.github/workflows/ci.yml`
5. `.github/workflows/cd.yml`

**SECURITY CHECKLIST**:
```
[ ] Secrets in environment variables, non in codice
[ ] HTTPS only
[ ] CORS configurato
[ ] Rate limiting attivo
[ ] Input validation su tutti gli endpoint
[ ] SQL injection prevention (parametrized queries)
[ ] Auth0 configurato con RBAC
```

**✅ CHECKPOINT 12**: Deploy su staging funziona, security scan passa

---

## 🔄 VALIDAZIONE FINALE

```bash
# Esegui tutti i test
cd backend && poetry run pytest --cov=app tests/
cd ../frontend && npm run test

# Build produzione
docker-compose -f docker-compose.prod.yml build

# Health check
curl https://api.staging.arbor.io/health
# Expected: {"status": "healthy", "version": "1.0.0"}

# Test discovery end-to-end
curl -X POST https://api.staging.arbor.io/api/v1/discover \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "Cerco un posto elegante", "limit": 3}'
# Expected: JSON con recommendations
```

---

## 📊 CRITERI DI COMPLETAMENTO

| Criterio | Requisito |
|----------|-----------|
| **Copertura Test** | >80% backend |
| **Latency P95** | <2.5 secondi |
| **Uptime Staging** | 99% su 24h |
| **Documentazione** | README + API docs |
| **Security** | 0 vulnerabilità critiche |
| **Build** | CI/CD green |

---

## 🚨 REGOLE PER L'AI EXECUTOR

1. **SEQUENZIALE**: Completa ogni fase prima di passare alla successiva
2. **CHECKPOINT**: Non procedere se il checkpoint fallisce
3. **TEST FIRST**: Scrivi test prima dell'implementazione dove possibile
4. **DOCUMENTA**: Ogni file deve avere docstring/commenti
5. **CHIEDI**: Se mancano informazioni (API keys, domain specifico), chiedi all'utente
6. **COMMIT**: Commit atomici alla fine di ogni fase

```bash
git add .
git commit -m "feat(fase-X): descrizione completamento fase"
```

---

> **INIZIO ESECUZIONE**: L'AI deve iniziare dalla FASE 0 e procedere sequenzialmente.
