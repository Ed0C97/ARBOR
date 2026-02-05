# 🚀 A.R.B.O.R. Enterprise - Guida Avvio Locale

## Requisiti

- **Docker Desktop** installato e in esecuzione
- **Node.js** 18+ installato
- **npm** o **yarn**

## ⚡ Avvio Rapido

### Opzione 1: Doppio Click (Facile)

1. **Windows CMD/Batch:**
   ```bash
   Double-click su: start-arbor.bat
   ```

2. **Windows PowerShell:**
   ```bash
   Right-click su: start-arbor.ps1 → Run with PowerShell
   ```

### Opzione 2: Manuale

#### Terminal 1 - Backend
```bash
cd arbor-enterprise
docker-compose up -d
```

#### Terminal 2 - Frontend
```bash
cd arbor-enterprise/frontend
npm run dev
```

---

## 📋 Cosa è Stato Fatto

✅ **Font locali JetBrains Mono** - Tutti i 16 weight sono stati integrati nel progetto
- Da Google Fonts → Font locali da `/assets/fonts/`
- Supporto completo: Thin, Light, Regular, SemiBold, Bold, ExtraBold (+ Italic)

✅ **Script di avvio unico** - Lancia backend e frontend insieme
- **start-arbor.bat** - Per Windows CMD
- **start-arbor.ps1** - Per Windows PowerShell

✅ **Layout aggiornato** - Next.js font configuration
- File: `arbor-enterprise/frontend/app/layout.tsx`
- Font face mappati ai file .woff2 locali

---

## 🌐 Accesso ai Servizi

| Servizio | URL | Note |
|----------|-----|------|
| **Frontend** | http://localhost:3000 | Next.js Dev Server |
| **Neo4j Browser** | http://localhost:7474 | User: `neo4j` / Pass: `arbor_dev_password` |
| **Qdrant API** | http://localhost:6333/docs | Swagger UI per Vector Search |
| **Temporal UI** | http://localhost:8088 | Workflow Management |
| **PostgreSQL** | localhost:5433 | DB: `arbor_db` / User: `arbor` / Pass: `arbor_password` |
| **Redis** | localhost:6379 | Caching & Sessions |

---

## 📦 Struttura Servizi

```
Docker Compose avvia:
├── PostgreSQL (arbor_db) - Database principale
├── Qdrant - Vector search engine
├── Neo4j - Knowledge graph
├── Redis - Caching
├── Temporal - Workflow engine
└── Temporal UI - Dashboard Temporal
```

---

## 🛑 Fermare i Servizi

```bash
cd arbor-enterprise
docker-compose down
```

Per pulire completamente (cancella volumi):
```bash
docker-compose down -v
```

---

## 📊 Comandi Utili

### Backend
```bash
# Vedere i log in tempo reale
docker-compose logs -f

# Log di un servizio specifico
docker-compose logs -f postgres-arbor
docker-compose logs -f neo4j
docker-compose logs -f qdrant

# Restart servizio specifico
docker-compose restart postgres-arbor

# Stop e start manuale
docker-compose stop
docker-compose start
```

### Frontend
```bash
# Development server
npm run dev

# Build di produzione
npm run build

# Start production build
npm start

# Linting
npm run lint

# Audit dipendenze
npm audit

# Fix vulnerabilità automaticamente
npm audit fix
```

---

## 🔧 Troubleshooting

### Docker non parte
```bash
# Verifica Docker Desktop sia running
docker ps

# Check status
docker-compose ps

# Rebuild images
docker-compose build --no-cache
```

### Porta già in uso
Se una porta è occupata, modifica `docker-compose.yml`:
```yaml
ports:
  - "5433:5432"  # Cambia il primo numero (5433 → 5434)
```

### Node modules corrotto
```bash
cd arbor-enterprise/frontend
rm -r node_modules package-lock.json
npm install
```

### Font non caricano
1. Verifica che `/assets/fonts/` esista
2. Riavvia il dev server (`npm run dev`)
3. Svuota cache browser (Ctrl+Shift+R)

---

## 📝 Note Sviluppo

- **Frontend** è in **hot reload** - Salva e il browser si aggiorna automaticamente
- **Backend** servizi sono containerizzati - Riavvi con `docker-compose restart`
- **Font JetBrains Mono** sono locali - Nessuna dipendenza da Google Fonts per i monospace
- **Database** persiste in volumi Docker - I dati rimangono tra i riavvi

---

## 🎯 Prossimi Passi

1. Apri http://localhost:3000
2. Esplora Neo4j Dashboard: http://localhost:7474
3. Controlla i workflow in Temporal UI: http://localhost:8088
4. Inizia a sviluppare! 🚀

---

**Creato:** 2026-02-04
**Ultima modifica:** `start-arbor.bat`, `start-arbor.ps1`, Font integration completata
