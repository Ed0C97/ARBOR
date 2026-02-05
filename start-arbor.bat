@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM A.R.B.O.R. Enterprise - Local Startup Script
REM Avvia backend (Docker Compose) e frontend (Next.js) in parallelo
REM ═══════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

echo.
echo   ╔═════════════════════════════════════════════════════════════╗
echo   ║         A.R.B.O.R. Enterprise - Local Development           ║
echo   ║          Advanced Reasoning By Ontological Rules            ║
echo   ╚═════════════════════════════════════════════════════════════╝
echo.

REM Set working directory
cd /d "%~dp0arbor-enterprise" || (
    echo ✗ Errore: Impossibile accedere alla cartella arbor-enterprise
    pause
    exit /b 1
)

echo ⏳ Verifiche pre-avvio...
echo.

REM Check Docker
echo ⦿ Verifica Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ Docker non trovato. Assicurati che Docker Desktop sia installato.
    pause
    exit /b 1
)
echo ✓ Docker OK

REM Check Node.js
echo ⦿ Verifica Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ Node.js non trovato. Assicurati che Node.js sia installato.
    pause
    exit /b 1
)
echo ✓ Node.js OK

echo.
echo ═══════════════════════════════════════════════════════════════════════════
echo.

REM Start Docker Compose in background
echo 📦 Avvio servizi backend (Docker Compose)...
start "ARBOR Backend" cmd /k "docker-compose up -d && echo. && echo ✓ Backend avviato! Controlla: http://localhost:7474 (Neo4j) && pause"

REM Give Docker time to start
echo ⏳ Attesa 5 secondi per l'avvio dei servizi...
timeout /t 5 /nobreak

REM Install frontend dependencies if needed
if not exist "frontend\node_modules" (
    echo.
    echo 📥 Installazione dipendenze frontend...
    cd frontend
    call npm install
    cd ..
)

REM Start Next.js frontend
echo.
echo 🚀 Avvio frontend (Next.js)...
start "ARBOR Frontend" cmd /k "cd frontend && npm run dev && pause"

echo.
echo ═══════════════════════════════════════════════════════════════════════════
echo.
echo ✓ Avvio completato!
echo.
echo 🌐 Frontend:   http://localhost:3000
echo 📊 Neo4j:      http://localhost:7474 (user: neo4j, password: arbor_dev_password)
echo 🔍 Qdrant:     http://localhost:6333/docs
echo ⏱️  Temporal:   http://localhost:8088
echo 📦 PostgreSQL: localhost:5433
echo 💾 Redis:      localhost:6379
echo.
echo ═══════════════════════════════════════════════════════════════════════════
echo.
echo 💡 COMANDI UTILI:
echo    docker-compose logs -f                   # Vedi i log dei servizi
echo    docker-compose down                      # Ferma tutti i servizi
echo    npm run build                            # Build di produzione (frontend)
echo.
echo Premi un tasto per chiudere questa finestra...
pause
