# ═══════════════════════════════════════════════════════════════════════════
# A.R.B.O.R. Enterprise - Local Startup Script (PowerShell)
# Avvia backend (Docker Compose) e frontend (Next.js) in parallelo
# ═══════════════════════════════════════════════════════════════════════════

Write-Host "`n"
Write-Host "  ╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║         A.R.B.O.R. Enterprise - Local Development            ║" -ForegroundColor Cyan
Write-Host "  ║  Advanced Reasoning By Ontological Rules                      ║" -ForegroundColor Cyan
Write-Host "  ╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host "`n"

# Set working directory
Set-Location -Path "$PSScriptRoot\arbor-enterprise" -ErrorAction Stop

Write-Host "⏳ Verifiche pre-avvio..." -ForegroundColor Yellow
Write-Host ""

# Check Docker
Write-Host "⦿ Verifica Docker..." -ForegroundColor Cyan
$dockerCheck = docker --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Docker non trovato. Assicurati che Docker Desktop sia installato." -ForegroundColor Red
    Read-Host "Premi Enter per chiudere"
    exit 1
}
Write-Host "✓ Docker OK" -ForegroundColor Green

# Check Node.js
Write-Host "⦿ Verifica Node.js..." -ForegroundColor Cyan
$nodeCheck = node --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Node.js non trovato. Assicurati che Node.js sia installato." -ForegroundColor Red
    Read-Host "Premi Enter per chiudere"
    exit 1
}
Write-Host "✓ Node.js OK" -ForegroundColor Green

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor DarkCyan
Write-Host ""

# Start Docker Compose
Write-Host "📦 Avvio servizi backend (Docker Compose)..." -ForegroundColor Cyan
Start-Process -NoNewWindow -FilePath "docker-compose" -ArgumentList "up -d" -PassThru | Out-Null

# Wait for services
Write-Host "⏳ Attesa 5 secondi per l'avvio dei servizi..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Install frontend dependencies if needed
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host ""
    Write-Host "📥 Installazione dipendenze frontend..." -ForegroundColor Cyan
    Set-Location "frontend"
    npm install
    Set-Location ".."
}

# Start Next.js frontend
Write-Host ""
Write-Host "🚀 Avvio frontend (Next.js)..." -ForegroundColor Cyan
Start-Process -FilePath "cmd" -ArgumentList "/k cd frontend && npm run dev"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "✓ Avvio completato!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Frontend:   http://localhost:3000" -ForegroundColor Magenta
Write-Host "📊 Neo4j:      http://localhost:7474 (user: neo4j, password: arbor_dev_password)" -ForegroundColor Magenta
Write-Host "🔍 Qdrant:     http://localhost:6333/docs" -ForegroundColor Magenta
Write-Host "⏱️  Temporal:   http://localhost:8088" -ForegroundColor Magenta
Write-Host "📦 PostgreSQL: localhost:5433" -ForegroundColor Magenta
Write-Host "💾 Redis:      localhost:6379" -ForegroundColor Magenta
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "💡 COMANDI UTILI:" -ForegroundColor Yellow
Write-Host "   docker-compose logs -f                   # Vedi i log dei servizi"
Write-Host "   docker-compose down                      # Ferma tutti i servizi"
Write-Host "   npm run build                            # Build di produzione (frontend)"
Write-Host ""
