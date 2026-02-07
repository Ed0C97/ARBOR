# ARBOR Test Sandbox

Infrastruttura di test diagnostica **non committata** per analisi completa del sistema.

## Quick Start

```bash
# Installa dipendenze
pip install -r requirements.txt

# Esegui diagnostica completa
python run_all_diagnostics.py

# Esegui solo un layer specifico
python -m pytest diagnostics/layer1_input_diagnostics.py -v
python -m pytest diagnostics/layer2_profile_diagnostics.py -v
python -m pytest diagnostics/layer3_ml_diagnostics.py -v
python -m pytest diagnostics/layer4_api_diagnostics.py -v

# Esegui benchmarks
python -m pytest benchmarks/ --benchmark-only
```

## Struttura

```
tests_sandbox/
├── data_generators/    # Generatori dati realistici (Faker + Hypothesis)
├── diagnostics/        # Test diagnostici per ogni layer
├── benchmarks/         # Performance benchmarks
├── ml_evaluation/      # Metriche ML e training
├── reports/            # Report generati automaticamente
└── fixtures/           # Dati di test predefiniti
```

## Layers Diagnosticati

| Layer | Cosa Testa | File |
|-------|-----------|------|
| **1** | Input validation, JSON schema | `layer1_input_diagnostics.py` |
| **2** | Domain profile, calibrazione | `layer2_profile_diagnostics.py` |
| **3** | ML pipeline, embeddings | `layer3_ml_diagnostics.py` |
| **4** | API, integration | `layer4_api_diagnostics.py` |

## Report

Dopo l'esecuzione, i report vengono generati in `reports/`:
- `diagnostic_report.html` - Report interattivo
- `diagnostic_report.json` - Dati strutturati per automazione
