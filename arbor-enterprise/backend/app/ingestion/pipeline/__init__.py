"""5-Layer Enterprise Enrichment Pipeline.

Layer 1: Multi-source data collection (collector.py)
Layer 2: Parallel fact-based analysis (text_analyzer, vision_analyzer, price_analyzer, context_analyzer)
Layer 3: Calibrated scoring engine (scoring_engine.py) + Gold Standard (gold_standard.py)
Layer 4: Confidence scoring & disagreement detection (confidence.py)
Layer 5: Continuous learning & feedback loop (feedback_loop.py)

Orchestrated by: enrichment_orchestrator.py
"""
