# Dedektimi i manipulimeve me Inteligjencë Artificiale në provat multimediale në sistemin gjyqësor

Tema e diplomës — Inxhinieri Telekomunikacioni, UPT-FTI.

## Struktura

- `src/data_pipeline/` — nxjerrja dhe përgatitja e imazheve nga datasetet (vetëm për trajnim)
- `src/integrity/` — hash, validim skedari, provenance (vetëm për inference real)
- `src/forensic_layer/` — ELA, CFA, DCT → S_forensic
- `src/dl_layer/` — CNN classifier → S_CNN
- `src/frequency_layer/` — FFT → S_frequency
- `src/fusion/` — logistic fusion, kalibrim, pragje (T1/T2)
- `src/explainability/` — Grad-CAM, ELA+Grad-CAM overlay
- `src/degradation/` — simulim kompresimi/rrjete sociale
- `src/reporting/` — raporti forenzik, chain of custody
- `src/training_pipeline.py` — orchestrator për trajnim mbi datasete
- `src/inference_pipeline.py` — orchestrator për një foto reale (evidence)

## Status

Struktura e projektit e finalizuar. Implementimi ende s'ka filluar.
Fillimi rekomanduar: `src/data_pipeline/video_level_split.py`.
