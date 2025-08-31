# Reporte de Comparación de Modelos de Embeddings

**Fecha**: 2025-08-03 14:47:01

## Modelo Base

**Nombre**: all-MiniLM-base

| Métrica | Valor |
|---------|-------|
| Accuracy@1 | 33.3% |
| Accuracy@5 | 52.1% |
| MRR | 0.406 |
| Similitud promedio | 54.6% |
| Velocidad | 0.0036s/query |

## Ranking Completo de Modelos

| # | Modelo | Acc@1 | vs Base | Acc@5 | MRR | Sim Avg | Vel (s) |
|---|--------|-------|---------|-------|-----|---------|-------|
| 1 | sapbert-umls/model-0_0029 | 64.6% | +31.3pp | 77.1% | 0.692 | 51.5% | 0.0059 |
| 2 | biomedical-roberta/model-0_0007 | 62.5% | +29.2pp | 66.7% | 0.642 | 56.8% | 0.0055 |
| 3 | sapbert-umls/model-0_0001 | 62.5% | +29.2pp | 75.0% | 0.674 | 52.1% | 0.0055 |
| 4 | sapbert-umls/model-0_0009 | 62.5% | +29.2pp | 75.0% | 0.678 | 51.5% | 0.0050 |
| 5 | sapbert-umls/model-0_0183 | 62.5% | +29.2pp | 77.1% | 0.677 | 51.3% | 0.0052 |
| 6 | sapbert-umls/model-0_0853 | 62.5% | +29.2pp | 81.2% | 0.685 | 48.7% | 0.0049 |
| 7 | biomedical-agressive/model-0_1000 | 60.4% | +27.1pp | 68.8% | 0.639 | 53.5% | 0.0049 |
| 8 | biomedical-roberta/model-0_0000 | 60.4% | +27.1pp | 68.8% | 0.630 | 59.5% | 0.0050 |
| 9 | biomedical-roberta/model-0_0000_1 | 60.4% | +27.1pp | 68.8% | 0.626 | 55.4% | 0.0047 |
| 10 | biomedical-roberta/model-0_0001 | 60.4% | +27.1pp | 68.8% | 0.630 | 59.5% | 0.0048 |
| 11 | sapbert-umls/model-0_0000_1 | 60.4% | +27.1pp | 72.9% | 0.660 | 52.1% | 0.0053 |
| 12 | sapbert-umls/model-0_2688 | 60.4% | +27.1pp | 79.2% | 0.670 | 48.7% | 0.0052 |
| 13 | sapbert-umls/model-0_4735 | 60.4% | +27.1pp | 79.2% | 0.678 | 58.0% | 0.0055 |
| 14 | pubmedbert-marco/model-0_1304 | 58.3% | +25.0pp | 75.0% | 0.626 | 57.1% | 0.0047 |
| 15 | sapbert-umls/model-0_0000 | 58.3% | +25.0pp | 72.9% | 0.651 | 52.0% | 0.0058 |
| 16 | sapbert-umls/model-0_0002 | 58.3% | +25.0pp | 75.0% | 0.650 | 52.1% | 0.0051 |
| 17 | all-mini-fine/model-0_0000 | 56.2% | +22.9pp | 60.4% | 0.583 | 58.0% | 0.0031 |
| 18 | all-mini-fine/model-0_0004 | 56.2% | +22.9pp | 60.4% | 0.576 | 52.2% | 0.0032 |
| 19 | all-mini-fine/model-0_0016 | 56.2% | +22.9pp | 58.3% | 0.568 | 54.5% | 0.0031 |
| 20 | pubmedbert-marco/model-0_0071 | 56.2% | +22.9pp | 60.4% | 0.574 | 63.6% | 0.0055 |
| 21 | all-mini-fine/model-0_0021 | 54.2% | +20.8pp | 62.5% | 0.565 | 51.6% | 0.0031 |
| 22 | all-mini-fine/model-0_0072 | 54.2% | +20.8pp | 60.4% | 0.567 | 54.1% | 0.0037 |
| 23 | all-mini-fine/model-0_0174 | 54.2% | +20.8pp | 60.4% | 0.560 | 53.2% | 0.0033 |
| 24 | biomedical-roberta/model-0_0035 | 54.2% | +20.8pp | 66.7% | 0.590 | 58.8% | 0.0049 |
| 25 | pubmedbert-marco/model-0_0000_1 | 54.2% | +20.8pp | 58.3% | 0.556 | 64.1% | 0.0053 |
| 26 | pubmedbert-marco/model-0_0006 | 54.2% | +20.8pp | 58.3% | 0.556 | 64.1% | 0.0050 |
| 27 | pubmedbert-marco/model-0_0007 | 54.2% | +20.8pp | 56.2% | 0.552 | 65.4% | 0.0049 |
| 28 | pubmedbert-marco/model-0_0019 | 54.2% | +20.8pp | 58.3% | 0.556 | 64.2% | 0.0058 |
| 29 | pubmedbert-marco/model-0_0020 | 54.2% | +20.8pp | 58.3% | 0.557 | 64.4% | 0.0054 |
| 30 | pubmedbert-marco/model-0_0035 | 54.2% | +20.8pp | 62.5% | 0.568 | 62.1% | 0.0055 |
| 31 | pubmedbert-marco/model-0_0434 | 54.2% | +20.8pp | 64.6% | 0.566 | 61.3% | 0.0059 |
| 32 | pubmedbert-marco/model-0_3372 | 54.2% | +20.8pp | 72.9% | 0.612 | 52.3% | 0.0053 |
| 33 | all-mini-fine/model-0_0001 | 52.1% | +18.8pp | 60.4% | 0.562 | 56.4% | 0.0032 |
| 34 | all-mini-fine/model-0_0438 | 52.1% | +18.8pp | 62.5% | 0.556 | 50.9% | 0.0031 |
| 35 | medcpt/model-0_0000_1 | 52.1% | +18.8pp | 62.5% | 0.554 | 68.4% | 0.0053 |
| 36 | medcpt/model-0_0147 | 52.1% | +18.8pp | 62.5% | 0.554 | 68.4% | 0.0051 |
| 37 | medcpt/model-0_0177 | 52.1% | +18.8pp | 62.5% | 0.554 | 68.3% | 0.0054 |
| 38 | medcpt/model-0_0203 | 52.1% | +18.8pp | 62.5% | 0.551 | 68.6% | 0.0053 |
| 39 | medcpt/model-0_0289 | 52.1% | +18.8pp | 62.5% | 0.552 | 68.8% | 0.0053 |
| 40 | pubmedbert-marco/model-0_0160 | 52.1% | +18.8pp | 64.6% | 0.565 | 60.4% | 0.0053 |
| 41 | sapbert-pubmed/model-0_0000 | 52.1% | +18.8pp | 58.3% | 0.541 | 59.2% | 0.0052 |
| 42 | sapbert-pubmed/model-0_0002 | 52.1% | +18.8pp | 56.2% | 0.532 | 60.5% | 0.0054 |
| 43 | sapbert-pubmed/model-0_0005 | 52.1% | +18.8pp | 56.2% | 0.535 | 63.6% | 0.0054 |
| 44 | biomedical-agressive/model-0_0079 | 50.0% | +16.7pp | 68.8% | 0.578 | 74.1% | 0.0055 |
| 45 | medcpt/model-0_0366 | 50.0% | +16.7pp | 64.6% | 0.543 | 68.6% | 0.0057 |
| 46 | medcpt/model-0_0589 | 50.0% | +16.7pp | 64.6% | 0.540 | 69.2% | 0.0053 |
| 47 | medcpt/model-0_0833 | 50.0% | +16.7pp | 62.5% | 0.539 | 70.4% | 0.0057 |
| 48 | paraphrase-agressive/model-0_0113 | 50.0% | +16.7pp | 54.2% | 0.521 | 68.0% | 0.0053 |
| 49 | sapbert-pubmed/model-0_0000_1 | 50.0% | +16.7pp | 60.4% | 0.531 | 58.6% | 0.0049 |
| 50 | sapbert-pubmed/model-0_0004 | 50.0% | +16.7pp | 58.3% | 0.518 | 60.2% | 0.0060 |
| 51 | sapbert-pubmed/model-0_0006 | 50.0% | +16.7pp | 56.2% | 0.525 | 61.0% | 0.0057 |
| 52 | sapbert-pubmed/model-0_0012 | 50.0% | +16.7pp | 60.4% | 0.528 | 62.0% | 0.0055 |
| 53 | sapbert-pubmed/model-0_0020 | 50.0% | +16.7pp | 58.3% | 0.527 | 59.1% | 0.0046 |
| 54 | sapbert-pubmed/model-0_0043 | 50.0% | +16.7pp | 54.2% | 0.517 | 66.7% | 0.0048 |
| 55 | all-mini-fine/model-0_0878 | 47.9% | +14.6pp | 60.4% | 0.522 | 50.3% | 0.0032 |
| 56 | biomedical-agressive/model-0_0088 | 47.9% | +14.6pp | 64.6% | 0.542 | 63.5% | 0.0050 |
| 57 | medcpt/model-0_1271 | 47.9% | +14.6pp | 58.3% | 0.516 | 71.8% | 0.0052 |
| 58 | sapbert-pubmed/model-0_0034 | 47.9% | +14.6pp | 58.3% | 0.512 | 59.2% | 0.0047 |
| 59 | sapbert-pubmed/model-0_0209 | 47.9% | +14.6pp | 58.3% | 0.515 | 62.1% | 0.0047 |
| 60 | sapbert-pubmed/model-0_0339 | 47.9% | +14.6pp | 52.1% | 0.494 | 60.2% | 0.0055 |
| 61 | sapbert-pubmed/model-0_0926 | 47.9% | +14.6pp | 54.2% | 0.494 | 57.8% | 0.0068 |
| 62 | sapbert-pubmed/model-0_0001 | 45.8% | +12.5pp | 54.2% | 0.484 | 61.3% | 0.0057 |
| 63 | sapbert-pubmed/model-0_0003 | 45.8% | +12.5pp | 56.2% | 0.488 | 60.7% | 0.0053 |
| 64 | sapbert-pubmed/model-0_0126 | 45.8% | +12.5pp | 56.2% | 0.492 | 59.0% | 0.0049 |
| 65 | sapbert-pubmed/model-0_2160 | 43.8% | +10.4pp | 58.3% | 0.476 | 56.8% | 0.0059 |
| 66 | sapbert-pubmed/model-0_4277 | 43.8% | +10.4pp | 60.4% | 0.500 | 56.7% | 0.0048 |
| 67 | all-mini-fine/model-0_1619 | 41.7% | +8.3pp | 60.4% | 0.472 | 52.3% | 0.0031 |
| 68 | biomedical-roberta/model-0_0157 | 41.7% | +8.3pp | 62.5% | 0.492 | 53.2% | 0.0050 |
| 69 | medcpt/model-0_2250 | 41.7% | +8.3pp | 62.5% | 0.487 | 71.8% | 0.0054 |
| 70 | sapbert-pubmed/model-0_1608 | 41.7% | +8.3pp | 56.2% | 0.473 | 58.4% | 0.0053 |
| 71 | pubmedbert-ms/model-0_0001 | 39.6% | +6.2pp | 52.1% | 0.450 | 75.4% | 0.0048 |
| 72 | pubmedbert-ms/model-0_0003 | 39.6% | +6.2pp | 58.3% | 0.462 | 75.5% | 0.0048 |
| 73 | pubmedbert-ms/model-0_0007 | 39.6% | +6.2pp | 50.0% | 0.438 | 73.1% | 0.0051 |
| 74 | roberta-catalan/model-0_0000 | 39.6% | +6.2pp | 47.9% | 0.428 | 59.6% | 0.0053 |
| 75 | roberta-catalan/model-0_0000_1 | 39.6% | +6.2pp | 47.9% | 0.425 | 59.9% | 0.0054 |
| 76 | biolinkbert/model-0_0001 | 37.5% | +4.2pp | 54.2% | 0.440 | 62.0% | 0.0050 |
| 77 | biolinkbert/model-0_0003 | 37.5% | +4.2pp | 54.2% | 0.439 | 60.7% | 0.0055 |
| 78 | medcpt/model-0_4646 | 37.5% | +4.2pp | 58.3% | 0.458 | 79.5% | 0.0052 |
| 79 | pubmedbert-ms/model-0_0000_1 | 37.5% | +4.2pp | 56.2% | 0.449 | 71.0% | 0.0051 |
| 80 | pubmedbert-ms/model-0_0006 | 37.5% | +4.2pp | 52.1% | 0.436 | 76.6% | 0.0055 |
| 81 | pubmedbert-ms/model-0_0009 | 37.5% | +4.2pp | 56.2% | 0.445 | 72.0% | 0.0049 |
| 82 | roberta-catalan/model-0_0005 | 37.5% | +4.2pp | 50.0% | 0.414 | 58.9% | 0.0051 |
| 83 | bioclinicalbert/model-0_0005 | 35.4% | +2.1pp | 60.4% | 0.449 | 71.5% | 0.0054 |
| 84 | biolinkbert/model-0_0000 | 35.4% | +2.1pp | 54.2% | 0.426 | 62.2% | 0.0050 |
| 85 | biolinkbert/model-0_0000_0 | 35.4% | +2.1pp | 52.1% | 0.425 | 63.4% | 0.0054 |
| 86 | biolinkbert/model-0_0013 | 35.4% | +2.1pp | 52.1% | 0.417 | 63.9% | 0.0047 |
| 87 | pubmedbert-ms/model-0_0000 | 35.4% | +2.1pp | 54.2% | 0.431 | 71.8% | 0.0054 |
| 88 | pubmedbert-ms/model-0_0024 | 35.4% | +2.1pp | 50.0% | 0.416 | 73.2% | 0.0055 |
| 89 | roberta-catalan/model-0_0001 | 35.4% | +2.1pp | 45.8% | 0.389 | 59.7% | 0.0057 |
| 90 | bioclinicalbert/model-0_0000 | 33.3% | 0.0pp | 60.4% | 0.434 | 72.0% | 0.0054 |
| 91 | bioclinicalbert/model-0_0000_1 | 33.3% | 0.0pp | 62.5% | 0.438 | 69.1% | 0.0051 |
| 92 | biolinkbert/model-0_0066 | 33.3% | 0.0pp | 54.2% | 0.406 | 61.6% | 0.0051 |
| 93 | paraphrase-agressive/model-0_0857 | 33.3% | 0.0pp | 58.3% | 0.415 | 53.7% | 0.0056 |
| 94 | roberta-catalan/model-0_0007 | 33.3% | 0.0pp | 43.8% | 0.368 | 59.9% | 0.0051 |
| 95 | bioclinicalbert/model-0_0002 | 31.2% | -2.1pp | 62.5% | 0.425 | 72.4% | 0.0052 |
| 96 | bioclinicalbert/model-0_0024 | 31.2% | -2.1pp | 68.8% | 0.437 | 64.1% | 0.0050 |
| 97 | biolinkbert/model-0_0527 | 31.2% | -2.1pp | 47.9% | 0.366 | 59.7% | 0.0046 |
| 98 | biomedical-roberta/model-0_1455 | 31.2% | -2.1pp | 64.6% | 0.431 | 58.3% | 0.0048 |
| 99 | bioclinicalbert/model-0_0034 | 29.2% | -4.2pp | 54.2% | 0.390 | 67.6% | 0.0052 |
| 100 | pubmedbert-ms/model-0_0062 | 29.2% | -4.2pp | 52.1% | 0.376 | 69.7% | 0.0046 |
| 101 | biobert-v1.2/model-0_0004 | 27.1% | -6.2pp | 33.3% | 0.302 | 73.3% | 0.0051 |
| 102 | bioclinicalbert/model-0_0265 | 27.1% | -6.2pp | 62.5% | 0.400 | 69.3% | 0.0056 |
| 103 | roberta-catalan/model-0_0067 | 27.1% | -6.2pp | 43.8% | 0.333 | 58.3% | 0.0054 |
| 104 | roberta-catalan/model-0_1372 | 27.1% | -6.2pp | 39.6% | 0.313 | 57.7% | 0.0058 |
| 105 | bioclinicalbert/model-0_0008 | 25.0% | -8.3pp | 62.5% | 0.394 | 69.5% | 0.0049 |
| 106 | bluebert/model-0_0027 | 22.9% | -10.4pp | 27.1% | 0.245 | 68.4% | 0.0052 |
| 107 | scibert/model-0_0001 | 22.9% | -10.4pp | 33.3% | 0.265 | 63.5% | 0.0060 |
| 108 | biobert-v1.2/model-0_0000 | 20.8% | -12.5pp | 35.4% | 0.264 | 71.5% | 0.0058 |
| 109 | biobert-v1.2/model-0_0000_1 | 20.8% | -12.5pp | 35.4% | 0.263 | 71.6% | 0.0055 |
| 110 | biobert-v1.2/model-0_0001 | 20.8% | -12.5pp | 35.4% | 0.263 | 71.4% | 0.0058 |
| 111 | biobert-v1.2/model-0_0002 | 20.8% | -12.5pp | 33.3% | 0.259 | 74.3% | 0.0054 |
| 112 | bluebert/model-0_0001 | 20.8% | -12.5pp | 29.2% | 0.232 | 67.9% | 0.0048 |
| 113 | bluebert/model-0_0014 | 20.8% | -12.5pp | 31.2% | 0.248 | 66.8% | 0.0050 |
| 114 | pubmedbert-ms/model-0_0175 | 20.8% | -12.5pp | 52.1% | 0.322 | 71.3% | 0.0051 |
| 115 | pubmedbert-ms/model-0_1013 | 20.8% | -12.5pp | 43.8% | 0.296 | 72.7% | 0.0051 |
| 116 | biobert-v1.2/model-0_0016 | 18.8% | -14.6pp | 31.2% | 0.243 | 79.3% | 0.0050 |
| 117 | bioclinicalbert/model-0_0017 | 18.8% | -14.6pp | 50.0% | 0.301 | 68.6% | 0.0048 |
| 118 | bluebert/model-0_0000 | 18.8% | -14.6pp | 25.0% | 0.215 | 67.8% | 0.0051 |
| 119 | bluebert/model-0_0000_1 | 18.8% | -14.6pp | 27.1% | 0.214 | 67.9% | 0.0053 |
| 120 | bluebert/model-0_0118 | 18.8% | -14.6pp | 33.3% | 0.239 | 63.2% | 0.0048 |
| 121 | bluebert/model-0_0020 | 16.7% | -16.7pp | 31.2% | 0.219 | 61.8% | 0.0052 |
| 122 | scibert/model-0_0000_1 | 16.7% | -16.7pp | 35.4% | 0.233 | 59.1% | 0.0057 |
| 123 | scibert/model-0_0000 | 14.6% | -18.7pp | 33.3% | 0.220 | 61.8% | 0.0052 |
| 124 | scibert/model-0_0002 | 14.6% | -18.7pp | 29.2% | 0.206 | 61.4% | 0.0055 |
| 125 | scibert/model-0_0004 | 14.6% | -18.7pp | 29.2% | 0.198 | 59.9% | 0.0059 |
| 126 | scibert/model-0_0017 | 14.6% | -18.7pp | 35.4% | 0.227 | 58.5% | 0.0052 |
| 127 | biobert-v1.2/model-0_0065 | 12.5% | -20.8pp | 29.2% | 0.180 | 72.6% | 0.0050 |
| 128 | bioclinicalbert/model-0_0003 | 12.5% | -20.8pp | 43.8% | 0.226 | 65.2% | 0.0048 |
| 129 | bioclinicalbert/model-0_0032 | 12.5% | -20.8pp | 37.5% | 0.216 | 68.8% | 0.0051 |
| 130 | biolinkbert/model-0_2622 | 12.5% | -20.8pp | 39.6% | 0.217 | 70.0% | 0.0056 |
| 131 | bluebert/model-0_0310 | 12.5% | -20.8pp | 25.0% | 0.173 | 64.9% | 0.0054 |
| 132 | scibert/model-0_0126 | 12.5% | -20.8pp | 37.5% | 0.223 | 56.7% | 0.0058 |
| 133 | biobert/model-0_0000 | 8.3% | -25.0pp | 31.2% | 0.151 | 68.6% | 0.0052 |
| 134 | biobert/model-0_0000_1 | 8.3% | -25.0pp | 31.2% | 0.151 | 69.4% | 0.0053 |
| 135 | biobert/model-0_0001 | 8.3% | -25.0pp | 31.2% | 0.151 | 68.7% | 0.0051 |
| 136 | biobert/model-0_0002 | 8.3% | -25.0pp | 35.4% | 0.165 | 67.6% | 0.0056 |
| 137 | biobert-v1.2/model-0_1077 | 8.3% | -25.0pp | 27.1% | 0.142 | 78.3% | 0.0051 |
| 138 | bioclinicalbert/model-0_0001 | 8.3% | -25.0pp | 43.8% | 0.201 | 59.7% | 0.0051 |
| 139 | bioclinicalbert/model-0_0107 | 8.3% | -25.0pp | 39.6% | 0.176 | 68.2% | 0.0057 |
| 140 | bioclinicalbert/model-0_0340 | 8.3% | -25.0pp | 27.1% | 0.130 | 60.5% | 0.0058 |
| 141 | bioclinicalbert/model-0_1554 | 8.3% | -25.0pp | 31.2% | 0.140 | 68.5% | 0.0053 |
| 142 | bluebert/model-0_1034 | 8.3% | -25.0pp | 22.9% | 0.139 | 65.0% | 0.0051 |
| 143 | bluebert/model-0_2036 | 8.3% | -25.0pp | 18.8% | 0.114 | 80.5% | 0.0054 |
| 144 | deberta-v3-conservative/model-0_0010 | 8.3% | -25.0pp | 14.6% | 0.106 | 81.9% | 0.0127 |
| 145 | biobert/model-0_0004 | 6.2% | -27.1pp | 35.4% | 0.156 | 68.5% | 0.0045 |
| 146 | biobert/model-0_0015 | 6.2% | -27.1pp | 35.4% | 0.152 | 71.0% | 0.0052 |
| 147 | biobert/model-0_0073 | 6.2% | -27.1pp | 27.1% | 0.129 | 75.3% | 0.0048 |
| 148 | biobert/model-0_0292 | 6.2% | -27.1pp | 35.4% | 0.139 | 73.4% | 0.0051 |
| 149 | biobert/model-0_1459 | 6.2% | -27.1pp | 22.9% | 0.113 | 75.4% | 0.0048 |
| 150 | biobert-v1.2/model-0_0173 | 6.2% | -27.1pp | 29.2% | 0.128 | 75.0% | 0.0051 |
| 151 | deberta-v3-conservative/model-0_0027 | 6.2% | -27.1pp | 14.6% | 0.092 | 83.2% | 0.0133 |
| 152 | deberta-v3-conservative/model-0_0095 | 6.2% | -27.1pp | 10.4% | 0.072 | 86.5% | 0.0119 |
| 153 | electra-agressive/model-0_0000 | 6.2% | -27.1pp | 12.5% | 0.082 | 100.0% | 0.0062 |
| 154 | scibert/model-0_0929 | 6.2% | -27.1pp | 41.7% | 0.173 | 55.2% | 0.0053 |
| 155 | scibert/model-0_2747 | 6.2% | -27.1pp | 29.2% | 0.123 | 64.4% | 0.0052 |
| 156 | biobert/model-0_0020 | 4.2% | -29.2pp | 35.4% | 0.136 | 70.2% | 0.0057 |
| 157 | bioclinicalbert/model-0_1698 | 4.2% | -29.2pp | 31.2% | 0.116 | 71.4% | 0.0049 |
| 158 | bioclinicalbert/model-0_2932 | 4.2% | -29.2pp | 22.9% | 0.101 | 80.6% | 0.0057 |
| 159 | biomedical-agressive/model-0_0000 | 4.2% | -29.2pp | 10.4% | 0.069 | 100.0% | 0.0060 |
| 160 | bluebert/model-0_2849 | 4.2% | -29.2pp | 16.7% | 0.089 | 86.3% | 0.0058 |
| 161 | deberta-v3-conservative/model-0_0000 | 4.2% | -29.2pp | 16.7% | 0.081 | 81.3% | 0.0136 |
| 162 | deberta-v3-conservative/model-0_0000_1 | 4.2% | -29.2pp | 14.6% | 0.078 | 81.1% | 0.0132 |
| 163 | deberta-v3-conservative/model-0_0001 | 4.2% | -29.2pp | 14.6% | 0.078 | 82.3% | 0.0121 |
| 164 | deberta-v3-conservative/model-0_0002 | 4.2% | -29.2pp | 12.5% | 0.075 | 81.0% | 0.0132 |
| 165 | deberta-v3-conservative/model-0_0004 | 4.2% | -29.2pp | 16.7% | 0.092 | 82.4% | 0.0125 |
| 166 | deberta-v3-conservative/model-0_0213 | 4.2% | -29.2pp | 10.4% | 0.066 | 86.3% | 0.0136 |
| 167 | deberta-v3-conservative/model-0_0363 | 4.2% | -29.2pp | 14.6% | 0.079 | 85.5% | 0.0118 |
| 168 | deberta-v3-conservative/model-0_0993 | 4.2% | -29.2pp | 20.8% | 0.097 | 93.5% | 0.0131 |
| 169 | electra-agressive/model-0_1317 | 4.2% | -29.2pp | 18.8% | 0.088 | 68.6% | 0.0071 |
| 170 | electra-agressive/model-0_1767 | 4.2% | -29.2pp | 18.8% | 0.094 | 83.6% | 0.0066 |
| 171 | biobert-v1.2/model-0_2813 | 2.1% | -31.2pp | 25.0% | 0.079 | 80.8% | 0.0049 |
| 172 | deberta-v3-conservative/model-0_0019 | 2.1% | -31.2pp | 18.8% | 0.085 | 80.1% | 0.0134 |
| 173 | deberta-v3-conservative/model-0_0041 | 2.1% | -31.2pp | 14.6% | 0.077 | 81.6% | 0.0128 |
| 174 | deberta-v3-conservative/model-0_0057 | 2.1% | -31.2pp | 18.8% | 0.078 | 78.8% | 0.0132 |
| 175 | deberta-v3-conservative/model-0_0060 | 2.1% | -31.2pp | 16.7% | 0.065 | 80.4% | 0.0131 |
| 176 | deberta-v3-conservative/model-0_0678 | 2.1% | -31.2pp | 20.8% | 0.084 | 86.4% | 0.0129 |
| 177 | pubmedbert-ms/model-0_2918 | 2.1% | -31.2pp | 29.2% | 0.100 | 82.7% | 0.0047 |
| 178 | biobert/model-0_2945 | 0.0% | -33.3pp | 20.8% | 0.078 | 81.6% | 0.0051 |
| 179 | deberta-v3-conservative/model-0_0123 | 0.0% | -33.3pp | 14.6% | 0.057 | 84.3% | 0.0130 |
| 180 | deberta-v3-conservative/model-0_0950 | 0.0% | -33.3pp | 20.8% | 0.065 | 90.0% | 0.0131 |
| 181 | paraphrase-agressive/model-0_0000 | 0.0% | -33.3pp | 16.7% | 0.059 | 100.0% | 0.0054 |

## Top 5 Mejores Modelos - Análisis Detallado

### 1. sapbert-umls/model-0_0029

**Ruta**: `models\sapbert-umls\model-0_0029`

| Métrica | Valor | Comparación con Base |
|---------|-------|---------------------|
| Accuracy@1 | 64.6% | +31.3 pp |
| Accuracy@5 | 77.1% | - |
| MRR | 0.692 | +0.286 |
| Velocidad | 0.0059s | 0.60x más lento |
| Queries no encontradas | 11/48 | - |

### 2. biomedical-roberta/model-0_0007

**Ruta**: `models\biomedical-roberta\model-0_0007`

| Métrica | Valor | Comparación con Base |
|---------|-------|---------------------|
| Accuracy@1 | 62.5% | +29.2 pp |
| Accuracy@5 | 66.7% | - |
| MRR | 0.642 | +0.236 |
| Velocidad | 0.0055s | 0.65x más lento |
| Queries no encontradas | 16/48 | - |

### 3. sapbert-umls/model-0_0001

**Ruta**: `models\sapbert-umls\model-0_0001`

| Métrica | Valor | Comparación con Base |
|---------|-------|---------------------|
| Accuracy@1 | 62.5% | +29.2 pp |
| Accuracy@5 | 75.0% | - |
| MRR | 0.674 | +0.268 |
| Velocidad | 0.0055s | 0.65x más lento |
| Queries no encontradas | 12/48 | - |

### 4. sapbert-umls/model-0_0009

**Ruta**: `models\sapbert-umls\model-0_0009`

| Métrica | Valor | Comparación con Base |
|---------|-------|---------------------|
| Accuracy@1 | 62.5% | +29.2 pp |
| Accuracy@5 | 75.0% | - |
| MRR | 0.678 | +0.272 |
| Velocidad | 0.0050s | 0.72x más lento |
| Queries no encontradas | 12/48 | - |

### 5. sapbert-umls/model-0_0183

**Ruta**: `models\sapbert-umls\model-0_0183`

| Métrica | Valor | Comparación con Base |
|---------|-------|---------------------|
| Accuracy@1 | 62.5% | +29.2 pp |
| Accuracy@5 | 77.1% | - |
| MRR | 0.677 | +0.271 |
| Velocidad | 0.0052s | 0.69x más lento |
| Queries no encontradas | 11/48 | - |

