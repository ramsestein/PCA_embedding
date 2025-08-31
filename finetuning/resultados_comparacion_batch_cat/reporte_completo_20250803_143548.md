# Reporte de Comparación de Modelos de Embeddings

**Fecha**: 2025-08-03 14:35:48

## Modelo Base

**Nombre**: all-MiniLM-base

| Métrica | Valor |
|---------|-------|
| Accuracy@1 | 39.6% |
| Accuracy@5 | 64.6% |
| MRR | 0.492 |
| Similitud promedio | 50.9% |
| Velocidad | 0.0033s/query |

## Ranking Completo de Modelos

| # | Modelo | Acc@1 | vs Base | Acc@5 | MRR | Sim Avg | Vel (s) |
|---|--------|-------|---------|-------|-----|---------|-------|
| 1 | sapbert-umls/model-0_0000 | 70.8% | +31.3pp | 79.2% | 0.738 | 55.2% | 0.0057 |
| 2 | sapbert-umls/model-0_0000_1 | 70.8% | +31.3pp | 79.2% | 0.738 | 55.3% | 0.0050 |
| 3 | sapbert-umls/model-0_0001 | 70.8% | +31.3pp | 79.2% | 0.740 | 56.2% | 0.0052 |
| 4 | sapbert-umls/model-0_0009 | 70.8% | +31.3pp | 79.2% | 0.743 | 55.5% | 0.0054 |
| 5 | sapbert-umls/model-0_0183 | 70.8% | +31.3pp | 81.2% | 0.743 | 53.7% | 0.0051 |
| 6 | sapbert-umls/model-0_0002 | 68.8% | +29.2pp | 79.2% | 0.729 | 56.1% | 0.0049 |
| 7 | all-mini-fine/model-0_0000 | 66.7% | +27.1pp | 66.7% | 0.667 | 55.7% | 0.0036 |
| 8 | pubmedbert-marco/model-0_3372 | 66.7% | +27.1pp | 79.2% | 0.705 | 47.3% | 0.0052 |
| 9 | sapbert-umls/model-0_0029 | 66.7% | +27.1pp | 79.2% | 0.719 | 56.0% | 0.0051 |
| 10 | all-mini-fine/model-0_0072 | 64.6% | +25.0pp | 70.8% | 0.667 | 49.3% | 0.0032 |
| 11 | sapbert-umls/model-0_0853 | 64.6% | +25.0pp | 81.2% | 0.694 | 52.8% | 0.0050 |
| 12 | all-mini-fine/model-0_0001 | 60.4% | +20.8pp | 66.7% | 0.627 | 54.2% | 0.0043 |
| 13 | sapbert-umls/model-0_2688 | 60.4% | +20.8pp | 70.8% | 0.634 | 52.5% | 0.0051 |
| 14 | all-mini-fine/model-0_0174 | 58.3% | +18.8pp | 68.8% | 0.622 | 49.1% | 0.0037 |
| 15 | paraphrase-agressive/model-0_0113 | 58.3% | +18.8pp | 68.8% | 0.617 | 66.4% | 0.0053 |
| 16 | pubmedbert-marco/model-0_0434 | 58.3% | +18.8pp | 70.8% | 0.617 | 53.7% | 0.0049 |
| 17 | sapbert-pubmed/model-0_0005 | 58.3% | +18.8pp | 62.5% | 0.593 | 59.4% | 0.0052 |
| 18 | all-mini-fine/model-0_0438 | 56.2% | +16.7pp | 70.8% | 0.626 | 46.3% | 0.0032 |
| 19 | medcpt/model-0_0000_1 | 56.2% | +16.7pp | 75.0% | 0.637 | 67.6% | 0.0054 |
| 20 | medcpt/model-0_0147 | 56.2% | +16.7pp | 75.0% | 0.637 | 67.6% | 0.0053 |
| 21 | medcpt/model-0_0177 | 56.2% | +16.7pp | 72.9% | 0.633 | 68.1% | 0.0060 |
| 22 | medcpt/model-0_0203 | 56.2% | +16.7pp | 75.0% | 0.637 | 67.7% | 0.0058 |
| 23 | medcpt/model-0_0289 | 56.2% | +16.7pp | 75.0% | 0.638 | 67.8% | 0.0062 |
| 24 | medcpt/model-0_0366 | 56.2% | +16.7pp | 75.0% | 0.643 | 68.0% | 0.0059 |
| 25 | medcpt/model-0_0589 | 56.2% | +16.7pp | 75.0% | 0.647 | 68.3% | 0.0058 |
| 26 | medcpt/model-0_0833 | 56.2% | +16.7pp | 75.0% | 0.633 | 68.4% | 0.0052 |
| 27 | pubmedbert-marco/model-0_0000_1 | 56.2% | +16.7pp | 62.5% | 0.584 | 56.4% | 0.0049 |
| 28 | pubmedbert-marco/model-0_0006 | 56.2% | +16.7pp | 62.5% | 0.584 | 56.4% | 0.0049 |
| 29 | pubmedbert-marco/model-0_0007 | 56.2% | +16.7pp | 62.5% | 0.584 | 56.4% | 0.0050 |
| 30 | pubmedbert-marco/model-0_0019 | 56.2% | +16.7pp | 62.5% | 0.584 | 56.5% | 0.0059 |
| 31 | pubmedbert-marco/model-0_0020 | 56.2% | +16.7pp | 62.5% | 0.587 | 56.5% | 0.0058 |
| 32 | pubmedbert-marco/model-0_0071 | 56.2% | +16.7pp | 62.5% | 0.589 | 57.5% | 0.0058 |
| 33 | pubmedbert-marco/model-0_1304 | 56.2% | +16.7pp | 77.1% | 0.635 | 52.5% | 0.0059 |
| 34 | sapbert-umls/model-0_4735 | 56.2% | +16.7pp | 79.2% | 0.653 | 58.7% | 0.0051 |
| 35 | all-mini-fine/model-0_0004 | 54.2% | +14.6pp | 66.7% | 0.578 | 48.2% | 0.0035 |
| 36 | all-mini-fine/model-0_0016 | 54.2% | +14.6pp | 62.5% | 0.568 | 50.3% | 0.0035 |
| 37 | medcpt/model-0_1271 | 54.2% | +14.6pp | 72.9% | 0.616 | 70.8% | 0.0049 |
| 38 | pubmedbert-marco/model-0_0035 | 54.2% | +14.6pp | 62.5% | 0.576 | 56.4% | 0.0054 |
| 39 | roberta-catalan/model-0_0000 | 54.2% | +14.6pp | 58.3% | 0.557 | 69.1% | 0.0053 |
| 40 | roberta-catalan/model-0_0000_1 | 54.2% | +14.6pp | 58.3% | 0.557 | 69.9% | 0.0064 |
| 41 | roberta-catalan/model-0_0001 | 54.2% | +14.6pp | 58.3% | 0.557 | 69.5% | 0.0050 |
| 42 | sapbert-pubmed/model-0_0000 | 54.2% | +14.6pp | 62.5% | 0.566 | 56.1% | 0.0057 |
| 43 | sapbert-pubmed/model-0_0000_1 | 54.2% | +14.6pp | 64.6% | 0.566 | 54.8% | 0.0052 |
| 44 | sapbert-pubmed/model-0_0001 | 54.2% | +14.6pp | 60.4% | 0.563 | 56.5% | 0.0052 |
| 45 | sapbert-pubmed/model-0_0002 | 54.2% | +14.6pp | 60.4% | 0.563 | 57.2% | 0.0048 |
| 46 | sapbert-pubmed/model-0_0004 | 54.2% | +14.6pp | 58.3% | 0.556 | 58.7% | 0.0049 |
| 47 | sapbert-pubmed/model-0_0006 | 54.2% | +14.6pp | 58.3% | 0.556 | 59.1% | 0.0049 |
| 48 | sapbert-pubmed/model-0_0012 | 54.2% | +14.6pp | 62.5% | 0.573 | 60.1% | 0.0047 |
| 49 | sapbert-pubmed/model-0_0034 | 54.2% | +14.6pp | 56.2% | 0.552 | 59.1% | 0.0056 |
| 50 | sapbert-pubmed/model-0_0126 | 54.2% | +14.6pp | 56.2% | 0.552 | 58.6% | 0.0056 |
| 51 | biomedical-agressive/model-0_0079 | 52.1% | +12.5pp | 62.5% | 0.551 | 82.6% | 0.0056 |
| 52 | biomedical-agressive/model-0_0088 | 52.1% | +12.5pp | 60.4% | 0.562 | 71.8% | 0.0053 |
| 53 | biomedical-roberta/model-0_0000 | 52.1% | +12.5pp | 66.7% | 0.582 | 65.5% | 0.0059 |
| 54 | biomedical-roberta/model-0_0001 | 52.1% | +12.5pp | 66.7% | 0.582 | 65.5% | 0.0050 |
| 55 | biomedical-roberta/model-0_0007 | 52.1% | +12.5pp | 66.7% | 0.564 | 64.7% | 0.0051 |
| 56 | pubmedbert-marco/model-0_0160 | 52.1% | +12.5pp | 66.7% | 0.571 | 55.0% | 0.0049 |
| 57 | sapbert-pubmed/model-0_0003 | 52.1% | +12.5pp | 60.4% | 0.550 | 56.6% | 0.0053 |
| 58 | sapbert-pubmed/model-0_0020 | 52.1% | +12.5pp | 56.2% | 0.542 | 59.2% | 0.0058 |
| 59 | sapbert-pubmed/model-0_0209 | 52.1% | +12.5pp | 60.4% | 0.549 | 57.7% | 0.0049 |
| 60 | sapbert-pubmed/model-0_0339 | 52.1% | +12.5pp | 58.3% | 0.549 | 55.7% | 0.0054 |
| 61 | all-mini-fine/model-0_0878 | 50.0% | +10.4pp | 68.8% | 0.582 | 46.1% | 0.0035 |
| 62 | biomedical-agressive/model-0_1000 | 50.0% | +10.4pp | 66.7% | 0.555 | 65.7% | 0.0055 |
| 63 | biomedical-roberta/model-0_0000_1 | 50.0% | +10.4pp | 62.5% | 0.540 | 65.1% | 0.0054 |
| 64 | medcpt/model-0_2250 | 50.0% | +10.4pp | 79.2% | 0.590 | 70.7% | 0.0049 |
| 65 | sapbert-pubmed/model-0_0043 | 50.0% | +10.4pp | 60.4% | 0.540 | 62.0% | 0.0051 |
| 66 | all-mini-fine/model-0_0021 | 47.9% | +8.3pp | 70.8% | 0.572 | 46.5% | 0.0034 |
| 67 | roberta-catalan/model-0_0005 | 47.9% | +8.3pp | 56.2% | 0.521 | 70.5% | 0.0057 |
| 68 | sapbert-pubmed/model-0_0926 | 47.9% | +8.3pp | 60.4% | 0.531 | 53.5% | 0.0054 |
| 69 | all-mini-fine/model-0_1619 | 45.8% | +6.2pp | 70.8% | 0.553 | 47.0% | 0.0033 |
| 70 | biomedical-roberta/model-0_0035 | 45.8% | +6.2pp | 62.5% | 0.517 | 67.5% | 0.0053 |
| 71 | roberta-catalan/model-0_0007 | 45.8% | +6.2pp | 56.2% | 0.504 | 68.9% | 0.0051 |
| 72 | sapbert-pubmed/model-0_1608 | 45.8% | +6.2pp | 60.4% | 0.508 | 53.6% | 0.0060 |
| 73 | sapbert-pubmed/model-0_2160 | 45.8% | +6.2pp | 58.3% | 0.509 | 52.7% | 0.0052 |
| 74 | roberta-catalan/model-0_0067 | 43.8% | +4.2pp | 60.4% | 0.499 | 67.7% | 0.0049 |
| 75 | bioclinicalbert/model-0_0002 | 41.7% | +2.1pp | 60.4% | 0.479 | 73.6% | 0.0050 |
| 76 | medcpt/model-0_4646 | 39.6% | 0.0pp | 70.8% | 0.521 | 79.9% | 0.0048 |
| 77 | paraphrase-agressive/model-0_0857 | 39.6% | 0.0pp | 62.5% | 0.474 | 53.0% | 0.0052 |
| 78 | biomedical-roberta/model-0_0157 | 37.5% | -2.1pp | 62.5% | 0.465 | 59.4% | 0.0050 |
| 79 | pubmedbert-ms/model-0_0001 | 37.5% | -2.1pp | 58.3% | 0.449 | 73.5% | 0.0053 |
| 80 | sapbert-pubmed/model-0_4277 | 37.5% | -2.1pp | 58.3% | 0.451 | 52.9% | 0.0053 |
| 81 | biolinkbert/model-0_0003 | 35.4% | -4.2pp | 56.2% | 0.416 | 57.6% | 0.0059 |
| 82 | pubmedbert-ms/model-0_0009 | 35.4% | -4.2pp | 54.2% | 0.424 | 72.6% | 0.0050 |
| 83 | roberta-catalan/model-0_1372 | 35.4% | -4.2pp | 47.9% | 0.406 | 66.4% | 0.0055 |
| 84 | bioclinicalbert/model-0_0000 | 33.3% | -6.2pp | 62.5% | 0.423 | 71.7% | 0.0053 |
| 85 | biolinkbert/model-0_0000 | 33.3% | -6.2pp | 56.2% | 0.410 | 59.2% | 0.0054 |
| 86 | biolinkbert/model-0_0001 | 33.3% | -6.2pp | 58.3% | 0.414 | 58.8% | 0.0055 |
| 87 | pubmedbert-ms/model-0_0000 | 33.3% | -6.2pp | 52.1% | 0.400 | 71.6% | 0.0054 |
| 88 | pubmedbert-ms/model-0_0000_1 | 33.3% | -6.2pp | 50.0% | 0.402 | 73.4% | 0.0059 |
| 89 | pubmedbert-ms/model-0_0003 | 33.3% | -6.2pp | 47.9% | 0.396 | 79.3% | 0.0051 |
| 90 | pubmedbert-ms/model-0_0006 | 33.3% | -6.2pp | 45.8% | 0.391 | 79.5% | 0.0056 |
| 91 | biolinkbert/model-0_0000_0 | 31.2% | -8.3pp | 56.2% | 0.397 | 59.6% | 0.0053 |
| 92 | biolinkbert/model-0_0066 | 31.2% | -8.3pp | 56.2% | 0.388 | 57.9% | 0.0054 |
| 93 | pubmedbert-ms/model-0_0007 | 31.2% | -8.3pp | 50.0% | 0.381 | 74.7% | 0.0052 |
| 94 | biolinkbert/model-0_0013 | 29.2% | -10.4pp | 50.0% | 0.361 | 61.7% | 0.0056 |
| 95 | bioclinicalbert/model-0_0000_1 | 27.1% | -12.5pp | 60.4% | 0.381 | 69.4% | 0.0051 |
| 96 | bioclinicalbert/model-0_0005 | 27.1% | -12.5pp | 62.5% | 0.405 | 71.0% | 0.0059 |
| 97 | pubmedbert-ms/model-0_0024 | 27.1% | -12.5pp | 50.0% | 0.361 | 73.6% | 0.0051 |
| 98 | pubmedbert-ms/model-0_0062 | 27.1% | -12.5pp | 45.8% | 0.334 | 72.7% | 0.0053 |
| 99 | biobert-v1.2/model-0_0004 | 25.0% | -14.6pp | 35.4% | 0.293 | 73.9% | 0.0051 |
| 100 | bioclinicalbert/model-0_0008 | 25.0% | -14.6pp | 58.3% | 0.367 | 70.4% | 0.0054 |
| 101 | bioclinicalbert/model-0_0024 | 25.0% | -14.6pp | 66.7% | 0.411 | 65.7% | 0.0057 |
| 102 | scibert/model-0_0000_1 | 25.0% | -14.6pp | 41.7% | 0.318 | 62.6% | 0.0051 |
| 103 | scibert/model-0_0017 | 25.0% | -14.6pp | 45.8% | 0.319 | 61.4% | 0.0056 |
| 104 | biobert-v1.2/model-0_0000 | 22.9% | -16.7pp | 35.4% | 0.276 | 73.4% | 0.0053 |
| 105 | biobert-v1.2/model-0_0000_1 | 22.9% | -16.7pp | 35.4% | 0.275 | 73.6% | 0.0057 |
| 106 | biobert-v1.2/model-0_0001 | 22.9% | -16.7pp | 35.4% | 0.276 | 73.4% | 0.0052 |
| 107 | biobert-v1.2/model-0_0016 | 22.9% | -16.7pp | 35.4% | 0.286 | 79.7% | 0.0055 |
| 108 | biolinkbert/model-0_0527 | 22.9% | -16.7pp | 43.8% | 0.297 | 60.5% | 0.0059 |
| 109 | scibert/model-0_0000 | 22.9% | -16.7pp | 39.6% | 0.302 | 64.5% | 0.0049 |
| 110 | scibert/model-0_0126 | 22.9% | -16.7pp | 47.9% | 0.314 | 59.5% | 0.0054 |
| 111 | biobert-v1.2/model-0_0002 | 20.8% | -18.7pp | 35.4% | 0.262 | 75.5% | 0.0051 |
| 112 | bioclinicalbert/model-0_0017 | 20.8% | -18.7pp | 45.8% | 0.285 | 70.4% | 0.0055 |
| 113 | bioclinicalbert/model-0_0034 | 20.8% | -18.7pp | 56.2% | 0.329 | 68.9% | 0.0063 |
| 114 | bioclinicalbert/model-0_0265 | 20.8% | -18.7pp | 62.5% | 0.350 | 69.8% | 0.0054 |
| 115 | biomedical-roberta/model-0_1455 | 20.8% | -18.7pp | 58.3% | 0.341 | 64.1% | 0.0055 |
| 116 | bluebert/model-0_0000 | 20.8% | -18.7pp | 31.2% | 0.236 | 67.8% | 0.0049 |
| 117 | bluebert/model-0_0000_1 | 20.8% | -18.7pp | 29.2% | 0.232 | 67.6% | 0.0052 |
| 118 | bluebert/model-0_0001 | 20.8% | -18.7pp | 29.2% | 0.236 | 68.8% | 0.0050 |
| 119 | bluebert/model-0_0014 | 20.8% | -18.7pp | 29.2% | 0.233 | 66.7% | 0.0047 |
| 120 | bluebert/model-0_0027 | 20.8% | -18.7pp | 29.2% | 0.237 | 69.0% | 0.0052 |
| 121 | bluebert/model-0_0020 | 16.7% | -22.9pp | 31.2% | 0.209 | 66.7% | 0.0050 |
| 122 | bluebert/model-0_0118 | 16.7% | -22.9pp | 29.2% | 0.212 | 66.5% | 0.0050 |
| 123 | bluebert/model-0_0310 | 16.7% | -22.9pp | 20.8% | 0.181 | 72.6% | 0.0047 |
| 124 | pubmedbert-ms/model-0_1013 | 16.7% | -22.9pp | 50.0% | 0.267 | 72.9% | 0.0053 |
| 125 | scibert/model-0_0001 | 16.7% | -22.9pp | 39.6% | 0.271 | 65.6% | 0.0059 |
| 126 | biobert-v1.2/model-0_0065 | 14.6% | -25.0pp | 29.2% | 0.192 | 75.0% | 0.0055 |
| 127 | biolinkbert/model-0_2622 | 14.6% | -25.0pp | 33.3% | 0.212 | 70.9% | 0.0048 |
| 128 | pubmedbert-ms/model-0_0175 | 14.6% | -25.0pp | 50.0% | 0.266 | 73.0% | 0.0050 |
| 129 | scibert/model-0_0002 | 14.6% | -25.0pp | 39.6% | 0.247 | 63.3% | 0.0055 |
| 130 | scibert/model-0_0004 | 14.6% | -25.0pp | 45.8% | 0.261 | 62.1% | 0.0050 |
| 131 | biobert/model-0_0000 | 12.5% | -27.1pp | 39.6% | 0.195 | 68.4% | 0.0052 |
| 132 | biobert/model-0_0000_1 | 12.5% | -27.1pp | 39.6% | 0.199 | 69.3% | 0.0058 |
| 133 | biobert/model-0_0001 | 12.5% | -27.1pp | 39.6% | 0.195 | 68.5% | 0.0055 |
| 134 | biobert/model-0_0002 | 12.5% | -27.1pp | 43.8% | 0.211 | 67.4% | 0.0056 |
| 135 | biobert/model-0_0004 | 12.5% | -27.1pp | 41.7% | 0.204 | 68.7% | 0.0060 |
| 136 | biobert/model-0_0015 | 12.5% | -27.1pp | 35.4% | 0.191 | 72.8% | 0.0050 |
| 137 | biobert/model-0_0020 | 12.5% | -27.1pp | 33.3% | 0.185 | 73.0% | 0.0058 |
| 138 | biobert/model-0_0292 | 12.5% | -27.1pp | 31.2% | 0.166 | 75.5% | 0.0053 |
| 139 | biobert/model-0_1459 | 12.5% | -27.1pp | 22.9% | 0.148 | 76.7% | 0.0054 |
| 140 | deberta-v3-conservative/model-0_0019 | 12.5% | -27.1pp | 18.8% | 0.147 | 76.4% | 0.0120 |
| 141 | scibert/model-0_0929 | 12.5% | -27.1pp | 37.5% | 0.207 | 59.7% | 0.0053 |
| 142 | biobert/model-0_0073 | 10.4% | -29.2pp | 29.2% | 0.161 | 75.2% | 0.0052 |
| 143 | bioclinicalbert/model-0_0107 | 10.4% | -29.2pp | 39.6% | 0.187 | 68.0% | 0.0050 |
| 144 | bluebert/model-0_2036 | 10.4% | -29.2pp | 25.0% | 0.151 | 78.3% | 0.0058 |
| 145 | biobert/model-0_2945 | 8.3% | -31.2pp | 20.8% | 0.121 | 82.9% | 0.0056 |
| 146 | biobert-v1.2/model-0_0173 | 8.3% | -31.2pp | 29.2% | 0.152 | 77.4% | 0.0053 |
| 147 | biobert-v1.2/model-0_1077 | 8.3% | -31.2pp | 29.2% | 0.158 | 81.0% | 0.0051 |
| 148 | bioclinicalbert/model-0_0001 | 8.3% | -31.2pp | 45.8% | 0.199 | 59.0% | 0.0064 |
| 149 | bioclinicalbert/model-0_0003 | 8.3% | -31.2pp | 50.0% | 0.216 | 63.9% | 0.0051 |
| 150 | bioclinicalbert/model-0_0032 | 8.3% | -31.2pp | 41.7% | 0.197 | 66.8% | 0.0054 |
| 151 | bioclinicalbert/model-0_1554 | 8.3% | -31.2pp | 35.4% | 0.150 | 67.1% | 0.0047 |
| 152 | bluebert/model-0_1034 | 8.3% | -31.2pp | 22.9% | 0.132 | 67.1% | 0.0056 |
| 153 | deberta-v3-conservative/model-0_0002 | 8.3% | -31.2pp | 14.6% | 0.111 | 82.0% | 0.0116 |
| 154 | deberta-v3-conservative/model-0_0004 | 8.3% | -31.2pp | 18.8% | 0.118 | 78.5% | 0.0117 |
| 155 | deberta-v3-conservative/model-0_0010 | 8.3% | -31.2pp | 16.7% | 0.112 | 79.2% | 0.0120 |
| 156 | deberta-v3-conservative/model-0_0027 | 8.3% | -31.2pp | 16.7% | 0.120 | 79.3% | 0.0118 |
| 157 | deberta-v3-conservative/model-0_0950 | 8.3% | -31.2pp | 22.9% | 0.136 | 89.2% | 0.0124 |
| 158 | deberta-v3-conservative/model-0_0993 | 8.3% | -31.2pp | 18.8% | 0.122 | 93.1% | 0.0118 |
| 159 | scibert/model-0_2747 | 8.3% | -31.2pp | 29.2% | 0.144 | 67.7% | 0.0051 |
| 160 | biobert-v1.2/model-0_2813 | 6.2% | -33.3pp | 20.8% | 0.107 | 84.0% | 0.0053 |
| 161 | bioclinicalbert/model-0_0340 | 6.2% | -33.3pp | 25.0% | 0.116 | 61.8% | 0.0054 |
| 162 | biomedical-agressive/model-0_0000 | 6.2% | -33.3pp | 8.3% | 0.073 | 100.0% | 0.0060 |
| 163 | deberta-v3-conservative/model-0_0000 | 6.2% | -33.3pp | 20.8% | 0.117 | 79.7% | 0.0121 |
| 164 | deberta-v3-conservative/model-0_0000_1 | 6.2% | -33.3pp | 20.8% | 0.118 | 79.2% | 0.0119 |
| 165 | deberta-v3-conservative/model-0_0001 | 6.2% | -33.3pp | 20.8% | 0.127 | 79.6% | 0.0125 |
| 166 | deberta-v3-conservative/model-0_0041 | 6.2% | -33.3pp | 18.8% | 0.109 | 76.2% | 0.0119 |
| 167 | deberta-v3-conservative/model-0_0057 | 6.2% | -33.3pp | 20.8% | 0.118 | 76.2% | 0.0119 |
| 168 | deberta-v3-conservative/model-0_0095 | 6.2% | -33.3pp | 22.9% | 0.122 | 80.9% | 0.0121 |
| 169 | deberta-v3-conservative/model-0_0123 | 6.2% | -33.3pp | 18.8% | 0.106 | 79.4% | 0.0130 |
| 170 | deberta-v3-conservative/model-0_0213 | 6.2% | -33.3pp | 18.8% | 0.101 | 82.2% | 0.0120 |
| 171 | deberta-v3-conservative/model-0_0363 | 6.2% | -33.3pp | 22.9% | 0.118 | 82.0% | 0.0117 |
| 172 | electra-agressive/model-0_0000 | 6.2% | -33.3pp | 18.8% | 0.103 | 100.0% | 0.0063 |
| 173 | paraphrase-agressive/model-0_0000 | 6.2% | -33.3pp | 18.8% | 0.104 | 100.0% | 0.0053 |
| 174 | bioclinicalbert/model-0_1698 | 4.2% | -35.4pp | 33.3% | 0.120 | 72.1% | 0.0069 |
| 175 | bioclinicalbert/model-0_2932 | 4.2% | -35.4pp | 22.9% | 0.101 | 80.8% | 0.0053 |
| 176 | bluebert/model-0_2849 | 4.2% | -35.4pp | 16.7% | 0.087 | 79.0% | 0.0056 |
| 177 | deberta-v3-conservative/model-0_0060 | 4.2% | -35.4pp | 20.8% | 0.106 | 78.0% | 0.0120 |
| 178 | deberta-v3-conservative/model-0_0678 | 4.2% | -35.4pp | 27.1% | 0.120 | 84.8% | 0.0120 |
| 179 | electra-agressive/model-0_1317 | 4.2% | -35.4pp | 20.8% | 0.098 | 79.9% | 0.0079 |
| 180 | electra-agressive/model-0_1767 | 4.2% | -35.4pp | 20.8% | 0.090 | 80.8% | 0.0074 |
| 181 | pubmedbert-ms/model-0_2918 | 4.2% | -35.4pp | 35.4% | 0.121 | 82.3% | 0.0056 |

## Top 5 Mejores Modelos - Análisis Detallado

### 1. sapbert-umls/model-0_0000

**Ruta**: `models\sapbert-umls\model-0_0000`

| Métrica | Valor | Comparación con Base |
|---------|-------|---------------------|
| Accuracy@1 | 70.8% | +31.3 pp |
| Accuracy@5 | 79.2% | - |
| MRR | 0.738 | +0.246 |
| Velocidad | 0.0057s | 0.58x más lento |
| Queries no encontradas | 10/48 | - |

### 2. sapbert-umls/model-0_0000_1

**Ruta**: `models\sapbert-umls\model-0_0000_1`

| Métrica | Valor | Comparación con Base |
|---------|-------|---------------------|
| Accuracy@1 | 70.8% | +31.3 pp |
| Accuracy@5 | 79.2% | - |
| MRR | 0.738 | +0.246 |
| Velocidad | 0.0050s | 0.66x más lento |
| Queries no encontradas | 10/48 | - |

### 3. sapbert-umls/model-0_0001

**Ruta**: `models\sapbert-umls\model-0_0001`

| Métrica | Valor | Comparación con Base |
|---------|-------|---------------------|
| Accuracy@1 | 70.8% | +31.3 pp |
| Accuracy@5 | 79.2% | - |
| MRR | 0.740 | +0.248 |
| Velocidad | 0.0052s | 0.64x más lento |
| Queries no encontradas | 10/48 | - |

### 4. sapbert-umls/model-0_0009

**Ruta**: `models\sapbert-umls\model-0_0009`

| Métrica | Valor | Comparación con Base |
|---------|-------|---------------------|
| Accuracy@1 | 70.8% | +31.3 pp |
| Accuracy@5 | 79.2% | - |
| MRR | 0.743 | +0.251 |
| Velocidad | 0.0054s | 0.61x más lento |
| Queries no encontradas | 10/48 | - |

### 5. sapbert-umls/model-0_0183

**Ruta**: `models\sapbert-umls\model-0_0183`

| Métrica | Valor | Comparación con Base |
|---------|-------|---------------------|
| Accuracy@1 | 70.8% | +31.3 pp |
| Accuracy@5 | 81.2% | - |
| MRR | 0.743 | +0.251 |
| Velocidad | 0.0051s | 0.64x más lento |
| Queries no encontradas | 9/48 | - |

