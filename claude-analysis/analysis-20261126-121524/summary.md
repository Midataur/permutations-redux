## g_m10  (general, 5 blocks, checkpoint step 200000)

Planted-chain accuracy (teacher-forced, at the planted slot):

| series | L | s | correct | % | median margin |
|---|---|---|---|---|---|
| in support | 4 | 5 | 300/300 | 100.0 | 17.6 |
| in support | 8 | 9 | 300/300 | 100.0 | 17.1 |
| in support | 12 | 10 | 300/300 | 100.0 | 12.7 |
| in support | 14 | 10 | 298/300 | 99.3 | 10.4 |
| in support | 16 | 10 | 296/300 | 98.7 | 8.9 |
| in support | 18 | 10 | 295/300 | 98.3 | 7.7 |
| in support | 20 | 10 | 295/300 | 98.3 | 6.1 |
| in support | 22 | 10 | 280/300 | 93.3 | 5.3 |
| in support | 24 | 10 | 265/300 | 88.3 | 4.1 |
| beyond support | 16 | 11 | 299/300 | 99.7 | 9.2 |
| beyond support | 16 | 12 | 298/300 | 99.3 | 8.9 |
| beyond support | 16 | 17 | 298/300 | 99.3 | 8.9 |
| beyond support | 20 | 11 | 293/300 | 97.7 | 6.6 |
| beyond support | 20 | 12 | 295/300 | 98.3 | 6.6 |
| beyond support | 20 | 17 | 297/300 | 99.0 | 6.8 |
| beyond support | 20 | 21 | 294/300 | 98.0 | 6.9 |
| beyond support | 24 | 12 | 282/300 | 94.0 | 4.6 |
| beyond support | 24 | 17 | 290/300 | 96.7 | 5.1 |
| beyond support | 24 | 25 | 290/300 | 96.7 | 4.9 |

Probe at the word positions (accuracy of the continuation probe by remaining hops; chance 4% general / 6% elementary):

| layer | 1-1 | 2-2 | 3-3 | 4-4 | 5-6 | 7-8 | 9-12 | 13-24 |
|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 5 | 4 | 4 | 5 | 4 | 4 | 5 |
| 1 | 16 | 5 | 5 | 3 | 4 | 4 | 5 | 5 |
| 2 | 58 | 8 | 4 | 4 | 4 | 5 | 4 | 5 |
| 3 | 75 | 45 | 19 | 4 | 4 | 5 | 4 | 5 |
| 4 | 76 | 53 | 46 | 41 | 38 | 4 | 4 | 6 |
| 5 | 75 | 52 | 49 | 43 | 46 | 49 | 49 | 53 |

Deepest bin decoded at 50% or more and at least 15 points above layer 0, by layer: 0: none; 1: none; 2: up to h=1; 3: up to h=1; 4: up to h=2; 5: up to h=24

Heads whose zeroing at the word positions costs more than 10 points (of the 'word' ablation, blocks 1 to n-1; block n is never read):

| L | model right on planted | costly heads | three costliest (block, head: accuracy) |
|---|---|---|---|
| 8 | 100% | 2 of 24 | b3h5: 0.45, b2h3: 0.84, b2h1: 0.90 |
| 12 | 100% | 5 of 24 | b3h5: 0.38, b2h3: 0.51, b1h2: 0.52 |
| 16 | 99% | 6 of 24 | b1h2: 0.37, b3h5: 0.46, b2h3: 0.48 |

Strongest hop offsets per head at L=16 (head, offset d: times uniform attention; only heads with a peak above 8x):

- block 1: none
- block 2: h1: +1 (13x); h3: -1 (11x)
- block 3: h1: -1 (11x); h2: +1 (11x); h5: +1 (11x)
- block 4: none
- block 5: none

Greedy head ablation (zero, at each step, the one further word-position head that hurts most; L=16, 149 words that the intact model gets right):

+b3h5 -> 0.43, +b1h2 -> 0.30, +b2h3 -> 0.18, +b1h4 -> 0.17, +b3h4 -> 0.15, +b2h2 -> 0.15

## g_plant10  (general, 5 blocks, checkpoint step 200000)

Planted-chain accuracy (teacher-forced, at the planted slot):

| series | L | s | correct | % | median margin |
|---|---|---|---|---|---|
| in support | 4 | 5 | 300/300 | 100.0 | 16.0 |
| in support | 8 | 9 | 300/300 | 100.0 | 15.4 |
| in support | 12 | 10 | 300/300 | 100.0 | 13.9 |
| in support | 14 | 10 | 300/300 | 100.0 | 13.8 |
| in support | 16 | 10 | 300/300 | 100.0 | 13.7 |
| in support | 18 | 10 | 300/300 | 100.0 | 14.2 |
| in support | 20 | 10 | 300/300 | 100.0 | 14.8 |
| in support | 22 | 10 | 300/300 | 100.0 | 14.9 |
| in support | 24 | 10 | 300/300 | 100.0 | 15.1 |
| beyond support | 16 | 11 | 300/300 | 100.0 | 13.9 |
| beyond support | 16 | 12 | 300/300 | 100.0 | 13.7 |
| beyond support | 16 | 17 | 300/300 | 100.0 | 12.7 |
| beyond support | 20 | 11 | 300/300 | 100.0 | 14.5 |
| beyond support | 20 | 12 | 300/300 | 100.0 | 14.3 |
| beyond support | 20 | 17 | 300/300 | 100.0 | 13.2 |
| beyond support | 20 | 21 | 300/300 | 100.0 | 12.2 |
| beyond support | 24 | 12 | 300/300 | 100.0 | 14.9 |
| beyond support | 24 | 17 | 300/300 | 100.0 | 13.6 |
| beyond support | 24 | 25 | 300/300 | 100.0 | 11.2 |

Probe at the word positions (accuracy of the continuation probe by remaining hops; chance 4% general / 6% elementary):

| layer | 1-1 | 2-2 | 3-3 | 4-4 | 5-6 | 7-8 | 9-12 | 13-24 |
|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 5 | 4 | 5 | 5 | 4 | 4 | 5 |
| 1 | 14 | 5 | 5 | 6 | 4 | 5 | 4 | 5 |
| 2 | 56 | 5 | 4 | 3 | 4 | 3 | 4 | 5 |
| 3 | 72 | 34 | 6 | 4 | 3 | 5 | 4 | 6 |
| 4 | 65 | 50 | 46 | 30 | 6 | 4 | 7 | 6 |
| 5 | 63 | 50 | 47 | 32 | 14 | 17 | 26 | 38 |

Deepest bin decoded at 50% or more and at least 15 points above layer 0, by layer: 0: none; 1: none; 2: up to h=1; 3: up to h=1; 4: up to h=2; 5: up to h=2

Heads whose zeroing at the word positions costs more than 10 points (of the 'word' ablation, blocks 1 to n-1; block n is never read):

| L | model right on planted | costly heads | three costliest (block, head: accuracy) |
|---|---|---|---|
| 8 | 100% | 5 of 24 | b2h1: 0.48, b2h3: 0.68, b4h0: 0.81 |
| 16 | 100% | 7 of 24 | b2h1: 0.49, b2h4: 0.60, b2h3: 0.68 |
| 24 | 100% | 4 of 24 | b2h1: 0.46, b2h3: 0.61, b2h4: 0.64 |

Strongest hop offsets per head at L=24 (head, offset d: times uniform attention; only heads with a peak above 8x):

- block 1: none
- block 2: h1: +1 (13x); h2: +1 (9x); h3: +1 (13x); h4: -1 (13x); h5: -1 (12x)
- block 3: h1: +1 (12x); h2: +1 (10x); h4: +2 (9x); h5: -1 (10x)
- block 4: none
- block 5: none

Greedy head ablation (zero, at each step, the one further word-position head that hurts most; L=24, 150 words that the intact model gets right):

+b2h1 -> 0.51, +b2h5 -> 0.39, +b2h3 -> 0.31, +b2h2 -> 0.25, +b3h3 -> 0.20, +b1h0 -> 0.19, +b1h4 -> 0.20, +b1h5 -> 0.21

## g_plant10_b8  (general, 8 blocks, checkpoint step 200000)

Planted-chain accuracy (teacher-forced, at the planted slot):

| series | L | s | correct | % | median margin |
|---|---|---|---|---|---|
| in support | 4 | 5 | 300/300 | 100.0 | 17.1 |
| in support | 8 | 9 | 300/300 | 100.0 | 16.3 |
| in support | 12 | 10 | 300/300 | 100.0 | 13.9 |
| in support | 14 | 10 | 300/300 | 100.0 | 14.3 |
| in support | 16 | 10 | 300/300 | 100.0 | 14.1 |
| in support | 18 | 10 | 300/300 | 100.0 | 14.2 |
| in support | 20 | 10 | 300/300 | 100.0 | 14.5 |
| in support | 22 | 10 | 300/300 | 100.0 | 14.7 |
| in support | 24 | 10 | 300/300 | 100.0 | 14.9 |
| beyond support | 16 | 11 | 300/300 | 100.0 | 14.1 |
| beyond support | 16 | 12 | 300/300 | 100.0 | 13.8 |
| beyond support | 16 | 17 | 300/300 | 100.0 | 13.2 |
| beyond support | 20 | 11 | 300/300 | 100.0 | 14.3 |
| beyond support | 20 | 12 | 300/300 | 100.0 | 14.4 |
| beyond support | 20 | 17 | 300/300 | 100.0 | 13.7 |
| beyond support | 20 | 21 | 300/300 | 100.0 | 13.0 |
| beyond support | 24 | 12 | 300/300 | 100.0 | 14.8 |
| beyond support | 24 | 17 | 300/300 | 100.0 | 14.1 |
| beyond support | 24 | 25 | 300/300 | 100.0 | 12.7 |

Probe at the word positions (accuracy of the continuation probe by remaining hops; chance 4% general / 6% elementary):

| layer | 1-1 | 2-2 | 3-3 | 4-4 | 5-6 | 7-8 | 9-12 | 13-24 |
|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 4 | 3 | 5 | 5 | 3 | 5 | 5 |
| 1 | 6 | 5 | 4 | 5 | 5 | 4 | 4 | 5 |
| 2 | 22 | 5 | 5 | 5 | 6 | 5 | 4 | 6 |
| 3 | 24 | 5 | 5 | 5 | 4 | 4 | 4 | 5 |
| 4 | 43 | 5 | 5 | 6 | 6 | 5 | 5 | 5 |
| 5 | 61 | 8 | 5 | 6 | 4 | 5 | 5 | 7 |
| 6 | 79 | 42 | 8 | 5 | 4 | 5 | 4 | 6 |
| 7 | 79 | 57 | 45 | 35 | 17 | 20 | 28 | 29 |
| 8 | 78 | 55 | 45 | 39 | 45 | 54 | 61 | 69 |

Deepest bin decoded at 50% or more and at least 15 points above layer 0, by layer: 0: none; 1: none; 2: none; 3: none; 4: none; 5: up to h=1; 6: up to h=1; 7: up to h=2; 8: up to h=24

Heads whose zeroing at the word positions costs more than 10 points (of the 'word' ablation, blocks 1 to n-1; block n is never read):

| L | model right on planted | costly heads | three costliest (block, head: accuracy) |
|---|---|---|---|
| 8 | 100% | 1 of 42 | b6h2: 0.56, b4h1: 0.94, b7h2: 0.96 |
| 16 | 100% | 4 of 42 | b2h5: 0.68, b6h2: 0.80, b4h1: 0.83 |
| 24 | 100% | 2 of 42 | b2h5: 0.69, b2h3: 0.86, b6h2: 0.91 |

Strongest hop offsets per head at L=24 (head, offset d: times uniform attention; only heads with a peak above 8x):

- block 1: none
- block 2: h3: +1 (10x); h5: -1 (10x)
- block 3: h1: +1 (10x)
- block 4: h1: -1 (10x)
- block 5: h5: -1 (10x)
- block 6: h2: +1 (9x); h3: +1 (10x)
- block 7: none
- block 8: none

Greedy head ablation (zero, at each step, the one further word-position head that hurts most; L=24, 150 words that the intact model gets right):

+b2h5 -> 0.75, +b2h3 -> 0.38, +b1h3 -> 0.29, +b5h3 -> 0.24, +b2h1 -> 0.23, +b1h0 -> 0.23, +b1h1 -> 0.23, +b1h2 -> 0.23

## e_base  (elementary, 5 blocks, checkpoint step 106250)

Planted-chain accuracy (teacher-forced, at the planted slot):

| series | L | width | correct | % | median margin |
|---|---|---|---|---|---|
| width 8 | 12 | 8 | 300/300 | 100.0 | 17.8 |
| width 8 | 24 | 8 | 300/300 | 100.0 | 17.9 |
| width 8 | 40 | 8 | 300/300 | 100.0 | 15.4 |
| width 8 | 56 | 8 | 299/300 | 99.7 | 13.9 |
| width 8 | 70 | 8 | 297/300 | 99.0 | 11.2 |
| width 8 | 78 | 8 | 291/300 | 97.0 | 9.3 |
| width 8 | 80 | 8 | 283/300 | 94.3 | 8.5 |
| width 8 | 90 | 8 | 280/300 | 93.3 | 6.3 |
| width 8 | 100 | 8 | 265/300 | 88.3 | 4.2 |
| width 8 | 110 | 8 | 246/300 | 82.0 | 3.3 |
| width 8 | 118 | 8 | 256/300 | 85.3 | 3.3 |
| L=24 | 24 | 4 | 300/300 | 100.0 | 17.2 |
| L=24 | 24 | 8 | 300/300 | 100.0 | 17.3 |
| L=24 | 24 | 11 | 291/300 | 97.0 | 16.3 |
| L=24 | 24 | 12 | 287/300 | 95.7 | 14.7 |
| L=24 | 24 | 14 | 259/300 | 86.3 | 11.2 |
| L=24 | 24 | 16 | 200/300 | 66.7 | 3.7 |
| L=40 | 40 | 8 | 300/300 | 100.0 | 15.6 |
| L=40 | 40 | 11 | 292/300 | 97.3 | 13.6 |
| L=40 | 40 | 12 | 291/300 | 97.0 | 12.7 |
| L=40 | 40 | 16 | 221/300 | 73.7 | 6.1 |

Probe at the word positions (accuracy of the continuation probe by remaining hops; chance 4% general / 6% elementary):

| layer | 1-1 | 2-2 | 3-3 | 4-4 | 5-6 | 7-8 | 9-12 | 13-24 | 25-48 | 49-120 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 57 | 54 | 45 | 44 | 20 | 17 | 15 | 15 | 15 | 11 |
| 1 | 84 | 56 | 49 | 43 | 20 | 18 | 16 | 15 | 15 | 11 |
| 2 | 95 | 68 | 55 | 45 | 21 | 18 | 16 | 15 | 14 | 11 |
| 3 | 95 | 79 | 70 | 55 | 26 | 19 | 16 | 14 | 15 | 10 |
| 4 | 93 | 77 | 71 | 62 | 47 | 42 | 31 | 23 | 28 | 14 |
| 5 | 92 | 77 | 71 | 61 | 47 | 42 | 31 | 27 | 56 | 57 |

Deepest bin decoded at 50% or more and at least 15 points above layer 0, by layer: 0: none; 1: up to h=1; 2: up to h=1; 3: up to h=3; 4: up to h=4; 5: up to h=120

Heads whose zeroing at the word positions costs more than 10 points (of the 'word' ablation, blocks 1 to n-1; block n is never read):

| L | model right on planted | costly heads | three costliest (block, head: accuracy) |
|---|---|---|---|
| 12 | 100% | 7 of 24 | b1h4: 0.45, b4h4: 0.48, b3h2: 0.56 |
| 40 | 100% | 10 of 24 | b2h1: 0.40, b1h5: 0.41, b1h4: 0.46 |
| 80 | 96% | 17 of 24 | b1h5: 0.39, b2h3: 0.43, b1h4: 0.45 |

Strongest hop offsets per head at L=80 (head, offset d: times uniform attention; only heads with a peak above 8x):

- block 1: h1: +1 (14x), +2 (10x); h2: -1 (11x), +1 (8x); h3: -1 (10x), -2 (9x); h4: -1 (64x), +1 (34x); h5: +1 (66x), -1 (34x)
- block 2: h0: -2 (32x), +2 (15x); h1: -1 (52x), +1 (25x); h2: -2 (10x), -3 (9x); h3: +1 (50x), -1 (25x); h4: +1 (26x), +2 (17x); h5: -3 (11x), -2 (10x)
- block 3: h2: -1 (42x), -3 (12x); h3: +1 (40x), +3 (12x); h4: -2 (18x), -4 (14x); h5: +2 (16x), +4 (13x)
- block 4: h3: +6 (10x), +4 (9x); h4: -8 (9x), -6 (8x)
- block 5: none

Greedy head ablation (zero, at each step, the one further word-position head that hurts most; L=80, 146 words that the intact model gets right):

+b1h5 -> 0.33, +b3h1 -> 0.32, +b2h3 -> 0.31, +b2h1 -> 0.28, +b4h0 -> 0.27, +b1h3 -> 0.25, +b2h2 -> 0.24, +b3h2 -> 0.25
