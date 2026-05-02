04/28 - 80d8a55ddc9afcda297bc67be0c2a93ad66a33b4 - Avg 247 at step 933
04/30 - 03e0433 - adaptive real_ratio — peak 442.6, Q4 avg +108.4, recent 20 avg +147.4 (still climbing at end)
04/30 - 80ea77a - extend real_ratio to ep 400 + faster Q ramp — peak 880.4, Q4 avg +280.8, recent 20 avg +214.8 (new all-time best)
  Changes vs main: adaptive real_ratio (1.0→0.5 over ep 0-400), WM:Q schedule [4,1]<100 [2,2]<200 [2,3]+ (was 200/400 thresholds)
05/01 - run 109 (cleanup, same config as main) — peak 782.6, Q4 avg +443.2, recent 20 avg +550.8, Q4 min +25.8 (no crashes in final quarter)
  Strong confirmation of consistency. Better Q4/recent-20 than run 107 despite lower peak.
05/01 - runs 110+111 (cleanup, fixed WM:Q [2,2]) — one soared (~500+ peak), one collapsed. Same variance as dynamic schedule. Item 6 accepted as neutral simplification.
05/01 - runs 113+114 (cleanup, epsilon recovery) — run 113 data inaccessible (API limitation). Run 114: peak 282.9, Q4 avg -0.3, recent 20 avg -15.1. Epsilon recovery fired twice (~ep 707, ~1007).
  Recovery prevented collapse (Q4 avg -0.3 vs ~-70 in collapsed runs) but peak far below best (282.9 vs 782-880).
  Assessment: mechanism works as designed but disrupts Q consolidation — too many recovery cycles cap peak performance.
