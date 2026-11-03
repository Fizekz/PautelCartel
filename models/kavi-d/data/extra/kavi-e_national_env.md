# kavi-e_national_env.json (borrowed from kavi-e)

Copied verbatim from `origin/kavi/kavi-e-1101:out/national_env.json` (kavi-e v2, disclosure 1) on 2026-11-02.
kavi-e's pre-election national House environment estimate N_hat (D-R points) and its sd for the midterms 2010, 2014,
2018, 2022 and 2026: inverse-variance combination of the final generic ballot, special-election over-performance,
the previous year's VA/NJ governor races and a fundamentals signal, each signal's bias/sd from EARLIER cycles only
(kavi-e METHODS section 3). `actual` is the realised MIT national House margin (not used from this file).

Used by kavi-d v3 only in the tested-and-reverted `env_e*` variants (src/models/kavi-d/environment.py); the final
v3 model (final_v3 = logit_gbmid) does not use it. Caveat (kavi-e's own): its signal set was designed knowing
roughly how 2018 and 2022 ended.
