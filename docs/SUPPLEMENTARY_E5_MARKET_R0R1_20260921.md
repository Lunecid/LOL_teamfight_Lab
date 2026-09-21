# Supplementary E5 — market_event + R0/R1 (§8)

**generated:** 2026-09-21T10:37:17.499306+00:00  

## market_event (same window as SVI)

n_ok=32981; tie/drop=0; flip vs SVI=0.13398623449865074; ΔBrier q−PT on market=0.0014860359941252232.

Same (engage,s)→endpoint_h90+1 window; v3.3 market_event. Not original CoG corpus numbers.

## R0 / R1 nextobj models

TRAIN n=39605; Q_SELECT n=10195; TEST n=32981.

- R0 TEST multiclass Brier=0.63265 (C=10.0)
- R1 TEST multiclass Brier=0.62541 (C=10.0)
- R1−R0 = -0.00724
- Blue|Red selected: R0=0.4733183653582429, R1=0.46708864301748093

R1 adds ΔV to R0 at endpoint; p_post=p_pre+ΔV reparameterization caveat applies. Not causal proof that SVI is the true fight value. Selected Blue|Red is a restricted denominator.

## E6

`INCOMPLETE` — No pre-registered unseen match list / new scrape for confirmatory eval.
