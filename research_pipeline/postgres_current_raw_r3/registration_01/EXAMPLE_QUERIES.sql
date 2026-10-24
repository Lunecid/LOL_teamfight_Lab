-- Read-only examples. Metadata/audit masks must not enter model features.
-- Existing NPZ loader remains the authoritative feature/label reader.

-- Current retained FIT sources with raw paths and historical source hashes.
SELECT * FROM raw_current_r3_20261018.source_lookup
WHERE split_role='FIT_15.20' AND disposition='RETAIN_MASKED'
ORDER BY match_id LIMIT 20;

-- Dispositions retain no-query matches; count matches separately from queries.
SELECT split_role,patch,disposition,count(*) AS matches,sum(coalesce(n_queries,0)) AS queries
FROM raw_current_r3_20261018.match_catalog GROUP BY 1,2,3 ORDER BY 1,2,3;

-- All 60/120/180 second support denominators from exact copied masks.
SELECT * FROM raw_current_r3_20261018.support_summary ORDER BY split_role,patch,horizon_s;

-- Trace a supported 120s net-gold outcome back to the existing immutable shard.
SELECT q.row_id,m.split_role,m.patch,s.absolute_path,q.shard_row,q.t0_ms,
 h.horizon_s,h.endpoint_ms,h.economic_sign,h.economic_supported
FROM raw_current_r3_20261018.queries q
JOIN raw_current_r3_20261018.matches m USING(match_id)
JOIN raw_current_r3_20261018.horizons h USING(row_id)
JOIN raw_current_r3_20261018.shards s ON s.family=q.family AND s.shard=q.shard
WHERE m.split_role='TEST_15.22' AND h.horizon_s=120 AND h.economic_supported
ORDER BY q.row_id LIMIT 20;

-- Original inclusive mask coordinates; a full-match audit is not a causal input.
SELECT match_id,split_role,hard_reasons,population_reasons,event_masks,field_masks,rule_masks
FROM raw_current_r3_20261018.match_catalog
WHERE disposition='QUARANTINE' ORDER BY match_id;

-- Blue team's Baron target support: team index 0, event index 4.
SELECT h.row_id,h.horizon_s,get_bit(h.event_support,4)=1 AS blue_baron_supported
FROM raw_current_r3_20261018.horizons h ORDER BY h.row_id,h.horizon_s LIMIT 20;
