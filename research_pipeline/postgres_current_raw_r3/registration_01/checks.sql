DO $$ BEGIN
 IF EXISTS(SELECT 1 FROM sources s LEFT JOIN (SELECT match_id,count(*) n FROM queries GROUP BY match_id) q USING(match_id) WHERE s.n_queries<>coalesce(q.n,0)) THEN RAISE EXCEPTION 'Source query count mismatch'; END IF;
 IF EXISTS(SELECT 1 FROM shards s LEFT JOIN (SELECT family,shard,count(*) n FROM queries GROUP BY 1,2) q USING(family,shard) WHERE s.n_queries<>coalesce(q.n,0)) THEN RAISE EXCEPTION 'Shard query count mismatch'; END IF;
 IF EXISTS(SELECT 1 FROM queries q LEFT JOIN (SELECT row_id,count(*) n FROM horizons GROUP BY row_id) h USING(row_id) WHERE coalesce(n,0)<>3) THEN RAISE EXCEPTION 'Missing horizon'; END IF;
 IF EXISTS(SELECT 1 FROM sources s JOIN matches m USING(match_id) WHERE m.disposition<>'RETAIN_MASKED') THEN RAISE EXCEPTION 'Excluded source entered ledger'; END IF;
 IF EXISTS(SELECT 1 FROM matches m LEFT JOIN sources s USING(match_id) WHERE m.disposition='RETAIN_MASKED' AND s.match_id IS NULL) THEN RAISE EXCEPTION 'Retained source missing'; END IF;
 IF EXISTS(SELECT 1 FROM horizons h JOIN queries q USING(row_id) WHERE NOT q.selection_supported AND (h.economic_supported OR h.terminal_supported OR h.hazard_supported OR h.pareto_supported OR bit_count(h.event_support)>0 OR bit_count(h.slice_event_support)>0 OR bit_count(h.gold_support)>0 OR bit_count(h.slice_gold_support)>0)) THEN RAISE EXCEPTION 'Unsupported query exposes target'; END IF;
 IF get_bit(decode('01','hex'),0)<>1 OR get_bit(decode('80','hex'),7)<>1 THEN RAISE EXCEPTION 'Bit order mismatch'; END IF;
END $$;
