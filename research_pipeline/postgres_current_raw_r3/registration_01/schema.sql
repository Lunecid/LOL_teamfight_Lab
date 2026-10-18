-- A new schema only. Deliberately no IF NOT EXISTS, DROP, UPDATE or DELETE.
\set schema_name research_catalog
CREATE SCHEMA :"schema_name";
SET LOCAL search_path= :"schema_name",pg_catalog;
CREATE TABLE catalog_info (
 key text PRIMARY KEY,payload_text text NOT NULL,
 payload jsonb GENERATED ALWAYS AS (payload_text::jsonb) STORED
);
CREATE TABLE audits (
 family text NOT NULL CHECK(family IN ('FIT','HOLDOUT')),
 match_id text NOT NULL CHECK(match_id ~ '^[A-Za-z0-9_]+$'),
 record_text text NOT NULL,
 record jsonb GENERATED ALWAYS AS (record_text::jsonb) STORED,
 PRIMARY KEY(family,match_id),
 CHECK(record->>'match_id'=match_id)
);
CREATE TABLE matches (
 match_id text PRIMARY KEY,patch text,split_role text,macro_role text NOT NULL,
 bucket smallint NOT NULL CHECK(bucket BETWEEN 0 AND 99),
 disposition text NOT NULL CHECK(disposition IN ('RETAIN_MASKED','EXCLUDE_POPULATION','QUARANTINE')),
 audit_family text NOT NULL,
 FOREIGN KEY(audit_family,match_id) REFERENCES audits(family,match_id),
 CHECK((patch IS NULL AND split_role IS NULL) OR (patch IS NOT NULL AND split_role IS NOT NULL))
);
CREATE TABLE files (
 match_id text NOT NULL REFERENCES matches,file_kind text NOT NULL CHECK(file_kind IN ('detail','timeline')),
 absolute_path text NOT NULL UNIQUE,present boolean NOT NULL,size_bytes bigint,mtime_ns bigint,
 audit_sha256 text CHECK(audit_sha256 ~ '^[a-f0-9]{64}$'),audit_size_bytes bigint,
 PRIMARY KEY(match_id,file_kind),
 CHECK((present AND size_bytes>=0 AND mtime_ns IS NOT NULL) OR (NOT present AND size_bytes IS NULL AND mtime_ns IS NULL)),
 CHECK(audit_size_bytes IS NULL OR audit_size_bytes=size_bytes)
);
CREATE TABLE sources (
 match_id text PRIMARY KEY REFERENCES matches,family text NOT NULL,
 n_queries integer NOT NULL CHECK(n_queries>=0),record_text text NOT NULL,
 record jsonb GENERATED ALWAYS AS (record_text::jsonb) STORED,
 UNIQUE(match_id,family),FOREIGN KEY(family,match_id) REFERENCES audits(family,match_id),
 CHECK(record->>'match_id'=match_id),CHECK((record->>'n_queries')::integer=n_queries)
);
CREATE TABLE shards (
 family text NOT NULL CHECK(family IN ('FIT','HOLDOUT')),shard text NOT NULL,absolute_path text NOT NULL UNIQUE,
 sha256 text NOT NULL CHECK(sha256 ~ '^[a-f0-9]{64}$'),n_queries integer NOT NULL CHECK(n_queries>=0),
 PRIMARY KEY(family,shard)
);
CREATE TABLE queries (
 row_id text PRIMARY KEY,match_id text NOT NULL,family text NOT NULL,shard text NOT NULL,
 shard_row integer NOT NULL CHECK(shard_row>=0),t0_ms bigint NOT NULL,tau_ms bigint NOT NULL,
 selection_supported boolean NOT NULL,history_ms_text text NOT NULL,
 player_support bytea NOT NULL CHECK(octet_length(player_support)=113),
 prefix_support bytea NOT NULL CHECK(octet_length(prefix_support)=5),
 skill_support bytea NOT NULL CHECK(octet_length(skill_support)=5),
 buff_support bytea NOT NULL CHECK(octet_length(buff_support)=3),
 focal_support bytea NOT NULL CHECK(octet_length(focal_support)=1),
 soul_status_text text NOT NULL,availability_status_text text NOT NULL,
 FOREIGN KEY(match_id,family) REFERENCES sources(match_id,family),
 FOREIGN KEY(family,shard) REFERENCES shards(family,shard),UNIQUE(family,shard,shard_row),
 CHECK(tau_ms-t0_ms BETWEEN 0 AND 10000)
);
CREATE TABLE horizons (
 row_id text NOT NULL REFERENCES queries,horizon_s smallint NOT NULL CHECK(horizon_s IN (60,120,180)),
 endpoint_ms bigint NOT NULL,endpoint_supported boolean NOT NULL,actual_elapsed_ms bigint NOT NULL,
 terminal_boundary_shift boolean NOT NULL,
 gold_support bytea NOT NULL CHECK(octet_length(gold_support)=1),
 slice_gold_support bytea NOT NULL CHECK(octet_length(slice_gold_support)=1),
 event_support bytea NOT NULL CHECK(octet_length(event_support)=5),
 slice_event_support bytea NOT NULL CHECK(octet_length(slice_event_support)=5),
 economic_supported boolean NOT NULL,economic_sign smallint NOT NULL,
 terminal_class smallint NOT NULL CHECK(terminal_class BETWEEN -1 AND 2),terminal_supported boolean NOT NULL,
 hazard_class smallint NOT NULL CHECK(hazard_class BETWEEN -1 AND 2),hazard_supported boolean NOT NULL,
 pareto_class smallint NOT NULL CHECK(pareto_class BETWEEN -1 AND 3),pareto_supported boolean NOT NULL,
 direct_reward_support bytea NOT NULL CHECK(octet_length(direct_reward_support)=1),remaining_value_supported boolean NOT NULL,
 PRIMARY KEY(row_id,horizon_s),
 CHECK((economic_supported AND economic_sign IN (-1,0,1)) OR (NOT economic_supported AND economic_sign=-9))
);
CREATE INDEX matches_split_patch_disposition_idx ON matches(split_role,patch,disposition);
CREATE INDEX files_kind_present_idx ON files(file_kind,present);
CREATE INDEX queries_match_idx ON queries(match_id);
CREATE INDEX horizons_window_supported_idx ON horizons(horizon_s,economic_supported);
CREATE VIEW match_catalog AS
 SELECT m.*,s.n_queries,a.record->'hard_reasons' AS hard_reasons,a.record->'population_reasons' AS population_reasons,
 a.record->'field_masks' AS field_masks,a.record->'event_masks' AS event_masks,a.record->'rule_masks' AS rule_masks
 FROM matches m JOIN audits a ON (a.family,a.match_id)=(m.audit_family,m.match_id)
 LEFT JOIN sources s ON s.match_id=m.match_id;
CREATE VIEW source_lookup AS
 SELECT m.match_id,m.patch,m.split_role,m.macro_role,m.disposition,s.n_queries,
 d.absolute_path AS detail_path,t.absolute_path AS timeline_path,d.audit_sha256 AS detail_audit_sha256,t.audit_sha256 AS timeline_audit_sha256
 FROM matches m LEFT JOIN sources s USING(match_id)
 JOIN files d ON d.match_id=m.match_id AND d.file_kind='detail'
 JOIN files t ON t.match_id=m.match_id AND t.file_kind='timeline';
COMMENT ON SCHEMA  :"schema_name" IS 'Generic index of locally configured research splits, raw files, audits, immutable ledger links and masks. No training.';
COMMENT ON COLUMN files.audit_sha256 IS 'Historical audited content hash. Current registration checks file existence and audit byte size, not a fresh raw-content rehash.';
COMMENT ON COLUMN queries.player_support IS '900 logical bits in C order [5,10,18], packbits little. PostgreSQL get_bit gives matching logical index. Full-match audit disposition is never a model input.';

CREATE VIEW support_summary AS SELECT m.split_role,
m.patch,
h.horizon_s,
count(*) AS queries,
count(*) FILTER(WHERE q.selection_supported) AS selection_supported,
count(*) FILTER(WHERE h.economic_supported) AS economic_supported,
count(*) FILTER(WHERE h.terminal_supported) AS terminal_supported,
count(*) FILTER(WHERE h.pareto_supported) AS pareto_supported,
count(*) FILTER(WHERE bit_count(h.event_support)=40) AS all_event_supported,
count(*) FILTER(WHERE get_bit(h.event_support,2)=1 AND get_bit(h.event_support,3)=1 AND get_bit(h.event_support,4)=1 AND get_bit(h.event_support,5)=1 AND get_bit(h.event_support,6)=1 AND get_bit(h.event_support,7)=1 AND get_bit(h.event_support,22)=1 AND get_bit(h.event_support,23)=1 AND get_bit(h.event_support,24)=1 AND get_bit(h.event_support,25)=1 AND get_bit(h.event_support,26)=1 AND get_bit(h.event_support,27)=1) AS all_epic_supported,
count(*) FILTER(WHERE get_bit(h.event_support,0)=1) AS event_0_kills,
count(*) FILTER(WHERE get_bit(h.event_support,1)=1) AS event_0_assists,
count(*) FILTER(WHERE get_bit(h.event_support,2)=1) AS event_0_elem_dragon,
count(*) FILTER(WHERE get_bit(h.event_support,3)=1) AS event_0_elder_dragon,
count(*) FILTER(WHERE get_bit(h.event_support,4)=1) AS event_0_baron,
count(*) FILTER(WHERE get_bit(h.event_support,5)=1) AS event_0_atakhan,
count(*) FILTER(WHERE get_bit(h.event_support,6)=1) AS event_0_herald,
count(*) FILTER(WHERE get_bit(h.event_support,7)=1) AS event_0_horde,
count(*) FILTER(WHERE get_bit(h.event_support,8)=1) AS event_0_outer_turret,
count(*) FILTER(WHERE get_bit(h.event_support,9)=1) AS event_0_inner_turret,
count(*) FILTER(WHERE get_bit(h.event_support,10)=1) AS event_0_base_turret,
count(*) FILTER(WHERE get_bit(h.event_support,11)=1) AS event_0_nexus_turret,
count(*) FILTER(WHERE get_bit(h.event_support,12)=1) AS event_0_inhibitor,
count(*) FILTER(WHERE get_bit(h.event_support,13)=1) AS event_0_plate,
count(*) FILTER(WHERE get_bit(h.event_support,14)=1) AS event_0_item_purchased,
count(*) FILTER(WHERE get_bit(h.event_support,15)=1) AS event_0_item_sold,
count(*) FILTER(WHERE get_bit(h.event_support,16)=1) AS event_0_item_undo,
count(*) FILTER(WHERE get_bit(h.event_support,17)=1) AS event_0_ward_placed,
count(*) FILTER(WHERE get_bit(h.event_support,18)=1) AS event_0_ward_kill,
count(*) FILTER(WHERE get_bit(h.event_support,19)=1) AS event_0_level_up,
count(*) FILTER(WHERE get_bit(h.event_support,20)=1) AS event_1_kills,
count(*) FILTER(WHERE get_bit(h.event_support,21)=1) AS event_1_assists,
count(*) FILTER(WHERE get_bit(h.event_support,22)=1) AS event_1_elem_dragon,
count(*) FILTER(WHERE get_bit(h.event_support,23)=1) AS event_1_elder_dragon,
count(*) FILTER(WHERE get_bit(h.event_support,24)=1) AS event_1_baron,
count(*) FILTER(WHERE get_bit(h.event_support,25)=1) AS event_1_atakhan,
count(*) FILTER(WHERE get_bit(h.event_support,26)=1) AS event_1_herald,
count(*) FILTER(WHERE get_bit(h.event_support,27)=1) AS event_1_horde,
count(*) FILTER(WHERE get_bit(h.event_support,28)=1) AS event_1_outer_turret,
count(*) FILTER(WHERE get_bit(h.event_support,29)=1) AS event_1_inner_turret,
count(*) FILTER(WHERE get_bit(h.event_support,30)=1) AS event_1_base_turret,
count(*) FILTER(WHERE get_bit(h.event_support,31)=1) AS event_1_nexus_turret,
count(*) FILTER(WHERE get_bit(h.event_support,32)=1) AS event_1_inhibitor,
count(*) FILTER(WHERE get_bit(h.event_support,33)=1) AS event_1_plate,
count(*) FILTER(WHERE get_bit(h.event_support,34)=1) AS event_1_item_purchased,
count(*) FILTER(WHERE get_bit(h.event_support,35)=1) AS event_1_item_sold,
count(*) FILTER(WHERE get_bit(h.event_support,36)=1) AS event_1_item_undo,
count(*) FILTER(WHERE get_bit(h.event_support,37)=1) AS event_1_ward_placed,
count(*) FILTER(WHERE get_bit(h.event_support,38)=1) AS event_1_ward_kill,
count(*) FILTER(WHERE get_bit(h.event_support,39)=1) AS event_1_level_up FROM horizons h JOIN queries q USING(row_id) JOIN matches m USING(match_id) GROUP BY m.split_role,m.patch,h.horizon_s;
