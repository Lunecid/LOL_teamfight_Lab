"""Preserve executor packet; create a version clarifying change vs absolute advantage."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/label_validity_full_20260915/review_packet'
OUT=ROOT/'outputs/label_observation_audit_20260915/review_packet_clarified'
OUT.mkdir(parents=True,exist_ok=True)
source=SRC/'REVIEWER_PACKET_KO.md'
text=source.read_text(encoding='utf8')
old='- (b) 표시된 구간 전체의 전략적 우세: `Blue` / `Red` / `neutral` / `insufficient`'
new='- (b) 구간 시작과 비교해 종료 시점에 어느 팀 쪽으로 전략적 이점이 개선되었는가: `Blue` / `Red` / `neutral` / `insufficient`. 종료 시점에 단순히 더 유리한 팀을 고르는 문항이 아니다. 원래 앞서던 팀의 우세가 줄었다면 상대 팀 쪽 개선일 수 있다. 관측 부족으로 변화 방향을 판단할 수 없으면 `insufficient`로 기록한다.'
assert text.count(old)==1
corrected=text.replace(old,new)
corrected=corrected.replace('## 안내','## 안내\n\n문항 b를 절대 우세가 아닌 **시작 대비 종료의 변화 방향**으로 명확히 한 설계 보완본이다. 사례 데이터와 사례 순서는 원본과 같다. 표시 시각은 읽기 편하게 반올림되어 pre와 s가 같아 보일 수 있지만, 실제 계산은 pre=s−1ms를 유지한다.\n',1)
target=OUT/'REVIEWER_PACKET_KO.md'
target.write_text(corrected,encoding='utf8')
with (SRC/'review_form_blank.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.reader(f))
assert 'b_strategic_advantage' in rows[0]
rows[0]=['b_strategic_change' if c=='b_strategic_advantage' else c for c in rows[0]]
assert len(rows)==121 and all(not any(r[1:]) for r in rows[1:])
with (OUT/'review_form_blank.csv').open('w',encoding='utf-8-sig',newline='') as f:
    csv.writer(f).writerows(rows)
# Exact case body identity (wording edits occur only before LV-001).
assert corrected.split('## LV-001',1)[1]==text.split('## LV-001',1)[1]
rec={'role':'Codex design clarification; no case or scientific model change','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'clarified_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'case_bodies_identical':True,'cases':120,'human_review':'UNPERFORMED','ratings_filled':0,'changed':'Question b now asks pre/post strategic change, not absolute team advantage; matching blank CSV field renamed; timestamp display precision disclosed','source_preserved':True}
(OUT/'clarification_manifest.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(rec,ensure_ascii=False))
