"""Bounded, reproducible public terrain source discovery. No match data access."""
import concurrent.futures, hashlib, json, pathlib, urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parent
class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(v for k,v in attrs if k == 'href')

def fetch(job):
    name,url = job
    receipt = {'name':name,'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Research-source-audit/1.0'})
        with urllib.request.urlopen(req, timeout=25) as r:
            blob=r.read(); receipt['final_url']=r.url
        path=ROOT/'sources'/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(blob)
        receipt.update(bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),status='DOWNLOADED')
        if name.endswith('.html'):
            p=Links(); p.feed(blob.decode('utf-8',errors='replace')); receipt['links']=p.links
    except Exception as e:
        receipt.update(status='FAILED',error=str(e))
    return receipt

if __name__=='__main__':
    import sys
    jobs=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(fetch,jobs))
    out=ROOT/sys.argv[2]; out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in result:
        print(json.dumps({k:v for k,v in r.items() if k not in ('links',)},ensure_ascii=True))
        if 'links' in r and r['name'].startswith('cdragon'): print(json.dumps([x for x in r['links'] if not x.startswith(('http','?'))],ensure_ascii=True))
