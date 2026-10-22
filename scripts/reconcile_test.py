import json, time, requests

BASE = "https://wikidata-reconciliation.wmcloud.org/en/api"
UA   = "entity-equivalence-repro/1.0 (academic reproduction; you@example.edu)"

session = requests.Session()
session.headers.update({"User-Agent": UA, "Accept": "application/json"})

def check_manifest():
    r = session.get(BASE, timeout=30)
    r.raise_for_status()
    m = r.json()
    assert "versions" in m or "defaultTypes" in m, "not a reconciliation manifest"
    return m

def reconcile(names, typ=None, limit=5, retries=5):
    """names -> {name: [ {id,name,score,type} ... ]}"""
    queries = {}
    for i, n in enumerate(names):
        q = {"query": n, "limit": limit}
        if typ:
            q["type"] = typ            # e.g. Q5 human, Q515 city, Q4830453 business
        queries[f"q{i}"] = q

    for attempt in range(retries):
        try:
            r = session.post(BASE, data={"queries": json.dumps(queries)}, timeout=60)
            if r.status_code in (429, 502, 503, 504):
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            data = r.json()
            out = {}
            for i, n in enumerate(names):
                out[n] = data.get(f"q{i}", {}).get("result", [])
            return out
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)   # 1,2,4,8s

print(reconcile(["Mexico City", "Mexico, D.F.", "México City", "New Mexico"]))
