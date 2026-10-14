import sys
import time

import requests

url = "https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json"
out = sys.argv[1] if len(sys.argv) > 1 else "wiki_cn.json"
t0 = time.time()
r = requests.get(url, stream=True, timeout=180)
print("HTTP", r.status_code, "size", r.headers.get("content-length"), flush=True)
r.raise_for_status()
n = 0
with open(out, "wb") as f:
    for chunk in r.iter_content(1 << 20):
        f.write(chunk)
        n += len(chunk)
        if n % (20 << 20) < (1 << 20):
            print(f"{n/2**20:.0f}MB {time.time()-t0:.0f}s", flush=True)
print("DONE", n, flush=True)
