import urllib.request
import re
import json

url = "https://data.mendeley.com/datasets/zkzghjvpn2/6"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
try:
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
        print("Page length:", len(html))
        # Look for download links
        matches = set(re.findall(r'https://[^\s"\'<>]+', html))
        for m in sorted(matches):
            if any(k in m.lower() for k in ["download", "zip", "files", "public"]):
                print("Matched URL:", m)
except Exception as e:
    print("Error:", e)
