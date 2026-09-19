import urllib.request
import os

curated_snos = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 18, 30, 45, 52, 60, 68, 70, 75]
for sno in curated_snos:
    url = f"http://localhost:8000/samples/subject_{sno}_0S.jpg"
    try:
        resp = urllib.request.urlopen(url)
        print(f"Subject #{sno}: HTTP {resp.status}, size={len(resp.read())} bytes")
    except Exception as e:
        print(f"Subject #{sno} FAILED: {e}")
