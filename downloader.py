import os
from pathlib import Path
import urllib.request
from bs4 import BeautifulSoup

url = "https://{}.myanmarexam.org/{}"
years = [ 2016]
links = [
    "ygn.html", "mdy.html", "npw.html", "ayy.html", "sgg.html", "mgy.html", "bgo.html", "tni.html", "kcn.html",
    "kyh.html", "kyn.html", "chn.html", "mon.html", "rke.html", "shn.html",
]


for y in years:
    Path('data/{}'.format(y)).mkdir(parents=True, exist_ok=True)
    for l in links:
        u = url.format(y, l)
        print(u)
        with urllib.request.urlopen(u) as p:
            page = p.read()
            soap = BeautifulSoup(page, 'html.parser')
            links = soap.find_all('a')
            for l in links:
                pdf_lnk = l.get('href')
                if not pdf_lnk.endswith('pdf'):
                    continue

                response = urllib.request.urlopen(pdf_lnk)
                print(Path(pdf_lnk).name)
                with open('data/{}/{}'.format(y, Path(pdf_lnk).name), 'wb') as f:
                    f.write(response.read())
