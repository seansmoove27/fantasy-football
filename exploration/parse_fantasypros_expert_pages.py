import sys,re,json
from bs4 import BeautifulSoup
out={}
for f in sys.argv[1:]:
    h=open(f).read(); s=BeautifulSoup(h,'html.parser')
    t=s.find('table'); title=s.title.get_text() if s.title else ''
    if not t or 'DST Rankings' not in title: continue
    wk=re.search(r'Week \d+ - [A-Za-z]{3} \d+, \d{4}',h)
    rows=[[c.get_text(' ',strip=True) for c in r.find_all('td')] for r in t.find_all('tr')[1:]]
    out[title.split('|')[0].strip()]={'week':wk.group(0) if wk else None,'rows':[r[:3] for r in rows if r]}
json.dump(out,open('ranks.json','w'),indent=1)
for k,v in out.items(): print(k,v["week"],len(v["rows"]),v["rows"][:3])
