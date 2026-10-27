"""Freeze eight small public-transcript excerpts before constructing any retrieval model."""
import hashlib,json
from pathlib import Path
import regex as re
P=Path(__file__).parent
ROOT=P.parent
W=re.compile(r'(?V1)\p{L}+(?:[\'’]\p{L}+)*')
examples=[
 ('IMVO_1885-08-26_p001.txt','https://cdm21048.contentdm.oclc.org/digital/collection/p21048coll37/id/1477/',
  'Nomtengisi wempahla ezinjengemisesane njalo njalo. Unentlobo ntlobo zemisesane, iwotshi, Nentsimbi ezinkulu zamaxeeba, nentvvane ezintle Zokinika amabaso ngamaxesha ae.'),
 ('IMVO_1889-05-30_p001.txt','https://cdm21048.contentdm.oclc.org/digital/collection/p21048coll37/id/2921/',
  'Saneke Ityali ezitshipu kanye, ifishini ezona zintsha, ziqala kwi 3/- inye. /kaliko Emhlope ne Brown, iqala 2^d. yd.'),
 ('IMVO_1889-06-13_p001.txt','https://cdm21048.contentdm.oclc.org/digital/collection/p21048coll37/id/3103/',
  'GANTSHO — XANTI. — Kutshatiswe e Ndwana ngomhla we 23rd April, u Mr. Boot J. Gantsho, wase Queen’s Town, no Miss Ellen M. Xanti'),
 ('IMVO_1890-09-04_p003.txt','https://cdm21048.contentdm.oclc.org/digital/collection/p21048coll37/id/3450/',
  'Emva kwalenteto ingxoxo irwaqelele pezulu kwizifundo ze University, ete yala- twa amadlala ngo Messrs Marais, Merriman, yatetelelwa ngo Messrs Fuller, Ohlsson.'),
 ('IMVO_1892-08-18_p001.txt','https://cdm21048.contentdm.oclc.org/digital/collection/p21048coll37/id/501/',
  'KUNG-OKU BABONISA Ezona Fashini Zizezintsha, Nengubo Ezifezekileyo ngamaxabiso apantsi anokufumaneka. Ihempe, Amaqhina, Izitadi, Izihlangu, ne Shuzi; Ingcawa, Amafelane, ItyaiPli'),
 ('IMVO_1893-06-14_p001.txt','https://cdm21048.contentdm.oclc.org/digital/collection/p21048coll37/id/268/',
  'KUNGOKU BABONISA Ezona Fashini Zizezintsha, Nengubo Ezifezekileyo ngamaxabiso apantsi anokufumaneka. Ihempe, Amaqhina, Izitadi, Izihlangu, ne Shuzi Ingcawa, Amafelane, Itya i Pli'),
 ('ISIG_1870-10-14_p001.txt','https://cdm21048.contentdm.oclc.org/digital/collection/p21048coll72/id/9/',
  'TELEGRAPHIC SUPPLEMENT TO (Ênjlislj-|iïtffir journal. LOVEDALE, FRIDAY, 4 p.m., 14th Oct., 1870.'),
 ('ISIG_1882-06-01_p004.txt','https://cdm21048.contentdm.oclc.org/digital/collection/p21048coll72/id/963/',
  'Beku- sand’uku puma u Mr. W. E. Forster kulendawo, wabe ku- tshwa kukungavumelani nabanye abalaula kunye naye, ngendawo yoku kululwa kwamadoda.')
]
D={json.loads(l)['id']:json.loads(l) for l in (ROOT/'pages.jsonl').open()}
assert len(D)==3608
files=[ROOT/'isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl',ROOT/'phase19/queries_precommit.jsonl',ROOT/'phase20/queries_frozen.jsonl',ROOT/'phase21/queries_frozen.jsonl']
previous=[{json.loads(l)['issue'] for l in f.open()} for f in files]
records=[]
for page_id,url,excerpt in examples:
 assert page_id in D
 nt=len(W.findall(excerpt))
 assert nt<=24, (page_id,nt)
 rec={'page_id':page_id,'issue':D[page_id]['issue'],'source':D[page_id]['source'],'query_external':excerpt,
  'url':url,'external_excerpt_alphabetic_words':nt,'prior_query_origin_phases':[18+i for i,S in enumerate(previous) if D[page_id]['issue'] in S],
  'material_type':'NLSA_page_linked_transcript_excerpt','ocr_provenance_verified':False}
 records.append(rec)
with (P/'queries_external_frozen.jsonl').open('w') as f:
 for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
sha=hashlib.sha256((P/'queries_external_frozen.jsonl').read_bytes()).hexdigest()
(P/'QUERY_SHA256.txt').write_text(sha+'  queries_external_frozen.jsonl\n')
print('Frozen',len(records),'external excerpts',sha)
for r in records:print(r['page_id'],r['external_excerpt_alphabetic_words'],'words, prior-origin',r['prior_query_origin_phases'])
