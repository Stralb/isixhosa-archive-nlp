"""Blind-to-retrieval manual query records and literal source-span multi-gold definition."""
from pathlib import Path
import json,hashlib,re,collections
import regex as rx
BASE=Path('/mnt/data/isiXhosa_continuation');OUT=BASE/'phase19'
I=[json.loads(x) for x in (BASE/'review_IMVO.jsonl').open()]
S=[json.loads(x) for x in (BASE/'review_ISIG.jsonl').open()]
T=[json.loads(x) for x in (BASE/'review_ISIG_extra.jsonl').open()]
queries_imvo={
 0:'Schreiner Kimberley ministry agriculture portfolio Cape parliament',
 1:'Franchise reform debate Sprigg Schermbrucker legislative council',
 2:'Victoria East candidate William Hay insurance manager',
 3:'Victor Sampson Cala meeting Jabavu Rhodes Tembuland',
 4:'Liquor trade election candidate Bailey Bradfield Peacock Rogers',
 5:'George Hay reform party parliamentary opposition principles',
 6:'Governor Loch portrait issued to Imvo subscribers',
 7:'Transkei Fingoes forestry dispute Kaffrarian Watchman newspaper',
 8:'George Bond electoral branch Robertson Vincent de Smidt',
 9:'1895 school syllabus Milton Shakespeare Latin history examinations',
 10:'Tamplin Peddie election constituency backing Victoria East',
 11:'Sauer newspaper review Wobblers political party dispute',
 12:'Rhodes legislation Queen royal assent native petition opposition',
 13:'Hutton parliamentary remarks Glen Grey Rhodes Sprigg legislation',
 14:'Pondoland annexation by imperial authorities Bechuanaland Basutoland',
 15:'Native administration twelve year government system proposed reforms',
 16:'Kokstad Advertiser allegations violence Pondoland newspaper reporting',
 17:'Local option liquor policy native voter mobilisation restrictions',
 18:'Glen Grey land dispossession proposed after census returns',
 19:'Kingwilliamstown election Schermbrucker Warren Hay vote totals',
 20:'Uitenhage election Bidwell Nixon Bond Humansdorp candidate',
 21:'Hofmeyr franchise proposal Cape cabinet instability rumours',
 22:'Pondoland threatened military manoeuvres border government pressure',
 23:'Pondo ultimatum war invasion Matebeleland precedent',
 24:'Bond party electoral organising independent candidates losses',
 25:'Murray disputes Cape Argus over surveys in Glen Grey',
 26:'Newspaper ownership bill enabling corporate stock shareholding',
 27:'Kimberley by election O Leary Solomon candidacy',
 28:'Cradock Farmers Congress participation political affairs Bond',
 29:'Tembuland Molteno Griffith legislative representation criticism',
 30:'Sesutu language teaching Basuto literary development missionaries',
 31:'Benson Kawa Peddie trading licence Civil Commissioner decision',
 32:'Maraisburg meeting Joubert Glen Grey farmland controversy',
 33:'Kingwilliamstown Victoria East franchise property valuation pasture',
 34:'Somerset East Moolman de Wet Bond franchise qualification',
 35:'Quitrent land annual rental valuation seven acre holdings',
 36:'Kimberley Congress Bond Farmers Association delegates excluded',
 37:'Namaqualand representative Charles Lewis banking collapse resignation',
 38:'Uitenhage John Mackay Bond Vanes Rautenbach election',
 39:'Lovedale Scottish patient nursing sickness education 1870'
}
queries_isig={
 ('main',0):'Cape Pondo peaceful dispute settlement government Sprigg resolution',
 ('main',1):'Telegraph delivery charges messenger porterage distances miles',
 ('main',2):'Napoleon defeated emperor captive German castle power loss',
 ('main',3):'Dismissed teacher missionary recommendation disciplinary misconduct allegation',
 ('main',5):'Crawshaw introductory Kafir grammar word structure textbook evaluation',
 ('main',7):'Alice hotel licence Richard Austen Victoria East magistrate hearing',
 ('main',9):'Victoria East timber theft forest regulations reward pounds',
 ('main',10):'Alice district surgeon free Tuesday vaccination children',
 ('main',11):'South African congratulations Queen survived assassination attempt',
 ('main',12):'Municipal regulation beer sellers canteens brandy near town locations',
 ('extra',1):'Matriculation curriculum modern languages Sesutu English Latin',
 ('extra',7):'Daily labour vacancies wage three shillings workers',
 ('extra',10):'Mrs Brownlee obituary London missionary January Brownlee',
 ('extra',12):'Binfield Park poundmaster Humphreys Tyumie council notice',
 ('extra',15):'Colonial house tax revenue payable July occupancy notice'
}
W=rx.compile(r'(?V1)\p{L}+(?:[\'’]\p{L}+)*')
records=[]
for idx, query in queries_imvo.items():records.append(('IMVO',str(idx),I[idx],query))
for (which,idx), query in queries_isig.items():records.append(('ISIG',which+str(idx),(S if which=='main' else T)[idx],query))
# Source evidence snippet from the paragraph, distinct from query text; choose a 13-word first-central excerpt.
# All source-pages containing it as a contiguous raw paragraph get counted as a verified-equivalent gold page.
arr=[]
for k,(source,idx,d,query) in enumerate(records):
 ref=d['paragraph'];words=[x.lower() for x in W.findall(ref)];qwords=[x.lower() for x in W.findall(query)]
 qgrams=set(qwords[i:i+3][0]+'|'+qwords[i:i+3][1]+'|'+qwords[i:i+3][2] for i in range(len(qwords)-2))
 sgrams=set(words[i]+'|'+words[i+1]+'|'+words[i+2] for i in range(len(words)-2))
 overlap=sorted(qgrams&sgrams)
 if overlap:print('THREE_GRAM_OVERLAP',source,idx,overlap)
 # Anchor evidence span: beginning of a 13-word window near the center, independent of ranking
 mid=max(0,(len(words)-13)//2)
 ev=' '.join(words[mid:mid+13]); assert len(ev.split())==13
 assert len(qwords)>=5
 arr.append(dict(qid=f'P19_{source}_{idx}',source=source,issue=d['issue'],source_page=d['id'],source_para_index=d['index'],text=query,
                anchor_evidence_tokens=ev,view='en',word_count=len(qwords),threegram_source_overlap=overlap,
                source_paragraph=ref,category='newspaper_editorial_or_notice',query_origin='assistant_composed_from_source_paragraph_before_retrieval'))
# Freeze only if no threegram matches. All distinct issue IDs and outside Phase18 source issue selection.
blocked={z['issue'] for z in map(json.loads,(BASE/'isiXhosa_phase18_evidence_routing/balanced_en_queries.jsonl').open())}
assert all(a['issue'] not in blocked for a in arr)
assert len(set(a['issue'] for a in arr))==len(arr)
assert all(not a['threegram_source_overlap'] for a in arr),'Must correct overlaps before freezing'
P=OUT/'queries_precommit.jsonl'
P.write_text(''.join(json.dumps({k:v for k,v in z.items() if k!='source_paragraph'},ensure_ascii=False)+'\n' for z in arr))
(OUT/'anchor_paragraphs_audit.jsonl').write_text(''.join(json.dumps({'qid':z['qid'],'source_page':z['source_page'],'paragraph':z['source_paragraph']},ensure_ascii=False)+'\n' for z in arr))
print('queries',len(arr),'source',collections.Counter(a['source'] for a in arr),'page numbers',collections.Counter(a['source_page'].split('_')[-1] for a in arr))
print('QUERY SHA256',hashlib.sha256(P.read_bytes()).hexdigest())
