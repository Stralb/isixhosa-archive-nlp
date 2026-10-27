#!/usr/bin/env python3
"""Build OFFLINE annotator HTML from archived pages. Local copy contains historical text.

Do not distribute the rendered HTML until corpus rights/permissions are checked.
The HTML does NOT include retrieval system names, ranks, or ground truth labels.
"""
from pathlib import Path
import csv,json,argparse,html

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--pages',required=True);p.add_argument('--assessor-sheet',required=True)
 p.add_argument('--out',required=True);p.add_argument('--max-pairs',type=int,default=0)
 a=p.parse_args()
 with Path(a.assessor_sheet).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 if a.max_pairs:rows=rows[:a.max_pairs]
 needed={r['document_id'] for r in rows}
 docs={}
 for s in Path(a.pages).open(encoding='utf-8'):
  rec=json.loads(s)
  if rec['id'] in needed:
   docs[rec['id']]={'body':rec['raw'],'source':rec['source'],'date':rec['date']}
 missing=needed-set(docs)
 if missing:raise SystemExit(f'Missing documents: {sorted(missing)[:4]}')
 content={'rows':rows,'docs':docs}
 data=json.dumps(content,ensure_ascii=False).replace('</','<\\/')
 page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Blind historical relevance annotation</title>
<style>body{font:16px/1.48 system-ui;margin:auto;max-width:1000px;padding:20px;background:#faf9f6;color:#212121} header{background:#e4e7f0;padding:12px;border-radius:8px}pre{white-space:pre-wrap;overflow:auto;max-height:56vh;border:1px solid #bbb;background:white;padding:14px;border-radius:6px}select,input,textarea,button{font:inherit;padding:7px;margin:4px}button{cursor:pointer}label{display:block;margin:8px}fieldset{border:1px solid #c7c7c7}#stamp{font-size:12px;color:#555}</style>
<header><strong>Blind relevance-judgment worksheet</strong> — Rank and retrieval systems hidden.<p>Independent assessment only. Exact historical text may contain discriminatory or colonial language. Semantic evidence, exact textual reuse, and edition provenance are separate questions.</p></header>
<div id="stamp"></div><h2 id="qid"></h2><p id="query"></p><p id="source"></p><div><button onclick="prev()">Previous</button><button onclick="next()">Save and next</button><button onclick="saveCSV()">Download all judgments CSV</button></div>
<p><strong>Candidate page:</strong> <span id="docid"></span></p><pre id="body"></pre>
<fieldset><legend>Evidence judgments</legend>
<label>Answer-bearing semantic relevance: <select id="relevance"><option value="">Not yet judged</option><option value="0">0 — unrelated</option><option value="1">1 — partially answers</option><option value="2">2 — directly answers with evidence</option><option value="U">U — uncertain</option></select></label>
<label>Correct original edition/provenance for query: <select id="provenance_match"><option value="">Not yet judged</option><option value="0">0 — no</option><option value="1">1 — yes</option><option value="U">U — uncertain</option></select></label>
<label>Meets exact textual reuse requested by query: <select id="verbatim_match"><option value="">Not yet judged</option><option value="0">0 — no</option><option value="1">1 — yes</option><option value="U">U — uncertain</option></select></label>
<label>Confidence: <select id="confidence"><option value="">Not yet judged</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label>
<label>Quote evidence span / explain uncertainty: <textarea id="adjudication_notes" rows="3" style="width:95%"></textarea></label></fieldset>
<p><strong>Saving:</strong> Browser storage is not used. Press “Download all judgments CSV” repeatedly during work; keep the newest file. Finished CSV is the ONLY saved evidence.</p>
<script id="corpus" type="application/json">___DATA___</script><script>
const payload=JSON.parse(document.getElementById('corpus').textContent);const rows=payload.rows;const docs=payload.docs;let i=0;
const fields=['relevance','provenance_match','verbatim_match','confidence','adjudication_notes'];
function capture(){if(!rows.length)return;for(let f of fields){rows[i][f]=document.getElementById(f).value;}}
function render(){const r=rows[i];document.getElementById('stamp').textContent=(i+1)+' / '+rows.length+' — Query '+r.query_id+', candidate '+r.candidate_id;
document.getElementById('qid').textContent=r.query_id+' · '+r.intent;
document.getElementById('query').textContent='Search need: '+r.query_text;
document.getElementById('source').textContent='Requested newspaper(s): '+r.source_scope+' · query language '+r.language;
document.getElementById('docid').textContent=r.document_id+' (publication '+docs[r.document_id].source+'; date '+docs[r.document_id].date+')';
document.getElementById('body').textContent=docs[r.document_id].body;
for(let f of fields){document.getElementById(f).value=r[f]||'';}window.scrollTo(0,0);}
function next(){capture();if(i<rows.length-1)i++;render();}function prev(){capture();if(i>0)i--;render();}
function csvLine(a){return a.map(x=>'"'+String(x??'').replaceAll('"','""')+'"').join(',');}
function saveCSV(){capture();const names=['query_id','candidate_id','query_text','source_scope','intent','language','document_id',...fields];const data=[csvLine(names),...rows.map(r=>csvLine(names.map(n=>r[n])))].join('\r\n');const blob=new Blob([data],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='completed_judgments.csv';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),3000);}
render();</script></html>'''.replace('___DATA___',data)
 dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(page,encoding='utf-8')
 print(f'Generated {len(rows)} blinded candidates, {len(docs)} distinct page texts, HTML {dest.stat().st_size:,} bytes')

if __name__=='__main__':main()
