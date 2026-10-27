"""Copy ONLY 12 already-trained BPE model binaries from original Phase30 handoff into portable zip."""
from pathlib import Path
import zipfile,hashlib,json
p=Path(__file__).resolve().parent
src=Path('/mnt/data/isiXhosa_Phase30_Shared_Word_Frequency_BPE_2026-10-24.zip')
manifest=json.loads((p/'RESULTS.json').read_text())['audit']['model_sha256']
names=[v['original_zip_member'] for v in manifest.values()]
out=p/'phase31_frozen_models_only.zip'
with zipfile.ZipFile(src) as si, zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as so:
 for name in sorted(names):
  b=si.read(name)
  assert hashlib.sha256(b).hexdigest() in [v['sha256'] for v in manifest.values()]
  so.writestr(name,b)
print('Portable model archive',out,'bytes',out.stat().st_size,'models',len(names))
