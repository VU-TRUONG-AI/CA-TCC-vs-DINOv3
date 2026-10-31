"""Upload small model-release parts sequentially; resume a draft without replacing assets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gh', default='gh')
parser.add_argument('--repo', default='VU-TRUONG-AI/CA-TCC-vs-DINOv3')
parser.add_argument('--tag', default='model-assets-v1')
args=parser.parse_args()
transport_env = {**os.environ, 'GODEBUG': 'http2client=0'}
prefix=[args.gh,'release']
value=subprocess.run([*prefix,'view',args.tag,'--repo',args.repo,'--json','assets,isDraft'],check=True,capture_output=True,text=True,env=transport_env)
release=json.loads(value.stdout)
if not release['isDraft']:
    raise SystemExit('Only resume a draft. Published model versions are immutable.')
existing={a['name']:a['size'] for a in release['assets']}
output=ROOT/'output/deployment'
index=json.loads((output/'deployment-assets.index.json').read_text())
paths=[output/p['url'].rsplit('/',1)[1] for p in index['parts']]
paths += [output/'deployment-assets.index.json', output/'deployment-assets.zip.sha256']
for number,path in enumerate(paths,1):
    if path.name in existing:
        if existing[path.name] != path.stat().st_size:
            raise SystemExit(f'Existing asset size mismatch: {path.name}')
        print(f'Already uploaded {number}/{len(paths)}: {path.name}',flush=True)
        continue
    for attempt in range(3):
        result=subprocess.run([*prefix,'upload',args.tag,str(path),'--repo',args.repo],capture_output=True,text=True,env=transport_env)
        if result.returncode==0:
            print(f'Uploaded {number}/{len(paths)}: {path.name}',flush=True)
            break
        if attempt==2:
            raise SystemExit(f'Upload failed for {path.name}; draft retained for resume.')
        print(f'Retrying {path.name} after upload error',flush=True)
        time.sleep(1)
print('All parts uploaded; verify the asset inventory before publishing.',flush=True)
