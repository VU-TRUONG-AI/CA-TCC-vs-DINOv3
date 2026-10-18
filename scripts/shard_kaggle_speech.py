"""Split the verified Speech Commands archive for reliable browser uploads."""
from pathlib import Path
import csv, hashlib, json, zipfile, shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/kaggle_raw_upload'
p=BASE/'speech-commands-v002-raw'
source=p/'speech_commands_v002_raw.zip'
payload=p/'payload';payload.mkdir(exist_ok=True)
map_paths={};shards=[]
with zipfile.ZipFile(source) as original:
    entries=original.infolist();groups=[];group=[];size=0
    for entry in entries:
        if group and size+entry.file_size>550*1024*1024:
            groups.append(group);group=[];size=0
        group.append(entry);size+=entry.file_size
    if group:groups.append(group)
    for i,group in enumerate(groups,1):
        name=f'speech_commands_raw_part_{i:02d}.zip';target=payload/name
        if target.exists():raise RuntimeError('Shard already exists: '+name)
        hashes={}
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for entry in group:
                data=original.read(entry);z.writestr(entry.filename,data)
                hashes[entry.filename]=hashlib.sha256(data).hexdigest()
                map_paths[entry.filename]=(target.stem+'/'+entry.filename,name,entry.filename)
        with zipfile.ZipFile(target) as z:
            for entry in z.infolist():assert hashlib.sha256(z.read(entry)).hexdigest()==hashes[entry.filename]
        with target.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        shards.append({'file':name,'member_count':len(group),'bytes':target.stat().st_size,'sha256':digest})
        print(name,target.stat().st_size,flush=True)
rows=list(csv.DictReader((p/'files.csv').open()))
for row in rows:
    relative,archive,inside=map_paths[row['relative_path']]
    row.update(relative_path=relative,archive_file=archive,archive_relative_path=inside)
with (p/'files.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=json.loads((p/'inventory_summary.json').read_text())
summary['original_single_archive_bytes']=summary.pop('archive_bytes');summary['original_single_archive_sha256']=summary.pop('archive_sha256');summary['upload_shards']=shards;summary['archive_bytes']=sum(s['bytes'] for s in shards);summary['archive_round_trip_all_files_verified']=True;summary['relative_path_base']='Kaggle dataset root after Kaggle expands each ZIP into a directory named after the ZIP stem'
(p/'inventory_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
with (p/'FILE_SHA256SUMS.txt').open() as f:lines=f.read().splitlines()
with (p/'FILE_SHA256SUMS.txt').open('w') as f:
    for line in lines:
        sha,old=line.split('  ',1);f.write(sha+'  '+map_paths[old][0]+'\n')
cache=BASE/'archive_cache';cache.mkdir(exist_ok=True);shutil.move(str(source),str(cache/source.name))
print(json.dumps({'shards':len(shards),'WAV':len(rows),'data_bytes_unchanged':True},indent=2))
