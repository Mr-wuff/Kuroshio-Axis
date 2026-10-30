#!/usr/bin/env python3
"""Validate files copied to a GitHub archive against the generated SHA256 manifest."""
from pathlib import Path
import sys,hashlib,csv,collections
root=Path(sys.argv[1] if len(sys.argv)>1 else '.').expanduser().resolve()
manifest=root/'manifests'/'uploaded_files_sha256.csv'
if not manifest.is_file():raise SystemExit(f'Manifest missing: {manifest}')
n=0;fail=[]
with manifest.open(encoding='utf-8-sig',newline='') as f:
 for r in csv.DictReader(f):
  p=root/r['repo_path'];n+=1
  if not p.is_file():fail.append((r['repo_path'],'missing'));continue
  hasher=hashlib.sha256()
  with p.open('rb') as fh:
   for b in iter(lambda:fh.read(4*1024*1024),b''):hasher.update(b)
  if hasher.hexdigest()!=r['sha256']:fail.append((r['repo_path'],'SHA256_mismatch'))
print('Checked',n,'files; problems:',len(fail))
for p,e in fail[:30]:print(e,p)
raise SystemExit(1 if fail else 0)
