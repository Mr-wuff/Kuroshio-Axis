#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct losslessly split derived Kuroshio datasets from a published repo.

No original NetCDF required. Output defaults outside the Git repo tree. Uses only
Python's standard library; refuses unsafe paths and unexpected overwrite.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path, PurePosixPath


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for buf in iter(lambda:stream.read(4*1024*1024),b''):h.update(buf)
    return h.hexdigest()


def under(root:Path, rel:str)->Path:
    p=PurePosixPath(rel)
    if p.is_absolute() or not p.parts or any(seg in ('','.','..') for seg in p.parts):
        raise ValueError(f'Unsafe manifest path: {rel!r}')
    dest=root.joinpath(*p.parts)
    resolved=dest.resolve(strict=False)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes root: {rel!r}')
    return dest


def restore(repo_root: Path, output_root: Path):
    repo_root=repo_root.resolve();output_root=output_root.resolve()
    mfile=repo_root/'manifests'/'split_files_restore.json'
    if not mfile.is_file():raise FileNotFoundError(mfile)
    meta=json.loads(mfile.read_text(encoding='utf-8'))
    entries=meta.get('files',[])
    if not isinstance(entries,list):raise ValueError('Bad split-file manifest')
    done=[]
    for ent in entries:
        dest=under(output_root,ent['original_path'])
        exp=int(ent['original_size_bytes']);hash_exp=ent['original_sha256']
        if dest.exists():
            if dest.stat().st_size==exp and sha256(dest)==hash_exp:
                print('SKIP matching original:',dest);done.append(str(dest));continue
            raise FileExistsError(f'Existing file differs from expected SHA256; no overwrite: {dest}')
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_name(dest.name+'.partial_restore')
        if temp.exists():raise FileExistsError(f'Leftover partial restore: {temp}; review/delete it before rerunning')
        h=hashlib.sha256();written=0
        try:
            with temp.open('xb') as output:
                for part in ent['parts']:
                    f=under(repo_root,part['path'])
                    length=int(part['size_bytes'])
                    if not f.is_file() or f.stat().st_size!=length:
                        raise ValueError(f'Broken part: {f}')
                    with f.open('rb') as stream:
                        while True:
                            data=stream.read(2*1024*1024)
                            if not data:break
                            output.write(data);h.update(data);written+=len(data)
            if written!=exp or h.hexdigest()!=hash_exp:
                raise ValueError(f'FULL SHA256 mismatch: {dest.name}')
            os.replace(temp,dest)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        print('RESTORED:',dest,'bytes=',written)
        done.append(str(dest))
    print(f'CHECKED: {len(done)}/{len(entries)} derived files; no source NetCDF needed')
    return done

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Restore Kuroshio derived datasets from lossless GitHub chunks')
    ap.add_argument('--repo-root',default='.',help='Cloned repository archive subtree')
    ap.add_argument('--output-root',default='./restored_derived_data',help='New safe destination root, not existing input files')
    args=ap.parse_args()
    restore(Path(args.repo_root),Path(args.output_root))
