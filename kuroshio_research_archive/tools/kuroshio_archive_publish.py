#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kuroshio research archival/publishing tool for Windows WSL.

Collects final data, human corrections, reproducible scripts, and terminal
experimental results into a safe, traceable GitHub repository subtree.

Default: publishes ONLY derived/frozen datasets, human annotations, source
code and final reviewer-facing results. Raw CMEMS/ADT NetCDF are NEVER scanned,
indexed, copied, staged or published. Large derived files are split losslessly
by default so Git LFS is not required. Daily axes can be grouped into yearly ZIPs.
Only the managed `kuroshio_research_archive/` subtree is staged in Git.

Python 3.10+, standard library only. No existing source data are modified.
"""
from __future__ import annotations

import argparse
import collections
import csv
import contextlib
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

TOOL_VERSION = '1.2.0'
GB = 1024 ** 3
MB = 1024 ** 2
SUBTREE = 'kuroshio_research_archive'
CODE_SUFFIXES = {'.py', '.ipynb', '.m', '.sh', '.bat', '.ps1', '.r', '.jl'}
DOC_SUFFIXES = {'.md', '.txt', '.csv', '.json', '.yaml', '.yml', '.toml'}
RESULT_SUFFIXES = {'.csv', '.json', '.txt', '.md', '.png', '.pdf', '.svg', '.parquet'}
EXPERIMENT_EXT = RESULT_SUFFIXES | {'.zip'}
RAW_SUFFIXES = {'.nc', '.nc4', '.cdf'}
BAD_DIRS = {
    '.git', '.venv', 'venv', 'env', 'node_modules', '__pycache__', '.ipynb_checkpoints',
    '.cache', '.pytest_cache', '.mypy_cache', 'tmp', 'temp', 'temporary', 'swap',
    'checkpoints', 'checkpoint', 'model_weights', 'models', 'weights', 'logs',
    'train_cache', 'datasets_cache', 'training_cache', 'output_cache', 'cached',
    '模型权重', '检查点', '模型缓存', '中间权重', '中间结果', '候选权重',
    'wandb', 'tensorboard', 'lightning_logs', 'runs', 'cache',
    'raw_nc', 'raw_netcdf', 'netcdf', 'ncfile', 'source_nc', '源数据',
    'adt_axes_npz', 'saved_forecast_paths', 'tracks_auto', 'tracks_final', 'manual_cache_50m',
}
BLOCKED_PARTS = {
    '.git', '.venv', 'venv', 'env', '__pycache__', 'checkpoints', 'checkpoint',
    'model_weights', 'models', 'weights', 'train_cache', 'cache', 'wandb',
    'tensorboard', '模型权重', '候选权重', '中间结果', 'debug', 'logs',
}
BLOCKED_FILE_NAMES = {
    '.env', '.env.local', '.env.production', '.env.development', 'id_rsa',
    'id_ed25519', 'credentials.json', 'secrets.json', 'token.json',
    '.npmrc', '.pypirc', 'authorized_keys', 'known_hosts', 'config.ini',
}
SECRET_RE = [
    re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
    re.compile(r'github_pat_[A-Za-z0-9_]{40,}'),
    re.compile(r'sk-(?:proj-)?[A-Za-z0-9_-]{30,}'),
    re.compile(r'hf_[A-Za-z0-9]{30,}'),
    re.compile(r'AKIA[0-9A-Z]{16}'),
    re.compile(r'AIza[0-9A-Za-z_-]{35}'),
    re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    re.compile(r'(?im)^\s*(?:password|passwd|secret_key|access_token|api_key)\s*[:=]\s*[\'\"]?[^\s#\'\"]{18,}'),
]
SOURCE_DATASET_PATTERNS = ('*daily_dual_branch*FINAL*', '*_FINAL.npz', '*_FINAL.csv')


def log(*a):
    print('[Kuroshio Archive]', *a, flush=True)


def call(cmd, cwd=None, capture=False, check=True):
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None,
                       check=False)
    if check and p.returncode:
        message = (p.stderr or '').strip() if capture else f'error code {p.returncode}'
        raise RuntimeError(f'Command failed: {" ".join(map(str, cmd))}\n{message}')
    return p.stdout.strip() if capture else p.returncode


def has_tool(command):
    return shutil.which(command) is not None


def safe_path(p: Path) -> bool:
    if p.is_symlink():
        return False
    if p.name.lower() in BLOCKED_FILE_NAMES:
        return False
    if p.suffix.lower() in ('.pem', '.key', '.p12', '.pfx', '.sqlite', '.db'):
        return False
    if any(x.lower() in ('.ssh', '.aws', '.config', '.kube', 'credential', 'secrets') for x in p.parts):
        return False
    return True


def allowed_regular(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and safe_path(path)
    except (OSError, ValueError):
        return False


def walk_safe(root: Path, *, max_depth=12, skip=BAD_DIRS):
    root = Path(root)
    if not root.is_dir():
        return
    base_depth = len(root.parts)
    for folder, dirs, names in os.walk(root, topdown=True, followlinks=False):
        here = Path(folder)
        dirs[:] = [d for d in sorted(dirs) if d.lower() not in {x.lower() for x in skip}
                   and not d.startswith('.') and not (here / d).is_symlink()]
        if len(here.parts) - base_depth >= max_depth:
            dirs[:] = []
        for name in sorted(names):
            f = here / name
            if allowed_regular(f):
                yield f


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * MB), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def fmt_size(n):
    if n >= GB:
        return f'{n / GB:.2f} GiB'
    return f'{n / MB:.2f} MiB'


def secret_signatures(data: bytes) -> list[str]:
    text = data.decode('utf-8', errors='replace')
    return [f'pattern_{i + 1}' for i, pat in enumerate(SECRET_RE) if pat.search(text)]


def sanitize_notebook(src: Path) -> bytes:
    """Strip outputs, traceback, widgets, and execution state; keep source cells."""
    doc = json.loads(src.read_text(encoding='utf-8-sig'))
    if not isinstance(doc, dict) or not isinstance(doc.get('cells'), list):
        raise ValueError('Invalid Jupyter notebook structure')
    for cell in doc['cells']:
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        cell['metadata'] = {k: v for k, v in cell.get('metadata', {}).items()
                            if k not in ('execution', 'collapsed', 'scrolled', 'widgets', 'trusted')}
    meta = doc.get('metadata', {})
    doc['metadata'] = {k: v for k, v in meta.items()
                       if k not in ('widgets', 'vscode', 'execution', 'colab')}
    return (json.dumps(doc, ensure_ascii=False, indent=1) + '\n').encode('utf-8')


def relative_label(root: Path, i: int) -> str:
    return f'{i + 1:02d}_{re.sub(r"[^a-zA-Z0-9_-]", "_", root.name)[:45]}'


def code_category(path: Path) -> str:
    v = path.name.lower()
    parts = ' '.join(p.lower() for p in path.parts[-5:])
    if any(k in v + parts for k in ('dataset', 'builder', 'data_build', 'databuild', '源合并', '构建', 'ndbc', 'cmems', 'glo_phy')):
        return '01_dataset_build'
    if any(k in v + parts for k in ('manual', 'annotation', '人工标', 'manualqc', 'manual_qc', '复核', '交互标注')):
        return '02_manual_annotation'
    if any(k in v + parts for k in ('figure', 'figure2', 'fig', 'plot', '画图', '绘图', 'r80', 'r81', 'visual')):
        return '05_visualization'
    if any(k in v + parts for k in ('r70', 'r71', 'r7_0', 'r7_1', 'adt', 'baseline', 'hindcast', 'blocked', 'cv', 'qualification', 'uncertainty', '评估', '验证')):
        return '04_experiment_evaluation'
    if any(k in v + parts for k in ('train', 'lstm', 'r65', 'r6_', 'model', '预测', '训练', 'crossdrop')):
        return '03_model_training'
    return '06_other_scripts'


def final_result_dir(name: str) -> bool:
    lo = name.lower()
    return any(x in lo for x in (
        'final_retrain', 'final_results', 'finaltraining', 'formal', '正式', '最终实验',
        'blockedcv', 'blocked_cv', 'blocked-cv', 'rolling', 'hindcast', '基线', '多年度',
        'r6_5_', 'r65_', 'r70_', 'r71_', 'r7_0_', 'r7_1_', 'r80_', 'r81_', 'r7.0', 'r7.1',
        'adt_independent', 'adt_验证', '独立验证', 'adt结果', 'review_figures',
        'uncertainty', 'stable', 'stability', '统计检验', '论文最终',
    ))


def final_result_zip(name: str) -> bool:
    lo = name.lower()
    return (any(x in lo for x in (
        'final_retrain', 'final_results', 'rolling', 'hindcast', 'blockedcv',
        'r65_', 'r70_', 'r71_', 'r7_0_', 'r7_1_', 'r80_', 'r81_',
        'adt独立', '正式', '最终', '审稿人', '投稿图件', 'figure2-8',
    )) and not re.search(r'(?i)\br(?:1|2|3|4|5|51|52|53|54|55|56|57|60|61|62|63|64)[_-]', lo))


def is_intermediate_output(p: Path):
    if any(x.lower() in {b.lower() for b in BLOCKED_PARTS} for x in p.parts):
        return True
    text = p.name.lower()
    return (p.suffix.lower() in {'.pt', '.pth', '.ckpt', '.safetensors', '.h5', '.onnx'} or
            text.startswith(('temp_', 'tmp_', 'debug_', 'latest_', 'epoch_', 'step_')) or
            text.endswith(('.partial', '.tmp', '.bak', '.swp')))


def gh_known_repos():
    """Return owned/accessible repo names in the authenticated WSL gh CLI.

    Display-only; never choose a repository based on a numeric list without
    an explicit user action. No tokens or auth files are read by this code.
    """
    if not has_tool('gh'):
        return []
    try:
        rows = json.loads(call(['gh', 'repo', 'list', '--limit', '200',
                                '--json', 'nameWithOwner'], capture=True))
        return sorted({normalize_repo(row['nameWithOwner'])
                       for row in rows if row.get('nameWithOwner')})
    except Exception as e:
        log('Cannot list gh repositories:', type(e).__name__)
        return []


def discover_repositories(args):
    if args.repo:
        return normalize_repo(args.repo)
    found = []
    for p in [Path.cwd(), Path(args.root).expanduser(),
              *[Path(x).expanduser() for x in args.extra_root]]:
        if p.exists() and has_tool('git'):
            try:
                result = call(['git', 'remote', 'get-url', 'origin'], cwd=p, capture=True)
                if 'github.com' in result:
                    found.append(normalize_repo(result))
            except Exception:
                pass
    for name in gh_known_repos():
        if re.search('kuroshio|black.?current|axis|黑潮', name, re.I):
            found.append(name)
    found = sorted(set(found))
    if len(found) == 1:
        log('Automatically found repository:', found[0]);return found[0]
    if len(found)>1:
        log('Multiple possible repositories found; explicit selection required:')
        for name in found:
            log('  -', name)
    return None


def normalize_repo(repo: str) -> str:
    repo = repo.strip()
    if not repo:
        raise ValueError('仓库地址为空。请传 --repo Mr-wuff/实际仓库名，或在提示时粘贴完整 https://github.com/owner/repo URL。')
    if repo.startswith('git@github.com:'):
        repo = repo.split(':', 1)[1]
    elif repo.startswith(('https://github.com/', 'http://github.com/')):
        repo = urlparse(repo).path.lstrip('/')
    elif repo.startswith('ssh://git@github.com/'):
        repo = urlparse(repo).path.lstrip('/')
    repo = repo.removesuffix('.git').strip('/')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo):
        raise ValueError('仓库地址格式错误：必须为 Mr-wuff/仓库名 或 https://github.com/Mr-wuff/仓库名；请不要只填写用户名。')
    return repo


def prompt_repository(args):
    """Explicitly choose a remote. Blank input reprompts instead of crashing."""
    detected=discover_repositories(args)
    if detected:
        return detected
    if not sys.stdin.isatty():
        raise RuntimeError('需要 GitHub 仓库名。请带 --repo Mr-wuff/实际仓库名 后重新运行。')
    known=gh_known_repos()
    if known:
        log('GitHub CLI lists these accessible repositories:')
        for i,repo in enumerate(known[:60],1):
            print(f'    {i:2d}. {repo}',flush=True)
        log('输入编号可选择对应仓库；也可输入完整 owner/repo。务必确认新建的目标仓库。')
    for attempt in range(5):
        try:
            reply=input('请输入 GitHub 仓库地址 / 编号（q 取消）：').strip()
        except EOFError:
            raise RuntimeError('终端无法读取仓库输入。改用 --repo Mr-wuff/实际仓库名。') from None
        if reply.lower() in ('q','quit','exit'):
            return None
        if not reply:
            log('收到空输入，尚未上传。请输入实际仓库地址，或 q 取消。')
            continue
        if reply.isdecimal() and 1 <= int(reply) <= min(len(known),60):
            reply=known[int(reply)-1]
        try:
            return normalize_repo(reply)
        except ValueError as e:
            log(str(e))
    raise RuntimeError('5 次输入均未提供有效仓库名。请显式传 --repo Mr-wuff/实际仓库名。')


def repository_preflight(repo):
    """Validate a user-selected repository before expensive file inventory."""
    if not has_tool('gh'):
        log('gh not installed; Git clone will validate the supplied repository later.')
        return
    try:
        name=call(['gh','repo','view',repo,'--json','nameWithOwner',
                   '--jq','.nameWithOwner'],capture=True)
    except Exception as e:
        raise RuntimeError(
            f'GitHub CLI 无法访问仓库 {repo}。请先运行 gh repo list 或检查 --repo；原始数据未修改。'
        ) from e
    if name.lower() != repo.lower():
        raise RuntimeError(f'GitHub 目标仓库异常：请求 {repo}，返回 {name}')
    log('Remote repository verified:',name)

@dataclass
class Item:
    src: Path
    dest: str
    group: str
    origin: str
    size: int
    subtype: str = 'file'  # notebook, normal, zip_member
    member: str = ''
    generated: bytes | None = None
    use_lfs: bool = False
    chunk_offset: int = 0
    bundle_files: tuple[str, ...] = ()


class Collector:
    def __init__(self, args):
        self.args = args
        self.roots = [Path(args.root).expanduser().resolve()]
        # Discover sibling legacy Kuroshio experiment folders (R1, old WSL runs).
        candidates=list(args.extra_root)
        if not args.no_auto_extra_roots:
            parent=self.roots[0].parent
            if parent.is_dir():
                for p in sorted(parent.iterdir()):
                    lo=p.name.lower()
                    if (p.is_dir() and p!=self.roots[0] and
                        ('kuroshio' in lo or '黑潮' in lo) and
                        not any(w in lo for w in ('github','archive','备份','_repo','publish'))):
                        candidates.append(str(p))
        for x in candidates:
            p = Path(x).expanduser().resolve()
            if p not in self.roots and len(self.roots)<16:
                self.roots.append(p)
        self.roots = [x for x in self.roots if x.is_dir()]
        if not self.roots:
            raise FileNotFoundError(f'Project directory not found: {args.root}')
        self.items: dict[str, Item] = {}
        self.omitted: list[dict] = []
        self.raw_info: list[dict] = []
        self.project_summary: dict = {}
        self.data_budget = int(args.max_data_gb * GB)
        self.data_used = 0
        self.results_budget = int(args.max_results_gb * GB)
        self.results_used = 0
        self.raw_bytes = 0
        self.splits: list[dict] = []
        self.manual_dates: set[str] = set()
        self.report_dir = Path(args.local_report_dir).expanduser().resolve() if args.local_report_dir else self.roots[0] / '.kuroshio_archive_reports'
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def origin(self, root):
        if root in self.roots:
            return relative_label(root, self.roots.index(root))
        return root.name

    def add_chunks(self, f: Path, dest: str, group: str, origin: str, size: int, reserve: bool):
        """Losslessly split one oversize *derived* file; do not use Git LFS.

        Records the original full-file hash and each part in the managed manifest.
        Keeps only source offsets in memory, not the source file content.
        """
        chunk_size = int(self.args.chunk_size_mb * MB)
        part_count = (size + chunk_size - 1) // chunk_size
        original_hash = sha256(f)
        parts = []
        for i in range(part_count):
            offset = i * chunk_size
            n = min(chunk_size, size - offset)
            output = f'{dest}.parts/part-{i+1:04d}.bin'
            if output in self.items:
                raise RuntimeError(f'Chunk destination collision: {output}')
            self.items[output] = Item(f,output,group,origin,n,'chunk',chunk_offset=offset)
            parts.append({'path':output,'size_bytes':n})
        self.splits.append({'original_path':dest,'original_name':f.name,'original_size_bytes':size,
                            'original_sha256':original_hash,'parts':parts})
        if reserve and group.startswith('data/'):
            self.data_used += size
        if group.startswith('results/'):
            self.results_used += size
        log(f'Lossless split: {f.name}, {fmt_size(size)} -> {part_count} pieces; reconstruction manifest saved.')
        return True

    def add_year_bundles(self, folder: Path, dest_base: str, group: str, origin: str):
        """Bundle each year's already compressed daily .npz files into deterministic ZIPs.

        Avoid tens of thousands of Git objects; no original NPZ is changed.
        Uses multiple parts per year if the estimated bundle would exceed cap.
        """
        groups = collections.defaultdict(list)
        for file in sorted(folder.glob('*.npz')):
            if not allowed_regular(file):continue
            if not re.fullmatch(r'\d{8}', file.stem):
                self.omit(file,group,'unexpected_daily_axis_filename');continue
            groups[file.stem[:4]].append(file)
        max_size = int(self.args.bundle_max_mb * MB)
        batches=[]
        for year,files in sorted(groups.items()):
            part=0;batch=[];total=0
            for file in files:
                estimate = file.stat().st_size + 512
                if batch and total + estimate > max_size:
                    batches.append((year,part,batch,total))
                    part+=1;batch=[];total=0
                if estimate > max_size:
                    self.omit(file,group,'single_daily_axis_exceeds_bundle_max');continue
                batch.append(file);total+=estimate
            if batch:batches.append((year,part,batch,total))
        estimate_total=sum(b[3] for b in batches)
        if self.data_used + estimate_total > self.data_budget:
            self.omit(folder,group,f'all_or_none_derived_tracks_exceed_budget_{fmt_size(estimate_total)}')
            self.project_summary[dest_base+'_excluded_original_files']=sum(len(g) for g in groups.values())
            return
        for year,part,files,approx in batches:
            name=f'{year}/daily_axes_{year}_part{part+1:02d}.zip'
            dest=f'{dest_base}/{name}'
            self.items[dest]=Item(folder,dest,group,origin,approx,'bundle',bundle_files=tuple(str(f) for f in files))
            self.data_used+=approx
        self.project_summary[dest_base+'_original_files']=sum(len(g) for g in groups.values())
        self.project_summary[dest_base+'_year_zip_count']=len(batches)
        log('Final derived axes:',dest_base,'original files',sum(len(g) for g in groups.values()),'year ZIPs',len(batches))

    def add(self, f, dest, group, origin, subtype='file', member='', generated=None, reserve=True, force_lfs=False):
        f = Path(f)
        if not allowed_regular(f):
            self.omit(f, group, 'not_regular_or_sensitive_path'); return False
        if dest in self.items:
            self.omit(f, group, 'duplicate_destination');return False
        size = len(generated) if generated is not None else f.stat().st_size
        if reserve and group.startswith('data/') and self.data_used + size > self.data_budget:
            self.omit(f, group, f'data_budget_exceeded_{self.args.max_data_gb}_GiB');return False
        if group.startswith('results/') and self.results_used + size > self.results_budget:
            self.omit(f, group, f'results_budget_exceeded_{self.args.max_results_gb}_GiB');return False
        if subtype == 'notebook':
            try:
                generated = sanitize_notebook(f)
                size = len(generated)
            except Exception as e:
                self.omit(f, group, f'invalid_notebook_{type(e).__name__}');return False
        if generated is not None:
            problems = secret_signatures(generated)
        elif f.suffix.lower() in CODE_SUFFIXES | {'.md', '.json', '.txt', '.csv', '.yaml', '.yml', '.toml', '.bat'}:
            # File size cap prevents giant log or binary scanning.
            problems = secret_signatures(f.read_bytes()) if size <= 15 * MB else ['text_too_large_review']
        else:
            problems = []
        if problems:
            self.omit(f, group, f'secret_review_required:{",".join(problems)}');return False
        # Derived data larger than our safe ordinary-Git threshold are split losslessly,
        # never silently discarded. Raw NetCDF do not reach this function.
        if (not self.args.lfs_enabled and not force_lfs and subtype=='file' and
            group.startswith(('data/','results/')) and size>self.args.lfs_threshold_mb*MB and
            f.suffix.lower() in {'.npz','.npy','.csv','.json','.parquet','.zip'}):
            if size>self.args.max_split_gb*GB:
                self.omit(f,group,f'oversize_derived_split_limit_{self.args.max_split_gb}GiB')
                return False
            return self.add_chunks(f,dest,group,origin,size,reserve)
        lfs = bool(force_lfs or size > self.args.lfs_threshold_mb * MB)
        if lfs and not self.args.lfs_enabled:
            self.omit(f, group, 'requires_git_lfs_not_enabled');return False
        if lfs and not has_tool('git-lfs') and not (has_tool('git') and call(['git','lfs','version'],check=False)==0):
            self.omit(f, group, 'git_lfs_not_installed');return False
        self.items[dest] = Item(f, dest, group, origin, size, subtype, member, generated, lfs)
        if reserve and group.startswith('data/'):
            self.data_used += size
        if group.startswith('results/'):
            self.results_used += size
        return True

    def omit(self, file, group, reason):
        self.omitted.append({'category': group, 'name': Path(file).name, 'reason': reason})

    def collect_code(self):
        # 'models/' can contain essential model classes, not only checkpoints.
        # Permit source-code traversal there while .pt/.pth weights stay excluded.
        code_skip = set(BAD_DIRS) - {'models'}
        for root in self.roots:
            lab = self.origin(root)
            count = 0
            scanned = 0
            log(f'Code scan begins for {lab}: {root}')
            for f in walk_safe(root, max_depth=12, skip=code_skip):
                scanned += 1
                if scanned % 25000 == 0:
                    log(f'Code scan progress {lab}: checked {scanned} files, collected {count} scripts')
                if f.suffix.lower() not in CODE_SUFFIXES:
                    continue
                # Exclude generated published archive from subsequent scans.
                if SUBTREE in f.parts or self.report_dir in f.parents:
                    continue
                # Code under temporary/intermediate data directories is not relevant to reproducibility.
                if (is_intermediate_output(f) and
                    not (f.suffix.lower() in CODE_SUFFIXES and
                         'models' in (part.lower() for part in f.parts) and
                         not any(part.lower() in (BLOCKED_PARTS-{'models'}) for part in f.parts))):
                    continue
                rel = f.relative_to(root).as_posix()
                cat = code_category(f)
                self.add(f, f'code/{cat}/{lab}/{rel}', 'code', lab,
                         subtype='notebook' if f.suffix.lower() == '.ipynb' else 'file', reserve=False)
                count += 1
            log(f'Code scan {lab}: discovered {count} source scripts')

    def collect_final_dataset(self):
        for root in self.roots:
            lab = self.origin(root)
            ds = root / 'final' / 'dataset'
            if ds.is_dir():
                for f in sorted(ds.iterdir()):
                    if f.is_file() and f.suffix.lower() in {'.npz', '.csv', '.json'} and 'FINAL' in f.name.upper():
                        self.add(f, f'data/frozen/{lab}/{f.name}', 'data/frozen', lab)
            for name in ('LONG_DAILY_1993_2025_HANDOFF.json', '最终逐日路径索引.csv',
                         '最终年度有效路径统计.csv', '最终连续有效时间段.csv'):
                f = root / 'final' / 'reports' / name
                if f.is_file():
                    self.add(f, f'data/frozen_reports/{lab}/{f.name}', 'data/frozen_reports', lab)

    def collect_manual(self):
        for root in self.roots:
            lab = self.origin(root)
            manual = root / 'manual_qc'
            if manual.is_dir():
                for sub in ('metadata', 'reports'):
                    d = manual / sub
                    if not d.is_dir():continue
                    for f in sorted(d.glob('*')):
                        if f.suffix.lower() in {'.json', '.csv', '.md', '.txt'}:
                            self.add(f, f'data/annotations/{lab}/{sub}/{f.name}', 'data/annotations', lab)
                for sub in ('tracks_manual', 'corrected', 'manual_tracks'):
                    d = manual / sub
                    if d.is_dir():
                        for f in sorted(d.glob('*.npz')):
                            year = f.stem[:4] if re.fullmatch(r'\d{8}', f.stem) else 'unknown'
                            self.add(f, f'data/annotations/{lab}/corrected_tracks/{year}/{f.name}',
                                     'data/annotations', lab)
                if self.args.include_qa_figures:
                    qdir = manual / 'figures'
                    if qdir.is_dir():
                        for f in walk_safe(qdir,max_depth=3):
                            if f.suffix.lower() in {'.png','.jpg','.jpeg','.pdf'}:
                                self.add(f, f'data/annotations/{lab}/review_figures/{f.relative_to(qdir).as_posix()}',
                                         'data/annotations',lab)
            # Important: final data can mark human-edited days even when manual_qc folder was moved.
            metapath = root / 'final' / 'dataset' / 'daily_dual_branch_meta_1993_2025_FINAL.csv'
            if metapath.is_file():
                with metapath.open('r',encoding='utf-8-sig',newline='') as stream:
                    for r in csv.DictReader(stream):
                        status = str(r.get('source','')).lower()
                        day = re.sub(r'\D','',str(r.get('date','')))[:8]
                        if status in ('manual_corrected','manual_qc_corrected') and re.fullmatch(r'\d{8}', day):
                            self.manual_dates.add(day)
                if self.manual_dates:
                    for day in sorted(self.manual_dates):
                        f = root / 'final' / 'tracks_final' / f'{day}.npz'
                        dest = f'data/annotations/{lab}/final_corrected_days/{day[:4]}/{day}.npz'
                        if dest not in self.items and f.is_file():
                            self.add(f, dest, 'data/annotations', lab)
        self.project_summary['human_corrected_days_from_final_metadata'] = len(self.manual_dates)

    def collect_tracks(self):
        if not self.args.include_final_tracks:return
        for root in self.roots:
            lab=self.origin(root)
            folder=root/'final'/'tracks_final'
            if not folder.is_dir():continue
            if self.args.individual_tracks:
                for f in sorted(folder.glob('*.npz')):
                    if allowed_regular(f):
                        self.add(f,f'data/final_tracks/{lab}/{f.stem[:4]}/{f.name}','data/final_tracks',lab)
            else:
                self.add_year_bundles(folder,f'data/final_tracks/{lab}','data/final_tracks',lab)

    def collect_adt_reference(self):
        if not self.args.include_adt_axes:return
        for root in self.roots:
            lab=self.origin(root)
            folder=root/'R70_ADT_Independent_Python'/'adt_axes_npz'
            if not folder.is_dir():continue
            if self.args.individual_tracks:
                for f in sorted(folder.glob('*.npz')):
                    if allowed_regular(f):
                        self.add(f,f'data/adt_axes/{lab}/{f.stem[:4]}/{f.name}','data/adt_axes',lab)
            else:
                self.add_year_bundles(folder,f'data/adt_axes/{lab}','data/adt_axes',lab)

    def collect_terminal_results(self):
        for root in self.roots:
            lab = self.origin(root)
            # Terminal experiments only: do NOT include R1-R64 model improvement datasets.
            result_dirs = [d for d in root.iterdir() if d.is_dir() and final_result_dir(d.name)]
            result_dirs += [root/'final'/'reports', root/'final'/'results', root/'results'/'final', root/'reports'/'final']
            seen=set()
            for d in result_dirs:
                if not d.is_dir() or d in seen:continue
                seen.add(d)
                for f in walk_safe(d, max_depth=7):
                    if f.suffix.lower() not in RESULT_SUFFIXES or is_intermediate_output(f):continue
                    rel=f.relative_to(d).as_posix()
                    self.add(f, f'results/{lab}/{d.name}/{rel}', 'results/final', lab, reserve=False)
            # Terminal experimental result archives often contain complete CSV+PNG
            # even when the original experiment directory was deleted.
            for f in sorted(root.glob('*.zip')):
                if not final_result_zip(f.name):continue
                self.extract_terminal_archive(f, lab)
            for name in ('Revised_Manuscript-Clean.docx', 'Revised_Manuscript_Fig2-8_Unified_Fonts.docx'):
                # Preserve only lightweight scientific data; drafts remain in original project.
                pass

    def extract_terminal_archive(self, archive: Path, lab: str):
        """Extract only CSV/JSON/plots from terminal ZIP; no checkpoints, no nested ZIPs."""
        try:
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None:
                    self.omit(archive,'results/final','zip_crc_fail');return
                for zi in z.infolist():
                    if zi.is_dir():continue
                    # Check zip paths AS POSIX after normalizing backslashes.
                    # Avoid zip-slip path traversal that could escape the archive subtree.
                    normalized=zi.filename.replace('\\','/')
                    parts=PurePosixPath(normalized).parts
                    if (normalized.startswith('/') or ':' in normalized or
                        any(seg in ('', '.', '..') or seg.startswith('.') for seg in parts)):
                        self.omit(archive,'results/final','unsafe_zip_member_path');continue
                    if any(seg.lower() in {x.lower() for x in BLOCKED_PARTS} for seg in parts):continue
                    if PurePosixPath(normalized).suffix.lower() not in RESULT_SUFFIXES:continue
                    if zi.file_size>self.args.lfs_threshold_mb*MB and not self.args.lfs_enabled:
                        self.omit(archive,'results/final',f'large_member_lfs_required:{zi.filename}');continue
                    # Prevent compressed-archive expansion attacks.
                    if zi.file_size > 100*MB:continue
                    # Avoid huge videos and unintended intermediate logs.
                    name=normalized
                    if len(name)>220:continue
                    dest=f'results/from_archives/{lab}/{archive.stem}/{name}'
                    if dest in self.items:continue
                    data=z.read(zi)
                    # For textual result JSON/CSV, scan for actual secrets before upload.
                    if Path(name).suffix.lower() in ('.json','.csv','.md','.txt') and secret_signatures(data):
                        self.omit(archive,'results/final',f'secret_in_zip_member:{name}');continue
                    if self.results_used + len(data) > self.results_budget:
                        self.omit(archive,'results/final',f'results_budget_exceeded_member:{name}');continue
                    # Never retain the entire unpacked archive in memory; files are streamed at staging.
                    self.items[dest]=Item(archive,dest,'results/final',lab,len(data),'zip_member',member=name,
                                          use_lfs=(len(data)>self.args.lfs_threshold_mb*MB))
                    self.results_used+=len(data)
        except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
            self.omit(archive,'results/final',f'zip_read_error:{type(exc).__name__}')

    def collect_raw(self):
        """Explicit publication scope: original NetCDF never scanned or distributed."""
        self.raw_info=[]
        self.raw_bytes=0
        self.project_summary['source_policy']='No source NetCDF scan/index/upload; derived data and code only'
        self.project_summary['raw_netcdf_files_scanned']=0
        self.project_summary['raw_netcdf_bytes']=0
        log('Raw CMEMS / ADT NetCDF: intentionally NOT scanned, indexed or uploaded (publication mode).')

    def collect_all(self):
        self.collect_raw()
        self.collect_code()
        self.collect_final_dataset()
        self.collect_manual()
        self.collect_tracks()
        self.collect_adt_reference()
        self.collect_terminal_results()
        # Keep this archiver itself in the repository for future reproducible refreshes.
        self.add(Path(__file__), 'tools/kuroshio_archive_publish.py', 'code', 'archiver', reserve=False)
        self.add(Path(__file__).with_name('restore_split_data.py'),
                 'tools/restore_split_data.py','code','archiver',reserve=False)
        self.add(Path(__file__).with_name('verify_kuroshio_archive.py'),
                 'tools/verify_kuroshio_archive.py','code','archiver',reserve=False)
        log('Planned files:',len(self.items),'total selected:',fmt_size(sum(x.size for x in self.items.values())))
        log('Raw NetCDF: not scanned or included (0 by design).')
        log('Manual-corrected final dates:',len(self.manual_dates))
        log('Skipped items needing review:',len(self.omitted))

    def audit_local(self):
        self.report_dir.mkdir(parents=True,exist_ok=True)
        # Do not write local original NetCDF inventories for paper-publication archives.
        (self.report_dir/'raw_training_data_inventory.csv').unlink(missing_ok=True)
        self.write_csv(self.report_dir/'selected_for_github.csv',
                       [dict(category=x.group,repo_path=x.dest,size_bytes=x.size,origin=x.origin,
                             git_lfs=x.use_lfs) for x in self.items.values()],
                       ('category','repo_path','size_bytes','origin','git_lfs'))
        self.write_csv(self.report_dir/'omitted_review.csv',self.omitted,('category','name','reason'))
        (self.report_dir/'summary.json').write_text(json.dumps(self.summary(),ensure_ascii=False,indent=2),encoding='utf-8')
        log('Local audit:',self.report_dir)

    @staticmethod
    def write_csv(dest,rows,keys):
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)

    def summary(self):
        stats=collections.Counter(x.group for x in self.items.values())
        return {'tool_version':TOOL_VERSION,'included_files':len(self.items),
                'included_size_bytes':sum(x.size for x in self.items.values()),
                'included_counts_by_category':dict(stats),
                'manual_corrected_days_from_frozen_metadata':len(self.manual_dates),
                'missing_or_excluded_entries':len(self.omitted),
                'data_budget_bytes':self.data_budget,
                'results_budget_bytes':self.results_budget,
                'raw_netcdf_inventory_size_bytes':self.raw_bytes,
                'raw_netcdf_inventory_file_count':len(self.raw_info),
                'raw_upload_policy':'none (deliberately excluded)',
                'lossless_split_original_files':len(self.splits),
                'project':self.project_summary}

    def stage(self, folder: Path):
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'manifests').mkdir(parents=True,exist_ok=True)
        # A refresh may supersede the prior managed collection (for example
        # individual 12k files -> yearly ZIPs). Remove only previously archived
        # unchanged managed copies, NEVER files from original project roots.
        old_csv=folder/'manifests'/'uploaded_files_sha256.csv'
        if old_csv.exists():
            new_files=set(self.items)
            obsolete=[];preserved=[]
            with old_csv.open(encoding='utf-8-sig',newline='') as stream:
                for entry in csv.DictReader(stream):
                    rel=entry.get('repo_path','')
                    if rel in new_files:continue
                    parts=PurePosixPath(rel).parts
                    if (not parts or any(seg in ('','.','..') for seg in parts)
                        or rel.startswith('/') or ':' in rel):continue
                    file=folder.joinpath(*parts)
                    if not file.resolve(strict=False).is_relative_to(folder.resolve()):continue
                    if file.is_file() and not file.is_symlink():
                        if sha256(file)==entry.get('sha256',''):
                            obsolete.append(file)
                        else:
                            preserved.append(rel)
            for path in obsolete:path.unlink()
            if obsolete:log('Pruned previously archived unchanged managed copies:',len(obsolete))
            if preserved:log('WARNING: changed former archived files preserved for review:',len(preserved))
        # Audit CSVs are included with safe relative paths only (no full local paths).
        (folder/'manifests'/'raw_cmems_inventory.csv').unlink(missing_ok=True)
        (folder/'manifests'/'original_CMEMS_data_policy.md').write_text(
            '# Original source data policy\n\nNo CMEMS GLORYS/DUACS raw NetCDF files or raw file lists are scanned or distributed. '
            'The repository contains derived Kuroshio datasets, annotations, source code and final results only.\n'
            'Download original public products from https://data.marine.copernicus.eu/ when reproducing extraction.\n',
            encoding='utf-8')
        (folder/'manifests'/'split_files_restore.json').write_text(
            json.dumps({'format_version':1,'files':self.splits},ensure_ascii=False,indent=2),encoding='utf-8')
        self.write_csv(folder/'manifests'/'excluded_files_review.csv',self.omitted,
                       ('category','name','reason'))
        (folder/'manifests'/'experiment_summary.json').write_text(
            json.dumps(self.summary(),ensure_ascii=False,indent=2),encoding='utf-8')
        (folder/'.gitignore').write_text('''# Machine-specific files and intermediate training data\n*.pt\n*.pth\n*.ckpt\n*.safetensors\n*.onnx\n*.log\n.env*\n__pycache__/\n.ipynb_checkpoints/\n.venv/\ncache/\nwandb/\n''',encoding='utf-8')
        lfs_paths=[];audit=[]
        for idx,(dest,item) in enumerate(sorted(self.items.items()),1):
            out=folder/dest
            out.parent.mkdir(parents=True,exist_ok=True)
            if item.subtype=='bundle':
                # Source-derived year ZIP has per-member checksums for perfect restoration.
                with tempfile.NamedTemporaryFile(dir=out.parent,suffix='.zip.tmp',delete=False) as fh:
                    tmp=Path(fh.name)
                try:
                    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                        rows=[]
                        for file_name in item.bundle_files:
                            f=Path(file_name)
                            z.write(f,arcname=f.name)
                            rows.append((f.name,f.stat().st_size,sha256(f)))
                        manifest='filename,size_bytes,sha256\n'+''.join(f'{n},{size},{digest}\n' for n,size,digest in rows)
                        z.writestr('_FILE_SHA256_MANIFEST.csv',manifest)
                    if not out.exists() or sha256(tmp)!=sha256(out):
                        os.replace(tmp,out)
                finally:
                    tmp.unlink(missing_ok=True)
                digest=sha256(out)
            elif item.subtype in ('chunk','zip_member'):
                with tempfile.NamedTemporaryFile(dir=out.parent,delete=False) as fh:
                    temp_path=Path(fh.name)
                    if item.subtype=='chunk':
                        with item.src.open('rb') as f:
                            f.seek(item.chunk_offset)
                            left=item.size
                            while left:
                                b=f.read(min(2*MB,left))
                                if not b:raise EOFError(f'Incomplete source chunk: {item.src}')
                                fh.write(b);left-=len(b)
                    else:
                        with zipfile.ZipFile(item.src) as z, z.open(item.member) as member:
                            shutil.copyfileobj(member,fh,length=2*MB)
                if not out.exists() or sha256(temp_path)!=sha256(out):
                    os.replace(temp_path,out)
                temp_path.unlink(missing_ok=True)
                digest=sha256(out)
            elif item.generated is not None:
                data=item.generated
                digest=hashlib.sha256(data).hexdigest()
                if not out.exists() or out.stat().st_size!=len(data) or sha256(out)!=digest:
                    with tempfile.NamedTemporaryFile(mode='wb',dir=out.parent,delete=False) as fh:
                        fh.write(data);tmp=Path(fh.name)
                    os.replace(tmp,out)
            else:
                if not out.exists() or out.stat().st_size!=item.size or sha256(out)!=sha256(item.src):
                    shutil.copy2(item.src,out)
                digest=sha256(out)
            if item.use_lfs:lfs_paths.append(dest)
            audit.append({'group':item.group,'repo_path':dest,'size_bytes':out.stat().st_size,
                          'sha256':digest,'lfs':int(item.use_lfs)})
            if idx % 200==0:log(f'Copied/verified {idx}/{len(self.items)} files')
        self.write_csv(folder/'manifests'/'uploaded_files_sha256.csv',audit,
                       ('group','repo_path','size_bytes','sha256','lfs'))
        attrs=['# Git LFS tracking; explicit archival-only paths']
        for dest in sorted(lfs_paths):
            # Git wildmatch escaping: spaces and special glob chars in paths.
            esc=dest.replace('\\','\\\\').replace(' ','\\ ')
            for ch in '[]*?':esc=esc.replace(ch,'\\'+ch)
            attrs.append(f'{esc} filter=lfs diff=lfs merge=lfs -text')
        (folder/'.gitattributes').write_text('\n'.join(attrs)+'\n',encoding='utf-8')
        (folder/'README.md').write_text(self.readme_text(),encoding='utf-8')
        (folder/'docs').mkdir(exist_ok=True)
        (folder/'docs'/'REPRODUCE_AND_RESTORE.md').write_text('''# Restore & verification workflow

1. Clone the GitHub repository. If Git LFS pointers are used, install Git LFS and run `git lfs pull`.
2. Read `manifests/experiment_summary.json`, `manifests/excluded_files_review.csv` and `manifests/split_files_restore.json` before interpreting file coverage. Use `tools/restore_split_data.py` to reconstruct any split large datasets.
3. Verify included file checksums using `manifests/uploaded_files_sha256.csv`.
4. Frozen path features are under `data/frozen/`. Accepted per-day tracks are stored in lossless year ZIPs under `data/final_tracks/` and `data/adt_axes/`; each ZIP contains per-file SHA256. Human-corrected track records are under `data/annotations/`.
5. CMEMS raw NetCDF are intentionally not scanned or distributed. Obtain original reanalysis and ADT from CMEMS independently.
6. To rerun scripts: create a WSL Python environment; check original `.ipynb` paths/configuration, replace machine-specific absolute paths, and install packages used by each Notebook. All execution outputs have been removed without altering code cells.
7. Research results are snapshots. R1–R64 model-search experiment outputs were intentionally excluded; the existence of training/evaluation code does not imply its trial data are included.
8. Human review/correction records are not equivalent to independent ground-truth observational labels; treat the source `source` field as provenance, not perfect truth.

Run `python3 tools/kuroshio_archive_publish.py --root /mnt/d/KuroshioDaily1993_2025 --repo OWNER/REPO` to refresh the archive on its source computer.
''',encoding='utf-8')
        log('Stage ready:',folder,'bytes:',fmt_size(sum(x.size for x in self.items.values())))

    def readme_text(self):
        t=self.summary();cnt=t['included_counts_by_category']
        listing='\n'.join(f'- `{k}`: {n} files' for k,n in sorted(cnt.items()))
        omitted = len(self.omitted)
        rawtxt = ('The repository intentionally contains NO raw CMEMS/ADT NetCDF or NetCDF file inventories. '
                  'Original sources are available separately from CMEMS. Only derived datasets are distributed.')
        return f'''# Kuroshio research reproducibility archive\n\nGenerated by `kuroshio_archive_publish.py` v{TOOL_VERSION}. Files are organized from the local WSL research project, not created from synthetic data. **All paths and contents must be reviewed before publication.**\n\n## Included material\n\n{listing}\n\n- `code/01_dataset_build`: NetCDF acquisition, path extraction, dataset assembly scripts.\n- `code/02_manual_annotation`: human QC and correction tools.\n- `code/03_model_training`, `code/04_experiment_evaluation`, `code/05_visualization`: research training, verification and plots. Python notebooks have their generated outputs and notebook execution state removed.\n- `data/frozen`: **final** fixed-dimensional daily dual-branch path dataset and metadata.\n- `data/final_tracks`: full accepted axis coordinates in per-year lossless ZIP bundles (with in-ZIP checksums).\n- `data/annotations`: manual-corrected days, metadata and review decisions.\n- `results/`: formal/final rolling hindcasts, comparative baselines, satellite ADT and figure summaries.\n- `manifests/`: derived dataset SHA256, split-file restoration and excluded-file reports; original NetCDF inventories are not created.\n\n## Training input policy\n\n{rawtxt}\n\nKnown original CMEMS historical directory: `GLOBAL_MULTIYEAR_PHY_001_030 / cmems_mod_glo_phy_myint_0.083deg_P1D-m / YYYY`. Newer supplementary daily data may be in an `E:/YYYY` structure.\n\nOriginal NetCDF deliberately not scanned. Excluded/review items: **{omitted}**; see the machine-readable manifest. Oversize derived training files may be stored as checksummed parts requiring restoration. No intermediate R1–R64 experimental checkpoints or repeated optimization result datasets were intentionally included.\n\n## Attribution / interpretation\n\nThe final dual-branch dataset is constructed from CMEMS GLORYS daily 50-m velocity-derived Kuroshio axes. ADT-derived reference axes are a distinct observational product used for cross-product consistency validation; they are not a direct substitute for independent in-situ truth. Display filtering of paths is not a model accuracy improvement. Research histograms and forecasts must be interpreted within their respective evaluation contracts.\n\n## Privacy, storage & provenance\n\n- Never commit API keys, SSH credentials, personal VPN configs, local `.env` files or secret-bearing notebooks.\n- GitHub ordinary Git has 100 MiB per-file hard limit; large research binaries require Git LFS and sufficient storage/bandwidth quota.\n- Original source NetCDF files are absent by design; no F:/E: mount is needed for publishing.\n- SHA256 hashes refer only to files actually included. Other original files remain at their local WSL source location.\n- Do not present the `human corrected days` total as all human-reviewed samples; manual review without path correction is counted separately in metadata.\n'''


def repo_public_status(repo):
    if not has_tool('gh'):
        return 'unknown'
    try:
        result = call(['gh','repo','view',repo,'--json','visibility','--jq','.visibility'],capture=True)
        return result.lower()
    except Exception:
        return 'unknown'


def prepare_worktree(repo, workspace):
    """Never force-push, delete branches, or reset an existing user worktree."""
    if not has_tool('git'):
        raise RuntimeError('git is required in WSL. Install with: sudo apt install git')
    workspace=Path(workspace).expanduser().resolve()
    if workspace.exists() and (workspace/'.git').exists():
        origin=call(['git','remote','get-url','origin'],cwd=workspace,capture=True)
        if normalize_repo(origin)!=repo:
            raise RuntimeError(f'Workspace has different GitHub remote: {origin}. Choose a new --workspace.')
        # Do not fetch/rebase or overwrite unrelated files automatically.
        return workspace
    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError(f'Workspace exists but is not an empty clone: {workspace}')
    workspace.parent.mkdir(parents=True,exist_ok=True)
    if has_tool('gh'):
        call(['gh','repo','clone',repo,str(workspace),'--','--depth','1'])
    else:
        call(['git','clone','--depth','1',f'https://github.com/{repo}.git',str(workspace)])
    return workspace


def git_publish(workspace, need_lfs, yes=False):
    if need_lfs:
        if not has_tool('git') or call(['git','lfs','version'],check=False)!=0:
            raise RuntimeError('Git LFS is required. In WSL: sudo apt install git-lfs && git lfs install')
        call(['git','lfs','install','--local'],cwd=workspace)
    user = call(['git','config','user.name'],cwd=workspace,capture=True,check=False)
    email= call(['git','config','user.email'],cwd=workspace,capture=True,check=False)
    if not (user and email):
        raise RuntimeError('Please configure Git user.name and user.email before publishing.\n'
                           'git config --global user.name "YOUR NAME"\n'
                           'git config --global user.email "YOUR GITHUB EMAIL"')
    status=call(['git','status','--porcelain','--',SUBTREE],cwd=workspace,capture=True)
    if not status:
        log('No changes under archival subtree; repository is already up to date.')
        return
    log('Files changed:',len(status.splitlines()))
    # Commit/push only in the designated archive directory, excluding unrelated user changes.
    call(['git','add','-A','--',SUBTREE],cwd=workspace)
    staged=call(['git','diff','--cached','--name-only','--',SUBTREE],cwd=workspace,capture=True)
    if not staged:
        log('No staged diff after indexing; nothing to publish.')
        return
    # Guard against Git accidentally staging huge files that were not LFS filtered.
    staged_paths=staged.splitlines()
    for rel in staged_paths:
        path=workspace/rel
        if not path.is_file():continue
        if path.stat().st_size>99*MB:
            attr=call(['git','check-attr','filter','--',rel],cwd=workspace,capture=True)
            if not attr.strip().endswith('lfs'):
                raise RuntimeError(f'Unsafe ordinary-Git file >99MiB staged: {rel}; configure LFS first.')
    call(['git','commit','-m',f'archive: reproducible Kuroshio research data and results ({dt.date.today()})','--',SUBTREE],cwd=workspace)
    # Push fast-forward to existing default branch only; respect protection rules.
    branch=call(['git','branch','--show-current'],cwd=workspace,capture=True)
    if not branch:
        branch='main'
    call(['git','push','-u','origin',f'HEAD:refs/heads/{branch}'],cwd=workspace)
    log('Push finished; branch:',branch)


def get_args(argv=None):
    p=argparse.ArgumentParser(description='WSL Kuroshio final research archive -> GitHub (safe, resumable)')
    p.add_argument('--root',default='/mnt/d/KuroshioDaily1993_2025',help='Primary WSL project directory')
    p.add_argument('--extra-root',action='append',default=[],help='Additional Kuroshio project root, repeatable')
    p.add_argument('--no-auto-extra-roots',action='store_true',help='Do not detect adjacent Kuroshio project directories')
    p.add_argument('--repo',default='',help='GitHub owner/repo or URL. Empty: detect remote/gh, else prompt')
    p.add_argument('--allow-empty-raw-inventory',action='store_true',help=argparse.SUPPRESS) # compatibility no-op
    p.add_argument('--workspace',default=str(Path.home()/'KuroshioAxis_GitHub_Archive_V12'),
                   help='Local Git clone/staging directory, outside the source project')
    p.add_argument('--local-report-dir',default='',help='Private audit output; defaults project/.kuroshio_archive_reports')
    p.add_argument('--max-data-gb',type=float,default=1.0,help='Cap for derived frozen+annotated+final axes (default 1GiB)')
    p.add_argument('--max-results-gb',type=float,default=0.5,help='Cap for final figures/experimental results (default 0.5GiB)')
    p.add_argument('--lfs-enabled',action='store_true',help='Use Git LFS for large derived files; otherwise losslessly split into ordinary Git chunks')
    p.add_argument('--lfs-threshold-mb',type=float,default=40.0,help='Regular-Git file threshold, default 40 MiB')
    p.add_argument('--chunk-size-mb',type=float,default=20.0,help='Lossless chunk size for oversize derived files (default 20 MiB)')
    p.add_argument('--max-split-gb',type=float,default=4.0,help='Maximum single derived file eligible for lossless split (default 4 GiB)')
    p.add_argument('--bundle-max-mb',type=float,default=24.0,help='Maximum year-axis bundle size (default 24 MiB)')
    p.add_argument('--individual-tracks',action='store_true',help='Keep every daily axis as individual NPZ (much slower Git staging)')
    p.add_argument('--no-final-tracks',action='store_false',dest='include_final_tracks',default=True,
                   help='Upload frozen dataset only, not year-grouped final daily tracks')
    p.add_argument('--no-adt-axes',action='store_false',dest='include_adt_axes',default=True,
                   help='Do not include accepted independent ADT reference axes')
    p.add_argument('--include-qa-figures',action='store_true',help='Also upload manually reviewed image evidence')
    p.add_argument('--plan-only',action='store_true',help='Inventory only, no clone/copy/push')
    p.add_argument('--no-push',action='store_true',help='Stage into workspace but do not push to GitHub')
    p.add_argument('--yes',action='store_true',help='Explicitly approve publishing; otherwise interactive prompt')
    return p.parse_args(argv)


def main(argv=None):
    args=get_args(argv)
    args.raw_mode='none'  # original CMEMS/ADT raw scans disabled without exception
    args.raw_root=[]
    if args.max_data_gb<=0 or args.max_results_gb<=0:
        raise SystemExit('--max-data-gb/--max-results-gb must be positive')
    if not (1 <= args.chunk_size_mb <= 32 and args.chunk_size_mb < args.lfs_threshold_mb <= 95):
        raise SystemExit('Require 1<=chunk-size-mb<=32 < lfs-threshold-mb<=95 (default 20/40)')
    if not 1<=args.bundle_max_mb<=32:
        raise SystemExit('--bundle-max-mb must be in [1,32]')

    # Check target FIRST: prevents expensive rescans after a blank repo prompt.
    repo=None
    if not args.plan_only and not args.no_push:
        repo=prompt_repository(args)
        if repo is None:
            log('仓库选择已取消；没有上传任何数据。');return 0
        repository_preflight(repo)
        visibility=repo_public_status(repo)
        log('Publish target:',repo,'visibility:',visibility)
        if visibility=='public':
            log('WARNING: PUBLIC repository! Source code, final paths and annotations will be visible to everyone.')

    log('Starting project inventory; original files are READ-ONLY.')
    c=Collector(args)
    c.collect_all()
    c.audit_local()
    nonlfs_size=sum(x.size for x in c.items.values() if not x.use_lfs)
    if nonlfs_size > 1.5*GB and not args.plan_only:
        raise RuntimeError('Non-LFS selected data exceed 1.5GiB; use smaller budgets or explicit Git LFS.')
    # Zero original NetCDF files is the REQUIRED successful result in this paper-release mode.
    assert c.raw_info==[] and c.raw_bytes==0, 'Raw NetCDF scan must stay disabled.'
    if args.plan_only:
        log('Plan-only complete. Inspect the private reports, then run without --plan-only.')
        return 0
    if not args.no_push:
        log('Selected:',len(c.items),'derived/code/result files; original NetCDF not scanned, policy:',args.raw_mode)
        log('Review private audit report first:',c.report_dir)
        if not args.yes:
            if not sys.stdin.isatty():
                raise SystemExit('Noninteractive publish needs --yes after reviewing audit.')
            try:
                reply=input(f'确认复制所选科研资料并 PUSH 到 {repo}？输入大写 PUBLISH 继续：').strip()
            except EOFError:
                raise RuntimeError('交互确认无法读取。请在真实终端运行，或审核后加 --yes。') from None
            if reply!='PUBLISH':
                log('Cancelled safely. Nothing was uploaded.');return 0
        workspace=prepare_worktree(repo,args.workspace)
    else:
        workspace=Path(args.workspace).expanduser().resolve()
        workspace.mkdir(parents=True,exist_ok=True)
    dest=workspace/SUBTREE
    c.stage(dest)
    if not args.no_push:
        git_publish(workspace,any(x.use_lfs for x in c.items.values()),args.yes)
        branch=call(['git','branch','--show-current'],cwd=workspace,capture=True)
        log(f'GitHub: https://github.com/{repo}/tree/{branch}/{SUBTREE}/')
    else:
        log('No-push mode. Stage prepared at:',dest)
    log('Done. Check manifests/experiment_summary.json and omitted files review.')
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log('Interrupted. Original data unchanged; rerun the same command to resume.');sys.exit(130)
    except Exception as exc:
        log('FAILED:',type(exc).__name__,str(exc))
        sys.exit(1)
