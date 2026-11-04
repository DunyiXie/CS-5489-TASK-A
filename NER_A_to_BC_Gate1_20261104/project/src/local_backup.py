"""Verified, portable NER backups. Uses local files and the Python standard library."""
import hashlib
import json
import re
import shutil
import stat
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

FORMAT = 'cs5489-ner-local-backup-v1'
ALLOWED_FOLDERS = {'config', 'logs', 'src', 'results', 'figures', 'predictions', 'notebooks'}
ALLOWED_ROOT_FILES = {'README.md', 'requirements.txt', 'requirements-step0.txt'}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Invalid backup member path.')
    parts = value.split('/')
    if any(part in ('', '.', '..') or part.startswith('.') for part in parts):
        raise ValueError('Unsafe backup member path.')
    if PurePosixPath(value).is_absolute():
        raise ValueError('Absolute backup member paths are forbidden.')
    return PurePosixPath(value)


def _is_project_output(relative, include_checkpoints):
    parts = relative.parts
    if any(part.startswith('.') or part == '__pycache__' for part in parts):
        return False
    filename = relative.name.lower()
    if ('credential' in filename or 'oauth' in filename or filename.startswith('client_secret') or
            filename in {'token.json', 'tokens.json'} or
            any(secret in filename for secret in ('access_token', 'refresh_token', 'id_token')) or
            filename.endswith(('.zip', '.pyc', '.pem', '.key'))):
        return False
    if len(parts) == 1:
        return relative.name in ALLOWED_ROOT_FILES
    return parts[0] in ALLOWED_FOLDERS or (include_checkpoints and parts[0] == 'checkpoints')


def verify_backup(archive_file, max_total_bytes=20 * 1024**3):
    """Check structure, file sizes and hashes before accepting a local ZIP.

    Hashes detect corruption; they do not establish who created an archive.
    Restore only backups from a source you trust.
    """
    with zipfile.ZipFile(archive_file) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or 'manifest.json' not in names:
            raise ValueError('Backup has duplicate members or no manifest.')
        if sum(item.file_size for item in members) > max_total_bytes:
            raise ValueError('Backup exceeds the configured uncompressed-size limit.')
        for item in members:
            _relative_path(item.filename)
            if stat.S_ISLNK(item.external_attr >> 16) or item.is_dir():
                raise ValueError('Links and directory entries are not valid backup members.')
        if archive.getinfo('manifest.json').file_size > 10 * 1024**2:
            raise ValueError('Backup manifest is too large.')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('format') != FORMAT or not isinstance(manifest.get('files'), list):
            raise ValueError('Unknown backup format.')
        expected_names = {'manifest.json'}
        paths = set()
        for record in manifest['files']:
            relative = _relative_path(record['path'])
            if relative.as_posix() in paths:
                raise ValueError('Duplicate path in backup manifest.')
            paths.add(relative.as_posix())
            member_name = 'project/' + relative.as_posix()
            expected_names.add(member_name)
            if member_name not in names:
                raise ValueError('A manifest file is missing from the ZIP.')
            info = archive.getinfo(member_name)
            if info.file_size != record['size_bytes']:
                raise ValueError(f'Backup size check failed: {relative}')
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != record['sha256']:
                raise ValueError(f'Backup checksum check failed: {relative}')
        if set(names) != expected_names:
            raise ValueError('Backup contains files not listed in the manifest.')
        # Reading every member to EOF above also checks each ZIP CRC.
    return manifest


def create_backup(project_root, export_dir, step, revision, include_checkpoints=False):
    """Archive selected project outputs and verify the completed ZIP by reading it."""
    root = Path(project_root).resolve(strict=True)
    export = Path(export_dir).resolve()
    if not root.is_dir() or export.is_relative_to(root):
        raise ValueError('Use an existing project directory and an export directory outside it.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', step):
        raise ValueError('Use a simple step name, e.g. step0.')
    export.mkdir(parents=True, exist_ok=True)
    records, selected = [], []
    folders = ALLOWED_FOLDERS | ({'checkpoints'} if include_checkpoints else set())
    candidates = [root / name for name in ALLOWED_ROOT_FILES if (root / name).is_file()]
    for name in sorted(folders):
        folder = root / name
        if folder.is_symlink():
            raise ValueError(f'Refusing to back up a linked project directory: {name}')
        if folder.is_dir():
            candidates.extend(folder.rglob('*'))
    for path in sorted(candidates):
        relative = path.relative_to(root)
        if not _is_project_output(relative, include_checkpoints):
            continue
        if path.is_symlink() or any((root / Path(*relative.parts[:i])).is_symlink()
                                    for i in range(1, len(relative.parts))):
            raise ValueError(f'Refusing to back up a link: {relative}')
        if not path.is_file():
            continue
        path.resolve(strict=True).relative_to(root)
        records.append({'path': relative.as_posix(), 'size_bytes': path.stat().st_size,
                        'sha256': sha256_file(path)})
        selected.append((path, relative))
    if not records:
        raise ValueError('No project outputs were selected for backup.')
    created = datetime.now(timezone.utc)
    manifest = {'format': FORMAT, 'step': step, 'revision': revision,
                'created_at_utc': created.isoformat(), 'files': records,
                'includes_checkpoints': include_checkpoints}
    name = f'NER_{step}_{created:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.zip'
    output = export / name
    partial = output.with_suffix('.partial')
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=2, strict_timestamps=False) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
        for path, relative in selected:
            archive.write(path, arcname='project/' + relative.as_posix())
    verify_backup(partial)
    if output.exists():
        raise FileExistsError(output)
    partial.rename(output)
    return {'path': str(output), 'name': output.name, 'size_bytes': output.stat().st_size,
            'sha256': sha256_file(output), 'file_count': len(records),
            'archive_verified': True, 'download_confirmed': False}


def restore_backup(archive_file, destination):
    """Restore verified bytes into a new directory without executing any code."""
    manifest = verify_backup(archive_file)
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError('Choose a new, empty destination name for restoration.')
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_file) as archive:
        for record in manifest['files']:
            relative = _relative_path(record['path'])
            path = target.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open('project/' + relative.as_posix()) as source, path.open('xb') as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            if sha256_file(path) != record['sha256']:
                raise ValueError(f'Restored file checksum failed: {relative}')
    return {'project_root': str(target), 'step': manifest['step'], 'file_count': len(manifest['files'])}
