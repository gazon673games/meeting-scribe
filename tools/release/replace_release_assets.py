from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReleaseAsset:
    id: int
    name: str
    size: int
    state: str


class ReleaseAssets(Protocol):
    def list_assets(self) -> list[ReleaseAsset]: ...
    def upload(self, path: Path) -> None: ...
    def rename(self, asset: ReleaseAsset, name: str) -> None: ...
    def delete(self, asset: ReleaseAsset) -> None: ...


def replace_asset(release: ReleaseAssets, path: Path) -> None:
    previous = next(
        (asset for asset in release.list_assets() if asset.name == path.name),
        None,
    )
    ticket = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='release-upload-') as directory:
        staged_path = Path(directory) / f'pending-{ticket}-{path.name}'
        shutil.copyfile(path, staged_path)
        release.upload(staged_path)
        staged = next(
            (
                asset
                for asset in release.list_assets()
                if asset.name == staged_path.name
            ),
            None,
        )
        if staged is None:
            raise RuntimeError(f'Uploaded asset is missing: {staged_path.name}')
        if (
            staged.state != 'uploaded'
            or staged.size != staged_path.stat().st_size
        ):
            try:
                release.delete(staged)
            except Exception:
                _logger.exception(
                    'Cannot remove incomplete release asset %s', staged.name
                )
            raise RuntimeError(f'Uploaded asset is incomplete: {path.name}')

        if previous is not None:
            release.rename(previous, f'previous-{ticket}-{path.name}')
        try:
            release.rename(staged, path.name)
        except Exception:
            # Retain both byte streams if the rename outcome is uncertain.
            if previous is not None:
                try:
                    release.rename(previous, path.name)
                except Exception:
                    _logger.exception(
                        'Cannot restore release asset %s', path.name
                    )
            raise
        if previous is not None:
            release.delete(previous)


def _gh(*arguments: str) -> str:
    return subprocess.run(
        ['gh', *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
    ).stdout


class GitHubReleaseAssets:
    def __init__(self, repository: str, tag: str) -> None:
        self._repository = repository
        self._tag = tag
        self._endpoint = f'repos/{repository}/releases'
        release = json.loads(
            _gh('api', f'{self._endpoint}/tags/{quote(tag, safe="")}')
        )
        self._release_id = int(release['id'])

    def list_assets(self) -> list[ReleaseAsset]:
        pages = json.loads(
            _gh(
                'api',
                f'{self._endpoint}/{self._release_id}/assets?per_page=100',
                '--paginate',
                '--slurp',
            )
        )
        return [
            ReleaseAsset(
                int(item['id']), item['name'], int(item['size']), item['state']
            )
            for page in pages
            for item in page
        ]

    def upload(self, path: Path) -> None:
        _gh(
            'release',
            'upload',
            self._tag,
            str(path),
            '--repo',
            self._repository,
        )

    def rename(self, asset: ReleaseAsset, name: str) -> None:
        _gh(
            'api',
            '--method',
            'PATCH',
            f'{self._endpoint}/assets/{asset.id}',
            '--raw-field',
            f'name={name}',
        )

    def delete(self, asset: ReleaseAsset) -> None:
        _gh('api', '--method', 'DELETE', f'{self._endpoint}/assets/{asset.id}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage release assets before replacement'
    )
    parser.add_argument('--repository', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('assets', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.assets:
        if not path.is_file() or path.stat().st_size <= 0:
            parser.error(f'Nonempty release asset required: {path}')
    release = GitHubReleaseAssets(args.repository, args.tag)
    for path in args.assets:
        replace_asset(release, path)


if __name__ == '__main__':
    main()
