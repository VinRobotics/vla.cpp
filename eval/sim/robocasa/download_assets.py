# Copyright 2026 VinRobotics - Apache-2.0
"""Install the minimal upstream asset packs with explicit completion markers."""

from pathlib import Path
from zipfile import ZipFile

PACKS = ("tex", "tex_generative", "fixtures_lw", "objs_lw")


def install_pack(name, config, download_url):
    folder = Path(config["folder"])
    complete = folder / f".vla-{name}.complete"
    if complete.is_file():
        print(f"[assets] reusing {name}: {folder}", flush=True)
        return
    folder.parent.mkdir(parents=True, exist_ok=True)
    archive = folder.parent / f".vla-{name}.zip"
    print(f"[assets] installing {name}: {folder}", flush=True)
    # Use the upstream downloader, but own extraction and the success boundary.
    # download_and_extract_zip silently returns after exhausted network retries;
    # that cannot distinguish success from pre-existing fixture metadata.
    download_url(config["url"], str(folder.parent), archive.name, check_overwrite=False)
    with ZipFile(archive) as contents:
        if not any(item.filename.startswith(folder.name + "/") and not item.is_dir()
                   for item in contents.infolist()):
            raise RuntimeError(f"Asset archive {name} contains no files for {folder.name}")
        contents.extractall(folder.parent)
    if not folder.is_dir() or not any(folder.iterdir()):
        raise RuntimeError(f"Asset extraction incomplete: {name}: {folder}")
    complete.touch()
    archive.unlink()


def main():
    from robocasa.scripts.download_kitchen_assets import DOWNLOAD_ASSET_REGISTRY, download_url
    for name in PACKS:
        for attempt in range(3):
            try:
                install_pack(name, DOWNLOAD_ASSET_REGISTRY[name], download_url)
                break
            except Exception:
                if attempt == 2:
                    raise
                print(f"[assets] retrying {name} after failed attempt {attempt + 1}", flush=True)


if __name__ == "__main__":
    main()
