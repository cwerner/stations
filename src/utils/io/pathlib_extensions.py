from pathlib import Path

__all__ = ["Path"]


def latest(self: Path, pattern: str = "*"):
    """return the last file that matches the provided pattern"""
    files = self.glob(pattern)
    try:
        latest_file = max(files, key=lambda x: x.stat().st_ctime)
    except ValueError:
        raise FileNotFoundError
    return latest_file


def filter_files(files, include=[], exclude=[]):
    for incl in include:
        files = [f for f in files if incl in f.name]
    for excl in exclude:
        files = [f for f in files if excl not in f.name]
    return sorted(files)


def ls(self, recursive=False, include=[], exclude=[]):
    if not recursive:
        out = list(self.iterdir())
    else:
        out = [o for o in self.glob("**/*")]
    out = filter_files(out, include=include, exclude=exclude)
    return out


Path.latest = latest
Path.ls = ls
