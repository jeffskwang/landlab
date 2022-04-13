import argparse
import os
import pathlib
import sys
import tarfile
from packaging.version import Version
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import urlopen


def main(version=None):
    notebooks = NotebookFetcher(version)

    out(f"fetching notebooks for landlab {notebooks.version}")
    out(f"{notebooks.url}")

    try:
        stream = notebooks.open()
    except NotebookError as error:
        err(str(error))
        return -1

    with tarfile.open(fileobj=stream, mode="r|gz") as tfile:
        base = NotebookExtractor(tfile).extract()

    out(f"notebooks have been extracted into {base}")
    print(base)

    return 0


class NotebookError(Exception):
    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        return self._msg


class NotebookFetcher:

    URL = "https://github.com/landlab/landlab/archive/refs"

    def __init__(self, version=None):
        if not version:
            self._version = "master"
        else:
            self._version = "v" + Version(version).base_version

    @property
    def version(self):
        return self._version

    @property
    def url(self):
        return urljoin(NotebookFetcher.URL, f"{self.version}.tar.gz")

    def open(self):
        try:
            stream = urlopen(self.url)
        except HTTPError as error:
            if error.code == 404:
                msg = f"unable to find notebooks for requested landlab version ({self.version})"
            else:
                msg = f"unable to fetch notebooks ({error.reason})"
            raise NotebookError(msg)
        else:
            return stream


class NotebookExtractor:
    def __init__(self, tfile):
        self._tfile = tfile
        self._names = []

    def extract(self):
        self._tfile.extractall(members=self._notebooks())
        return self.base

    def _notebooks(self):
        for tarinfo in self._tfile:
            parts = pathlib.Path(tarinfo.name).parts
            if len(parts) > 1 and parts[1] == "notebooks":
                self._names.append(tarinfo.name)
                yield tarinfo

    @property
    def names(self):
        return sorted(self._names)

    @property
    def base(self):
        return pathlib.Path(os.path.commonprefix(self.names))


def out(text):
    print("\033[1m" + text + "\033[0m", file=sys.stderr)


def err(text):
    print("\033[91m" + text + "\033[0m", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fetch landlab notebooks")
    parser.add_argument("version", nargs="?", default="")
    args = parser.parse_args()

    sys.exit(main(args.version))
