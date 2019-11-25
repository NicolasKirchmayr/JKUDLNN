import os
import tempfile
from urllib.request import urlopen
from urllib.error import HTTPError, URLError


class CachedDownload:
    """
    State of a possibly already cached download
    from a file at some URL to the local filesystem.
    """

    def __init__(self, base_url, file_name, base_path=None,
                 overwrite=False, block_size=4096):
        """
        Set up the cached download.

        Parameters
        ----------
        base_url : str
            URL that points to the directory where the file is located.
        file_name : str
            Name of the file that is to be downloaded.
        base_path : str, optional
            Path to the location where the downloaded file should be stored.
            If not specified, the file is stored in a temporary directory.
        overwrite : bool, optional
            Whether or not an existing local file should be overwritten.
        block_size : int, optional
            Number of bytes to read at once.
        """
        if base_path is None:
            base_path = tempfile.gettempdir()
        elif not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        self.url = '/'.join([base_url.rstrip('/'), file_name])
        self.file = os.path.join(base_path, file_name)
        self.block_size = block_size
        self.overwrite = overwrite

        try:
            self._response = urlopen(self.url)
            self._response.close()
        except HTTPError:
            raise ValueError("wrong URL? {}".format(self.url))
        except URLError:
            raise RuntimeError("could not connect to URL: {}".format(self.url))

    @property
    def file_name(self):
        """ Name of the file that is downloaded. """
        return os.path.basename(self.file)

    def _download_file(self):
        """ Download to file. """
        with open(self.file, 'wb') as fp:
            chunk = self._response.read(self.block_size)
            while chunk:
                fp.write(chunk)
                yield chunk
                chunk = self._response.read(self.block_size)

    def _read_file(self):
        """ Read from existing file. """
        with open(self.file, 'rb') as fp:
            chunk = fp.read(self.block_size)
            while chunk:
                yield chunk
                chunk = fp.read(self.block_size)

    def __enter__(self):
        self._response = urlopen(self.url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._response.close()

    def __iter__(self):
        if self.overwrite or not os.path.exists(self.file):
            return self._download_file()
        else:
            return self._read_file()

    def __len__(self):
        content_length = int(self._response.getheader('Content-Length', 0))
        return 1 + (content_length - 1) // self.block_size
