from pathlib import Path
import bisect
import pickle
import numpy as np
import lmdb

from torch.utils.data import Dataset
from scdp.common.typing import assert_is_instance as aii

class LmdbDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path

        if isinstance(self.path, str):
            self.path = Path(self.path)

        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
            self._keys = []
            self.envs = []
            for db_path in db_paths:
                cur_env = self.connect_db(db_path)
                self.envs.append(cur_env)

                # If "length" encoded as ascii is present, use that
                length_entry = cur_env.begin().get("length".encode("ascii"))
                if length_entry is not None:
                    num_entries = pickle.loads(length_entry)
                else:
                    # Get the number of stores data from the number of entries
                    # in the LMDB
                    num_entries = cur_env.stat()["entries"]

                # Append the keys (0->num_entries) as a list
                self._keys.append(list(range(num_entries)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.env = self.connect_db(self.path)
            num_entries = aii(self.env.stat()["entries"], int)
            # If "length" encoded as ascii is present, we have one fewer
            # data than the stats suggest
            if self.env.begin().get("length".encode("ascii")) is not None:
                num_entries -= 1
            self._keys = list(range(num_entries))
            self.num_samples = num_entries

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )

            data_object = pickle.loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(
                f"{self._keys[idx]}".encode("ascii")
            )
            data_object = pickle.loads(datapoint_pickled)

        return data_object

    def get_metadata(self, num_samples):
        pass

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()