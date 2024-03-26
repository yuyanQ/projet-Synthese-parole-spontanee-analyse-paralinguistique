# Copyright (C) 2017-2018 Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
#
# This file is part of auDeep.
#
# auDeep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# auDeep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep. If not, see <http://www.gnu.org/licenses/>.

"""Parser for the Compare 2018 Crying data set"""
from pathlib import Path
from typing import Optional, Mapping, Sequence

import pandas as pd

from audeep.backend.data.data_set import Partition
from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser, _InstanceMetadata

import math

_COMPARE18_CRYING_LABEL_MAP = {
    "0": 0,
    "1": 1,
    "2": 2
}


class Compare18CryingParser(LoggingMixin, Parser):

    def __init__(self, basedir: Path):
        super().__init__(basedir)

        self._metadata_cache = None
        self._audio_dir = basedir / "wav"

    def _metadata(self) -> pd.DataFrame:
        if not self.can_parse():
            raise IOError("unable to parse the ComParE 2018 Crying dataset at {}".format(self._basedir))
        if self._metadata_cache is None:
            metadata_file = self._basedir / "lab" / "ComParE2018_Crying.tsv"
            metadata_file_confidential = self._basedir / "lab" / "ComParE2018_Crying_confidential.tsv"

            if (metadata_file_confidential.exists()):
                self.log.warn("using confidential metadata file")

                self._metadata_cache = pd.read_csv(metadata_file_confidential, sep="\t")
            else:
                self._metadata_cache = pd.read_csv(metadata_file, sep="\t")

        return self._metadata_cache

    def can_parse(self) -> bool:
        metadata_file = self._basedir / "lab" / "ComParE2018_Crying.tsv"
        metadata_file_confidential = self._basedir / "lab" / "ComParE2018_Crying_confidential.tsv"

        if not self._audio_dir.exists():
            self.log.debug("cannot parse: audio directory at %s missing", self._audio_dir)

            return False

        if not metadata_file_confidential.exists() and not metadata_file.exists():
            self.log.debug("cannot parse: metadata file at %s missing", metadata_file)

            return False

        return True

    @property
    def label_map(self) -> Optional[Mapping[str, int]]:
        return _COMPARE18_CRYING_LABEL_MAP

    @property
    def num_instances(self) -> int:
        if not self.can_parse():
            raise IOError("Unable to parse ComParE 2018 Crying data set at {}".format(self._basedir))

        return len(self._metadata())

    @property
    def num_folds(self) -> int:
        if not self.can_parse():
            raise IOError("Unable to parse ComParE 2018 Crying data set at {}".format(self._basedir))

        return 0

    def parse(self) -> Sequence[_InstanceMetadata]:
        if not self.can_parse():
            raise IOError("unable to parse dataset at {}".format(self._basedir))

        meta_list = []

        metadata = self._metadata()
        inverse_label_map = dict(map(reversed, self.label_map.items()))

        for file in sorted(self._audio_dir.glob("*.*")):
            label_numeric = float(metadata.loc[metadata["file_name"] == file.name]["label"])

            # test labels are missing and parsed as NaN
            if not math.isnan(label_numeric):
                label_numeric = int(label_numeric)
            else:
                label_numeric = None

            instance_metadata = _InstanceMetadata(
                path=file,
                filename=file.name,
                label_nominal=None if label_numeric is None else inverse_label_map[label_numeric],
                label_numeric=None,  # inferred from label map
                cv_folds=[],
                partition=Partition.TRAIN if file.name.startswith("train") else Partition.TEST
            )

            self.log.debug("parsed instance %s: label = %s", file.name, label_numeric)
            meta_list.append(instance_metadata)

        return meta_list
