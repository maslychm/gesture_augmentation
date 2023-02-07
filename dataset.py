import glob
import os
from typing import List, Tuple
import xml.etree.ElementTree as ET
import numpy as np

import zipfile
import urllib.request

class Sample(object):
    """Class containing the low-level sample information.

    sname:      subject name
    gname:      gesture name
    fname:      full path to the sample file
    pid:        participant id(subject id)
    trajectory: original sample trajectory
    timestamps: time stamps
    """

    def __init__(self, sname: str, gname: str, fname: str, pid: int, trajectory: np.ndarray, timestamps: np.ndarray) -> None:
        self.sname: str = sname
        self.gname: str = gname
        self.fname: str = fname
        self.pid: int = pid
        self.trajectory: List[List[float]] = trajectory
        self.timestamps: List[float] = timestamps

    def __str__(self) -> str:
        return f"{self.gname} ({self.sname}): len={len(self.trajectory)}"

    def __eq__(self, __o: object) -> bool:
        return self.fname == __o.fname

    def __hash__(self) -> int:
        return hash(self.fname)

    @classmethod
    def parse_xml(cls, file):
        """
        Reads one sample file
        :param file: the file to read
        :return: a Sample instance
        """
        gesture = ET.parse(file).getroot()

        sname = "sub" + str(gesture.attrib["Subject"])
        pid = int(
            "".join([i for i in gesture.attrib["Subject"] if i.isdigit()]))
        label = "".join([i for i in gesture.attrib["Name"] if not i.isdigit()])
        pts = []
        timestamps = []

        pts = [[float(pt.attrib["X"]), float(pt.attrib["Y"]), ]
               for pt in gesture]
        first_t = float(gesture[0].attrib["T"])
        timestamps = [float(pt.attrib["T"]) - first_t for pt in gesture]

        pts = np.array(pts)
        timestamps = np.array(timestamps)

        sample = cls(
            sname=sname,
            gname=label,
            fname=str(file),
            pid=pid,
            trajectory=pts,
            timestamps=timestamps,
        )

        return sample


class GDSDataset:
    """
    Underlying dataset

    name:       Name of the dataset
    subjects:   All samples per subject
    sgestures:  All samples per subject per gesture type
    samples:    All samples, sorted by filename
    gnames:     All gesture type names
    snames:     All subject names
    """

    def __init__(self, path: str, sub_idx: int = None) -> None:
        self.name: str = "gds"
        self.subjects = {}
        self.sgestures = {}
        self.samples: List[Sample] = []
        self.gnames: List[str] = []
        self.snames: List[str] = []

        samples = self._load(path, sub_idx)
        self._fill(samples)

    def __str__(self) -> str:
        info = [
            f"Dataset: {self.name}",
            f"Subjects: {len(self.subjects)}",
            f"Gesture types: {len(self.gnames)}",
            f"Samples: {len(self.samples)}",
        ]
        return "\n".join(info)

    def _load(self, path: str, sub_idx: int = None) -> List[Sample]:
        """
        Loads the $1-GDS dataset from path.
        Args:
        `path`: path to the dataset
        `sub_idx`: if not None, only load the subject with the given index
        """
        dataset_dir = os.path.join(path, "xml_logs")

        if not os.path.exists(dataset_dir):
            if os.path.exists("../" + path + "/xml_logs"):
                dataset_dir = os.path.join("../" + path, "xml_logs")
            else:
                print("Dataset not found. Download? (y/n)")

                choice = input().lower()
                if choice.startswith('y'):
                    self._download_dollar_gds(path)
                    dataset_dir = os.path.join(path, "xml_logs")
                else:
                    raise FileNotFoundError("Dataset not found.")

        print("Loading dataset from", dataset_dir)

        dirs = sorted(glob.glob(os.path.join(dataset_dir, "*")))
        samples: List[Sample] = []

        for i, dr in enumerate(dirs):
            if "pilot" in dr or (sub_idx is not None and i != sub_idx):
                continue

            for speed in ["slow", "medium", "fast"]:
                for file in sorted(glob.glob(os.path.join(dr, speed, "*.xml"))):
                    sample = Sample.parse_xml(file)
                    samples.append(sample)

        return samples

    def _fill(self, samples: List[Sample]) -> None:
        """
        Fills the dataset with the given samples.
        """
        for sample in samples:
            sname = sample.sname
            gname = sample.gname

            if sname not in self.subjects:
                self.subjects[sname] = []
                self.sgestures[sname] = {}

            if gname not in self.sgestures[sname]:
                self.sgestures[sname][gname] = []

            self.subjects[sname].append(sample)
            self.sgestures[sname][gname].append(sample)

        self.samples = sorted(samples, key=lambda x: x.fname)
        self.snames = sorted(self.subjects.keys())
        self.gnames = sorted(self.sgestures[self.snames[0]].keys())

    @staticmethod
    def _download_dollar_gds(root):
        print("Downloading...")
        filepath = os.path.join(root, "xml.zip")
        urllib.request.urlretrieve(
            "https://depts.washington.edu/acelab/proj/dollar/xml.zip", filepath)
        print("Extracting...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(root)
        print("Done Extracting.")

    def ud_split(self, k: int, fixed: bool = False):
        """
        Get u User-Dependent train, validation, and test splits.
        When `fixed = true`, sample the same way for a consistent test.
        Isolate the test samples from validation; Extract k per class for training.

        Test indices will always be the same irrespective of `fixed` or gname.
        Train and validation indices will be different for each gname, but same
        across runs if `fixed = True`.
        args:
            `k` (int): number of samples per class to use for training.
            `fixed` (bool): if True, use the same train and val indices across runs.

        returns: `train, val, test` as Lists of Tuples[points, label]
        """

        if len(self.snames) > 1:
            raise ValueError("More than 1 subject. Cannot use UD split.")

        sname = self.snames[0]
        rand = np.random.RandomState(42) if fixed else np.random.RandomState(None)

        # Isolate test sample indices (20%) -> always same
        test_indeces = set(
            [i * 5 for i in range(len(self.sgestures[sname][self.gnames[0]]) // 5)]
        )

        test, train, val = [], [], []
        for gname in self.gnames:

            # Different train and val indices for each gname.
            # If `fixed == True`, same across runs but still diff for gnames.
            val_indeces = set(range(len(self.sgestures[sname][gname]))) - test_indeces
            train_indeces = set(list(rand.choice(list(val_indeces), k, replace=False)))
            val_indeces = val_indeces - set(train_indeces)

            # check for no intersections between the three sets
            assert (
                len(train_indeces & val_indeces) == 0
                and len(train_indeces & test_indeces) == 0
                and len(val_indeces & test_indeces) == 0
            )

            for i, sample in enumerate(self.sgestures[sname][gname]):
                if i in test_indeces:
                    test.append(sample)
                elif i in train_indeces:
                    train.append(sample)
                elif i in val_indeces:
                    val.append(sample)

        return samples_to_np(train), samples_to_np(val), samples_to_np(test)

    def ui_split(self, split_idx: int, p: int, k: int, fixed: bool = False):
        """
        User-Independent split.
        args:
            `split_idx` (int): index of the subject to use for training.
            `p` (int): number of subjects to use for validation.
            `k` (int): number of samples per class to use for training.
            `fixed` (bool): if True, use the same train and val indices across runs.
        """
        if len(self.snames) < 10:
            raise ValueError("Not whole set is loaded. Cannot use UI split.")

        train_pids, val_pids, test_pids = self._get_ui_split_indeces(
            split_idx, p=p, fixed=fixed
        )

        rand = np.random.RandomState(42) if fixed else np.random.RandomState(None)

        train, val, test = [], [], []

        for pid in train_pids:
            sname = self.snames[pid]
            for gname in self.gnames:
                train.extend(
                    rand.choice(self.sgestures[sname][gname], k, replace=False)
                )

        for pid in val_pids:
            sname = self.snames[pid]
            for gname in self.gnames:
                val.extend(self.sgestures[sname][gname])

        for pid in test_pids:
            sname = self.snames[pid]
            for gname in self.gnames:
                test.extend(self.sgestures[sname][gname])

        return samples_to_np(train), samples_to_np(val), samples_to_np(test)
    
    def _get_ui_split_indeces(
        self, split_idx: int, p: int, fixed: bool = False
    ) -> Tuple[List, List, List]:

        """
        args:
            `split_idx` [1, 11) is the index of the split to use.
            `p` [1, 2, 4] is the number of train subjects per split.
            `fixed` is whether to use a random seed or a fixed one.

        returns: `train_pids, val_plids, test_pids` as Lists of ints.
        """

        if split_idx not in list(range(1, 11)):
            raise ValueError("Split index must be in range [1, 10]")

        rand = np.random.RandomState(42) if fixed else np.random.RandomState()

        all_indices = set(range(0, 10))

        # always add the "indexed" participant to train
        train_indices = set([split_idx - 1])
        all_indices -= train_indices

        train_indices.update(list(rand.choice(list(all_indices), p - 1, replace=False)))
        all_indices -= train_indices

        val_indices = set(list(rand.choice(list(all_indices), 2, replace=False)))
        all_indices -= val_indices

        test_indices = set(list(rand.choice(list(all_indices), 2, replace=False)))
        all_indices -= test_indices

        return list(train_indices), list(val_indices), list(test_indices)


def samples_to_np(samples: List[Sample]):
    return [(s.trajectory.copy(), s.gname) for s in samples]