import glob
import os
from typing import List
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

        # number = int(
        #     "".join([i for i in gesture.attrib["Number"] if i.isdigit()]))
        # sample.number = number
        # sample.speed = gesture.attrib["Speed"]

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

def samples_to_np(samples: List[Sample]):
    return [(s.trajectory.copy(), s.gname) for s in samples]