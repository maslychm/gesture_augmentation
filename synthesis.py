from functools import partial
from typing import List, Tuple, Union
import numpy as np
from numpy.random.mtrand import RandomState

from dataset import Sample, samples_to_np

# from copy import deepcopy

from utils import (
    # DP,
    resample,
    # gpsr,
    # rotate_by_angle,
    # get_x_3d_rotation_matrix,
    # get_y_3d_rotation_matrix
)


class DataFactory:
    rand: RandomState = RandomState(42)

    @classmethod
    def reset_random_state(cls, seed: int = 42) -> None:
        cls.rand = RandomState(seed)

    @classmethod
    def spatial_resample(
        cls,
        samples: List[Tuple],
        n: int,
        variance: float = 0.5
    ) -> List[Tuple[np.ndarray, Union[str, int]]]:
        resampled = []
        for pts, label in samples:
            for _ in range(n):
                resample_n = cls.rand.randint(
                    max(len(pts) // 3, 16),
                    len(pts) * 2
                )
                new_pts = resample(pts, resample_n, variance=variance)
                resampled.append((new_pts, label))
        return resampled

    @staticmethod
    def generate_chain(
        originals: Union[List[Sample], List[Tuple[np.ndarray, Union[str, int]]]],
        chain: List[str],
        n: int = 10,
    ) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Generate a chain of synthetic samples from a list of original samples.
        :param originals: list of original samples
        :param chain: list of method names
        :param n: number of samples per original to generate
        """

        if isinstance(originals[0], Sample):
            originals = samples_to_np(originals)

        methods = {
            # "uniform": partial(DataFactory.uniform, noise=0.2),
            # "gaussian": partial(DataFactory.gaussian, sigma=0.08),
            # "rotate": partial(DataFactory.rotate, degree=20),
            # "perspective": partial(DataFactory.rotate_xy, max_angle=30),
            # "scale": partial(DataFactory.scale, max_scale_ratio=0.2),
            # "shear": partial(DataFactory.shear, shear_ratio=0.3),
            "spatial": partial(DataFactory.spatial_resample, variance=0.5),
            # "temporal": partial(DataFactory.temporal_resample, force_resample_cnt=None, vary_intervals=True),
            # "duplicate": partial(DataFactory.point_duplicate, duplication_ratio=0.1),
            # "frame-skip": partial(DataFactory.frame_skip, skip_ratio=0.3),
            # "frame-jitter": partial(DataFactory.sample_with_jitter, option=0),
            # "gpsr": partial(DataFactory.gpsr, fixed_n=0, fixed_r=0),
            # "bezier": partial(DataFactory.bezier_deformation, _target_res=None),
        }

        synth = methods[chain[0]](originals, n=n)
        for method in chain[1:]:
            synth = methods[method](synth, n=1)

        return synth
