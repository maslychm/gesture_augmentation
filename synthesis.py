from functools import partial
from typing import List, Tuple, Union
import numpy as np
from numpy.random.mtrand import RandomState

from dataset import Sample, samples_to_np

from scipy import interpolate

# from copy import deepcopy

from utils import (
    DP,
    resample,
    # gpsr,
    rotate_by_angle,
    get_x_3d_rotation_matrix,
    get_y_3d_rotation_matrix
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
        """
        Spatially resample trajectories by representing them by a different number of points.
        args:
            `samples` (List): list of training samples
            `n` (int): num of augmented to generate per each original
            `variance` (float): variance of the number of points to generate. Defaults to 0.5.
        """
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

    @classmethod
    def temporal_resample(
        cls,
        samples: List[Tuple],
        n: int
    ) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Assume a fixed sampling rate. Generate random time intervals to resample the trajectories.
        args:
            `samples` (List): list of training samples
            `n` (int): num of augmented to generate per each original
        """

        time_resampled = []
        for pts, label in samples:
            for _ in range(n):

                resample_n = max(16, cls.rand.randint(
                    len(pts) // 2, len(pts) * 2))

                intervals = [cls.rand.uniform(0.25, 1)
                             for _ in range(resample_n)]
                intervals = np.cumsum(intervals)
                intervals = intervals / intervals[-1]

                new_sample = interpolate.interp1d(
                    np.linspace(0, 1, len(pts)), pts, kind="linear", axis=0
                )(intervals)

                time_resampled.append((new_sample, label))

        return time_resampled

    @classmethod
    def gaussian(
        cls,
        samples: List[Tuple],
        n: int,
        sigma: float = 0.08
    ) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Generate gaussian perturbations by shifting each point by a random amount.
        offsets = percent_of_bounding_box * np.random.normal(0, sigma, points.shape).
        args:
            `samples` (List): list of training samples
            `n` (int): num of augmented to generate per each original
            `sigma` (float): ratio of the bounding box to be equal to sigma (standard deviation). Defaults to 0.08.
        """
        gassed = []
        for points, label in samples:

            original_size = np.max(points, axis=0) - np.min(points, axis=0)
            max_shift = original_size * sigma

            for _ in range(n):
                rnd = cls.rand.normal(0, max_shift, points.shape)
                new_sample = points + rnd
                gassed.append((new_sample, label))

        return gassed

    @classmethod
    def uniform(
        cls, samples: List[Tuple], n: int, noise: float = 0.15
    ) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Generate uniform offsets (jitter) with random perturbations by shifting each
        point by a random amount in +-20% of the original bounding box.
        args:
            `samples` (List): list of training samples
            `n` (int): num of augmented to generate per each original
            `noise` (float): Defaults to 0.15. -15% and +15% of the original bounding box.
        """

        offset = []
        for points, label in samples:

            original_size = np.max(points, axis=0) - np.min(points, axis=0)
            max_shift = original_size * noise

            for _ in range(n):
                rnd = cls.rand.uniform(-max_shift, max_shift, points.shape)
                new_sample = points + rnd
                offset.append((new_sample, label))

        return offset

    @classmethod
    def scale(
        cls, samples: List[Tuple], n: int, max_scale_ratio: float = 0.2,
    ) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Generate scaled samples by scaling each dimension by a random value +-max_scale_ratio.
        args:
            `samples` (List): list of training samples
            `n` (int): num of augmented to generate per each original
            `max_scale_ratio` (float): when set to .2 samples will be generated
                of scales from 0.8 to 1.2 around the original.
        """

        # set to `True` to keep the trajectory centroid fixed
        fix_centroid = True

        scaled = []
        for points, label in samples:

            if fix_centroid:
                translation_reference = np.mean(points, axis=0)
            else:
                translation_reference = points[0]

            for _ in range(n):
                rnd = [
                    cls.rand.uniform(1 - max_scale_ratio, 1 + max_scale_ratio)
                    for _ in range(points.shape[-1])
                ]

                rnd = np.asanyarray(rnd)

                new_sample = points - translation_reference
                new_sample = new_sample * rnd
                new_sample += translation_reference

                scaled.append((new_sample, label))

        return scaled

    @classmethod
    def rotate(
        cls, samples: List[Tuple], n: int, degree: int = 20
    ) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Add rotations in a range of degrees.
        args:
            `samples` (List): list of training samples
            `n` (int): num of augmented to generate per each original
            `degree` (float): degree of max rotation. Defaults to 20.
        """

        rotated = []
        for points, label in samples:

            # Generate rotation choices
            rotations = np.linspace(-degree, degree, n)

            # If small number of samples to generate, randomize degree
            if n < 5:
                rotations = [cls.rand.randint(-degree, degree)
                             for _ in range(n)]

            for angle in rotations:
                new_sample = rotate_by_angle(
                    points, origin=np.mean(points, axis=0), angle=np.deg2rad(angle)
                )
                rotated.append((new_sample, label))

        return rotated

    @classmethod
    def shear(cls, samples: List[Tuple], n: int, shear_ratio: float = 0.3) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        `shear_ratio` (float) (-1, 1) is the range of the shear ratio.
        Multiply each point of the trajectory by a shearing matrix with a factor m.
        args:
            `samples` (List): list of training samples
            `n` (int): num of augmented to generate per each original
            `shear_ratio` (float): Defaults to 0.3.
        """
        sheared = []
        for pts, label in samples:
            for _ in range(n):

                m, p = 0.0, cls.rand.uniform(-shear_ratio, shear_ratio)
                if np.random.uniform(0, 1) > 0.5:
                    m, p = p, m
                shear_matrix = np.array([[1.0 + m * p, m], [p, 1.0]])

                sheared_trajectory = np.copy(pts)
                first_pt = pts[0]
                sheared_trajectory -= first_pt
                sheared_trajectory = sheared_trajectory @ shear_matrix
                sheared_trajectory += first_pt

                sheared.append((sheared_trajectory, label))

        return sheared

    @classmethod
    def perspective(cls, samples: List[Tuple], n: int, max_angle: float = 30.0) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Generate perspective transformations. For 2D trajectories this means rotation around the X and Y axes.
        args:
            `samples` (List): list of training samples
            `n` (int): num of augmented to generate per each original
            `max_angle` (float): Defaults to 30.0.
        """
        rotated = []
        for pts, label in samples:

            orig_trajectory = np.copy(pts)
            threed_trajectory_base = np.zeros((orig_trajectory.shape[0], 3))
            threed_trajectory_base[:, :2] = np.copy(orig_trajectory)

            rotation_origin = np.mean(threed_trajectory_base, axis=0)
            threed_trajectory_base -= rotation_origin

            for _ in range(n):

                threed_trajectory = np.copy(threed_trajectory_base)

                (x_angle, y_angle) = (
                    cls.rand.uniform(-max_angle, max_angle),
                    cls.rand.uniform(-max_angle, max_angle),
                )

                threed_trajectory = threed_trajectory @ get_x_3d_rotation_matrix(
                    x_angle)
                threed_trajectory = threed_trajectory @ get_y_3d_rotation_matrix(
                    y_angle)

                threed_trajectory += rotation_origin

                twod_trajectory = threed_trajectory[:, :2]
                rotated.append((twod_trajectory, label))

        return rotated

    @classmethod
    def sampling_with_frame_jitter(
        cls, samples: List[Tuple], n: int, option: int = 0
    ) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Get every ith frame with a random offset in a small bound.
        args:
            `samples` (List): list of training samples
            `n` (int): num of augmented to generate per each original
            `option` (int): 0: every 2nd frame, 1: every 3rd frame, 2: every 4th frame, 3: every 5th frame
        """

        option = int(option)

        # Keep every 2nd, 3rd, 4th, or 5th frame
        every_ith_options = [2, 3, 4, 5]

        # Range of selected frame offset
        offset_range_options = [(-1, 1), (-2, 2), (-3, 3), (-4, 4)]

        k = every_ith_options[option]
        st, en = offset_range_options[option]

        ith_noisy = []
        for points, label in samples:
            for _ in range(n):

                new_sample = []

                i = 0
                while i < len(points):
                    if i % k == 0:
                        selected_idx = i + cls.rand.randint(st, en + 1)
                        selected_idx = min(
                            max(0, selected_idx), len(points) - 1
                        )  # clamp
                        new_sample.append(points[selected_idx])
                    i += 1

                new_sample.append(points[-1])  # always add the last one
                new_sample = np.array(new_sample)

                ith_noisy.append((new_sample, label))

        return ith_noisy

    @classmethod
    def point_duplicate(
        cls, samples: List[Tuple], n: int, duplication_ratio: float = 0.1
    ) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """Duplicate points at random positions.
        args:
            `samples` (List): list of training samples
            `n` (int): num of augmented to generate per each original
            `duplication_ratio` (float): Defaults to 0.1.
        """
        point_duplicated = []
        for pts, label in samples:
            for _ in range(n):
                new_sample = []
                for pt in np.copy(pts):
                    new_sample.append(pt)
                    # duplicate the points if the random chance is met
                    if cls.rand.uniform() < duplication_ratio:
                        new_sample.append(pt)

                new_sample = np.array(new_sample)
                point_duplicated.append((new_sample, label))

        return point_duplicated

    @classmethod
    def frame_skip(
        cls, samples: List[Tuple], n: int, skip_ratio: float = 0.3,
    ) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Randomly skip a specified ratio of frames.
        args:
            `samples` (List): list of training samples
            `n` (int): num of augmented to generate per each original
            `skip_ratio` (float): Defaults to 0.3.
        """

        skipped = []
        for points, label in samples:
            for _ in range(n):

                new_sample = np.array(
                    [pt for i, pt in enumerate(
                        points) if cls.rand.rand() > skip_ratio]
                )

                skipped.append((new_sample, label))

        return skipped

    @classmethod
    def bezier_deformation(cls, samples: List[Tuple], n: int) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Apply Bezier deformation using B-spline.
        args:
            `samples` (List): list of training samples
            `n` (int): number of new samples to generate per original sample

        Apply Bezier deformation using B-spline. Steps:
        Step 1: select control points using Line Simplification algorithm.
            * keep first and last points as control points
            * use RDP to discard points that are too close to lines made by other control points (epsilon=1.5/127=~1.18)
        Step 2: pass control points to the bspline
            * uniformly resample points
            * pass the control knots and points to the b-spline
            * resample to a target "out_res" resolution using the spline
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.BSpline
        """

        extrapolate = False  # may go outside of original trajectory bounds
        gds_bb_len = 127.0
        gds_epsilon = 1.5
        epsilon_percent = gds_epsilon / gds_bb_len  # 1.5/127*100 =~1.18%
        resample_enhance_factor = 2
        noise = 0.07

        bezier_deformed = []

        for pts, label in samples:
            trajectory = np.copy(pts)

            # Select control points using Line Simplification algorithm.
            epsilon = (
                np.linalg.norm(np.max(trajectory, axis=0) - np.min(trajectory, axis=0))
                * epsilon_percent
            )

            dp = DP(trajectory)
            dp.run()
            indices_keep = sorted([r[0] for r in dp.results if r[1] > epsilon])
            simplified_points = trajectory[indices_keep]

            # Uniformly resample up (for bspline to work well)
            resampled_points = np.copy(
                resample(
                    simplified_points, len(simplified_points) * resample_enhance_factor
                )
            )
            selected_indices = np.arange(len(resampled_points) + 3)

            original_size = np.max(trajectory, axis=0) - np.min(trajectory, axis=0)
            max_shift = original_size * noise

            for _ in range(n):

                selected_points = np.copy(resampled_points)

                # Add random offsets to control points
                rnd = cls.rand.uniform(-max_shift, max_shift, selected_points.shape)
                selected_points = np.copy(selected_points + rnd)

                # Fit Bspline to knots and points
                spl = interpolate.BSpline(selected_indices, selected_points, 2)

                # What is the target resolution?
                out_res = cls.rand.randint(
                    len(trajectory) // 2, len(trajectory) * 2
                )

                # account for the NaN points due to no extrapolation
                out_res = int(out_res * 1.4)
                
                # Resample along the spline
                xx = np.linspace(0, selected_indices[-1], out_res)
                traj_out = spl(xx, extrapolate=extrapolate)
                traj_out = traj_out[~np.isnan(traj_out).any(axis=1)]

                bezier_deformed.append((traj_out, label))

        return bezier_deformed

    @classmethod
    def generate_avc(cls, samples: List[Tuple], n: int) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Generate chain of `gaussian` -> `frame-skip` -> `spatial` -> `perspective` -> `rotate` -> `scale`.
        args:
            `samples` (List): list of training samples
            `n` (int): number of new samples to generate per original sample
        """
        return cls.generate_chain(samples, ["gaussian", "frame-skip", "spatial", "perspective", "rotate", "scale"], n)

    @classmethod
    def generate_simple(cls, samples: List[Tuple], n: int) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Generate chaing of `rotate` -> `scale` -> `gaussian`.
        args:
            `samples` (List): list of training samples
            `n` (int): number of new samples to generate per original sample
        """
        return cls.generate_chain(samples, ["rotate", "scale", "gaussian"], n)

    @staticmethod
    def generate_chain(
        originals: Union[List[Sample], List[Tuple[np.ndarray, Union[str, int]]]],
        chain: List[str],
        n: int = 10,
    ) -> List[Tuple[np.ndarray, Union[str, int]]]:
        """
        Generate a chain of synthetic samples from a list of original samples.
        args:
            `originals` (List) list of original samples
            `chain` (List): list of method names
            `n` (int): number of samples per original to generate
        """

        if isinstance(originals[0], Sample):
            originals = samples_to_np(originals)

        methods = {
            "gaussian": partial(DataFactory.gaussian, sigma=0.08),
            "uniform": partial(DataFactory.uniform, noise=0.2),
            "rotate": partial(DataFactory.rotate, degree=20),
            "perspective": partial(DataFactory.perspective, max_angle=30),
            "scale": partial(DataFactory.scale, max_scale_ratio=0.2),
            "shear": partial(DataFactory.shear, shear_ratio=0.3),
            "spatial": partial(DataFactory.spatial_resample, variance=0.5),
            "temporal": partial(DataFactory.temporal_resample),
            "duplicate": partial(DataFactory.point_duplicate, duplication_ratio=0.1),
            "frame-skip": partial(DataFactory.frame_skip, skip_ratio=0.3),
            "frame-jitter": partial(DataFactory.sampling_with_frame_jitter, option=0),
            # "gpsr": partial(DataFactory.gpsr, fixed_n=0, fixed_r=0),
            "bezier": partial(DataFactory.bezier_deformation),
        }

        synth = methods[chain[0]](originals, n=n)
        for method in chain[1:]:
            synth = methods[method](synth, n=1)

        return synth
