import random
import numpy as np

class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        self.classes_ = sorted(list(set(y)))

    def transform(self, y):
        return [self.classes_.index(label) for label in y]


class DP(object):
    """
    Score points based on Ramer Douglas Peucker algorithm:
    https://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm

    Example:
        epsilon = 2.0

        dp = DP(trajectory)
        afterdp = dp.run()

        indices_keep = sorted([r[0] for r in dp.results if r[1] > epsilon])
        simplified_trajectory = trajectory[indices_keep]
    """

    def __init__(self, _pts):

        self.pts = np.copy(np.array(_pts))
        self.N = len(self.pts)
        self.results = [[idx, np.inf] for idx in range(self.N)]  # List[index, distance]

    def run(self):
        self.dp(0, self.N - 1)
        self.results = sorted(
            self.results, key=lambda r: r[1], reverse=True  # sort on score
        )
        indices = sorted([r[0] for r in self.results[:16]])  # keep top 16
        resampled = self.pts[indices]
        return resampled

    def dp(self, start, end):
        if start + 1 == end or start == end:
            return
        dmax = -np.inf
        index = -1

        # Find the point with the maximum distance
        for i in range(start + 1, end):
            d = self.point_line_dist(self.pts[i], self.pts[start], self.pts[end])
            if d > dmax:
                index = i
                dmax = d

        assert index != -1

        self.results[index][1] = dmax

        # print start, index, end
        self.dp(start, index)
        self.dp(index, end)

    def point_line_dist(self, point, start, end):
        """Calculate the distance between a point and a line"""
        if np.all(np.equal(start, end)):
            return np.linalg.norm(point - start)

        return np.divide(
            np.abs(np.linalg.norm(np.cross(end - start, start - point))),
            np.linalg.norm(end - start),
        )

def path_length(pts):
    """Path traveled by the points of a gesture: sum of Euclidean
    distances between each consecutive pair of points"""
    ret = 0
    pts = np.array(pts)
    for idx in range(1, len(pts)):
        ret += np.linalg.norm(pts[idx] - pts[idx - 1])

    return ret


def resample(pts, n, variance=0.0) -> np.ndarray:
    """Resample a trajectory in pts into n points"""

    # create random intervals
    scale = (12 * variance) ** 0.5
    intervals = [1.0 + random.uniform(0, 1) * scale for ii in range(n - 1)]
    total = sum(intervals)
    intervals = [val / total for val in intervals]

    # Setup place to store resampled points, and
    # store first point. jj indexes the return matrix
    ret = np.empty((n, len(pts[0])))
    ret[0] = pts[0]
    path_distance = path_length(pts)
    jj = 1

    # now do resampling
    accumulated_distance = 0.0
    interval = path_distance * intervals[jj - 1]

    for ii in range(1, len(pts)):

        distance = np.linalg.norm(pts[ii] - pts[ii - 1])

        if accumulated_distance + distance < interval:
            accumulated_distance += distance
            continue

        previous = pts[ii - 1]
        while accumulated_distance + distance >= interval:

            # Now we need to interpolate between the last point
            # and the current point.
            remaining = interval - accumulated_distance
            t = remaining / distance

            # Handle any precision errors. Note that the distance can
            # be zero if two samples are sufficiently close together,
            # which can result in nan.
            t = min(max(t, 0.0), 1.0)
            if not np.isfinite(t):
                t = 0.5

            ret[jj] = (1.0 - t) * previous + t * pts[ii]

            # Reduce the distances based on how much path we
            # just consumed.
            distance = distance - remaining
            accumulated_distance = 0.0
            previous = ret[jj]
            jj += 1

            # Exit early so we don't go past end
            # of the intervals array.
            if jj == n:
                break

            # select next interval
            interval = path_distance * intervals[jj - 1]

        accumulated_distance = distance

    if jj < n:
        ret[n - 1] = pts[-1]
        jj += 1

    assert jj == n
    return ret


def rotate_by_angle(p, origin=(0, 0), angle=0):
    """Rotate trajectory around origin by an angle"""
    R = np.array([[np.cos(angle), -np.sin(angle)],
                 [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def get_x_3d_rotation_matrix(theta):
    # if it's an angle in degrees, convert to radians, using the formula: rad = theta * pi / 180
    theta = theta * np.pi / 180
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def get_y_3d_rotation_matrix(theta):
    theta = theta * np.pi / 180
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def first_point_to_origin(trajectory):
    trajectory = trajectory - trajectory[0]
    return trajectory


def first_point_to_origin_whole_set(train, val, test):
    train = [(first_point_to_origin(t[0]), t[1]) for t in train]
    val = [(first_point_to_origin(t[0]), t[1]) for t in val]
    test = [(first_point_to_origin(t[0]), t[1]) for t in test]
    return train, val, test

