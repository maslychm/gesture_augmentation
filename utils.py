import random
import numpy as np

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