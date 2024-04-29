# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .total_suppression import calculate_iou_matrix
from .search_ver2 import get_box
from .detection import Detection
def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections,img):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections,img)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        # 注意这里features实际上不是记录总的track的features，而是同一帧下的traks的各个特征，随后进行清楚单个的track的特征，相当于每个track只是记录了最后的特征
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, coordinates, targets = [], [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            # print(np.asarray(track.features))
            features += track.features
            targets += [track.track_id for _ in track.features]
            coordinates.append(track.coordinates[-1])
            track.features = []
        if len(targets) != len(coordinates):
            targets = set(targets)
            targets = list(targets)
        # print(np.asarray(coordinates))
        # print(np.asarray(targets))

        self.metric.partial_fit(
            np.asarray(coordinates), np.asarray(targets), active_targets)


    def _match(self, detections,img):

        def now_metric(tracks, dets, track_indices, detection_indices):
            coordinate_det = np.array([dets[i].tlwh for i in detection_indices])
            center_det = np.array([coordinate_det[:, 1] + 0.5*coordinate_det[:, 2] , coordinate_det[:, 0] - 0.5 * coordinate_det[:,3]])
            center_det = center_det.T
            coordinate_tracks = np.array([tracks[i].coordinates[-1] for i in track_indices])
            test_track = np.array([tracks[i].mean for i in track_indices])
            center_track = np.array([coordinate_tracks[:, 1] + 0.5*coordinate_tracks[:, 2] , coordinate_tracks[:, 0] - 0.5 * coordinate_tracks[:,3]]).T
            distance = _pdist(center_det,center_track).T
            cost_matrix = distance
            return cost_matrix
        def last_metric(tracks, dets, track_indices, detection_indices):
            dets = np.array([dets[i].to_xyah() for i in detection_indices])
            center_dets = dets[:, :2]
            center_tracks = np.array([(tracks[i].lastmean[0], tracks[i].lastmean[1]) for i in track_indices])
            distance = _pdist(center_dets, center_tracks).T
            cost_matrix =distance
            return cost_matrix
        def combined_metric(tracks, dets, track_indices, detection_indices):
            cost1 = now_metric(tracks, dets, track_indices, detection_indices)
            cost2 = last_metric(tracks, dets, track_indices, detection_indices)
            cost_matrix = 0.7*cost2 + 0.3*cost1
            return cost_matrix
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        # for k in confirmed_tracks:
        #     print('age是：',self.tracks[k].time_since_update)
        # matches_a, unmatched_tracks_a, unmatched_detections = \
        #     linear_assignment.matching_cascade(
        #         dis_metric, 2000, self.max_age,
        #         self.tracks, detections, confirmed_tracks)
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                combined_metric, 2000, 2,
                self.tracks, detections, confirmed_tracks)
        print('matches_a是', matches_a)
        unmatched_detections = calculate_iou_matrix(detections, unmatched_detections, 0.4)

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.matching_cascade(
                last_metric, 2000, self.max_age,
                self.tracks, detections, unmatched_tracks_a,unmatched_detections)
        print('matches_b是', matches_b)
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        # unmatched_tracks_a = [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update != 1]
        # 这里首先筛除了重复框
        # unmatched_detections = calculate_iou_matrix(detections, unmatched_detections, 0.5)


        matches_c, unmatched_tracks_c, unmatched_detections = \
            linear_assignment.min_cost_matching(
                now_metric, 2000, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b + matches_c
        unmatched_tracks = list(set(unmatched_tracks_b + unmatched_tracks_c))
        print('matches_c是', matches_c)


        # Re_search_trackids = [
        #     k for k in unmatched_tracks_b if
        #     self.tracks[k].time_since_update == 1]
        # matches_d = []
        # for track_idx in Re_search_trackids:
        #     track = self.tracks[track_idx]
        #     box = get_box(track, detections, img)
        #     if len(box) != 0:
        #         t, l, w, h = box
        #         tlwh = [t, l, w, h]
        #         # print('tlwh是', t, l, w, h)
        #         new_detection = Detection(tlwh, 1, [], track.cls)
        #
        #         detections.append(new_detection)
        #         detection_idx = len(detections) - 1
        #         matches_d.append((track_idx, detection_idx))
        # unmatched_tracks = [x for x in unmatched_tracks if x not in matches_c]
        # print('matches_d是', matches_d)
        # matches = matches + matches_d
        # print('matches是', matches)

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, detection.tlwh))
        self._next_id += 1
