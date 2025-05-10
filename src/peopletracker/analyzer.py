import json
import os
import pathlib
import sys

import cv2
import numpy as np

import deep_sort.detection
import deep_sort.nn_matching
import deep_sort.preprocessing
import deep_sort.tracker

from tools import generate_detections
from utils import poses2boxes

if "OPENPOSE_PATH" not in os.environ:
    raise Exception("Environment variable 'OPENPOSE_PATH' is not set")

op_path = str((pathlib.Path(os.environ["OPENPOSE_PATH"]) / "build/python").resolve())
if op_path not in sys.path:
    sys.path.append(op_path)
del op_path

from openpose import pyopenpose as op


MAX_COSINE_DISTANCE = 1
NN_BUDGET = None
NMS_MAX_OVERLAP = 1.0
MAX_AGE = 100
N_INIT = 20


class EnhancedTracker(deep_sort.tracker.Tracker):
    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        super().__init__(metric, max_iou_distance, max_age, n_init)
        self.stats = {}

    def update(self, frame, detections, frame_number):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(frame, detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.trackerinuse, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        self._update_stats(frame_number)

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _stat(self, start_frame):
        return {"start_frame": start_frame, "duration": 1}

    def _update_stats(self, frame_number):
        for track in self.tracks:
            if track.track_id not in self.stats:
                self.stats[track.track_id] = self._stat(frame_number)
            else:
                stat = self.stats[track.track_id]
                stat["duration"] = track.age


class Input:
    def __init__(self, file_path=""):
        params = dict()
        params["model_folder"] = (
            pathlib.Path(os.environ["OPENPOSE_PATH"]) / "models"
        ).resolve()
        params["net_resolution"] = "-1x320"
        self.openpose = op.WrapperPython()
        self.openpose.configure(params)
        self.openpose.start()

        self.nms_max_overlap = NMS_MAX_OVERLAP

        model_filename = str(
            (pathlib.Path(__file__).parent / "model_data/mars-small128.pb").resolve()
        )
        self.encoder = generate_detections.create_box_encoder(
            model_filename, batch_size=1
        )
        metric = deep_sort.nn_matching.NearestNeighborDistanceMetric(
            "cosine", MAX_COSINE_DISTANCE, NN_BUDGET
        )
        self.tracker = EnhancedTracker(metric, max_age=MAX_AGE, n_init=N_INIT)

        self.capture = cv2.VideoCapture(file_path if file_path else 0)
        if self.capture.isOpened():
            self.frameSize = (
                int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            )
        else:
            self.frameSize = (0, 0)

        self.frame_number = 0

    def run(self):
        result, self.currentFrame = self.capture.read()

        if not result:
            return result

        self.frame_number += 1

        datum = op.Datum()
        datum.cvInputData = self.currentFrame
        self.openpose.emplaceAndPop(op.VectorDatum([datum]))

        keypoints, self.currentFrame = np.array(datum.poseKeypoints), datum.cvOutputData

        # Doesn't use keypoint confidence.
        try:
            poses = keypoints[:, :, :2]
        except:
            return False

        # Get containing box for each seen body.
        boxes = poses2boxes(poses)
        boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for [x1, y1, x2, y2] in boxes]
        features = self.encoder(self.currentFrame, boxes_xywh)

        nonempty = lambda xywh: xywh[2] != 0 and xywh[3] != 0
        detections = [
            deep_sort.detection.Detection(bbox, 1.0, feature, pose)
            for bbox, feature, pose in zip(boxes_xywh, features, poses)
            if nonempty(bbox)
        ]
        # Run non-maxima suppression.
        boxes_det = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = deep_sort.preprocessing.non_max_suppression(
            boxes_det, self.nms_max_overlap, scores
        )
        detections = [detections[i] for i in indices]
        # Call the tracker.
        self.tracker.predict()
        self.tracker.update(self.currentFrame, detections, self.frame_number)

        for track in self.tracker.tracks:
            color = None
            if not track.is_confirmed():
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)
            bbox = track.to_tlbr()

            cv2.rectangle(
                self.currentFrame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )
            cv2.putText(
                self.currentFrame,
                "id%s - ts%s" % (track.track_id, track.time_since_update),
                (int(bbox[0]), int(bbox[1]) - 20),
                0,
                5e-3 * 200,
                (0, 255, 0),
                2,
            )

        return result

    def get_stats(self):
        return self.tracker.stats


class Analyzer:
    def __init__(self, input_video_path="", output_video_path=""):
        self.input = Input(input_video_path)
        self.output_video_path = output_video_path

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(
            self.output_video_path,
            fourcc=fourcc,
            fps=25,
            frameSize=self.input.frameSize,
        )
        while True:
            result = self.input.run()
            if not result:
                break

            output_video.write(self.input.currentFrame)

        output_video.release()

    def get_stats(self, json_format=False):
        result = self.input.get_stats()
        if json_format:
            result = json.dumps(result)
        return result
