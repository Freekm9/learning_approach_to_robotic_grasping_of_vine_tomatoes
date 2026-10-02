import os
import time
import warnings

import cv2
import numpy as np

# Unmodified copy of Robert Babuska's detector (find_tomato_grasp.py). To update it,
# replace that file; nothing in it is specific to this package.
from find_tomato_grasp import find_tomato_grasp

# Every frame and its annotated result are kept here, so detections can be
# inspected (or rerun with the command-line version) afterwards.
OUTPUT_DIR = os.path.expanduser("~/.ros/babuska_grasp")

# Length (pixels) of the stem-direction vector handed to generate_grasp_pose;
# only its angle is used.
DIRECTION_LENGTH_PX = 50

WINDOW_NAME = 'babuska_pick_point'
GRASP_STATUSES = ("best_candidate", "weak_candidate", "at_point")


class BabuskaPicker():
    """Grasp point from find_tomato_grasp instead of two manual clicks.

    Click a truss: the detector looks for its free peduncle end and draws the grasp.
    'a' runs it on the whole image without a selection, 'g' toggles grasp-at-point
    (grasp next to the clicked point instead of at the peduncle end). Clicking again
    redoes the detection. Enter confirms, Esc cancels."""

    def reset(self, image):
        """image is RGB, as in PickTwoPoints."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.frame_file = os.path.join(OUTPUT_DIR, stamp + ".png")
        self.output_file = os.path.join(OUTPUT_DIR, stamp + "_grasp.png")
        cv2.imwrite(self.frame_file, image[..., ::-1])

        self.display = image[..., ::-1].copy()
        self.grasp_at_point = False
        self.pending = None   # ("auto", None) or ("click", (x, y)), run from the draw loop
        self.result = None    # (point, direction, info) of the last detection
        self.save = False

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending = ("click", (x, y))

    def detect(self, truss_point):
        """truss_point is a 0-based (x, y) pixel, or None for automatic detection."""
        mode = "automatic" if truss_point is None else (
            "grasp at point" if self.grasp_at_point else "truss selection")
        print(f"Running find_tomato_grasp ({mode})...")
        kwargs = dict(output_file=self.output_file)
        if truss_point is not None:
            # find_tomato_grasp uses MATLAB's 1-based pixel coordinates.
            kwargs.update(truss_point=[truss_point[0] + 1, truss_point[1] + 1],
                          grasp_at_point=self.grasp_at_point)
        start = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            point, direction, info = find_tomato_grasp(self.frame_file, **kwargs)
        for w in caught:
            print(f"  warning: {w.message}")
        print(f"  status={info['status']}  point={point}  direction={direction}  "
              f"({time.time() - start:.2f} s, saved {self.output_file})")

        self.result = (point - 1, direction, info)
        self.display = cv2.imread(self.output_file)

    def draw(self):
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.click_event)
        while True:
            shown = self.display.copy()
            self._draw_help(shown)
            cv2.imshow(WINDOW_NAME, shown)
            pressed_key = cv2.waitKey(20) & 0xFF

            if self.pending is not None:
                kind, xy = self.pending
                self.pending = None
                self._draw_busy(shown)
                self.detect(xy if kind == "click" else None)
                continue

            if pressed_key == 27:  # Esc cancels
                break
            if pressed_key == ord('a'):
                self.pending = ("auto", None)
            elif pressed_key == ord('g'):
                self.grasp_at_point = not self.grasp_at_point
                print(f"grasp at point: {self.grasp_at_point}")
            elif pressed_key in (10, 13):  # Enter confirms
                if self.has_grasp():
                    self.save = True
                    break
                print("No grasp detected yet: click a truss or press 'a'")
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            print("Window already closed. Ignoring")
        cv2.waitKey(100)

    def has_grasp(self):
        return (self.result is not None and self.result[2]["status"] in GRASP_STATUSES
                and np.all(np.isfinite(self.result[0])))

    def points(self):
        """Grasp point and a second point along the stem, in the (center, direction)
        format PickTwoPoints produces for SimplePickPoint.generate_grasp_pose."""
        point, direction, _ = self.result
        center = point
        ahead = point + DIRECTION_LENGTH_PX * direction
        return (center[0], center[1]), (ahead[0], ahead[1])

    def _draw_help(self, image):
        mode = "grasp at point" if self.grasp_at_point else "peduncle end"
        status = "" if self.result is None else f"   last: {self.result[2]['status']}"
        lines = [f"click truss | a: automatic | g: mode ({mode}) | Enter: grasp | Esc: cancel{status}"]
        for i, text in enumerate(lines):
            cv2.putText(image, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(image, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _draw_busy(self, image):
        cv2.putText(image, "detecting...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, image)
        cv2.waitKey(1)
