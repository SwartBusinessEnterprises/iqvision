# Field-hockey top view from a single photo

`hockey_topview.py` reconstructs a 2-D top view of an outdoor hockey pitch
from one spectator photo (Paris 2024, ESP on the scoreboard).

* Landmarks (goal frame, backline along the boards, apparent heights of people)
  are read from the photo as pixel coordinates.
* A pinhole camera (position, yaw, tilt, focal length) is fitted with
  `scipy.optimize.least_squares`; OpenCV builds the ground-plane homography.
* Every player's feet are back-projected onto the pitch and drawn on a
  regulation FIH pitch (91.40 x 55.00 m, landscape, goals left and right).

Outputs: `hockey_topview.png` and `positions.json` (metres from the left
goal line / far sideline). Run with `python3 analysis/hockey_topview.py`
from the repository root. Accuracy is roughly ±3 m; the ball was not
visible and is inferred from the players' sticks and gaze.
