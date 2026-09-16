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

## Second photo: warm-up (`warmup_topview.py`)

Camera on the near sideline about 35 m from the left backline, 4 m up, long
lens. Landmarks: corner flag at the far-left corner, the circle arc (start on
the backline, apex toward the centre), the keeper beside the just-off-frame
goal and the penalty spot at the left edge. Outputs `warmup_topview.png` and
`positions_warmup.json`. Both scripts share the same left-handed camera frame
(X along the pitch, Y toward the near sideline, Z up).

FIH pitch used throughout: 91.40 x 55.00 m, shooting circle radius 14.63 m
from each post joined by a 3.66 m straight, dotted line 5 m outside it
(radius 19.63 m), 23 m lines at 22.90 m, penalty spot 6.40 m from the goal
line, goal 3.66 m wide, 2.14 m high, 1.22 m deep.
