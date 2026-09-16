"""
Reconstruct a top-down view of an outdoor field-hockey pitch from a single
spectator photo (Paris 2024, ESP on the scoreboard).

Pipeline
  1. Landmarks read from the photo (pixel coords, 1280x960):
     goal posts + crossbar, the backline running along the far boards,
     and the apparent heights of several people (depth cue).
  2. Fit a pinhole camera (position, yaw, tilt, focal length) with
     scipy.optimize.least_squares (OpenCV is used for the final
     ground-plane homography).
  3. Back-project every player's feet to the pitch plane.
  4. Draw a regulation FIH pitch (91.40 x 55.00 m) landscape, goals
     left/right, with all players, GK, umpire and the inferred ball.
"""
import numpy as np, cv2, json
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, Circle

W, H = 1280, 960
CX, CY = W / 2, H / 2

# ---------------------------------------------------------------- pitch (FIH)
L, WID = 91.40, 55.00          # length, width
GOAL_W, GOAL_D, GOAL_H = 3.66, 1.22, 2.14
R_D, R_DOT = 14.63, 19.63      # shooting circle / 5 m dotted line radii
LINE_23 = 22.90
PEN_SPOT = 6.40
YC = WID / 2                   # y of the goal centre (Y=0 is the FAR sideline)
POST_FAR, POST_NEAR = YC - GOAL_W / 2, YC + GOAL_W / 2

# ------------------------------------------------------- image measurements
# world (X along pitch from the left backline, Y across from the far sideline,
# Z up) -> pixel.  Right post in the photo = post nearest the camera side.
pts_world = np.array([
    [0, POST_NEAR, 0],       # near post base
    [0, POST_FAR, 0],        # far post base
    [0, POST_FAR, GOAL_H],   # crossbar, far end
    [0, POST_NEAR, GOAL_H],  # crossbar, near end
], float)
pts_img = np.array([[345, 440], [192, 440], [192, 332], [345, 332]], float)

# the backline continues along the base of the far boards (345,440)->(1280,505)
backline_a, backline_b = np.array([345, 440.]), np.array([1280, 505.])
backline_samples = [(0, 35, 0), (0, 42, 0), (0, 49, 0)]
near_corner_px = 1390          # (0,55) sits just past the right edge

# people used as depth cues: (feet pixel, pixel height, real height)
height_cues = [((778, 512), 70, 1.80),   # umpire
               ((440, 855), 145, 1.80),  # white player, bottom of frame
               ((975, 748), 120, 1.80),  # white #10
               ((835, 690), 105, 1.80),  # red player, bent forward
               ((265, 440), 72, 1.85)]   # goalkeeper (padded)

# players: feet pixel positions read from the photo
players = {
    "R1": (515, 548), "R2": (750, 585), "R3": (835, 690), "R4": (1015, 650),
    "R5": (5, 500),                                    # partly out of frame
    "W1": (555, 545), "W2": (975, 748), "W3": (1115, 640), "W4": (440, 855),
    "GK": (265, 440), "UMP": (778, 512),
    "BALL": (900, 715),  # not visible; inferred from R3 / W2 stick + gaze
}

# ------------------------------------------------------------- camera model
def cam_axes(az, tilt):
    fwd = np.array([-np.cos(az), -np.sin(az), 0.0])
    fwd = np.array([fwd[0] * np.cos(tilt), fwd[1] * np.cos(tilt), -np.sin(tilt)])
    up0 = np.array([0, 0, 1.0])
    right = np.cross(fwd, up0); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return fwd, right, down

def project(P, prm):
    Xc, Yc, Zc, az, tilt, f = prm
    fwd, right, down = cam_axes(az, tilt)
    p = np.atleast_2d(P) - np.array([Xc, Yc, Zc])
    z = p @ fwd
    return np.c_[CX + f * (p @ right) / z, CY + f * (p @ down) / z]

def ground_homography(prm):
    """pixel -> ground (Z=0) as a 3x3 homography, computed with OpenCV."""
    g = np.array([[0, 0, 0], [80, 0, 0], [0, 55, 0], [80, 55, 0], [40, 27, 0], [20, 45, 0]], float)
    img = project(g, prm).astype(np.float32)
    Hm, _ = cv2.findHomography(img, g[:, :2].astype(np.float32))
    return Hm

def back_project(px, prm):
    Hm = ground_homography(prm)
    q = cv2.perspectiveTransform(np.array([[px]], np.float32), Hm)[0, 0]
    return float(q[0]), float(q[1])

def residuals(prm):
    r = []
    r += list((project(pts_world, prm) - pts_img).ravel())
    d = backline_b - backline_a; n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    for s in backline_samples:
        r.append(float((project(np.array(s), prm)[0] - backline_a) @ n))
    r.append(0.3 * (project(np.array([0, WID, 0]), prm)[0, 0] - near_corner_px))
    for (fx, fy), hpx, hm in height_cues:
        gx, gy = back_project((fx, fy), prm)
        head = project(np.array([gx, gy, hm]), prm)[0]
        r.append(0.5 * ((fy - head[1]) - hpx))
    Xc, Yc, Zc = prm[:3]
    r.append(0.2 * (Yc - 66))     # weak priors: camera in the near stand,
    r.append(0.2 * (Zc - 18))     # ~10 m behind the sideline, elevated
    return r

x0 = [80, 66, 18, np.deg2rad(20), np.deg2rad(13), 3600]
fit = least_squares(residuals, x0, bounds=([30, 56, 5, 0, 0, 800], [110, 90, 40, 1.2, 0.8, 9000]))
prm = fit.x
print("camera X=%.1f Y=%.1f Z=%.1f  az=%.1f deg  tilt=%.1f deg  f=%.0f px  (rms %.1f)" %
      (prm[0], prm[1], prm[2], np.degrees(prm[3]), np.degrees(prm[4]), prm[5],
       np.sqrt(np.mean(np.square(fit.fun)))))

# The goal in the photo is the RIGHT-hand goal of the drawn pitch, so mirror
# the along-pitch coordinate (the camera sits at the left end of the near stand).
pos = {k: (L - back_project(v, prm)[0], back_project(v, prm)[1]) for k, v in players.items()}
for k, (x, y) in pos.items():
    print("%-4s  X=%5.1f m from left goal line   Y=%5.1f m from far sideline" % (k, x, y))
json.dump({"camera": {"X": prm[0], "Y": prm[1], "Z": prm[2], "azimuth_deg": np.degrees(prm[3]),
                      "tilt_deg": np.degrees(prm[4]), "focal_px": prm[5]},
           "positions_m": pos, "pixels": players},
          open("analysis/positions.json", "w"), indent=2)

# ------------------------------------------------------------------- drawing
def draw_pitch(ax):
    ax.add_patch(Rectangle((-4, -3), L + 8, WID + 6, color="#1e5bb8", zorder=0))
    ax.add_patch(Rectangle((0, 0), L, WID, fill=False, ec="white", lw=2.2, zorder=2))
    lw = 2
    for x in (LINE_23, L / 2, L - LINE_23):
        ax.plot([x, x], [0, WID], color="white", lw=lw, zorder=2)
    for side in (0, 1):
        sgn = 1 if side == 0 else -1
        x0 = 0 if side == 0 else L
        for py in (POST_FAR, POST_NEAR):
            for r, ls in ((R_D, "-"), (R_DOT, (0, (2.5, 2.5)))):
                a0, a1 = (270, 90) if side == 0 else (90, 270)
                if py == POST_FAR:
                    t = (270, 360) if side == 0 else (180, 270)
                else:
                    t = (0, 90) if side == 0 else (90, 180)
                ax.add_patch(Arc((x0, py), 2 * r, 2 * r, theta1=t[0], theta2=t[1],
                                 color="white", lw=lw, ls=ls, zorder=2))
        for r, ls in ((R_D, "-"), (R_DOT, (0, (2.5, 2.5)))):
            ax.plot([x0 + sgn * r] * 2, [POST_FAR, POST_NEAR], color="white", lw=lw, ls=ls, zorder=2)
        ax.plot(x0 + sgn * PEN_SPOT, YC, "o", color="white", ms=5, zorder=3)
        gx = -GOAL_D if side == 0 else L
        ax.add_patch(Rectangle((gx, POST_FAR), GOAL_D, GOAL_W, fill=True, fc="#e8e8e8",
                               ec="white", lw=2, hatch="///", zorder=3))
    ax.plot(L / 2, YC, "o", color="white", ms=3, zorder=3)

fig, ax = plt.subplots(figsize=(15, 9.6))
draw_pitch(ax)
ax.set_xlim(-6, L + 6); ax.set_ylim(WID + 5, -5)   # invert Y: far sideline at top, like the photo
ax.set_aspect("equal"); ax.axis("off")
ax.text(L / 2, -2.2, "FAR sideline (spectator stand in the photo)", ha="center", color="white", fontsize=10)
ax.text(L / 2, WID + 2.8, "NEAR sideline (camera side)", ha="center", color="white", fontsize=10)

style = {"R": ("#d62828", "white"), "W": ("white", "black")}
for k, (x, y) in pos.items():
    if k == "BALL":
        ax.add_patch(Circle((x, y), 0.7, fc="white", ec="black", lw=1.2, zorder=9))
        ax.annotate("ball (inferred)", (x, y), (x + 4, y + 2.5), color="white", fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="white"), zorder=6)
    elif k == "GK":
        ax.add_patch(Circle((x, y), 1.4, fc="#2ecc71", ec="black", lw=1.5, zorder=6))
        ax.text(x, y, "GK", ha="center", va="center", fontsize=8, weight="bold", zorder=7)
    elif k == "UMP":
        ax.add_patch(Circle((x, y), 1.4, fc="#222222", ec="white", lw=1.5, zorder=6))
        ax.text(x, y, "U", ha="center", va="center", fontsize=8, color="white", weight="bold", zorder=7)
    else:
        fc, tc = style[k[0]]
        ax.add_patch(Circle((x, y), 1.2, fc=fc, ec="black", lw=1.5, zorder=6 + (1 if k[0]=="R" else 0)))
        ax.text(x, y, k, ha="center", va="center", fontsize=7, color=tc, weight="bold", zorder=8)

cam = (L - prm[0], prm[1])
ax.plot(cam[0], min(cam[1], WID + 4.2), marker=(3, 0, 180), ms=14, color="yellow", zorder=6)
ax.text(cam[0], min(cam[1], WID + 4.2) - 1.7, "camera", ha="center", color="yellow", fontsize=9)

ax.text(1, WID + 4.6, "Red = ESP (red shirts)   White = opponents   GK = goalkeeper   U = umpire   "
        "Positions estimated from one photo (approx. ±3 m)", color="white", fontsize=9)
fig.patch.set_facecolor("#1e5bb8")
plt.tight_layout()
fig.savefig("analysis/hockey_topview.png", dpi=110, facecolor=fig.get_facecolor())
print("saved analysis/hockey_topview.png")
