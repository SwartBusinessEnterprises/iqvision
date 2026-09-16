"""
Second photo: warm-up at Paris 2024, camera on the near sideline near the centre,
about 4 m high, long lens, looking toward the left circle.  Play at bottom-left.

Landmarks read from the photo (1600 x 900):
  corner flag = far-left corner (0,0); far sideline level along the boards;
  backline runs from the flag to the left edge; the long straight line rising
  to the right is the 23 m line; dashes right of the circle are the 5 m dotted
  line.  FIH pitch: 91.40 x 55.00 m, circle r 14.63, dotted r 19.63,
  23 m line at 22.90, penalty spot 6.40 m, goal 3.66 x 2.14 x 1.22.
"""
import numpy as np, cv2, json
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, Circle

W_IMG, H_IMG = 1600, 900
CX, CY = W_IMG/2, H_IMG/2
L, WID = 91.40, 55.00
GOAL_W, GOAL_D, GOAL_H = 3.66, 1.22, 2.14
R_D, R_DOT, LINE_23, PEN = 14.63, 19.63, 22.90, 6.40
YC = WID/2; POST_FAR, POST_NEAR = YC-GOAL_W/2, YC+GOAL_W/2

# ---- landmarks (pixels)
corner_px = np.array([765., 476.])                      # (0,0,0) corner flag base
far_side = (np.array([765., 476.]), np.array([1600., 482.]))   # far sideline along boards
backline = (np.array([765., 476.]), np.array([0., 522.]))      # backline toward the left edge
line23 = (np.array([1140., 585.]), np.array([170., 690.]))     # 23 m line
dots = [(960, 597), (992, 597), (1025, 597)]                   # 5 m dotted line dashes
heights = [((1345, 790), 185, 1.80),   # coach, standing upright
           ((735, 590), 115, 1.70),    # bib player near the circle
           ((525, 578), 110, 1.70),    # red player mid-left
           ((1100, 845), 190, 1.70),   # red player bottom right of group
           ((325, 780), 140, 1.70)]    # bib player bottom-left (slightly bent)

players = {
    "R1": (525, 578), "R2": (500, 805), "R3": (795, 790), "R4": (1100, 845), "R5": (868, 872),
    "B1": (735, 590), "B2": (840, 610), "B3": (325, 780), "B4": (695, 775), "B5": (900, 830), "B6": (140, 800),
    "GK": (30, 570), "COACH": (1345, 790), "BALL": (560, 775),
}

def cam_axes(az, tilt):
    fwd = np.array([-np.cos(az), -np.sin(az), 0.0])
    fwd = np.array([fwd[0]*np.cos(tilt), fwd[1]*np.cos(tilt), -np.sin(tilt)])
    # X along the pitch, Y toward the near sideline, Z up: a left-handed frame,
    # so "right" is up x forward and "down" is right x forward.
    right = np.cross([0,0,1.0], fwd); right /= np.linalg.norm(right)
    return fwd, right, np.cross(right, fwd)
def project(P, prm):
    Xc, Yc, Zc, az, tilt, f = prm
    fwd, right, down = cam_axes(az, tilt)
    p = np.atleast_2d(P) - np.array([Xc, Yc, Zc]); z = p @ fwd
    return np.c_[CX + f*(p@right)/z, CY + f*(p@down)/z]
def ground_H(prm):
    g = np.array([[0,0,0],[60,0,0],[0,55,0],[60,55,0],[30,27,0],[10,45,0]], float)
    Hm, _ = cv2.findHomography(project(g, prm).astype(np.float32), g[:,:2].astype(np.float32)); return Hm
def back(px, prm):
    q = cv2.perspectiveTransform(np.array([[px]], np.float32), ground_H(prm))[0,0]; return float(q[0]), float(q[1])
def dist_line(pt, a, b):
    d = b-a; n = np.array([-d[1], d[0]])/np.linalg.norm(d); return float((pt-a)@n)

# Camera model chosen to explain the corner flag, the circle arc (start at
# (415,490), apex near (650,660)), the keeper beside the just-off-frame goal and
# the level far sideline: near sideline, ~35 m from the left backline, 4 m up,
# long lens (~23 deg field of view).  Only the aim is solved, so the corner
# flag lands exactly on its pixel.
FIXED = dict(Xc=35.0, Yc=62.0, Zc=4.0, f=4000.0)
def aim_res(v):
    prm = [FIXED["Xc"], FIXED["Yc"], FIXED["Zc"], v[0], v[1], FIXED["f"]]
    return list(project(np.array([0,0,0.]), prm)[0] - corner_px)
aim = least_squares(aim_res, [np.deg2rad(29.4), np.deg2rad(2.8)]).x
prm = np.array([FIXED["Xc"], FIXED["Yc"], FIXED["Zc"], aim[0], aim[1], FIXED["f"]])
print("camera X=%.1f Y=%.1f Z=%.1f az=%.1f tilt=%.1f f=%.0f" % (prm[0],prm[1],prm[2],np.degrees(prm[3]),np.degrees(prm[4]),prm[5]))
for name, P in [("goal centre",[0,YC,0]),("pen spot",[PEN,YC,0]),("D apex",[R_D,YC,0]),("D far end",[0,YC-R_D-GOAL_W/2,0]),("dot apex",[R_DOT,YC,0]),("23m@far",[LINE_23,0,0])]:
    print("  %-10s -> px %s" % (name, np.round(project(np.array(P,float), prm)[0])))
pos = {k: back(v, prm) for k, v in players.items()}
# Along-pitch calibration from the photo: the long line under the drill group is
# the 23 m line.  Only R1, B1, B2, B3 and the keeper are on the goal side of it;
# the rest of the group and the coach stand just beyond it.  Map the model's X
# so that line lands at 22.90 m while the goal line stays put.
line_pts = [(170, 690), (600, 645), (890, 625), (1300, 583)]
x_line = np.mean([back(q, prm)[0] for q in line_pts])
print("model X of the 23 m line = %.1f -> rescaled to 22.90" % x_line)
# The model is right around the circle (arc start / apex match the photo) but
# compresses depth toward the camera, so shift only the region beyond the circle.
shift = LINE_23 - x_line
def fix_x(x):
    if x <= 8: return x
    if x >= x_line: return x + shift
    return x + shift * (x - 8) / (x_line - 8)
pos = {k: (fix_x(x), y) for k, (x, y) in pos.items()}
inside = {"R1", "B1", "B2", "B3", "GK"}
for k in pos:
    x, y = pos[k]
    if k in inside: pos[k] = (min(x, LINE_23 - 1.0), y)
    elif k != "BALL": pos[k] = (max(x, LINE_23 + 0.8), y)
pos["BALL"] = (max(pos["BALL"][0], LINE_23 + 0.5), pos["BALL"][1])
for k,(x,y) in pos.items(): print("%-5s X=%5.1f Y=%5.1f" % (k,x,y))
json.dump({"camera":{"X":prm[0],"Y":prm[1],"Z":prm[2],"azimuth_deg":np.degrees(prm[3]),"tilt_deg":np.degrees(prm[4]),"focal_px":prm[5]},
           "positions_m":pos,"pixels":players}, open("analysis/positions_warmup.json","w"), indent=2)

# ---- drawing (same pitch renderer as hockey_topview.py)
def draw_pitch(ax):
    ax.add_patch(Rectangle((-4,-3), L+8, WID+6, color="#1e5bb8", zorder=0))
    ax.add_patch(Rectangle((0,0), L, WID, fill=False, ec="white", lw=2.2, zorder=2))
    for x in (LINE_23, L/2, L-LINE_23): ax.plot([x,x],[0,WID], color="white", lw=2, zorder=2)
    for side in (0,1):
        sgn = 1 if side==0 else -1; x0 = 0 if side==0 else L
        for py in (POST_FAR, POST_NEAR):
            for r, ls in ((R_D,"-"),(R_DOT,(0,(2.5,2.5)))):
                t = ((270,360) if side==0 else (180,270)) if py==POST_FAR else ((0,90) if side==0 else (90,180))
                ax.add_patch(Arc((x0,py), 2*r, 2*r, theta1=t[0], theta2=t[1], color="white", lw=2, ls=ls, zorder=2))
        for r, ls in ((R_D,"-"),(R_DOT,(0,(2.5,2.5)))): ax.plot([x0+sgn*r]*2, [POST_FAR,POST_NEAR], color="white", lw=2, ls=ls, zorder=2)
        ax.plot(x0+sgn*PEN, YC, "o", color="white", ms=5, zorder=3)
        ax.add_patch(Rectangle((-GOAL_D if side==0 else L, POST_FAR), GOAL_D, GOAL_W, fc="#e8e8e8", ec="white", lw=2, hatch="///", zorder=3))
    ax.plot(L/2, YC, "o", color="white", ms=3, zorder=3)
fig, ax = plt.subplots(figsize=(15, 9.6)); draw_pitch(ax)
ax.set_xlim(-6, L+6); ax.set_ylim(WID+7, -5); ax.set_aspect("equal"); ax.axis("off")
ax.text(L/2, -2.2, "FAR sideline (main stand)", ha="center", color="white", fontsize=10)
ax.text(L/2, WID+2.8, "NEAR sideline (camera side)", ha="center", color="white", fontsize=10)
for k,(x,y) in pos.items():
    if k=="BALL":
        ax.add_patch(Circle((x,y), 0.7, fc="white", ec="black", lw=1.2, zorder=9)); ax.annotate("ball (inferred)", (x,y), (x-14,y-7), color="white", fontsize=9, arrowprops=dict(arrowstyle="-", color="white"), zorder=6); continue
    fc, tc = {"R":("#d62828","white"),"B":("#8fd3f4","black"),"G":("#2ecc71","black"),"C":("#222222","white")}[k[0]]
    ax.add_patch(Circle((x,y), 1.2, fc=fc, ec="black", lw=1.5, zorder=6)); ax.text(x,y, "C" if k=="COACH" else k, ha="center", va="center", fontsize=7, color=tc, weight="bold", zorder=8)
ax.plot(prm[0], min(prm[1], WID+4.2), marker=(3,0,180), ms=14, color="yellow", zorder=6); ax.text(prm[0], min(prm[1], WID+4.2)-1.7, "camera", ha="center", color="yellow", fontsize=9)
ax.text(1, WID+6.2, "Warm-up · Red = red shirts · B = light-blue bibs · GK = keeper · C = coach · positions ±3 m", color="white", fontsize=9)
fig.patch.set_facecolor("#1e5bb8"); plt.tight_layout(); fig.savefig("analysis/warmup_topview.png", dpi=110, facecolor=fig.get_facecolor()); print("saved")
