# Astro Walk

A browser app that turns the reconstructed player positions into a 3D
field-hockey pitch you can walk around in first person.

* `index.html` – open directly in a browser (needs internet for the 3D library).
* `astro-walk.html` – the same page in the form published as a Claude artifact.

Controls: click the astro to drop in, W A S D to move, mouse to look,
Shift to run, keys 1–9 jump to a player, Esc releases the mouse. On a phone,
drag the left half to move and the right half to look. "Load positions"
accepts the `positions.json` written by `analysis/hockey_topview.py`.

## Tactics board

Press T or the "Tactics board" button for a 2D editor: drag players and the
ball, add or delete red and white players, and draw lines, arrows, circles,
freehand strokes and text labels in five colours (solid or dashed). "Back to
3D" places the new layout and the drawings on the astro. Layout and drawings
are remembered in the browser and reset with "Reset layout".
