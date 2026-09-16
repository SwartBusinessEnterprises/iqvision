# Astro Walk

A browser app that turns the reconstructed player positions into a 3D
field-hockey pitch you can walk around in first person.

* `index.html` – open directly in a browser (needs internet for the 3D library).
* `astro-walk.html` – the same page in the form published as a Claude artifact.

Controls: click the astro to drop in, W A S D to move, mouse to look,
Shift to run, keys 1–9 jump to a player, Esc releases the mouse. On a phone,
drag the left half to move and the right half to look. "Load positions"
accepts the `positions.json` written by `analysis/hockey_topview.py`.
