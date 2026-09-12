"""
chord_diagrams.py
------------------------------
Draws standard open-chord fretboard diagrams onto an OpenCV frame.

Design note: every element (title, open/muted string markers, string
lines, fret lines, and finger dots) is positioned relative to ONE
shared anchor point (grid_x0, grid_y0) -- the top-left corner of the
actual string/fret grid. This is what keeps everything precisely
aligned; the earlier version computed marker positions from a
different, unrelated offset, which is why they drifted.
"""

import cv2

# Each chord is defined per-string (low E, A, D, G, B, high e).
# None = muted string ("x"), 0 = open string ("o"), 1-4 = fret to press.
CHORD_SHAPES = {
    "G":  [3, 2, 0, 0, 0, 3],
    "C":  [None, 3, 2, 0, 1, 0],
    "D":  [None, None, 0, 2, 3, 2],
    "Em": [0, 2, 2, 0, 0, 0],
    "A":  [None, 0, 2, 2, 2, 0],
}

NUM_STRINGS = 6
NUM_FRETS_SHOWN = 4

# Layout constants (all in pixels) -- change these to resize the whole
# diagram consistently, everything else derives from them.
PADDING = 14
TITLE_HEIGHT = 26
MARKER_ROW_HEIGHT = 22
MARKER_RADIUS = 6


def draw_chord_diagram(frame, chord_name, top_left=(20, 80), grid_width=130, grid_height=140):
    """
    Draws a fretboard diagram for chord_name, anchored with its
    top-left panel corner at `top_left`. grid_width/grid_height define
    the size of just the string/fret grid itself (the panel around it
    is slightly larger to fit the title and marker row).
    """
    if chord_name not in CHORD_SHAPES:
        return

    shape = CHORD_SHAPES[chord_name]
    panel_x0, panel_y0 = top_left

    # --- The single shared anchor everything else is built from ---
    grid_x0 = panel_x0 + PADDING
    grid_y0 = panel_y0 + PADDING + TITLE_HEIGHT + MARKER_ROW_HEIGHT

    panel_x1 = panel_x0 + grid_width + 2 * PADDING
    panel_y1 = grid_y0 + grid_height + PADDING

    # --- Background panel (covers title + markers + grid, all in one box) ---
    cv2.rectangle(frame, (panel_x0, panel_y0), (panel_x1, panel_y1), (30, 30, 30), -1)

    # --- Title, centered within the panel width ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(chord_name, font, 0.75, 2)[0]
    title_x = panel_x0 + (panel_x1 - panel_x0 - text_size[0]) // 2
    title_y = panel_y0 + PADDING + int(TITLE_HEIGHT * 0.7)
    cv2.putText(frame, chord_name, (title_x, title_y), font, 0.75, (0, 255, 255), 2)

    # --- Grid geometry, derived from the shared anchor ---
    string_spacing = grid_width / (NUM_STRINGS - 1)
    fret_spacing = grid_height / NUM_FRETS_SHOWN

    string_x_positions = [int(grid_x0 + i * string_spacing) for i in range(NUM_STRINGS)]

    # Vertical string lines
    for x in string_x_positions:
        cv2.line(frame, (x, grid_y0), (x, grid_y0 + grid_height), (200, 200, 200), 2)

    # Horizontal fret lines (top one thicker, representing the nut)
    for i in range(NUM_FRETS_SHOWN + 1):
        y = int(grid_y0 + i * fret_spacing)
        thickness = 4 if i == 0 else 1
        cv2.line(frame, (grid_x0, y), (grid_x0 + grid_width, y), (200, 200, 200), thickness)

    # --- Open/muted markers, vertically centered in the marker row,
    #     directly above the grid's top edge (same x as each string) ---
    marker_y = grid_y0 - MARKER_ROW_HEIGHT // 2

    for x, fret in zip(string_x_positions, shape):
        if fret is None:
            r = MARKER_RADIUS
            cv2.line(frame, (x - r, marker_y - r), (x + r, marker_y + r), (0, 0, 255), 2)
            cv2.line(frame, (x - r, marker_y + r), (x + r, marker_y - r), (0, 0, 255), 2)
        elif fret == 0:
            cv2.circle(frame, (x, marker_y), MARKER_RADIUS + 1, (0, 255, 0), 2)

    # --- Finger dots, centered within their fret's vertical band ---
    for x, fret in zip(string_x_positions, shape):
        if fret and fret > 0:
            dot_y = int(grid_y0 + (fret - 0.5) * fret_spacing)
            cv2.circle(frame, (x, dot_y), MARKER_RADIUS + 3, (0, 255, 255), -1)