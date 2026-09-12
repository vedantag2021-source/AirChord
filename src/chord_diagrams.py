"""
chord_diagrams.py
------------------------------
Draws simple, standard open-chord fretboard diagrams (like you'd see
in any beginner guitar chart) onto an OpenCV frame. Purely visual --
this has no effect on gesture logic, it's just a nice touch showing
the actual finger positions for the currently detected chord.
"""

import cv2

# Each chord is defined per-string (low E, A, D, G, B, high e).
# None = muted string (shown as "x"), 0 = open string (shown as "o"),
# 1-4 = fret number to press.
CHORD_SHAPES = {
    "G":  [3, 2, 0, 0, 0, 3],
    "C":  [None, 3, 2, 0, 1, 0],
    "D":  [None, None, 0, 2, 3, 2],
    "Em": [0, 2, 2, 0, 0, 0],
    "A":  [None, 0, 2, 2, 2, 0],
}

NUM_FRETS_SHOWN = 4


def draw_chord_diagram(frame, chord_name, top_left=(20, 100), width=140, height=170):
    """
    Draws a small fretboard diagram for chord_name at the given position.
    Does nothing if chord_name isn't in CHORD_SHAPES (e.g. None / "No Chord").
    """
    if chord_name not in CHORD_SHAPES:
        return

    shape = CHORD_SHAPES[chord_name]
    x0, y0 = top_left

    # Background panel so the diagram is readable over any video content
    cv2.rectangle(frame, (x0 - 10, y0 - 30), (x0 + width + 10, y0 + height + 10),
                  (30, 30, 30), -1)
    cv2.putText(frame, chord_name, (x0 + width // 2 - 15, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    num_strings = 6
    string_spacing = width / (num_strings - 1)
    fret_spacing = height / NUM_FRETS_SHOWN

    # Draw the 6 vertical string lines
    for i in range(num_strings):
        x = int(x0 + i * string_spacing)
        cv2.line(frame, (x, y0), (x, y0 + height), (200, 200, 200), 2)

    # Draw the horizontal fret lines
    for i in range(NUM_FRETS_SHOWN + 1):
        y = int(y0 + i * fret_spacing)
        thickness = 4 if i == 0 else 1  # thicker top line = the nut
        cv2.line(frame, (x0, y), (x0 + width, y), (200, 200, 200), thickness)

    # Draw open/muted string markers above the diagram, and finger dots on it
    for i, fret in enumerate(shape):
        x = int(x0 + i * string_spacing)

        if fret is None:
            cv2.putText(frame, "x", (x - 6, y0 - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        elif fret == 0:
            cv2.circle(frame, (x, y0 - 45), 7, (0, 255, 0), 2)
        else:
            # Position the dot in the middle of its fret's vertical space
            dot_y = int(y0 + (fret - 0.5) * fret_spacing)
            cv2.circle(frame, (x, dot_y), 9, (0, 255, 255), -1)