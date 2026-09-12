"""
app_ui.py
------------------------------
The full AirChord application with a proper UI shell:
- Start/Stop button to control the webcam pipeline
- Embedded live video feed with landmark overlay + chord diagram
- A running chord history log
- Clean window layout using Tkinter (Python's built-in GUI toolkit)

Requires:
- assets/models/hand_landmarker.task
- assets/audio/G.wav, C.wav, D.wav, Em.wav, A.wav
- pillow (pip install pillow)
"""

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import datetime

from gesture_engine import GestureEngine
from chord_diagrams import draw_chord_diagram


class AirChordApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AirChord")
        self.root.configure(bg="#1e1e1e")

        self.engine = None
        self.cap = None
        self.running = False

        self._build_ui()

    def _build_ui(self):
        # --- Top bar: title ---
        title_label = tk.Label(
            self.root, text="AirChord \U0001F3B8", font=("Segoe UI", 20, "bold"),
            fg="white", bg="#1e1e1e"
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(15, 5))

        # --- Left side: video feed ---
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.grid(row=1, column=0, padx=15, pady=10)

        # --- Right side: controls + chord history ---
        control_frame = tk.Frame(self.root, bg="#1e1e1e")
        control_frame.grid(row=1, column=1, sticky="n", padx=15, pady=10)

        self.start_button = ttk.Button(
            control_frame, text="Start", command=self.start
        )
        self.start_button.pack(fill="x", pady=(0, 5))

        self.stop_button = ttk.Button(
            control_frame, text="Stop", command=self.stop, state="disabled"
        )
        self.stop_button.pack(fill="x", pady=(0, 15))

        history_label = tk.Label(
            control_frame, text="Chord History", font=("Segoe UI", 12, "bold"),
            fg="white", bg="#1e1e1e"
        )
        history_label.pack(anchor="w")

        self.history_listbox = tk.Listbox(
            control_frame, width=28, height=18, bg="#2b2b2b", fg="white",
            font=("Consolas", 10), highlightthickness=0, borderwidth=0
        )
        self.history_listbox.pack(fill="both", expand=True)

        self.status_label = tk.Label(
            self.root, text="Status: Stopped", font=("Segoe UI", 10),
            fg="#aaaaaa", bg="#1e1e1e"
        )
        self.status_label.grid(row=2, column=0, columnspan=2, pady=(0, 10))

    def start(self):
        if self.running:
            return

        self.engine = GestureEngine()
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.status_label.config(text="Status: Could not access webcam")
            return

        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Status: Running")

        self._update_frame()

    def stop(self):
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Stopped")

        if self.cap:
            self.cap.release()
            self.cap = None
        if self.engine:
            self.engine.close()
            self.engine = None

        self.video_label.config(image="")

    def _update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Status: Failed to read webcam")
            self.stop()
            return

        frame = cv2.flip(frame, 1)
        confirmed_chord, hand_detected, chord_just_changed = self.engine.process_frame(frame)

        # Draw the chord diagram overlay (does nothing if chord is None)
        if confirmed_chord:
            draw_chord_diagram(frame, confirmed_chord)

        # Log to chord history only when the chord actually changes
        # (and isn't going back to "no chord")
        if chord_just_changed and confirmed_chord is not None:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.history_listbox.insert(tk.END, f"[{timestamp}]  {confirmed_chord}")
            self.history_listbox.yview(tk.END)  # auto-scroll to latest

        # --- Convert the OpenCV (BGR) frame to something Tkinter can display ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Keep a reference, otherwise Python garbage-collects the image
        # and the video would go blank
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # Schedule the next frame update (~30fps -> ~33ms delay)
        self.root.after(33, self._update_frame)

    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AirChordApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()