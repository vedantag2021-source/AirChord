"""
app_ui.py
------------------------------
The full AirChord application with a responsive UI shell:
- Window sized relative to the actual screen (not a fixed pixel size)
- Video feed scales dynamically to fill its available space, keeping
  aspect ratio (like CSS "object-fit: contain")
- Layout RE-FLOWS at a width breakpoint: side-by-side on wide windows,
  stacked on narrow ones -- conceptually the same idea as a CSS media
  query, implemented manually since Tkinter has no built-in equivalent
- Start/Stop button, live video with landmark overlay + chord diagram,
  and a scrolling chord history log

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
import os
import threading

from gesture_engine import GestureEngine
from chord_diagrams import draw_chord_diagram
from generate_ai_chord_audio import generate_all_chords

# Below this window width, switch from side-by-side to stacked layout.
# This is our "media query breakpoint".
NARROW_BREAKPOINT = 820


class AirChordApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AirChord")
        self.root.configure(bg="#1e1e1e")
        self.root.minsize(480, 400)

        # --- Responsive initial sizing: relative to the actual screen ---
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        window_w = int(screen_w * 0.75)
        window_h = int(screen_h * 0.75)
        pos_x = (screen_w - window_w) // 2
        pos_y = (screen_h - window_h) // 2
        self.root.geometry(f"{window_w}x{window_h}+{pos_x}+{pos_y}")

        self.engine = None
        self.cap = None
        self.running = False
        self.current_layout = None  # "wide" or "narrow" -- tracked to avoid re-laying-out every pixel

        self._build_ui()

        # Re-check layout whenever the window is resized
        self.root.bind("<Configure>", self._on_resize)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Root grid: row 0 = title, row 1 = main content, row 2 = status.
        # Column/row weights make widgets stretch to fill available space
        # instead of staying a fixed size when the window is resized.
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        title_label = tk.Label(
            self.root, text="AirChord \U0001F3B8", font=("Segoe UI", 18, "bold"),
            fg="white", bg="#1e1e1e"
        )
        title_label.grid(row=0, column=0, pady=(12, 4), sticky="ew")

        # main_frame holds video_frame + control_frame. We rearrange
        # THESE two inside main_frame depending on the layout breakpoint,
        # rather than rebuilding the whole UI each time.
        self.main_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        self.video_frame = tk.Frame(self.main_frame, bg="black")
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        self.control_frame = tk.Frame(self.main_frame, bg="#1e1e1e")

        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start)
        self.start_button.pack(fill="x", pady=(0, 5))

        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(fill="x", pady=(0, 15))

        history_label = tk.Label(
            self.control_frame, text="Chord History", font=("Segoe UI", 12, "bold"),
            fg="white", bg="#1e1e1e"
        )
        history_label.pack(anchor="w")

        # A frame + scrollbar for the history list, since it needs to
        # shrink/grow with available space too.
        list_container = tk.Frame(self.control_frame, bg="#1e1e1e")
        list_container.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")

        self.history_listbox = tk.Listbox(
            list_container, bg="#2b2b2b", fg="white", font=("Consolas", 10),
            highlightthickness=0, borderwidth=0, yscrollcommand=scrollbar.set
        )
        self.history_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.history_listbox.yview)

        # --- Settings: AI audio customization ---
        settings_label = tk.Label(
            self.control_frame, text="Sound Settings", font=("Segoe UI", 12, "bold"),
            fg="white", bg="#1e1e1e"
        )
        settings_label.pack(anchor="w", pady=(15, 4))

        tk.Label(self.control_frame, text="Style description:", font=("Segoe UI", 9),
                 fg="#cccccc", bg="#1e1e1e").pack(anchor="w")
        self.style_entry = tk.Entry(self.control_frame, bg="#2b2b2b", fg="white",
                                     insertbackground="white", font=("Segoe UI", 9))
        self.style_entry.insert(0, "warm acoustic, bright")
        self.style_entry.pack(fill="x", pady=(0, 8))

        tk.Label(self.control_frame, text="Duration (seconds):", font=("Segoe UI", 9),
                 fg="#cccccc", bg="#1e1e1e").pack(anchor="w")
        self.duration_scale = tk.Scale(
            self.control_frame, from_=0.5, to=4.0, resolution=0.5, orient="horizontal",
            bg="#1e1e1e", fg="white", highlightthickness=0, troughcolor="#2b2b2b"
        )
        self.duration_scale.set(2.0)
        self.duration_scale.pack(fill="x", pady=(0, 8))

        # API key: pre-fill from environment variable if already set,
        # so the user doesn't have to re-type it every session.
        tk.Label(self.control_frame, text="ElevenLabs API key:", font=("Segoe UI", 9),
                 fg="#cccccc", bg="#1e1e1e").pack(anchor="w")
        self.api_key_entry = tk.Entry(self.control_frame, bg="#2b2b2b", fg="white",
                                       insertbackground="white", font=("Segoe UI", 9), show="*")
        existing_key = os.environ.get("ELEVENLABS_API_KEY", "")
        if existing_key:
            self.api_key_entry.insert(0, existing_key)
        self.api_key_entry.pack(fill="x", pady=(0, 8))

        self.regenerate_button = ttk.Button(
            self.control_frame, text="Regenerate Chord Sounds", command=self._on_regenerate_clicked
        )
        self.regenerate_button.pack(fill="x", pady=(0, 4))

        self.generation_status_label = tk.Label(
            self.control_frame, text="", font=("Segoe UI", 9), fg="#00cc66", bg="#1e1e1e",
            wraplength=200, justify="left"
        )
        self.generation_status_label.pack(anchor="w")

        self.status_label = tk.Label(
            self.root, text="Status: Stopped", font=("Segoe UI", 10),
            fg="#aaaaaa", bg="#1e1e1e"
        )
        self.status_label.grid(row=2, column=0, pady=(0, 8))

        # Apply the initial layout based on current width
        self._apply_layout("wide")

    # ------------------------------------------------------------------
    # Responsive layout switching (our "media query")
    # ------------------------------------------------------------------
    def _on_resize(self, event):
        # Only react to resizes of the ROOT window, not every child widget
        # (Tkinter fires <Configure> for many widgets, so we filter this).
        if event.widget != self.root:
            return

        width = event.width
        desired_layout = "narrow" if width < NARROW_BREAKPOINT else "wide"

        if desired_layout != self.current_layout:
            self._apply_layout(desired_layout)

    def _apply_layout(self, layout):
        """
        Rearranges video_frame and control_frame within main_frame.
        'wide'   -> side-by-side (video left, controls right)
        'narrow' -> stacked (video on top, controls below)
        This is the manual equivalent of a CSS breakpoint reflow.
        """
        self.current_layout = layout

        # Detach both frames first (safe even if not currently placed)
        self.video_frame.grid_forget()
        self.control_frame.grid_forget()

        if layout == "wide":
            self.main_frame.columnconfigure(0, weight=3)
            self.main_frame.columnconfigure(1, weight=1)
            self.main_frame.rowconfigure(0, weight=1)

            self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
            self.control_frame.grid(row=0, column=1, sticky="nsew")
            self.control_frame.config(width=220)
        else:
            self.main_frame.columnconfigure(0, weight=1)
            self.main_frame.columnconfigure(1, weight=0)
            self.main_frame.rowconfigure(0, weight=3)
            self.main_frame.rowconfigure(1, weight=1)

            self.video_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
            self.control_frame.grid(row=1, column=0, sticky="nsew")

    # ------------------------------------------------------------------
    # Webcam / engine control
    # ------------------------------------------------------------------
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

        if confirmed_chord:
            draw_chord_diagram(frame, confirmed_chord)

        if chord_just_changed and confirmed_chord is not None:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.history_listbox.insert(tk.END, f"[{timestamp}]  {confirmed_chord}")
            self.history_listbox.yview(tk.END)

        # --- Scale the frame to fit the video_label's CURRENT size ---
        # (this is what makes the video responsive to window resizing,
        # similar to "object-fit: contain" in CSS -- fills the box
        # without distorting the aspect ratio)
        display_frame = self._resize_to_fit(frame)

        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(33, self._update_frame)

    def _resize_to_fit(self, frame):
        """
        Resizes the frame to fit within the video_label's current
        rendered size, preserving aspect ratio, and pads with black
        bars (letterboxing) so it never stretches or distorts.
        """
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()

        # On the very first frame(s), the widget may not have a real
        # size yet (still 1x1) -- fall back to a sensible default.
        if label_w <= 1 or label_h <= 1:
            label_w, label_h = 640, 480

        frame_h, frame_w = frame.shape[:2]
        scale = min(label_w / frame_w, label_h / frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        # Create a black canvas of the label's exact size and paste the
        # resized frame centered within it (the "letterbox" bars)
        canvas = cv2.copyMakeBorder(
            resized,
            top=(label_h - new_h) // 2,
            bottom=(label_h - new_h) - (label_h - new_h) // 2,
            left=(label_w - new_w) // 2,
            right=(label_w - new_w) - (label_w - new_w) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        return canvas

    def on_close(self):
        self.stop()
        self.root.destroy()

    # ------------------------------------------------------------------
    # AI audio regeneration (Settings panel)
    # ------------------------------------------------------------------
    def _on_regenerate_clicked(self):
        style = self.style_entry.get()
        duration = self.duration_scale.get()
        api_key = self.api_key_entry.get().strip()

        if not api_key:
            self.generation_status_label.config(
                text="Please enter an ElevenLabs API key.", fg="#ff5555"
            )
            return

        # Disable the button and show progress while generating, so the
        # user can't trigger overlapping requests.
        self.regenerate_button.config(state="disabled")
        self.generation_status_label.config(text="Starting generation...", fg="#ffcc00")

        # Run the actual network calls on a background thread.
        # IMPORTANT: Tkinter widgets must only be updated from the main
        # thread, so the worker thread never touches widgets directly --
        # it only calls self.root.after(0, ...) to schedule safe updates
        # back on the main thread.
        thread = threading.Thread(
            target=self._regenerate_worker, args=(style, duration, api_key), daemon=True
        )
        thread.start()

    def _regenerate_worker(self, style, duration, api_key):
        def progress_callback(chord_name, status):
            self.root.after(0, self._update_generation_status, chord_name, status)

        try:
            results = generate_all_chords(style, duration, api_key, progress_callback)
            failed_chords = [name for name, ok in results.items() if not ok]

            if failed_chords:
                message = f"Done with errors: {', '.join(failed_chords)} failed."
                color = "#ff5555"
            else:
                message = "All chords regenerated successfully!"
                color = "#00cc66"

            self.root.after(0, self._on_regeneration_finished, message, color)

        except Exception as e:
            self.root.after(0, self._on_regeneration_finished, f"Error: {e}", "#ff5555")

    def _update_generation_status(self, chord_name, status):
        status_text = {
            "generating": f"Generating '{chord_name}'...",
            "done": f"'{chord_name}' done.",
            "error": f"'{chord_name}' failed.",
        }.get(status, "")
        self.generation_status_label.config(text=status_text, fg="#ffcc00")

    def _on_regeneration_finished(self, message, color):
        self.generation_status_label.config(text=message, fg=color)
        self.regenerate_button.config(state="normal")

        # If the engine is currently running, reload the new sounds
        # immediately so the user doesn't need to Stop/Start.
        if self.engine:
            self.engine.reload_sounds()


def main():
    root = tk.Tk()
    app = AirChordApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()