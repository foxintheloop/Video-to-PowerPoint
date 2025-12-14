"""Main GUI application using CustomTkinter."""

import threading
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Callable
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from decksnag import __version__
from decksnag.config import Config
from decksnag.capture import ScreenCapture
from decksnag.comparison import ImageComparator
from decksnag.presentation import PresentationManager
from decksnag.exporter import Exporter
from decksnag.utils import setup_logging, format_duration

logger = logging.getLogger("decksnag")

# Set appearance mode and color theme
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"


class MiniModeWindow(ctk.CTkToplevel):
    """Compact floating window shown during capture.

    A small, always-on-top widget that displays recording status,
    slide count, and elapsed time without obstructing the capture area.
    """

    def __init__(
        self,
        parent: ctk.CTk,
        on_stop: Callable[[], None],
        on_expand: Callable[[], None],
    ) -> None:
        """Initialize the mini mode window.

        Args:
            parent: Parent window (main app).
            on_stop: Callback when stop button is clicked.
            on_expand: Callback when expand button is clicked.
        """
        super().__init__(parent)

        self._on_stop = on_stop
        self._on_expand = on_expand
        self._slide_count = 0
        self._start_time = time.time()
        self._is_running = True

        # Window configuration
        self.title("Recording")
        self.geometry("300x90")
        self.resizable(False, False)

        # Always on top
        self.attributes("-topmost", True)

        # Position in top-right corner of screen
        screen_width = self.winfo_screenwidth()
        x_pos = screen_width - 320
        y_pos = 20
        self.geometry(f"300x90+{x_pos}+{y_pos}")

        # Make window draggable
        self._drag_data = {"x": 0, "y": 0}

        # Create UI
        self._create_widgets()

        # Start timer updates
        self._update_display()

        # Bind drag events to the window
        self.bind("<Button-1>", self._start_drag)
        self.bind("<B1-Motion>", self._on_drag)

    def _create_widgets(self) -> None:
        """Create the mini mode UI."""
        # Main container with padding
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=8)

        # Top row: Status info
        info_frame = ctk.CTkFrame(container, fg_color="transparent")
        info_frame.pack(fill="x", pady=(0, 8))

        # Recording indicator (red dot + text)
        rec_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        rec_frame.pack(side="left")

        self.rec_dot = ctk.CTkLabel(
            rec_frame,
            text="●",
            text_color="#FF4444",
            font=ctk.CTkFont(size=16),
        )
        self.rec_dot.pack(side="left", padx=(0, 4))

        self.rec_label = ctk.CTkLabel(
            rec_frame,
            text="REC",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#FF4444",
        )
        self.rec_label.pack(side="left")

        # Separator
        ctk.CTkLabel(info_frame, text="|", text_color="gray").pack(side="left", padx=10)

        # Slide count
        self.slide_label = ctk.CTkLabel(
            info_frame,
            text="0 slides",
            font=ctk.CTkFont(size=13),
        )
        self.slide_label.pack(side="left")

        # Separator
        ctk.CTkLabel(info_frame, text="|", text_color="gray").pack(side="left", padx=10)

        # Elapsed time
        self.time_label = ctk.CTkLabel(
            info_frame,
            text="00:00",
            font=ctk.CTkFont(size=13),
        )
        self.time_label.pack(side="left")

        # Bottom row: Buttons
        btn_frame = ctk.CTkFrame(container, fg_color="transparent")
        btn_frame.pack(fill="x")

        # Stop button
        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="■ Stop",
            command=self._handle_stop,
            width=100,
            height=32,
            fg_color="#DC3545",
            hover_color="#BB2D3B",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.stop_btn.pack(side="left")

        # Expand button
        self.expand_btn = ctk.CTkButton(
            btn_frame,
            text="↗ Expand",
            command=self._handle_expand,
            width=100,
            height=32,
            fg_color="transparent",
            border_width=1,
            font=ctk.CTkFont(size=12),
        )
        self.expand_btn.pack(side="right")

    def _start_drag(self, event) -> None:
        """Record starting position for drag."""
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_drag(self, event) -> None:
        """Handle window dragging."""
        x = self.winfo_x() + (event.x - self._drag_data["x"])
        y = self.winfo_y() + (event.y - self._drag_data["y"])
        self.geometry(f"+{x}+{y}")

    def _handle_stop(self) -> None:
        """Handle stop button click."""
        self._is_running = False
        self._on_stop()

    def _handle_expand(self) -> None:
        """Handle expand button click."""
        self._on_expand()

    def update_slide_count(self, count: int) -> None:
        """Update the slide count display.

        Args:
            count: Number of slides captured.
        """
        self._slide_count = count
        slides_text = "1 slide" if count == 1 else f"{count} slides"
        self.slide_label.configure(text=slides_text)

    def _update_display(self) -> None:
        """Update the elapsed time display."""
        if self._is_running and self.winfo_exists():
            elapsed = time.time() - self._start_time
            minutes, seconds = divmod(int(elapsed), 60)
            hours, minutes = divmod(minutes, 60)

            if hours > 0:
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                time_str = f"{minutes:02d}:{seconds:02d}"

            self.time_label.configure(text=time_str)

            # Blink the recording dot
            current_color = self.rec_dot.cget("text_color")
            new_color = "#FF4444" if current_color == "#881111" else "#881111"
            self.rec_dot.configure(text_color=new_color)

            self.after(500, self._update_display)

    def close(self) -> None:
        """Close the mini mode window."""
        self._is_running = False
        self.destroy()


class RegionOverlay:
    """Visual border showing the capture region using a Canvas with dashed lines.

    Creates a single transparent window with a dashed rectangle around the
    selected capture region. Uses Windows transparency for click-through.
    """

    BORDER_WIDTH = 3
    DASH_PATTERN = (8, 4)  # (dash_length, gap_length)
    COLOR_SELECTED = "#3B8ED0"  # Blue - region selected
    COLOR_RECORDING = "#FF4444"  # Red - recording in progress

    def __init__(self, parent: ctk.CTk) -> None:
        """Initialize the region overlay.

        Args:
            parent: Parent window (main app).
        """
        self._parent = parent
        self._window: Optional[tk.Toplevel] = None
        self._canvas: Optional[tk.Canvas] = None
        self._region: Optional[Tuple[int, int, int, int]] = None
        self._color = self.COLOR_SELECTED
        self._visible = False
        self._rect_id: Optional[int] = None

    def show(self, region: Tuple[int, int, int, int], color: Optional[str] = None) -> None:
        """Show the overlay around the specified region.

        Args:
            region: Tuple of (x1, y1, x2, y2) coordinates.
            color: Optional color override.
        """
        # Clean up existing window
        self.hide()

        self._region = region
        if color:
            self._color = color

        x1, y1, x2, y2 = region
        region_width = x2 - x1
        region_height = y2 - y1

        # Position overlay OUTSIDE the capture region
        # Add padding so the dashed border doesn't overlap the capture area
        padding = self.BORDER_WIDTH + 2  # Border width + small gap

        overlay_x = x1 - padding
        overlay_y = y1 - padding
        overlay_width = region_width + (padding * 2)
        overlay_height = region_height + (padding * 2)

        # Create transparent window
        self._window = tk.Toplevel(self._parent)
        self._window.overrideredirect(True)  # No window decorations
        self._window.attributes("-topmost", True)  # Always on top
        self._window.attributes("-transparentcolor", "#000001")  # Windows transparency
        self._window.geometry(f"{overlay_width}x{overlay_height}+{overlay_x}+{overlay_y}")

        # Canvas with transparent background
        self._canvas = tk.Canvas(
            self._window,
            width=overlay_width,
            height=overlay_height,
            bg="#000001",
            highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True)

        # Draw dashed rectangle at canvas edges (which is outside capture region)
        offset = self.BORDER_WIDTH // 2
        self._rect_id = self._canvas.create_rectangle(
            offset,
            offset,
            overlay_width - offset - 1,
            overlay_height - offset - 1,
            outline=self._color,
            width=self.BORDER_WIDTH,
            dash=self.DASH_PATTERN,
        )

        self._visible = True

    def set_color(self, color: str) -> None:
        """Change the border color.

        Args:
            color: New color (hex string).
        """
        self._color = color
        if self._canvas and self._rect_id is not None:
            try:
                self._canvas.itemconfig(self._rect_id, outline=color)
            except Exception:
                pass  # Canvas might be destroyed

    def hide(self) -> None:
        """Hide and destroy the overlay window."""
        if self._window:
            try:
                self._window.destroy()
            except Exception:
                pass  # Window might already be destroyed
        self._window = None
        self._canvas = None
        self._rect_id = None
        self._visible = False

    def destroy(self) -> None:
        """Alias for hide() - destroy the overlay window."""
        self.hide()

    @property
    def is_visible(self) -> bool:
        """Check if the overlay is currently visible."""
        return self._visible


class DeckSnagApp(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        self.title(f"DeckSnag v{__version__}")
        self.geometry("900x700")
        self.minsize(800, 600)

        # State
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_capture = threading.Event()
        self._is_capturing = False
        self._region: Optional[Tuple[int, int, int, int]] = None
        self._slides: List[Image.Image] = []
        self._start_time: float = 0

        # Mini mode state
        self._mini_mode: Optional[MiniModeWindow] = None
        self._in_mini_mode = False

        # Region overlay state
        self._region_overlay: Optional[RegionOverlay] = None

        # Setup logging
        setup_logging(verbose=False)

        # Create UI
        self._create_widgets()
        self._update_monitor_list()

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header
        self._create_header()

        # Settings panel
        self._create_settings_panel()

        # Preview panel
        self._create_preview_panel()

        # Status bar
        self._create_status_bar()

    def _create_header(self) -> None:
        """Create the header section."""
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        title = ctk.CTkLabel(
            header,
            text="DeckSnag",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title.pack(side="left", padx=10, pady=10)

        # Control buttons on the right
        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right", padx=10)

        self.start_btn = ctk.CTkButton(
            btn_frame,
            text="Start Capture",
            command=self._start_capture,
            width=140,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="Stop Capture",
            command=self._stop_capture_session,
            width=140,
            height=40,
            font=ctk.CTkFont(size=14),
            state="disabled",
            fg_color="gray",
        )
        self.stop_btn.pack(side="left", padx=5)

        # Minimize button (only visible during capture)
        self.minimize_btn = ctk.CTkButton(
            btn_frame,
            text="Minimize",
            command=self._enter_mini_mode,
            width=100,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="transparent",
            border_width=1,
        )
        # Hidden by default - shown only during capture
        self.minimize_btn.pack(side="left", padx=5)
        self.minimize_btn.pack_forget()

    def _create_settings_panel(self) -> None:
        """Create the settings panel."""
        settings = ctk.CTkFrame(self)
        settings.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Row 1: Monitor and Region selection
        row1 = ctk.CTkFrame(settings, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=5)

        # Monitor selection
        ctk.CTkLabel(row1, text="Monitor:").pack(side="left", padx=(0, 5))
        self.monitor_var = ctk.StringVar(value="Primary")
        self.monitor_dropdown = ctk.CTkComboBox(
            row1,
            variable=self.monitor_var,
            values=["Primary"],
            width=200,
            state="readonly",
        )
        self.monitor_dropdown.pack(side="left", padx=(0, 20))

        # Region selection button
        self.region_btn = ctk.CTkButton(
            row1,
            text="Select Region",
            command=self._select_region,
            width=120,
        )
        self.region_btn.pack(side="left", padx=(0, 10))

        self.region_label = ctk.CTkLabel(row1, text="No region selected")
        self.region_label.pack(side="left")

        # Row 2: Interval and Sensitivity
        row2 = ctk.CTkFrame(settings, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=5)

        # Interval
        ctk.CTkLabel(row2, text="Interval:").pack(side="left", padx=(0, 5))
        self.interval_var = ctk.DoubleVar(value=5.0)
        self.interval_slider = ctk.CTkSlider(
            row2,
            from_=1,
            to=30,
            variable=self.interval_var,
            width=150,
            command=self._update_interval_label,
        )
        self.interval_slider.pack(side="left", padx=(0, 5))
        self.interval_label = ctk.CTkLabel(row2, text="5.0s", width=50)
        self.interval_label.pack(side="left", padx=(0, 20))

        # Sensitivity
        ctk.CTkLabel(row2, text="Sensitivity:").pack(side="left", padx=(0, 5))
        self.sensitivity_var = ctk.StringVar(value="Medium")
        self.sensitivity_dropdown = ctk.CTkComboBox(
            row2,
            variable=self.sensitivity_var,
            values=["Low", "Medium", "High"],
            width=120,
            state="readonly",
        )
        self.sensitivity_dropdown.pack(side="left", padx=(0, 20))

        # Comparison method
        ctk.CTkLabel(row2, text="Method:").pack(side="left", padx=(0, 5))
        self.method_var = ctk.StringVar(value="MSE (Fast)")
        self.method_dropdown = ctk.CTkComboBox(
            row2,
            variable=self.method_var,
            values=["MSE (Fast)", "SSIM (Fast)", "CLIP AI (Accurate)"],
            width=150,
            state="readonly",
        )
        self.method_dropdown.pack(side="left", padx=(0, 20))

        # Row 3: Output settings
        row3 = ctk.CTkFrame(settings, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=5)

        # Output format
        ctk.CTkLabel(row3, text="Format:").pack(side="left", padx=(0, 5))
        self.format_var = ctk.StringVar(value="PowerPoint (.pptx)")
        self.format_dropdown = ctk.CTkComboBox(
            row3,
            variable=self.format_var,
            values=["PowerPoint (.pptx)", "PDF (.pdf)", "Images (folder)", "All formats"],
            width=180,
            state="readonly",
        )
        self.format_dropdown.pack(side="left", padx=(0, 20))

        # Output path
        ctk.CTkLabel(row3, text="Output:").pack(side="left", padx=(0, 5))
        self.output_var = ctk.StringVar(value=str(Path.cwd() / "presentation"))
        self.output_entry = ctk.CTkEntry(row3, textvariable=self.output_var, width=300)
        self.output_entry.pack(side="left", padx=(0, 5))

        self.browse_btn = ctk.CTkButton(
            row3,
            text="Browse",
            command=self._browse_output,
            width=80,
        )
        self.browse_btn.pack(side="left")

    def _create_preview_panel(self) -> None:
        """Create the slide preview panel."""
        preview_frame = ctk.CTkFrame(self)
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(preview_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        ctk.CTkLabel(
            header,
            text="Captured Slides",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(side="left")

        self.slide_count_label = ctk.CTkLabel(header, text="0 slides")
        self.slide_count_label.pack(side="right")

        # Scrollable preview area
        self.preview_scroll = ctk.CTkScrollableFrame(
            preview_frame,
            orientation="horizontal",
            height=200,
        )
        self.preview_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Placeholder
        self.preview_placeholder = ctk.CTkLabel(
            self.preview_scroll,
            text="Captured slides will appear here",
            text_color="gray",
        )
        self.preview_placeholder.pack(expand=True, pady=50)

    def _create_status_bar(self) -> None:
        """Create the status bar."""
        self.status_frame = ctk.CTkFrame(self, height=30)
        self.status_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            anchor="w",
        )
        self.status_label.pack(side="left", padx=10, pady=5)

        self.time_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            anchor="e",
        )
        self.time_label.pack(side="right", padx=10, pady=5)

    def _update_monitor_list(self) -> None:
        """Update the monitor dropdown with available monitors."""
        try:
            with ScreenCapture() as capture:
                monitors = capture.list_monitors()

            values = []
            for mon in monitors:
                if mon.id == 0:
                    values.append(f"All Monitors ({mon.width}x{mon.height})")
                else:
                    values.append(f"Monitor {mon.id} ({mon.width}x{mon.height})")

            self.monitor_dropdown.configure(values=values)
            if values:
                self.monitor_var.set(values[1] if len(values) > 1 else values[0])
        except Exception as e:
            logger.error(f"Failed to list monitors: {e}")

    def _update_interval_label(self, value: float) -> None:
        """Update the interval label when slider changes."""
        self.interval_label.configure(text=f"{value:.1f}s")

    def _select_region(self) -> None:
        """Open region selection."""
        # Minimize window during selection
        self.iconify()
        time.sleep(0.3)  # Brief delay to ensure window is minimized

        try:
            with ScreenCapture() as capture:
                self._region = capture.select_region_interactive()

            x1, y1, x2, y2 = self._region
            width = x2 - x1
            height = y2 - y1
            self.region_label.configure(text=f"{width}x{height} at ({x1}, {y1})")
            self._set_status(f"Region selected: {width}x{height}")

            # Show blue overlay around the selected region
            if self._region_overlay is not None:
                self._region_overlay.destroy()
            self._region_overlay = RegionOverlay(self)
            self._region_overlay.show(self._region, color=RegionOverlay.COLOR_SELECTED)
        except Exception as e:
            logger.error(f"Region selection failed: {e}")
            messagebox.showerror("Error", f"Region selection failed: {e}")
        finally:
            self.deiconify()
            self.lift()

    def _browse_output(self) -> None:
        """Open file browser for output path."""
        fmt = self.format_var.get()

        if "Images" in fmt:
            path = filedialog.askdirectory(title="Select Output Directory")
        else:
            if "PDF" in fmt:
                filetypes = [("PDF files", "*.pdf"), ("All files", "*.*")]
                default_ext = ".pdf"
            else:
                filetypes = [("PowerPoint files", "*.pptx"), ("All files", "*.*")]
                default_ext = ".pptx"

            path = filedialog.asksaveasfilename(
                title="Save As",
                filetypes=filetypes,
                defaultextension=default_ext,
            )

        if path:
            self.output_var.set(path)

    def _get_format_code(self) -> str:
        """Convert format dropdown value to format code."""
        fmt = self.format_var.get()
        if "PDF" in fmt:
            return "pdf"
        elif "Images" in fmt:
            return "images"
        elif "All" in fmt:
            return "all"
        return "pptx"

    def _get_comparison_method(self) -> str:
        """Get comparison method from dropdown selection."""
        method_map = {
            "MSE (Fast)": "mse",
            "SSIM (Fast)": "ssim",
            "CLIP AI (Accurate)": "clip",
        }
        return method_map.get(self.method_var.get(), "mse")

    def _get_threshold(self) -> float:
        """Get threshold from sensitivity setting and comparison method."""
        sensitivity = self.sensitivity_var.get().lower()
        method = self._get_comparison_method()
        return ImageComparator.threshold_from_sensitivity(sensitivity, method)

    def _set_status(self, text: str) -> None:
        """Update status bar text."""
        self.status_label.configure(text=text)

    def _start_capture(self) -> None:
        """Start the capture session."""
        if self._is_capturing:
            return

        # Validate region
        if self._region is None:
            messagebox.showwarning(
                "No Region Selected",
                "Please select a capture region first.",
            )
            return

        # Validate output path
        output_path = self.output_var.get().strip()
        if not output_path:
            messagebox.showwarning("No Output Path", "Please specify an output path.")
            return

        # Update UI
        self._is_capturing = True
        self.start_btn.configure(state="disabled", fg_color="gray")
        self.stop_btn.configure(state="normal", fg_color=["#3B8ED0", "#1F6AA5"])
        self.minimize_btn.pack(side="left", padx=5)  # Show minimize button
        self._set_status("Capturing...")

        # Change region overlay to red (recording indicator)
        if self._region_overlay is not None:
            self._region_overlay.set_color(RegionOverlay.COLOR_RECORDING)

        # Clear previous slides
        self._slides = []
        for widget in self.preview_scroll.winfo_children():
            widget.destroy()

        # Start capture in background thread
        self._stop_capture.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # Start timer update
        self._start_time = time.time()
        self._update_timer()

        # Enter mini mode after a brief delay
        self.after(100, self._enter_mini_mode)

    def _stop_capture_session(self) -> None:
        """Stop the capture session."""
        self._stop_capture.set()

    def _capture_loop(self) -> None:
        """Background capture loop."""
        try:
            config = Config(
                output_path=Path(self.output_var.get()),
                output_format=self._get_format_code(),
                interval=self.interval_var.get(),
                threshold=self._get_threshold(),
                region=self._region,
            )

            # Get comparison method
            method = self._get_comparison_method()

            with ScreenCapture() as capture:
                comparator = ImageComparator(threshold=config.threshold, method=method)
                presentation = PresentationManager()
                exporter = Exporter()

                presentation.create(config.output_path)

                # First capture
                previous_image = capture.capture_region(config.region)
                presentation.add_slide(previous_image)
                exporter.add_slide(previous_image)
                self._add_slide_preview(previous_image)

                while not self._stop_capture.is_set():
                    self._stop_capture.wait(timeout=config.interval)

                    if self._stop_capture.is_set():
                        break

                    current_image = capture.capture_region(config.region)

                    if comparator.is_different(previous_image, current_image):
                        presentation.add_slide(current_image)
                        exporter.add_slide(current_image)
                        self._add_slide_preview(current_image)
                        previous_image = current_image

                # Save results
                self._save_results(presentation, exporter, config)

        except Exception as e:
            logger.error(f"Capture error: {e}")
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Capture failed: {error_msg}"))

        finally:
            self.after(0, self._capture_finished)

    def _add_slide_preview(self, image: Image.Image) -> None:
        """Add a slide preview thumbnail."""
        self._slides.append(image.copy())

        # Create thumbnail
        thumbnail = image.copy()
        thumbnail.thumbnail((200, 150), Image.Resampling.LANCZOS)

        def add_thumbnail() -> None:
            # Remove placeholder if present
            if self.preview_placeholder.winfo_exists():
                self.preview_placeholder.destroy()

            # Create frame for thumbnail
            frame = ctk.CTkFrame(self.preview_scroll)
            frame.pack(side="left", padx=5, pady=5)

            # Convert to PhotoImage
            photo = ctk.CTkImage(light_image=thumbnail, dark_image=thumbnail, size=thumbnail.size)

            label = ctk.CTkLabel(frame, image=photo, text="")
            label.image = photo  # Keep reference
            label.pack(padx=2, pady=2)

            # Slide number
            num_label = ctk.CTkLabel(
                frame,
                text=f"Slide {len(self._slides)}",
                font=ctk.CTkFont(size=11),
            )
            num_label.pack(pady=(0, 2))

            # Update count
            self.slide_count_label.configure(text=f"{len(self._slides)} slides")

            # Update mini mode if active
            self._update_mini_mode_slide_count()

        self.after(0, add_thumbnail)

    def _save_results(
        self, presentation: PresentationManager, exporter: Exporter, config: Config
    ) -> None:
        """Save the captured results."""
        try:
            if config.output_format == "all":
                results = exporter.export_all(config.output_path)
                msg = "Exported to:\n"
                for fmt, path in results.items():
                    if isinstance(path, list):
                        msg += f"  {fmt}: {path[0].parent}/\n"
                    else:
                        msg += f"  {fmt}: {path}\n"
            elif config.output_format == "images":
                images_dir = config.output_path.parent / f"{config.output_path.stem}_images"
                paths = exporter.export_images(images_dir)
                msg = f"Saved {len(paths)} images to:\n{images_dir}"
            elif config.output_format == "pdf":
                path = exporter.export_pdf(config.output_path)
                msg = f"Saved to:\n{path}"
            else:
                path = presentation.save()
                msg = f"Saved to:\n{path}"

            self.after(
                0, lambda: messagebox.showinfo("Capture Complete", msg)
            )

        except Exception as e:
            logger.error(f"Save error: {e}")
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to save: {error_msg}"))

    def _capture_finished(self) -> None:
        """Called when capture is complete."""
        self._is_capturing = False

        # Exit mini mode if active
        self._exit_mini_mode()

        # Hide region overlay
        if self._region_overlay is not None:
            self._region_overlay.destroy()
            self._region_overlay = None

        # Update UI
        self.start_btn.configure(state="normal", fg_color=["#3B8ED0", "#1F6AA5"])
        self.stop_btn.configure(state="disabled", fg_color="gray")
        self.minimize_btn.pack_forget()  # Hide minimize button
        self._set_status(f"Capture complete - {len(self._slides)} slides")

    def _update_timer(self) -> None:
        """Update the elapsed time display."""
        if self._is_capturing:
            elapsed = time.time() - self._start_time
            self.time_label.configure(text=f"Elapsed: {format_duration(elapsed)}")
            self.after(1000, self._update_timer)
        else:
            self.time_label.configure(text="")

    def _enter_mini_mode(self) -> None:
        """Enter mini mode - hide main window, show compact widget."""
        if self._in_mini_mode:
            return

        self._in_mini_mode = True

        # Create mini mode window
        self._mini_mode = MiniModeWindow(
            parent=self,
            on_stop=self._stop_capture_session,
            on_expand=self._expand_from_mini,
        )

        # Update mini mode with current slide count
        self._mini_mode.update_slide_count(len(self._slides))

        # Hide main window
        self.withdraw()

    def _exit_mini_mode(self) -> None:
        """Exit mini mode - close widget, show main window."""
        if not self._in_mini_mode:
            return

        self._in_mini_mode = False

        # Close mini mode window if it exists
        if self._mini_mode is not None:
            try:
                self._mini_mode.close()
            except Exception:
                pass  # Window might already be destroyed
            self._mini_mode = None

        # Show main window
        self.deiconify()
        self.lift()
        self.focus_force()

    def _expand_from_mini(self) -> None:
        """Expand from mini mode to full window while continuing capture."""
        self._in_mini_mode = False

        # Close mini mode window
        if self._mini_mode is not None:
            try:
                self._mini_mode.close()
            except Exception:
                pass
            self._mini_mode = None

        # Show main window (capture continues in background)
        self.deiconify()
        self.lift()
        self.focus_force()

    def _update_mini_mode_slide_count(self) -> None:
        """Update the slide count in mini mode if active."""
        if self._in_mini_mode and self._mini_mode is not None:
            try:
                self._mini_mode.update_slide_count(len(self._slides))
            except Exception:
                pass  # Window might be closing


def main() -> None:
    """Launch the GUI application."""
    app = DeckSnagApp()
    app.mainloop()


if __name__ == "__main__":
    main()
