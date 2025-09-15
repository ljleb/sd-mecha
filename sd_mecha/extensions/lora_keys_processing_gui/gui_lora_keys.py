import safetensors
import os
import sys
import tkinter as tk
from tkinter import ttk
import subprocess
try:
    from tkinterdnd2 import TkinterDnD
    BaseClass = TkinterDnD.Tk
except ImportError:
    import tkinter as tk
    BaseClass = tk.Tk

# IMPORTANT:
# Requires torch installation in the environment


class LoRAKeySaver(BaseClass):
    def __init__(self):
        super().__init__()
        self.title("LoRA Key Saver")
        self.geometry("1000x700")
        self.minsize(800, 600)  # Lock minimum size

        # Dark theme colors
        self.bg_color = "#2b2b2b"
        self.fg_color = "white"
        self.button_bg = "#404040"
        self.button_fg = "white"

        # Configure main window
        self.configure(bg=self.bg_color)

        # Define larger font
        self.large_font = ("Arial", 12)
        self.button_font = ("Arial", 14, "bold")

        # Configure grid weights for scaling
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=2)
        self.grid_rowconfigure(7, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Row 0: Input label
        self.input_label = tk.Label(self, text="LoRA File Paths", font=self.large_font, bg=self.bg_color, fg=self.fg_color)
        self.input_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10,0))

        # Row 1: Text input for paths
        self.input_text = tk.Text(self, height=5, font=self.large_font, bg=self.button_bg, fg=self.fg_color, insertbackground=self.fg_color)
        self.input_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))
        self.add_placeholder(self.input_text, "Enter file paths here, one per line...")

        # Row 2: Add button
        self.add_btn = tk.Button(self, text="Add to List Below", command=self.add_paths, font=self.button_font, height=2,
                                bg=self.button_bg, fg=self.button_fg, activebackground="#555555", activeforeground=self.button_fg)
        self.add_btn.grid(row=2, column=0, pady=5)

        # Row 3: Label for drag/drop and listbox
        self.drag_label = tk.Label(self, text="Files to Process (Drag .safetensors files here or use text input)", font=self.large_font,
                                  bg=self.bg_color, fg=self.fg_color)
        self.drag_label.grid(row=3, column=0, sticky="w", padx=10, pady=(10,5))

        # Row 4: Listbox for entries
        self.listbox = tk.Listbox(self, selectmode=tk.MULTIPLE, font=self.large_font, bg=self.button_bg, fg=self.fg_color,
                                 selectbackground="#555555", selectforeground=self.fg_color)
        self.listbox.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0,10))

        # Enable drag/drop if available
        if hasattr(self, 'dnd_bind'):
            self.drop_target_register("DND_Files")
            self.dnd_bind('<<Drop>>', self.drop_files)

        # Row 5: Buttons frame
        self.button_frame = tk.Frame(self, bg=self.bg_color)
        self.button_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=10)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)

        self.run_btn = tk.Button(self.button_frame, text="Process LoRAs", command=self.run_processing, font=self.button_font, height=3,
                                bg=self.button_bg, fg=self.button_fg, activebackground="#555555", activeforeground=self.button_fg)
        self.run_btn.grid(row=0, column=0, padx=(0,5), sticky="ew")

        self.clear_btn = tk.Button(self.button_frame, text="Clear List", command=self.clear_list, font=self.button_font, height=3,
                                  bg=self.button_bg, fg=self.button_fg, activebackground="#555555", activeforeground=self.button_fg)
        self.clear_btn.grid(row=0, column=1, padx=(5,0), sticky="ew")

        # Row 6: Summary label
        self.summary_label = tk.Label(self, text="Processing Summary", font=self.large_font, bg=self.bg_color, fg=self.fg_color)
        self.summary_label.grid(row=6, column=0, sticky="w", padx=10, pady=(10,0))

        # Row 7: Summary text
        self.summary_text = tk.Text(self, height=8, font=self.large_font, bg=self.button_bg, fg=self.fg_color, insertbackground=self.fg_color)
        self.summary_text.grid(row=7, column=0, sticky="nsew", padx=10, pady=(0,10))
        self.add_placeholder(self.summary_text, "Processing summary will appear here...")

    def add_placeholder(self, widget, placeholder_text):
        widget.placeholder = placeholder_text
        widget.placeholder_color = '#888888'  # Light gray for placeholder
        widget.default_color = self.fg_color  # White for actual text
        widget.insert("1.0", placeholder_text)
        widget.config(fg=widget.placeholder_color)

        def on_focus_in(event):
            if widget.get("1.0", tk.END).strip() == placeholder_text:
                widget.delete("1.0", tk.END)
                widget.config(fg=widget.default_color)

        def on_focus_out(event):
            if not widget.get("1.0", tk.END).strip():
                widget.insert("1.0", placeholder_text)
                widget.config(fg=widget.placeholder_color)

        widget.bind("<FocusIn>", on_focus_in)
        widget.bind("<FocusOut>", on_focus_out)

    def add_paths(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if text and text != self.input_text.placeholder:
            paths = [p.strip() for p in text.splitlines() if p.strip()]
            for path in paths:
                self.listbox.insert(tk.END, path)
            self.input_text.delete("1.0", tk.END)

    def drop_files(self, event):
        data = event.data
        if data:
            if isinstance(data, list):
                # data is a list of file paths
                files = data
            else:
                # data is a string, split on various separators
                import re
                files = re.split(r'[\n\r]+', data.strip())
                # If no newlines, try spaces
                if len(files) == 1:
                    files = data.strip().split()
            for file_path in files:
                file_path = file_path.strip()
                # Remove file:// or file:/// prefix if present
                if file_path.startswith('file:///'):
                    file_path = file_path[8:]
                elif file_path.startswith('file://'):
                    file_path = file_path[7:]
                # Remove surrounding braces if present (some systems)
                file_path = file_path.strip('{}')
                if file_path:  # Only add non-empty paths
                    self.listbox.insert(tk.END, file_path)

    def run_processing(self):
        items = list(self.listbox.get(0, tk.END))
        self.listbox.delete(0, tk.END)

        if not items:
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert(tk.END, "No files selected for processing.")
            return

        cli_script_path = os.path.join(os.path.dirname(__file__), "cli_lora_keys.py")
        command = ["python", cli_script_path] + items

        try:
            # Determine the Python executable from the current environment
            python_executable = sys.executable
            command = [python_executable, cli_script_path] + items

            # Run the CLI script
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            
            summary_output = result.stdout
            if result.stderr:
                summary_output += "\nErrors:\n" + result.stderr

            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert(tk.END, summary_output)

            # Re-add items that failed to process (if any) - this logic needs to be handled by parsing CLI output
            # For now, we'll just clear the list if successful, or show all if there were errors.
            # A more robust solution would involve the CLI script returning specific error codes or structured output.
            if result.returncode != 0:
                # If there was an error, assume all items might have failed or some were not processed
                # For simplicity, re-adding all items to the listbox if the CLI returned an error
                for item in items:
                    self.listbox.insert(tk.END, item)

        except FileNotFoundError:
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert(tk.END, "Error: Python executable or cli_lora_keys.py not found. Make sure Python is in your PATH and cli_lora_keys.py exists.")
        except Exception as e:
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert(tk.END, f"An unexpected error occurred: {e}")

    def clear_list(self):
        self.listbox.delete(0, tk.END)



if __name__ == "__main__":
    # GUI mode
    app = LoRAKeySaver()
    app.mainloop()