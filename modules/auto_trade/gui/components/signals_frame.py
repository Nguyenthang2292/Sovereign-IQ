import customtkinter as ctk
from tkinter import ttk
from typing import List, Dict


class SignalsFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self._create_header()
        self._create_table()

        self.refresh_label = ctk.CTkLabel(self, text="Auto-refresh: 30s", font=("Arial", 10), text_color="gray")
        self.refresh_label.pack(pady=5)

    def _create_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 0))

        title = ctk.CTkLabel(header, text="Live Signals", font=("Arial", 16, "bold"))
        title.pack(side="left")

        filters_frame = ctk.CTkFrame(header, fg_color="transparent")
        filters_frame.pack(side="right")

        self.filter_long = ctk.CTkCheckBox(filters_frame, text="LONG", command=self.apply_filters)
        self.filter_long.pack(side="left", padx=5)
        self.filter_long.select()

        self.filter_short = ctk.CTkCheckBox(filters_frame, text="SHORT", command=self.apply_filters)
        self.filter_short.pack(side="left", padx=5)
        self.filter_short.select()

        score_label = ctk.CTkLabel(filters_frame, text="Min Score:")
        score_label.pack(side="left", padx=(10, 5))

        self.min_score = ctk.CTkEntry(filters_frame, width=60)
        self.min_score.insert(0, "0.7")
        self.min_score.pack(side="left")

    def _create_table(self):
        table_frame = ctk.CTkFrame(self)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = ctk.CTkScrollbar(table_frame)
        scrollbar.pack(side="right", fill="y")

        columns = ("Symbol", "Signal", "Score", "Time")
        self.table = ttk.Treeview(
            table_frame, columns=columns, show="headings", yscrollcommand=scrollbar.set, height=10
        )

        self.table.heading("Symbol", text="Symbol")
        self.table.heading("Signal", text="Signal")
        self.table.heading("Score", text="Score")
        self.table.heading("Time", text="Time")

        self.table.column("Symbol", width=100)
        self.table.column("Signal", width=80)
        self.table.column("Score", width=80)
        self.table.column("Time", width=100)

        self.table.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=self.table.yview)

        self._configure_table_tags()

    def _configure_table_tags(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b")
        style.configure("Treeview.Heading", background="#1e1e1e", foreground="white")

    def update_signals(self, signals: List[Dict]):
        for item in self.table.get_children():
            self.table.delete(item)

        for signal in signals:
            tag = signal["signal"].lower()
            self.table.insert(
                "",
                "end",
                values=(signal["symbol"], signal["signal"], f"{signal['score']:.2f}", signal["time"]),
                tags=(tag,),
            )

        self.table.tag_configure("long", foreground="#00ff88")
        self.table.tag_configure("short", foreground="#ff4444")
        self.table.tag_configure("neutral", foreground="#888888")

    def apply_filters(self):
        pass
