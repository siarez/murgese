from __future__ import annotations

from PySide6 import QtWidgets, QtCore

from ..device_interface.cdc_link import CdcLink, auto_detect_port, list_serial_ports


class JournalTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._link: CdcLink | None = None

        # Top row: port selection and controls
        self.port_combo = QtWidgets.QComboBox(self)
        self.refresh_btn = QtWidgets.QPushButton("Refresh", self)
        self.auto_btn = QtWidgets.QPushButton("Auto-Detect", self)
        self.connect_btn = QtWidgets.QPushButton("Connect", self)
        self.status_lbl = QtWidgets.QLabel("Disconnected", self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Port:"))
        top.addWidget(self.port_combo, 1)
        top.addWidget(self.refresh_btn)
        top.addWidget(self.auto_btn)
        top.addWidget(self.connect_btn)
        top.addWidget(self.status_lbl)

        # Middle row: type/id inputs and actions
        self.type_edit = QtWidgets.QLineEdit(self)
        self.type_edit.setPlaceholderText("0x21")
        self.id_edit = QtWidgets.QLineEdit(self)
        self.id_edit.setPlaceholderText("0x00")
        self.jrd_btn = QtWidgets.QPushButton("Read (hex)", self)
        self.jrdb_btn = QtWidgets.QPushButton("Read (raw)", self)

        mid = QtWidgets.QHBoxLayout()
        mid.addWidget(QtWidgets.QLabel("Type:"))
        mid.addWidget(self.type_edit)
        mid.addSpacing(10)
        mid.addWidget(QtWidgets.QLabel("ID:"))
        mid.addWidget(self.id_edit)
        mid.addSpacing(20)
        mid.addWidget(self.jrd_btn)
        mid.addWidget(self.jrdb_btn)
        mid.addStretch(1)

        # Output area
        self.output = QtWidgets.QPlainTextEdit(self)
        self.output.setReadOnly(True)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(mid)
        root.addWidget(self.output, 1)

        # Wire signals
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.auto_btn.clicked.connect(self._on_auto)
        self.connect_btn.clicked.connect(self._on_connect)
        self.jrd_btn.clicked.connect(self._on_jrd)
        self.jrdb_btn.clicked.connect(self._on_jrdb)

        # Initial
        self._on_refresh()

    # --------------- Helpers ---------------
    def _set_status(self, text: str):
        self.status_lbl.setText(text)

    def _ensure_link(self) -> bool:
        if self._link is not None:
            return True
        port = self._current_port()
        if not port:
            QtWidgets.QMessageBox.warning(self, "Device", "Please select a serial port.")
            return False
        try:
            self._link = CdcLink(port)
            if not self._link.init():
                self._set_status("Init failed")
            else:
                self._set_status(f"Connected: {port}")
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Connect", f"Failed to open {port}:\n{e}")
            self._link = None
            return False

    def _close_link(self):
        if self._link is not None:
            try:
                self._link.close()
            except Exception:
                pass
            self._link = None
            self._set_status("Disconnected")

    def _current_port(self) -> str:
        idx = self.port_combo.currentIndex()
        if idx < 0:
            return ""
        return self.port_combo.itemData(idx) or ""

    def _parse_type_id(self) -> tuple[int, int] | None:
        t_text = self.type_edit.text().strip() or "0x21"
        i_text = self.id_edit.text().strip() or "0x00"
        try:
            typ = int(t_text, 0)
            _id = int(i_text, 0)
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Journal", "Type/ID must be integers (e.g., 0x21, 0x00)")
            return None
        if not (0 <= typ <= 255 and 0 <= _id <= 255):
            QtWidgets.QMessageBox.warning(self, "Journal", "Type/ID must be 0..255")
            return None
        return typ, _id

    def _populate_ports(self, auto_select: bool = False):
        ports = list_serial_ports()
        sel_index = -1
        self.port_combo.clear()
        for p in ports:
            label = f"{p.get('device','')} — {p.get('product','') or p.get('description','')} ({p.get('manufacturer','')})"
            self.port_combo.addItem(label, p.get('device',''))
        if auto_select:
            dev = auto_detect_port()
            if dev:
                for i in range(self.port_combo.count()):
                    if self.port_combo.itemData(i) == dev:
                        sel_index = i
                        break
        if sel_index >= 0:
            self.port_combo.setCurrentIndex(sel_index)

    # --------------- Slots ---------------
    def _on_refresh(self):
        self._populate_ports(auto_select=False)

    def _on_auto(self):
        self._populate_ports(auto_select=True)

    def _on_connect(self):
        if self._link is None:
            self._ensure_link()
            if self._link is not None:
                self.connect_btn.setText("Disconnect")
        else:
            self._close_link()
            self.connect_btn.setText("Connect")

    def _on_jrd(self):
        if not self._ensure_link():
            return
        ti = self._parse_type_id()
        if not ti:
            return
        typ, _id = ti
        data = self._link.jrd(typ, _id)
        if data is None:
            self.output.setPlainText("ERR JNF (no record)")
            return
        self._display_bytes(data, note="JRD")

    def _on_jrdb(self):
        if not self._ensure_link():
            return
        ti = self._parse_type_id()
        if not ti:
            return
        typ, _id = ti
        data = self._link.jrdb(typ, _id)
        if data is None:
            self.output.setPlainText("ERR JNF (no record)")
            return
        self._display_bytes(data, note="JRDB")

    def _display_bytes(self, data: bytes, note: str = ""):
        # Render as hex, grouped binary with offsets, and ASCII preview
        hex_str = " ".join(f"{b:02X}" for b in data)

        # Binary view: one byte per line with byte index (offset)
        bin_lines = [f"{i:04X}: {b:08b}" for i, b in enumerate(data)]
        bin_block = "\n".join(bin_lines)

        ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
        self.output.setPlainText(
            f"{note}: {len(data)} bytes\n\nHex:\n{hex_str}\n\nBinary (1/line with index):\n{bin_block}\n\nASCII:\n{ascii_str}\n"
        )
