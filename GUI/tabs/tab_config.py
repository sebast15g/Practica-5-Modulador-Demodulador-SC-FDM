import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QComboBox,
    QSpinBox, QMessageBox, QCheckBox, QTextEdit
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import numpy as np
from PIL import Image

# Core SC-FDMA / OFDM (nuevo core)
from core.scfdm_tx import OFDMConfig, load_rgb_image_to_bits


class ConfigTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state

        # ---------------------------
        # Widgets
        # ---------------------------
        self.lbl_image = QLabel("No hay imagen cargada")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setFixedHeight(420)
        self.lbl_image.setStyleSheet("border: 1px solid gray")

        btn_load_img = QPushButton("Cargar imagen")
        btn_load_img.clicked.connect(self.load_image)

        # Modulación
        self.combo_mod = QComboBox()
        self.combo_mod.addItems(["QPSK", "16-QAM", "64-QAM"])

        # Canal
        self.combo_channel = QComboBox()
        self.combo_channel.addItems([
            "IDEAL",
            "AWGN",
            "RAYLEIGH",
            "RAYLEIGH+AWGN"
        ])

        # SNR
        self.spin_snr = QSpinBox()
        self.spin_snr.setRange(0, 40)
        self.spin_snr.setValue(15)
        self.spin_snr.setSuffix(" dB")

        # Tamaño de imagen
        self.chk_original_size = QCheckBox("Usar tamaño original")
        self.chk_original_size.setChecked(True)
        self.chk_original_size.stateChanged.connect(self.toggle_image_size)

        self.spin_w = QSpinBox()
        self.spin_w.setRange(32, 2000)
        self.spin_w.setValue(256)

        self.spin_h = QSpinBox()
        self.spin_h.setRange(32, 2000)
        self.spin_h.setValue(256)

        self.spin_w.setEnabled(False)
        self.spin_h.setEnabled(False)

        # Parámetros OFDM
        self.spin_bw = QSpinBox()
        self.spin_bw.setRange(1, 50)
        self.spin_bw.setValue(10)
        self.spin_bw.setSuffix(" MHz")

        self.spin_df = QSpinBox()
        self.spin_df.setRange(5, 60)
        self.spin_df.setValue(15)
        self.spin_df.setSuffix(" kHz")

        self.spin_guard = QSpinBox()
        self.spin_guard.setRange(0, 50)
        self.spin_guard.setValue(10)
        self.spin_guard.setSuffix(" %")

        self.spin_cp = QSpinBox()
        self.spin_cp.setRange(1, 50)
        self.spin_cp.setValue(16)
        self.spin_cp.setSuffix(" us")

        self.txt_ofdm_info = QTextEdit()
        self.txt_ofdm_info.setReadOnly(True)
        self.txt_ofdm_info.setFixedHeight(220)

        btn_apply = QPushButton("Cargar parámetros")
        btn_apply.clicked.connect(self.apply_parameters)

        # ---------------------------
        # Layout
        # ---------------------------
        form = QVBoxLayout()
        form.addWidget(QLabel("Modulación"))
        form.addWidget(self.combo_mod)
        form.addWidget(QLabel("Canal"))
        form.addWidget(self.combo_channel)
        form.addWidget(QLabel("SNR"))
        form.addWidget(self.spin_snr)
        form.addStretch()

        left = QVBoxLayout()
        left.addWidget(self.lbl_image)
        left.addWidget(btn_load_img)

        main = QHBoxLayout()
        main.addLayout(left, 3)
        main.addLayout(form, 2)

        form.addWidget(QLabel("Tamaño de imagen"))
        form.addWidget(self.chk_original_size)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("W"))
        size_row.addWidget(self.spin_w)
        size_row.addWidget(QLabel("H"))
        size_row.addWidget(self.spin_h)
        form.addLayout(size_row)

        form.addWidget(QLabel("OFDM – Parámetros"))
        form.addWidget(QLabel("BW total"))
        form.addWidget(self.spin_bw)
        form.addWidget(QLabel("Δf"))
        form.addWidget(self.spin_df)
        form.addWidget(QLabel("Guard fraction"))
        form.addWidget(self.spin_guard)
        form.addWidget(QLabel("CP"))
        form.addWidget(self.spin_cp)

        form.addWidget(QLabel("OFDM – Valores calculados"))
        form.addWidget(self.txt_ofdm_info)

        form.addWidget(btn_apply)

        self.setLayout(main)

        # ---------------------------
        # Estado interno
        # ---------------------------
        self.image_path = None

    # =========================================================
    # Cargar imagen
    # =========================================================
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen",
            "",
            "Imágenes (*.png *.jpg *.bmp)"
        )

        if not path:
            return

        self.image_path = path

        pix = QPixmap(path)
        pix = pix.scaled(
            self.lbl_image.width(),
            self.lbl_image.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.lbl_image.setPixmap(pix)

    # =========================================================
    # Aplicar parámetros
    # =========================================================
    def apply_parameters(self):

        if self.image_path is None:
            QMessageBox.warning(
                self,
                "Falta imagen",
                "Debes cargar una imagen primero."
            )
            return

        # ---------------------------
        # Configuración OFDM
        # ---------------------------
        cfg = OFDMConfig(
            bw_mhz=self.spin_bw.value(),
            delta_f=self.spin_df.value() * 1e3,
            guard_fraction=self.spin_guard.value() / 100,
            cp_time_us=self.spin_cp.value()
        )

        self.txt_ofdm_info.setText(self.format_ofdm_info(cfg))

        # ---------------------------
        # Tamaño imagen
        # ---------------------------
        img = Image.open(self.image_path).convert("RGB")

        if self.chk_original_size.isChecked():
            resize_to = img.size
        else:
            resize_to = (self.spin_w.value(), self.spin_h.value())

        img = img.resize(resize_to, Image.Resampling.NEAREST)
        img_shape = (resize_to[1], resize_to[0], 3)

        # ---------------------------
        # Imagen → bits
        # ---------------------------
        bits = load_rgb_image_to_bits(
            self.image_path,
            resize_to
        )

        # ---------------------------
        # Modulación
        # ---------------------------
        mod_txt = self.combo_mod.currentText()
        M = {"QPSK": 4, "16-QAM": 16, "64-QAM": 64}[mod_txt]

        # ---------------------------
        # Canal y SNR
        # ---------------------------
        channel = self.combo_channel.currentText()
        snr_db = self.spin_snr.value()

        # ---------------------------
        # Guardar estado compartido
        # ---------------------------
        self.state.clear()
        self.state.update({
            "image_path": self.image_path,
            "bits_tx": bits,
            "img_shape": img_shape,
            "M": M,
            "channel": channel,
            "snr_db": snr_db,
            "cfg": cfg
        })

        QMessageBox.information(
            self,
            "Configuración lista",
            "Parámetros cargados correctamente.\n"
            "La simulación OFDM y SC-FDMA puede ejecutarse."
        )

        try:
            self.run_full_simulation() # <--- ESTA ES LA CLAVE
            QMessageBox.information(self, "Éxito", "Simulación completada. Revisa las pestañas TX/RX.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Falló la simulación:\n{str(e)}")


    def run_full_simulation(self):
        """
        Corre toda la cadena: TX -> Canal -> RX y guarda resultados en self.state
        """
        # Recuperar lo que guardaste
        cfg = self.state["cfg"]
        bits_tx = self.state["bits_tx"]
        M = self.state["M"]
        ch_type = self.state["channel"]
        snr = self.state["snr_db"]
        
        # 1. TX
        # Importamos aquí para evitar ciclos si es necesario, o arriba
        from core.scfdm_tx import build_tx
        
        X_ofdm, x_t_ofdm = build_tx(bits_tx, cfg, M, use_dft=False)
        X_scfdm, x_t_scfdm = build_tx(bits_tx, cfg, M, use_dft=True)
        
        # Concatenar para canal
        x_ofdm_full = np.concatenate(x_t_ofdm)
        x_scfdm_full = np.concatenate(x_t_scfdm)

        # 2. CANAL
        from core.scfdm_channel import channel_awgn, channel_rayleigh_critical
        
        h_channel = None
        
        # Lógica de canales según tu combo
        if "RAYLEIGH" in ch_type:
            # Generar h UNA VEZ
            y_ofdm_pre, h, _, _ = channel_rayleigh_critical(x_ofdm_full, cfg.fs)
            # Aplicar mismo h a SC-FDMA
            y_scfdm_pre = np.convolve(x_scfdm_full, h, mode="full")[:len(x_scfdm_full)]
            h_channel = h
            
            if "AWGN" in ch_type:
                y_ofdm = channel_awgn(y_ofdm_pre, snr)
                y_scfdm = channel_awgn(y_scfdm_pre, snr)
            else:
                y_ofdm = y_ofdm_pre
                y_scfdm = y_scfdm_pre
        
        elif ch_type == "AWGN":
            y_ofdm = channel_awgn(x_ofdm_full, snr)
            y_scfdm = channel_awgn(x_scfdm_full, snr)
            
        else: # IDEAL
            y_ofdm = x_ofdm_full
            y_scfdm = x_scfdm_full

        # 3. RX
        from core.scfdm_rx import rx_process
        
        # Detectar si usamos H real (podrías agregar un check en tu UI para esto)
        use_perfect_csi = False # O leer de un checkbox
        
        res_ofdm = rx_process(y_ofdm, cfg, M, False, "RAYLEIGH" if "RAYLEIGH" in ch_type else "AWGN", snr, h_channel, use_perfect_csi)
        res_scfdm = rx_process(y_scfdm, cfg, M, True, "RAYLEIGH" if "RAYLEIGH" in ch_type else "AWGN", snr, h_channel, use_perfect_csi)

        # Guardar TODO en state
        self.state.update({
            "cfg": cfg,
            "M": M,
            "snr": snr,
            "ch_type": ch_type,
            
            # Datos TX
            "x_ofdm": x_ofdm_full, 
            "x_scfdm": x_scfdm_full,
            "X_ofdm": X_ofdm, 
            "X_scfdm": X_scfdm,

            # --- NUEVO: GUARDAR SALIDA DEL CANAL PARA GRAFICAR ---
            "y_ofdm": y_ofdm,    # <--- AGREGAR ESTO
            "y_scfdm": y_scfdm,  # <--- AGREGAR ESTO
            # -----------------------------------------------------

            # Datos RX
            "res_ofdm": res_ofdm, 
            "res_scfdm": res_scfdm,
            "h_channel": h_channel,
        })
        
    # =========================================================
    # Auxiliares
    # =========================================================
    def toggle_image_size(self):
        use_original = self.chk_original_size.isChecked()
        self.spin_w.setEnabled(not use_original)
        self.spin_h.setEnabled(not use_original)

    def format_ofdm_info(self, cfg):
        return (
            f"BW TOTAL        : {cfg.BW_total/1e6:.3f} MHz\n"
            f"BW UTIL         : {cfg.BW_util/1e6:.3f} MHz\n"
            f"Guard Fraction  : {cfg.guard_fraction*100:.1f} %\n"
            f"Δf              : {cfg.delta_f/1e3:.2f} kHz\n"
            f"N_used          : {cfg.N_used}\n"
            f"Nfft            : {cfg.Nfft}\n"
            f"Fs              : {cfg.fs/1e6:.3f} MHz\n"
            f"CP samples      : {cfg.cp_len}\n"
        )

