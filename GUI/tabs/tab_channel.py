import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from gui_mpl import MplWidget

class ChannelTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Panel superior
        top_panel = QVBoxLayout()
        self.lbl_info = QLabel("Estado del Canal: Esperando simulación...")
        self.lbl_info.setStyleSheet("font-size: 14px; font-weight: bold; color: #444;")
        
        btn_refresh = QPushButton("Actualizar Gráficas de Canal")
        btn_refresh.setStyleSheet("padding: 6px; font-weight: bold;")
        btn_refresh.clicked.connect(self.plot_channel)
        
        top_panel.addWidget(self.lbl_info)
        top_panel.addWidget(btn_refresh)
        layout.addLayout(top_panel)

        # Widget de gráficos (Altura mayor)
        self.mpl = MplWidget(height=10) 
        layout.addWidget(self.mpl)
        
        self.setLayout(layout)

    def showEvent(self, event):
        self.plot_channel()

    def plot_channel(self):
        # 1. Validaciones
        if "y_ofdm" not in self.state:
            self.lbl_info.setText(" Primero ejecuta la simulación en la pestaña 'Configuración'.")
            return

        # Recuperar datos
        cfg = self.state["cfg"]
        x_tx = self.state["x_ofdm"]
        y_rx = self.state["y_ofdm"]
        h = self.state.get("h_channel", None)
        ch_type = self.state.get("ch_type", "IDEAL")
        snr = self.state.get("snr_db", 0) # Ojo con el nombre de la clave snr/snr_db

        self.lbl_info.setText(f"Visualizando Canal: {ch_type} | SNR: {snr} dB")

        # 2. Configurar Figura (2 Filas simples)
        self.mpl.clear()
        fig = self.mpl.fig
        # Layout: 2 filas, 1 columna
        axs = fig.subplots(2, 1)
        
        # --- GRÁFICA 1: TIEMPO (COMPARATIVA TX vs RX) ---
        ax_time = axs[0]
        
        # Zoom a los primeros 200 µs (o menos si la señal es corta)
        n_samples_zoom = int(200e-6 * cfg.fs)
        n_samples_zoom = min(n_samples_zoom, len(x_tx))
        
        t_us = np.arange(n_samples_zoom) * cfg.Ts * 1e6
        
        ax_time.plot(t_us, np.real(x_tx[:n_samples_zoom]), label="TX (Entrada)", linewidth=1.5)
        ax_time.plot(t_us, np.real(y_rx[:n_samples_zoom]), label="RX (Salida con Ruido/Fading)", alpha=0.7, linewidth=1.2)
        
        ax_time.set_title("Dominio del Tiempo: Efecto del Canal (Zoom inicial)")
        ax_time.set_ylabel("Amplitud")
        ax_time.legend(loc="upper right")
        ax_time.grid(True, alpha=0.3)
        # Quitamos etiqueta X de arriba para limpiar
        ax_time.set_xticklabels([])


        # --- GRÁFICA 2: RESPUESTA EN FRECUENCIA (Solo si es Rayleigh) ---
        ax_freq = axs[1]
        
        if "RAYLEIGH" in ch_type and h is not None:
            # FFT de h para ver los "huecos"
            N_fft_h = 2048
            H = np.fft.fftshift(np.fft.fft(h, N_fft_h))
            H_db = 20*np.log10(np.abs(H) + 1e-12)
            H_db -= np.max(H_db) # Normalizar a 0 dB para ver la atenuación relativa
            
            f_axis = np.linspace(-cfg.fs/2, cfg.fs/2, N_fft_h) / 1e6 # MHz
            
            ax_freq.plot(f_axis, H_db, color='tab:red', linewidth=1.5)
            ax_freq.fill_between(f_axis, H_db, -100, color='tab:red', alpha=0.1) # Relleno coqueto
            
            ax_freq.set_title("Respuesta en Frecuencia del Canal (|H(f)|)")
            ax_freq.set_xlabel("Frecuencia (MHz)")
            ax_freq.set_ylabel("Magnitud (dB)")
            ax_freq.set_ylim([-35, 2])
            ax_freq.grid(True, alpha=0.3)
            
            # Añadir una línea de umbral visual
            ax_freq.axhline(-10, color='k', linestyle='--', alpha=0.3, label="Desvanecimiento profundo")
            ax_freq.legend(loc="lower right")

        else:
            # CASO AWGN o IDEAL
            ax_freq.text(0.5, 0.5, "Canal Plano (Sin Desvanecimiento en Frecuencia)\nSolo se aplica ruido blanco (AWGN).", 
                        ha='center', va='center', fontsize=11, color='gray',
                        bbox=dict(boxstyle="round", fc="0.95"))
            ax_freq.set_title("Respuesta en Frecuencia")
            ax_freq.axis('off')

        self.mpl.fig.tight_layout()
        self.mpl.draw()