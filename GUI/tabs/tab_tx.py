import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QScrollArea, 
    QLabel, QFrame
)
from PyQt5.QtCore import Qt
from gui_mpl import MplWidget 

class TxTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.init_ui()

    def init_ui(self):
        # Layout principal que contendrá el ScrollArea
        main_layout = QVBoxLayout()
        
        # Botón superior
        self.btn_plot = QPushButton("Actualizar Visualización TX")
        self.btn_plot.setStyleSheet("padding: 5px; font-weight: bold;")
        self.btn_plot.clicked.connect(self.plot_all)
        main_layout.addWidget(self.btn_plot)

        # --- SCROLL AREA ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True) # Importante para que se ajuste a lo ancho
        
        # Widget contenedor dentro del scroll
        content_widget = QWidget()
        self.layout_plots = QVBoxLayout(content_widget)

        # --- WIDGETS DE MATPLOTLIB ---
        # Height=5 pulgadas (aprox), pero forzaremos pixeles abajo
        self.mpl_papr = MplWidget(height=5)
        self.mpl_grid = MplWidget(height=6)
        self.mpl_spec = MplWidget(height=5)

        # --- TRUCO PARA EVITAR EL "APLASTAMIENTO" ---
        # Forzamos una altura mínima en pixeles. 
        # Si la ventana es más chica, saldrá el scroll.
        self.mpl_papr.setMinimumHeight(400)
        self.mpl_grid.setMinimumHeight(500)
        self.mpl_spec.setMinimumHeight(400)

        # Agregamos al layout con etiquetas
        self.add_section_title("1. Comparación PAPR (Potencia Instantánea)", self.layout_plots)
        self.layout_plots.addWidget(self.mpl_papr)
        
        self.add_separator(self.layout_plots)
        
        self.add_section_title("2. Grid Tiempo-Frecuencia", self.layout_plots)
        self.layout_plots.addWidget(self.mpl_grid)
        
        self.add_separator(self.layout_plots)
        
        self.add_section_title("3. Espectro de Potencia", self.layout_plots)
        self.layout_plots.addWidget(self.mpl_spec)

        # Añadir un espacio al final para que no quede pegado
        self.layout_plots.addStretch()

        # Configurar el scroll
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        self.setLayout(main_layout)

    def add_section_title(self, text, layout):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #333; margin-top: 10px;")
        layout.addWidget(lbl)

    def add_separator(self, layout):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

    def plot_all(self):
        if "x_ofdm" not in self.state:
            return

        cfg = self.state["cfg"]
        x_ofdm = self.state["x_ofdm"]
        x_scfdm = self.state["x_scfdm"]
        
        # --- 1. PAPR ---
        self.mpl_papr.clear()
        ax = self.mpl_papr.fig.add_subplot(111)
        
        # Tomamos una muestra representativa (ej. 4 símbolos)
        samples = 4 * (cfg.Nfft + cfg.cp_len)
        p1 = np.abs(x_ofdm[:samples])**2
        p1 /= p1.mean() # Normalizar
        p2 = np.abs(x_scfdm[:samples])**2
        p2 /= p2.mean() # Normalizar
        
        t = np.arange(len(p1)) * cfg.Ts * 1e6 # microsegundos

        ax.plot(t, p1, label=f"OFDM (Pico={p1.max():.1f})", alpha=0.7, linewidth=1)
        ax.plot(t, p2, label=f"SC-FDMA (Pico={p2.max():.1f})", alpha=0.9, linewidth=1)
        ax.axhline(1, c='k', ls='--', alpha=0.5)
        ax.set_xlabel("Tiempo (µs)")
        ax.set_ylabel("Potencia Normalizada")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title("Reducción de PAPR en Dominio del Tiempo")
        
        self.mpl_papr.draw()

        # --- 2. GRID TF ---
        self.mpl_grid.clear()
        ax1, ax2 = self.mpl_grid.fig.subplots(1, 2)
        
        X_o = self.state["X_ofdm"]
        X_s = self.state["X_scfdm"]
        
        # Visualizar primeros 20 símbolos
        n_syms_view = 20
        # Transponer para que el tiempo sea el eje X
        g_o = np.abs(X_o[:n_syms_view]).T
        g_s = np.abs(X_s[:n_syms_view]).T
        
        # Centrar frecuencia (shift)
        g_o = np.fft.fftshift(g_o, axes=0)
        g_s = np.fft.fftshift(g_s, axes=0)
        
        vmax = max(g_o.max(), g_s.max())
        
        im1 = ax1.imshow(g_o, aspect='auto', cmap='viridis', vmax=vmax, origin='lower')
        ax1.set_title("OFDM\n(Energía localizada)")
        ax1.set_xlabel("Símbolo")
        ax1.set_ylabel("Subportadora")
        
        im2 = ax2.imshow(g_s, aspect='auto', cmap='viridis', vmax=vmax, origin='lower')
        ax2.set_title("SC-FDMA\n(Energía dispersa)")
        ax2.set_xlabel("Símbolo")
        ax2.set_yticks([]) # Quitar eje Y repetido
        
        # Barra de color compartida
        self.mpl_grid.fig.colorbar(im2, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.1)
        self.mpl_grid.draw()

        # --- 3. ESPECTRO ---
        self.mpl_spec.clear()
        ax = self.mpl_spec.fig.add_subplot(111)
        
        # FFT de alta resolución sobre la señal en tiempo concatenada
        N_fft = 4 * cfg.Nfft
        # Tomamos un trozo de señal
        chunk_len = min(len(x_ofdm), N_fft)
        
        f = np.fft.fftshift(np.fft.fftfreq(N_fft, cfg.Ts)) / 1e6
        
        # Periodograma simple
        S_o = np.fft.fftshift(np.fft.fft(x_ofdm[:chunk_len], N_fft))
        S_s = np.fft.fftshift(np.fft.fft(x_scfdm[:chunk_len], N_fft))
        
        # dB
        S_o_db = 20*np.log10(np.abs(S_o) + 1e-12)
        S_s_db = 20*np.log10(np.abs(S_s) + 1e-12)
        
        # Normalizar al máximo
        ref = S_o_db.max()
        S_o_db -= ref
        S_s_db -= ref
        
        ax.plot(f, S_o_db, label="OFDM", alpha=0.6, linewidth=1)
        ax.plot(f, S_s_db, label="SC-FDMA", alpha=0.6, ls='--', linewidth=1)
        
        # Marcar ancho de banda
        ax.axvline(cfg.bw_mhz/2, c='r', ls=':', alpha=0.5)
        ax.axvline(-cfg.bw_mhz/2, c='r', ls=':', alpha=0.5)
        
        ax.set_xlim(-cfg.bw_mhz, cfg.bw_mhz)
        ax.set_ylim(-60, 5)
        ax.set_xlabel("Frecuencia (MHz)")
        ax.set_ylabel("Magnitud (dB)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("Espectro de Potencia")
        
        self.mpl_spec.draw()