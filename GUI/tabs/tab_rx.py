import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QPushButton
from gui_mpl import MplWidget
from core.scfdm_rx import bits_to_image, get_qam_reference

class RxTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Botón manual por si acaso
        btn = QPushButton("Refrescar Resultados")
        btn.clicked.connect(self.plot_results)
        layout.addWidget(btn)

        self.tabs = QTabWidget()
        
        # Tab Imágenes
        self.tab_img = QWidget()
        l_img = QVBoxLayout(self.tab_img)
        self.mpl_img = MplWidget(height=6)
        l_img.addWidget(self.mpl_img)
        
        # Tab Constelaciones
        self.tab_const = QWidget()
        l_const = QVBoxLayout(self.tab_const)
        self.mpl_const = MplWidget(height=5)
        l_const.addWidget(self.mpl_const)

        self.tabs.addTab(self.tab_img, "Imágenes")
        self.tabs.addTab(self.tab_const, "Constelaciones")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def plot_results(self):
        if "res_ofdm" not in self.state:
            return

        # Desempaquetar
        # res = (bits, bits_raw, const_freq, const_time)
        b_o, b_o_raw, _, c_o = self.state["res_ofdm"]
        b_s, b_s_raw, c_s_freq, c_s = self.state["res_scfdm"]
        
        img_shape = self.state["img_shape"]
        
        # Reconstruir imágenes
        i_orig = bits_to_image(self.state["bits_tx"], img_shape)
        i_o_eq = bits_to_image(b_o, img_shape)
        i_o_raw = bits_to_image(b_o_raw, img_shape)
        i_s_eq = bits_to_image(b_s, img_shape)
        i_s_raw = bits_to_image(b_s_raw, img_shape)

        # --- PLOT IMÁGENES ---
        self.mpl_img.clear()
        
        # Lógica de layout según canal
        ch_type = self.state["channel"]
        if "RAYLEIGH" in ch_type:
            axs = self.mpl_img.fig.subplots(2, 3)
            # Fila 1 OFDM
            axs[0,0].imshow(i_orig); axs[0,0].set_title("Original")
            axs[0,1].imshow(i_o_raw); axs[0,1].set_title("OFDM Sin EQ")
            axs[0,2].imshow(i_o_eq); axs[0,2].set_title("OFDM Final")
            # Fila 2 SC-FDMA
            axs[1,0].axis('off')
            axs[1,1].imshow(i_s_raw); axs[1,1].set_title("SC-FDMA Sin EQ")
            axs[1,2].imshow(i_s_eq); axs[1,2].set_title("SC-FDMA Final")
            for ax in axs.flat: ax.axis('off')
        else:
            axs = self.mpl_img.fig.subplots(1, 3)
            axs[0].imshow(i_orig); axs[0].set_title("Original")
            axs[1].imshow(i_o_eq); axs[1].set_title("RX OFDM")
            axs[2].imshow(i_s_eq); axs[2].set_title("RX SC-FDMA")
            for ax in axs: ax.axis('off')
            
        self.mpl_img.draw()

        # --- PLOT CONSTELACIONES ---
        self.mpl_const.clear()
        axs = self.mpl_const.fig.subplots(1, 3)
        
        M = self.state["M"]
        ref = get_qam_reference(M)
        lim = 3000

        # 1. SC-FDMA Freq
        axs[0].scatter(c_s_freq[:lim].real, c_s_freq[:lim].imag, s=1, c='g', alpha=0.5)
        axs[0].set_title("SC-FDMA Freq (Gauss)")
        axs[0].grid(True)

        # 2. OFDM Final
        axs[1].scatter(c_o[:lim].real, c_o[:lim].imag, s=1, alpha=0.5)
        axs[1].scatter(ref.real, ref.imag, s=15, c='r', marker='x')
        axs[1].set_title("OFDM Final")
        axs[1].grid(True)

        # 3. SC-FDMA Final
        axs[2].scatter(c_s[:lim].real, c_s[:lim].imag, s=1, c='orange', alpha=0.5)
        axs[2].scatter(ref.real, ref.imag, s=15, c='r', marker='x')
        axs[2].set_title("SC-FDMA Final")
        axs[2].grid(True)
        
        self.mpl_const.draw()