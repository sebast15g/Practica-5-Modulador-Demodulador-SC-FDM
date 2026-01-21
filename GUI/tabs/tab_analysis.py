import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QSpinBox, QProgressBar, QScrollArea, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QCoreApplication
from gui_mpl import MplWidget

# --- IMPORTS ROBUSTOS ---
try:
    from core.scfdm_tx import OFDMConfig, build_tx, load_rgb_image_to_bits
    from core.scfdm_channel import channel_awgn
    from core.scfdm_rx import rx_process
except ImportError:
    # Fallback para imports locales si falla
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.scfdm_tx import OFDMConfig, build_tx, load_rgb_image_to_bits
    from core.scfdm_channel import channel_awgn
    from core.scfdm_rx import rx_process

class AnalysisTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # --- PANEL DE CONTROL ---
        control_panel = QHBoxLayout()
        
        control_panel.addWidget(QLabel("Repeticiones (Montecarlo):"))
        self.spin_repeats = QSpinBox()
        self.spin_repeats.setRange(1, 200)
        self.spin_repeats.setValue(10) # 10 es un buen balance velocidad/precisión
        control_panel.addWidget(self.spin_repeats)
        
        self.btn_run = QPushButton("▶ EJECUTAR ANÁLISIS")
        self.btn_run.setStyleSheet("font-weight: bold; background-color: #d1f2eb; padding: 8px;")
        self.btn_run.clicked.connect(self.run_montecarlo_analysis)
        control_panel.addWidget(self.btn_run)
        
        control_panel.addStretch()
        main_layout.addLayout(control_panel)

        # Barra de progreso
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress)

        # --- ÁREA DE SCROLL PARA GRÁFICAS ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        self.layout_plots = QVBoxLayout(content_widget)

        # Widgets de Matplotlib
        self.mpl_ber = MplWidget(height=5)
        self.mpl_papr = MplWidget(height=5)
        self.mpl_bar = MplWidget(height=5)
        
        # Altura mínima para forzar el scroll
        self.mpl_ber.setMinimumHeight(800)
        self.mpl_papr.setMinimumHeight(800)
        self.mpl_bar.setMinimumHeight(800)

        # Títulos y Widgets
        self.add_section_title("1. Curvas BER vs SNR (Waterfalls)", self.layout_plots)
        self.layout_plots.addWidget(self.mpl_ber)
        
        self.add_separator(self.layout_plots)
        
        self.add_section_title("2. CCDF del PAPR", self.layout_plots)
        self.layout_plots.addWidget(self.mpl_papr)
        
        self.add_separator(self.layout_plots)
        
        self.add_section_title("3. Comparativa de Desempeño (Intervalos de Confianza)", self.layout_plots)
        self.layout_plots.addWidget(self.mpl_bar)

        self.layout_plots.addStretch()
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        self.setLayout(main_layout)

    def add_section_title(self, text, layout):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 15px; color: #333;")
        layout.addWidget(lbl)

    def add_separator(self, layout):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

    # =========================================================
    # LÓGICA DE SIMULACIÓN
    # =========================================================
    def run_montecarlo_analysis(self):
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        QCoreApplication.processEvents()

        repeats = self.spin_repeats.value()
        
        # 1. Cargar imagen PEQUEÑA para velocidad (128x128 max)
        img_path = self.state.get("image_path", "core/cuenca.png")
        try:
            analysis_size = (128, 128) # Tamaño reducido para que sea rápido
            bits_tx = load_rgb_image_to_bits(img_path, analysis_size)
        except:
            bits_tx = np.random.randint(0, 2, 10000, dtype=np.uint8)

        cfg = OFDMConfig()
        
        # Rangos
        snr_range = np.arange(0, 26, 2)
        mods = [4, 16, 64]
        
        results_ber = {"OFDM": {}, "SC-FDMA": {}}
        results_papr = {"OFDM": {}, "SC-FDMA": {}}
        
        total_steps = len(mods) * 2 
        current_step = 0

        try:
            for M in mods:
                # --- A. CALCULO PAPR ---
                _, x_ofdm_blks = build_tx(bits_tx, cfg, M, use_dft=False)
                _, x_scfdm_blks = build_tx(bits_tx, cfg, M, use_dft=True)
                
                # Función auxiliar PAPR dB
                def get_papr_db(blocks):
                    vals = []
                    for b in blocks:
                        p = np.abs(b)**2
                        peak = np.max(p)
                        avg = np.mean(p)
                        if avg > 0: vals.append(10*np.log10(peak/avg))
                    return np.array(vals)

                results_papr["OFDM"][M] = self.calc_ccdf(get_papr_db(x_ofdm_blks))
                results_papr["SC-FDMA"][M] = self.calc_ccdf(get_papr_db(x_scfdm_blks))
                
                current_step += 1
                self.progress.setValue(int((current_step / total_steps) * 50))
                QCoreApplication.processEvents()

                # --- B. CALCULO BER ---
                res_o = self.simulate_ber_loop(bits_tx, cfg, M, snr_range, repeats, False)
                results_ber["OFDM"][M] = res_o
                
                res_s = self.simulate_ber_loop(bits_tx, cfg, M, snr_range, repeats, True)
                results_ber["SC-FDMA"][M] = res_s
                
                current_step += 1
                prog = 50 + int(((current_step - len(mods)) / len(mods)) * 50)
                self.progress.setValue(prog)
                QCoreApplication.processEvents()

            # --- PLOTEAR ---
            self.plot_ber(snr_range, mods, results_ber)
            self.plot_papr(mods, results_papr)
            self.plot_bars(snr_range, mods, results_ber)
            
            self.progress.setValue(100)
            QMessageBox.information(self, "Finalizado", "Análisis Montecarlo completado.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error Crítico", str(e))
        
        finally:
            self.btn_run.setEnabled(True)

    # =========================================================
    # HELPERS MATEMÁTICOS
    # =========================================================
    def calc_ccdf(self, papr_vals):
        papr_vals = np.sort(papr_vals)
        N = len(papr_vals)
        ccdf = 1 - np.arange(1, N+1) / N
        return papr_vals, ccdf

    def simulate_ber_loop(self, bits_tx, cfg, M, snr_range, repeats, use_scfdm):
        """Loop interno de simulación."""
        ber_mean = []
        ber_std = []
        
        _, x_blocks = build_tx(bits_tx, cfg, M, use_dft=use_scfdm)
        x_tx = np.concatenate(x_blocks)
        total_bits = len(bits_tx)

        for snr in snr_range:
            ber_run = []
            for _ in range(repeats):
                # Canal
                y = channel_awgn(x_tx, snr)
                
                # RX -> ¡CORRECCIÓN CRÍTICA AQUÍ!
                # rx_process devuelve tupla de 4 elementos.
                # [0]: bits_before (MALO para SC-FDMA)
                # [1]: bits_after (BUENO para SC-FDMA)
                rx_ret = rx_process(y, cfg, M, use_scfdm=use_scfdm)
                
                # Tomamos el índice que son los bits finales recuperados
                bits_rx = rx_ret[0]
                
                # Error Check
                n = min(len(bits_rx), total_bits)
                errs = np.sum(bits_rx[:n] != bits_tx[:n])
                errs += abs(len(bits_rx) - total_bits)
                ber_run.append(errs / total_bits)
            
            ber_mean.append(np.mean(ber_run))
            ber_std.append(np.std(ber_run))
            QCoreApplication.processEvents()
            
        return np.array(ber_mean), np.array(ber_std)

    # =========================================================
    # PLOTTING
    # =========================================================
    def plot_ber(self, snr_range, mods, res_ber):
        self.mpl_ber.clear()
        ax = self.mpl_ber.fig.add_subplot(111)
        colors = {4: 'b', 16: 'g', 64: 'r', 256: 'm'}
        
        for M in mods:
            c = colors.get(M, 'k')
            # OFDM
            ax.semilogy(snr_range, res_ber["OFDM"][M][0], f'{c}-o', label=f'OFDM {M}Q', lw=1.5, markersize=4)
            # SC-FDMA
            ax.semilogy(snr_range, res_ber["SC-FDMA"][M][0], f'{c}--x', label=f'SC-FDMA {M}Q', lw=1.5, markersize=4, alpha=0.7)
        
        ax.set_title("BER vs SNR (Canal AWGN)")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER (Log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        #ax.set_ylim([1e-5, 1])
        self.mpl_ber.draw()

    def plot_papr(self, mods, res_papr):
        self.mpl_papr.clear()
        ax = self.mpl_papr.fig.add_subplot(111)
        colors = {4: 'b', 16: 'g', 64: 'r', 256: 'm'}
        
        for M in mods:
            c = colors.get(M, 'k')
            po, co = res_papr["OFDM"][M]
            ps, cs = res_papr["SC-FDMA"][M]
            ax.semilogy(po, co, f'{c}-', label=f'OFDM {M}Q')
            ax.semilogy(ps, cs, f'{c}--', label=f'SC-FDMA {M}Q')
            
        ax.set_title("CCDF PAPR")
        ax.set_xlabel("PAPR (dB)")
        ax.set_ylabel("Probabilidad")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        self.mpl_papr.draw()

    def plot_bars(self, snr_range, mods, res_ber):
        self.mpl_bar.clear()
        ax = self.mpl_bar.fig.add_subplot(111)
        colors = {4: 'b', 16: 'g', 64: 'r', 256: 'm'}
        
        idx = np.argmin(np.abs(snr_range - 12)) # Target 12dB
        labels = []
        means = []
        stds = []
        bar_colors = []
        
        for M in mods:
            # OFDM
            means.append(res_ber["OFDM"][M][0][idx])
            stds.append(res_ber["OFDM"][M][1][idx])
            labels.append(f"OFDM\n{M}Q")
            bar_colors.append(colors.get(M))
            
            # SC-FDMA
            means.append(res_ber["SC-FDMA"][M][0][idx])
            stds.append(res_ber["SC-FDMA"][M][1][idx])
            labels.append(f"SC\n{M}Q")
            bar_colors.append(colors.get(M))

        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, means, yerr=np.array(stds)*1.96, capsize=5, 
               color=bar_colors, alpha=0.6, edgecolor='black')
        
        # Hatching para distinguir SC-FDMA visualmente
        for i, bar in enumerate(bars):
            if i % 2 == 1: # Es SC-FDMA
                bar.set_hatch('///')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("BER")
        ax.set_title(f"BER a SNR={snr_range[idx]}dB (con Intervalos de Confianza)")
        ax.set_yscale('log')
        ax.grid(True, axis='y', which='both', alpha=0.3)
        self.mpl_bar.draw()