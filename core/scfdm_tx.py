import numpy as np
import matplotlib.pyplot as plt
from math import log2
from dataclasses import dataclass, field
from PIL import Image

#plt.rcParams["figure.autolayout"] = True

# ============================================================
# PARÁMETROS GENERALES
# ============================================================
M = 4
IMAGE_PATH = "cuenca.png" # Asegúrate que esta imagen exista
IMAGE_SIZE = (936, 702)

# ============================================================
# CONFIGURACIÓN OFDM
# ============================================================
@dataclass
class OFDMConfig:
    bw_mhz: float = 10.0
    delta_f: float = 15e3
    guard_fraction: float = 0.10
    cp_time_us: float = 16.6

    N_used: int = field(init=False)
    Nfft: int = field(init=False)
    fs: float = field(init=False)
    Ts: float = field(init=False)
    cp_len: int = field(init=False)
    BW_total: float = field(init=False)
    BW_util: float = field(init=False)

    def __post_init__(self):
        self.BW_total = self.bw_mhz * 1e6
        self.BW_util  = self.BW_total * (1 - self.guard_fraction)
        
        # Ajuste para que sea par y manejable
        self.N_used = int(self.BW_util // self.delta_f)
        if self.N_used % 2:
            self.N_used -= 1

        Nfft_min = int(np.ceil(self.BW_total / self.delta_f))
        self.Nfft = 1 << int(np.ceil(np.log2(Nfft_min)))

        self.fs = self.Nfft * self.delta_f
        self.Ts = 1 / self.fs
        self.cp_len = int(round(self.cp_time_us * 1e-6 * self.fs))

# ============================================================
# SUBPORTADORAS
# ============================================================
def active_subcarrier_indices(cfg):
    half = cfg.N_used // 2
    # Mapeo estándar DC en el centro (null)
    return np.concatenate([
        np.arange(1, half + 1),
        np.arange(cfg.Nfft - half, cfg.Nfft)
    ])

def pilot_subcarrier_indices(cfg, sym_idx, spacing=3):
    # spacing=3 para mejor estimación en canales selectivos
    idx = active_subcarrier_indices(cfg)
    return idx[0::spacing]

# ============================================================
# UTILIDADES IMAGEN / MOD
# ============================================================
def load_rgb_image_to_bits(path, resize_to):
    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        print(f"Advertencia: No encontré {path}, usando ruido aleatorio.")
        img = Image.fromarray(np.random.randint(0,255, (resize_to[1], resize_to[0], 3), dtype=np.uint8))
        
    img = img.resize(resize_to, Image.Resampling.NEAREST)
    bits = np.unpackbits(np.array(img, dtype=np.uint8))
    return bits

def qam_mod(bits, M):
    k = int(log2(M))
    extra = len(bits) % k
    if extra > 0:
        bits = np.concatenate([bits, np.zeros(k - extra, dtype=np.uint8)])
        
    bits = bits.reshape((-1, k))
    ints = bits.dot(2**np.arange(k-1, -1, -1))
    m = int(np.sqrt(M))
    I = 2*(ints % m) - (m-1)
    Q = 2*(ints // m) - (m-1)
    return (I + 1j*Q) / np.sqrt((2/3)*(M-1))

# ============================================================
# TX CORE
# ============================================================
def build_tx(bits, cfg, M, use_dft=False):
    idx_all = active_subcarrier_indices(cfg)
    idx_pilots = pilot_subcarrier_indices(cfg, 0)
    idx_data_carriers = np.setdiff1d(idx_all, idx_pilots)
    
    N_data_per_sym = len(idx_data_carriers)
    k = int(log2(M))
    bps = N_data_per_sym * k 

    num_sym = int(np.ceil(len(bits) / bps))
    total_bits_needed = num_sym * bps
    if len(bits) < total_bits_needed:
        bits = np.concatenate([bits, np.zeros(total_bits_needed - len(bits), dtype=np.uint8)])
    
    bits_reshaped = bits.reshape(num_sym, bps)

    X_freq_blocks = []
    x_time_blocks = []

    for n in range(num_sym):
        b_sym = bits_reshaped[n, :]
        syms_qam = qam_mod(b_sym, M)

        if use_dft:
            # SC-FDMA: Spread con DFT
            syms_to_map = np.fft.fft(syms_qam) / np.sqrt(len(syms_qam))
        else:
            # OFDM: QAM directo
            syms_to_map = syms_qam

        X = np.zeros(cfg.Nfft, dtype=complex)
        X[idx_pilots] = 1.0 + 0j 
        X[idx_data_carriers] = syms_to_map

        x = np.fft.ifft(X)
        
        cp = x[-cfg.cp_len:]
        x_with_cp = np.concatenate([cp, x])
        
        X_freq_blocks.append(X)
        x_time_blocks.append(x_with_cp)

    return np.array(X_freq_blocks), np.array(x_time_blocks)

# ============================================================
# NUEVAS FUNCIONES DE VISUALIZACIÓN (TX)
# ============================================================

# ============================================================
# NUEVAS FUNCIONES DE VISUALIZACIÓN (TX) - SIN WARNINGS
# ============================================================

def plot_papr_time_comparison(x_ofdm, x_scfdm, cfg):
    samples_to_plot = 4 * (cfg.Nfft + cfg.cp_len)
    p_ofdm = np.abs(x_ofdm[:samples_to_plot])**2
    p_scfdm = np.abs(x_scfdm[:samples_to_plot])**2
    
    p_ofdm_norm = p_ofdm / np.mean(p_ofdm)
    p_scfdm_norm = p_scfdm / np.mean(p_scfdm)
    
    t = np.arange(len(p_ofdm)) * cfg.Ts * 1e6 

    plt.figure(figsize=(10, 5)) # Tamaño un poco más compacto
    plt.plot(t, p_ofdm_norm, label=f"OFDM (Max={np.max(p_ofdm_norm):.2f})", alpha=0.7, lw=1)
    plt.plot(t, p_scfdm_norm, label=f"SC-FDMA (Max={np.max(p_scfdm_norm):.2f})", alpha=0.9, lw=1.2)
    plt.axhline(1, color='black', linestyle='--', label="Promedio")
    
    plt.title("Comparación de Potencia Instantánea (PAPR)")
    plt.ylabel("Potencia Normalizada")
    plt.xlabel("Tiempo (µs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout() # <--- El arreglo manual


def plot_tf_grids(X_ofdm, X_scfdm, cfg, n_syms=20):
    grid_ofdm = np.abs(X_ofdm[:n_syms, :]).T
    grid_scfdm = np.abs(X_scfdm[:n_syms, :]).T
    
    grid_ofdm = np.fft.fftshift(grid_ofdm, axes=0)
    grid_scfdm = np.fft.fftshift(grid_scfdm, axes=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True) # <--- constrained_layout es mejor aquí
    
    vmax = max(np.max(grid_ofdm), np.max(grid_scfdm))
    
    im1 = ax1.imshow(grid_ofdm, aspect='auto', interpolation='nearest', cmap='viridis', vmax=vmax)
    ax1.set_title("Grid TF - OFDM")
    ax1.set_xlabel("Símbolo")
    ax1.set_ylabel("Subportadora")
    
    im2 = ax2.imshow(grid_scfdm, aspect='auto', interpolation='nearest', cmap='viridis', vmax=vmax)
    ax2.set_title("Grid TF - SC-FDMA")
    ax2.set_xlabel("Símbolo")
    ax2.set_yticks([]) 
    
    fig.colorbar(im2, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.1, label="Magnitud")


def plot_spectrum_continuous(x_ofdm, x_scfdm, cfg):
    n_points = min(len(x_ofdm), 4 * cfg.Nfft)
    N_fft_high = 8 * cfg.Nfft 
    
    f_axis = np.fft.fftfreq(N_fft_high, cfg.Ts)
    f_axis = np.fft.fftshift(f_axis) / 1e6 
    
    X_spec_ofdm = np.fft.fft(x_ofdm[:n_points] * np.hanning(n_points), n=N_fft_high)
    X_spec_ofdm = np.fft.fftshift(X_spec_ofdm)
    psd_ofdm = 20 * np.log10(np.abs(X_spec_ofdm) + 1e-12)
    psd_ofdm -= np.max(psd_ofdm)
    
    X_spec_scfdm = np.fft.fft(x_scfdm[:n_points] * np.hanning(n_points), n=N_fft_high)
    X_spec_scfdm = np.fft.fftshift(X_spec_scfdm)
    psd_scfdm = 20 * np.log10(np.abs(X_spec_scfdm) + 1e-12)
    psd_scfdm -= np.max(psd_scfdm)
    
    plt.figure(figsize=(10, 5))
    plt.plot(f_axis, psd_ofdm, label="OFDM", alpha=0.6)
    plt.plot(f_axis, psd_scfdm, label="SC-FDMA", alpha=0.6, ls='--')
    
    bw_half = cfg.bw_mhz / 2
    plt.axvline(bw_half, c='r', ls=':', alpha=0.5)
    plt.axvline(-bw_half, c='r', ls=':', alpha=0.5)
    
    plt.title("Espectro Continuo")
    plt.xlabel("Frecuencia (MHz)")
    plt.ylabel("Magnitud (dB)")
    plt.ylim([-60, 5])
    plt.xlim([-cfg.bw_mhz, cfg.bw_mhz])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout() # <--- El arreglo manual


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    
    print("--- Generando Señales TX ---")
    cfg = OFDMConfig()
    
    # Cargar datos
    bits = load_rgb_image_to_bits(IMAGE_PATH, IMAGE_SIZE)
    
    # Generar OFDM
    X_freq_ofdm, x_time_ofdm_blocks = build_tx(bits, cfg, M, use_dft=False)
    x_ofdm_full = np.concatenate(x_time_ofdm_blocks)
    
    # Generar SC-FDMA
    X_freq_scfdm, x_time_scfdm_blocks = build_tx(bits, cfg, M, use_dft=True)
    x_scfdm_full = np.concatenate(x_time_scfdm_blocks)
    
    print(f"Símbolos generados: {len(X_freq_ofdm)}")
    
    # --- VISUALIZACIONES ---
    
    # 1. PAPR en tiempo continuo (Energía)
    plot_papr_time_comparison(x_ofdm_full, x_scfdm_full, cfg)
    
    # 2. Grid Tiempo-Frecuencia
    plot_tf_grids(X_freq_ofdm, X_freq_scfdm, cfg, n_syms=20)
    
    # 3. Espectro Continuo
    plot_spectrum_continuous(x_ofdm_full, x_scfdm_full, cfg)
    
    plt.show()