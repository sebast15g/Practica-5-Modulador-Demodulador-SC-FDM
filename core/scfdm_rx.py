import numpy as np
import matplotlib.pyplot as plt
from math import log2
from PIL import Image
from scipy.interpolate import CubicSpline

# ============================================================
# IMPORTS BLINDADOS (Funciona desde GUI y Standalone)
# ============================================================
try:
    # Intento 1: Si corremos desde la GUI (la carpeta 'core' es un paquete visible)
    from core.scfdm_tx import (
        OFDMConfig,
        build_tx,
        load_rgb_image_to_bits,
        active_subcarrier_indices,
        pilot_subcarrier_indices,
        M,
        IMAGE_PATH, 
        IMAGE_SIZE
    )
    from core.scfdm_channel import (
        channel_awgn,
        channel_rayleigh_critical
    )
except ImportError:
    # Intento 2: Si corremos el script directo dentro de la carpeta 'core'
    from scfdm_tx import (
        OFDMConfig,
        build_tx,
        load_rgb_image_to_bits,
        active_subcarrier_indices,
        pilot_subcarrier_indices,
        M,
        IMAGE_PATH, 
        IMAGE_SIZE
    )
    from scfdm_channel import (
        channel_awgn,
        channel_rayleigh_critical
    )

plt.rcParams["figure.autolayout"] = True

# ============================================================
# DEMODULADOR QAM & UTILIDADES
# ============================================================
def qam_demod(symbols, M):
    k = int(log2(M))
    m = int(np.sqrt(M))
    symbols = symbols * np.sqrt((2/3)*(M-1))
    I = np.real(symbols)
    Q = np.imag(symbols)
    I_idx = np.clip(np.round((I + (m-1)) / 2), 0, m-1)
    Q_idx = np.clip(np.round((Q + (m-1)) / 2), 0, m-1)
    ints = (Q_idx * m + I_idx).astype(int)
    bits = ((ints[:, None] & (1 << np.arange(k-1, -1, -1))) > 0).astype(np.uint8)
    return bits.reshape(-1)

def get_qam_reference(M):
    """Genera los puntos ideales de la constelación (puntos rojos)."""
    m = int(np.sqrt(M))
    vals = np.arange(-(m-1), m, 2)
    # Producto cartesiano para generar la grilla
    const = np.array([x + 1j*y for x in vals for y in vals])
    # Normalizamos igual que en el TX
    return const / np.sqrt((2/3)*(M-1))

def bits_to_image(bits, shape):
    """Convierte chorro de bits a array de imagen."""
    # Recortar o rellenar si no cuadra exacto (seguridad)
    expected = np.prod(shape) * 8
    if len(bits) > expected:
        bits = bits[:expected]
    elif len(bits) < expected:
        bits = np.concatenate([bits, np.zeros(expected - len(bits), dtype=np.uint8)])
    
    return np.packbits(bits).reshape(shape)

# ============================================================
# ESTIMACIÓN Y ECUALIZACIÓN
# ============================================================
def get_perfect_channel_response(h_time, cfg):
    return np.fft.fft(h_time, cfg.Nfft)

def estimate_channel_cubic(Y_rx, cfg, sym_idx):
    p_idx = pilot_subcarrier_indices(cfg, sym_idx)
    all_idx = active_subcarrier_indices(cfg)
    
    H_pilots = Y_rx[p_idx] 
    
    mag_pilots = np.abs(H_pilots)
    ang_pilots = np.unwrap(np.angle(H_pilots))
    
    cs_mag = CubicSpline(p_idx, mag_pilots)
    cs_ang = CubicSpline(p_idx, ang_pilots)
    
    H_est = np.zeros(cfg.Nfft, dtype=complex)
    H_est[all_idx] = cs_mag(all_idx) * np.exp(1j * cs_ang(all_idx))
    return H_est

def equalize_mmse(Y_rx, H_est, snr_db):
    snr_lin = 10**(snr_db/10)
    N0 = 1.0 / snr_lin 
    num = np.conj(H_est)
    den = (np.abs(H_est)**2 + N0)
    return Y_rx * num / den

# ============================================================
# RECEPTOR (RETORNA TAMBIÉN BITS SIN EQ)
# ============================================================
def rx_process(y_rx, cfg, M, use_scfdm, channel_mode="AWGN", snr_db=30, h_true_time=None, use_perfect_csi=False):
    
    idx_all = active_subcarrier_indices(cfg)
    idx_pilots = pilot_subcarrier_indices(cfg, 0)
    idx_data = np.setdiff1d(idx_all, idx_pilots)
    
    sym_len = cfg.Nfft + cfg.cp_len
    num_syms = len(y_rx) // sym_len
    
    bits_final_out = [] # Bits finales ()
    bits_raw_out = []   # Bits sin ecualizar ()
    
    const_freq = []     # Constelación en frecuencia
    const_final = []    # Constelación final

    if channel_mode == "RAYLEIGH" and use_perfect_csi and h_true_time is not None:
        H_perfect = get_perfect_channel_response(h_true_time, cfg)
    else:
        H_perfect = None

    for n in range(num_syms):
        y_segment = y_rx[n*sym_len : (n+1)*sym_len]
        y_no_cp = y_segment[cfg.cp_len:]
        Y = np.fft.fft(y_no_cp)
        
        # 1. Extraer datos CRUDOS (sin EQ) para visualización
        # Esto nos mostrará el efecto del canal (OFDM) o la mezcla DFT (SC-FDMA)
        syms_raw = Y[idx_data]
        #if not use_scfdm:
        bits_raw_out.append(qam_demod(syms_raw, M))
        #else:
            #bits_raw_out.append(np.zeros(len(syms_raw) * int(np.log2(M)), dtype=np.uint8))

        # 2. Selección de H
        if channel_mode == "RAYLEIGH":
            H_use = H_perfect if use_perfect_csi else estimate_channel_cubic(Y, cfg, n)
        else:
            H_use = np.ones(cfg.Nfft, dtype=complex)
        
        # 3. Ecualización
        Y_eq = equalize_mmse(Y, H_use, snr_db)
        syms_data_eq = Y_eq[idx_data]
        const_freq.append(syms_data_eq)

        # 4. Despreading (Solo SC-FDMA)
        if use_scfdm:
            syms_final = np.fft.ifft(syms_data_eq) * np.sqrt(len(syms_data_eq))
        else:
            syms_final = syms_data_eq
            
        const_final.append(syms_final)
        bits_final_out.append(qam_demod(syms_final, M))

    return (
        np.concatenate(bits_final_out),
        np.concatenate(bits_raw_out),
        np.concatenate(const_freq),
        np.concatenate(const_final)
    )

# ============================================================
# FUNCIONES DE VISUALIZACIÓN (GUI READY)
# ============================================================

def plot_images_analysis(img_orig, img_ofdm_raw, img_ofdm_eq, img_scfdm_raw, img_scfdm_eq, channel_type):
    """
    Genera la FIGURA 1 con las imágenes comparativas.
    Lógica dinámica según el canal seleccionado.
    """
    fig = plt.figure(figsize=(14, 8))
    
    if channel_type == "AWGN":
        # Layout simple: 1 fila, 3 columnas
        # Muestra: Original | OFDM Final | SC-FDMA Final
        axs = fig.subplots(1, 3)
        list_imgs = [
            (img_orig, "Imagen Original"),
            (img_ofdm_eq, "RX OFDM (Final)"),
            (img_scfdm_eq, "RX SC-FDMA (Final)")
        ]
        
        for ax, (im, tit) in zip(axs, list_imgs):
            ax.imshow(im)
            ax.set_title(tit)
            ax.axis('off')
            
    else:
        # Layout complejo (RAYLEIGH): 2 filas, 3 columnas
        axs = fig.subplots(2, 3)
        
        # --- FILA 1: OFDM ---
        # Original
        axs[0,0].imshow(img_orig)
        axs[0,0].set_title("Referencia (Original)")
        axs[0,0].axis('off')
        
        # OFDM Sin EQ (Efecto del Canal)
        axs[0,1].imshow(img_ofdm_raw)
        axs[0,1].set_title("OFDM - Sin Ecualizar\n(Efecto del Canal)")
        axs[0,1].axis('off')
        
        # OFDM Final
        axs[0,2].imshow(img_ofdm_eq)
        axs[0,2].set_title("OFDM - Ecualizada")
        axs[0,2].axis('off')
        
        # --- FILA 2: SC-FDMA ---
        # Repetimos original o dejamos en blanco (Repetimos para simetría visual)
        axs[1,0].imshow(img_orig)
        axs[1,0].set_title("Referencia")
        axs[1,0].axis('off')
        
        # SC-FDMA Sin EQ (Mezcla DFT)
        axs[1,1].imshow(img_scfdm_raw)
        axs[1,1].set_title("SC-FDMA - Sin Ecualizar\n(Datos Mezclados en Frecuencia)")
        axs[1,1].axis('off')
        
        # SC-FDMA Final
        axs[1,2].imshow(img_scfdm_eq)
        axs[1,2].set_title("SC-FDMA - Ecualizada + IDFT")
        axs[1,2].axis('off')

    plt.suptitle(f"Análisis de Imagen Recuperada - Canal {channel_type}", fontsize=14)


def plot_constellations_analysis(c_ofdm, c_scfdm_freq, c_scfdm_time, M, snr):
    """
    Genera la FIGURA 2 con las constelaciones y PUNTOS ROJOS de referencia.
    """
    ref_points = get_qam_reference(M)
    limit = 4000 # Para no saturar la gráfica
    
    fig = plt.figure(figsize=(15, 5))
    axs = fig.subplots(1, 3)
    
    # Estilo común
    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
    
    # 1. SC-FDMA Frecuencia
    # Aquí NO pintamos los puntos rojos QAM porque esto está en dominio frecuencia
    # (distribución casi gaussiana). No tiene sentido comparar con QAM cuadrada.
    axs[0].scatter(c_scfdm_freq[:limit].real, c_scfdm_freq[:limit].imag, s=2, alpha=0.5, color='green')
    axs[0].set_title("SC-FDMA (Dominio Frecuencia)\nAntes de IDFT")
    
    # 2. OFDM Final
    axs[1].scatter(c_ofdm[:limit].real, c_ofdm[:limit].imag, s=2, alpha=0.5, color='tab:blue', label='RX')
    axs[1].scatter(ref_points.real, ref_points.imag, s=20, c='red', marker='x', label='TX Ideal')
    axs[1].set_title(f"OFDM Final (Eq) - SNR={snr}dB")
    axs[1].legend(loc='upper right', fontsize='small')
    
    # 3. SC-FDMA Final
    axs[2].scatter(c_scfdm_time[:limit].real, c_scfdm_time[:limit].imag, s=2, alpha=0.5, color='tab:orange', label='RX')
    axs[2].scatter(ref_points.real, ref_points.imag, s=20, c='red', marker='x', label='TX Ideal')
    axs[2].set_title(f"SC-FDMA Final (Eq + IDFT) - SNR={snr}dB")
    axs[2].legend(loc='upper right', fontsize='small')

    plt.suptitle("Diagramas de Constelación", fontsize=14)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    
    cfg = OFDMConfig()
    SNR_DB = 10
    CHANNEL_TYPE = "AWGN"  
    USE_PERFECT_CSI = False
    
    print(f"--- RUN: {CHANNEL_TYPE} | PerfectCSI={USE_PERFECT_CSI} | {SNR_DB}dB ---")

    # 1. TX
    bits_orig = load_rgb_image_to_bits(IMAGE_PATH, IMAGE_SIZE)
    img_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    img_orig_arr = np.array(Image.open(IMAGE_PATH).convert("RGB").resize(IMAGE_SIZE))

    _, x_tx_ofdm_blocks = build_tx(bits_orig, cfg, M, use_dft=False)
    x_tx_ofdm = np.concatenate(x_tx_ofdm_blocks)
    
    _, x_tx_scfdm_blocks = build_tx(bits_orig, cfg, M, use_dft=True)
    x_tx_scfdm = np.concatenate(x_tx_scfdm_blocks)

    # 2. CANAL
    h_channel = None
    if CHANNEL_TYPE == "RAYLEIGH":
        y_ofdm_pre, h, _, _ = channel_rayleigh_critical(x_tx_ofdm, cfg.fs)
        y_scfdm_pre = np.convolve(x_tx_scfdm, h, mode="full")[:len(x_tx_scfdm)]
        h_channel = h
        y_ofdm = channel_awgn(y_ofdm_pre, SNR_DB)
        y_scfdm = channel_awgn(y_scfdm_pre, SNR_DB)
    else:
        y_ofdm = channel_awgn(x_tx_ofdm, SNR_DB)
        y_scfdm = channel_awgn(x_tx_scfdm, SNR_DB)

    # 3. RX
    # Nota: Ahora rx_process devuelve 4 cosas
    b_ofdm, b_ofdm_raw, _, c_ofdm_final = rx_process(
        y_ofdm, cfg, M, False, CHANNEL_TYPE, SNR_DB, h_channel, USE_PERFECT_CSI
    )
    
    b_scfdm, b_scfdm_raw, c_scfdm_freq, c_scfdm_final = rx_process(
        y_scfdm, cfg, M, True, CHANNEL_TYPE, SNR_DB, h_channel, USE_PERFECT_CSI
    )

    # 4. RECONSTRUCCIÓN IMÁGENES
    # OFDM
    img_ofdm_eq = bits_to_image(b_ofdm, img_shape)
    img_ofdm_raw = bits_to_image(b_ofdm_raw, img_shape)
    
    # SC-FDMA
    img_scfdm_eq = bits_to_image(b_scfdm, img_shape)
    img_scfdm_raw = bits_to_image(b_scfdm_raw, img_shape)

    # 5. VISUALIZACIÓN (¡Aquí está lo nuevo!)
    plot_images_analysis(
        img_orig_arr, 
        img_ofdm_raw, img_ofdm_eq, 
        img_scfdm_raw, img_scfdm_eq, 
        CHANNEL_TYPE
    )
    
    plot_constellations_analysis(
        c_ofdm_final, 
        c_scfdm_freq, c_scfdm_final, 
        M, SNR_DB
    )

    plt.show()