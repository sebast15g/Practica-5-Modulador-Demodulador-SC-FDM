# ============================================================
# SC-FDMA vs OFDM – ARCHIVO DE ANÁLISIS FINAL
# Compatible con TX / RX / CANAL actuales
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scfdm_tx import (
    OFDMConfig,
    build_tx,
    load_rgb_image_to_bits
)

from scfdm_rx import rx_process
from scfdm_channel import channel_awgn

plt.rcParams["figure.autolayout"] = True


# ============================================================
# PAPR
# ============================================================

def compute_papr(x):
    p = np.abs(x)**2
    return np.max(p) / np.mean(p)


def papr_ccdf(x, num_blocks=2000):
    """
    CCDF del PAPR en dB
    """
    papr_vals = []
    L = len(x) // num_blocks

    for i in range(num_blocks):
        blk = x[i*L:(i+1)*L]
        papr_vals.append(10 * np.log10(compute_papr(blk)))

    papr_vals = np.array(papr_vals)
    papr_axis = np.linspace(papr_vals.min(), papr_vals.max(), 200)
    ccdf = np.array([np.mean(papr_vals > p) for p in papr_axis])

    return papr_axis, ccdf


# ============================================================
# BER vs SNR – AWGN
# ============================================================

def ber_vs_snr_awgn(
    bits_tx,
    cfg,
    M,
    snr_db_range,
    repeats=1,
    use_scfdm=False
):
    ber_out = []

    # TX fijo (correcto para Monte Carlo)
    _, x_blocks = build_tx(bits_tx, cfg, M, use_dft=use_scfdm)
    x_tx = np.concatenate(x_blocks)

    for snr_db in snr_db_range:

        ber_rep = []

        for _ in range(repeats):

            # Canal
            y = channel_awgn(x_tx, snr_db)

            # RX
            bits_rx, _, _, _ = rx_process(
                y, cfg, M, use_scfdm=use_scfdm
            )

            bits_rx = bits_rx[:len(bits_tx)]
            ber_rep.append(np.mean(bits_rx != bits_tx))

        ber_out.append(np.mean(ber_rep))

    return np.array(ber_out)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Parámetros
    # --------------------------------------------------------
    snr_db_range = np.arange(0, 25, 1)
    mods = [4, 16, 64]     # QPSK, 16QAM, 64QAM
    repeats = 10           # subir a 10–50 si tienes tiempo

    IMAGE_PATH = "cuenca.png"
    IMAGE_SIZE = (256, 256)

    # --------------------------------------------------------
    # bits TX
    # --------------------------------------------------------
    bits_tx = load_rgb_image_to_bits(IMAGE_PATH, IMAGE_SIZE)

    # --------------------------------------------------------
    # Configuración OFDM
    # --------------------------------------------------------
    cfg = OFDMConfig()

    ber_results = {"OFDM": {}, "SC-FDMA": {}}
    papr_results = {"OFDM": {}, "SC-FDMA": {}}

    # --------------------------------------------------------
    # Simulaciones
    # --------------------------------------------------------
    for M in mods:
        print(f"Simulando {M}-QAM...")

        # ===== BER =====
        ber_results["OFDM"][M] = ber_vs_snr_awgn(
            bits_tx, cfg, M,
            snr_db_range,
            repeats=repeats,
            use_scfdm=False
        )

        ber_results["SC-FDMA"][M] = ber_vs_snr_awgn(
            bits_tx, cfg, M,
            snr_db_range,
            repeats=repeats,
            use_scfdm=True
        )

        # ===== PAPR =====
        _, x_ofdm = build_tx(bits_tx, cfg, M, use_dft=False)
        _, x_scfdm = build_tx(bits_tx, cfg, M, use_dft=True)

        papr_results["OFDM"][M] = papr_ccdf(np.concatenate(x_ofdm))
        papr_results["SC-FDMA"][M] = papr_ccdf(np.concatenate(x_scfdm))

    # --------------------------------------------------------
    # GRÁFICAS
    # --------------------------------------------------------

    # ===== BER vs SNR =====
    plt.figure(figsize=(10, 6))
    for M in mods:
        plt.semilogy(
            snr_db_range,
            ber_results["OFDM"][M],
            marker="o",
            label=f"OFDM – {M}-QAM"
        )
        plt.semilogy(
            snr_db_range,
            ber_results["SC-FDMA"][M],
            linestyle="--",
            marker="s",
            label=f"SC-FDMA – {M}-QAM"
        )

    plt.grid(True, which="both")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR – Canal AWGN")
    plt.legend()

    # ===== CCDF PAPR =====
    plt.figure(figsize=(10, 6))
    for M in mods:
        p_o, c_o = papr_results["OFDM"][M]
        p_s, c_s = papr_results["SC-FDMA"][M]

        plt.semilogy(p_o, c_o, label=f"OFDM – {M}-QAM")
        plt.semilogy(p_s, c_s, '--', label=f"SC-FDMA – {M}-QAM")

    plt.grid(True, which="both")
    plt.xlabel("PAPR (dB)")
    plt.ylabel("CCDF")
    plt.title("CCDF del PAPR")
    plt.legend()

    plt.show()
