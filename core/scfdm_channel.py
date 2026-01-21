import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True


# ============================================================
# 1) CANAL IDEAL
# ============================================================

def channel_ideal(x):
    return x.copy()


# ============================================================
# 2) CANAL AWGN
# ============================================================

def channel_awgn(x, snr_db):
    Px = np.mean(np.abs(x)**2)
    snr_lin = 10**(snr_db/10)
    noise_var = Px / snr_lin

    noise = np.sqrt(noise_var/2) * (
        np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape)
    )
    return x + noise


# ============================================================
# 3) CANAL RAYLEIGH – ITU PEDESTRIAN A (NO CRÍTICO)
# ============================================================

def channel_rayleigh_pedA(x, fs):
    """
    ITU Pedestrian A
    - Delay spread pequeño
    - Canal suavemente selectivo
    """

    delays_ns = np.array([0, 300, 700, 1200, 2200, 3700])
    powers_db = np.array([0, -3, -9, -10, -15, -20])

    delays_s = delays_ns * 1e-9
    powers_lin = 10**(powers_db / 10)
    powers_lin /= np.sum(powers_lin)

    delays_samples = np.round(delays_s * fs).astype(int)

    h = np.zeros(delays_samples[-1] + 1, dtype=complex)
    g = (np.random.randn(len(delays_samples)) +
         1j*np.random.randn(len(delays_samples))) / np.sqrt(2)

    for i, d in enumerate(delays_samples):
        h[d] += g[i] * np.sqrt(powers_lin[i])

    y = np.convolve(x, h, mode="full")[:len(x)]

    return y, h, delays_ns, powers_db


# ============================================================
# 4) CANAL RAYLEIGH SELECTIVO (CRÍTICO)
# ============================================================

def channel_rayleigh_critical(x, fs):
    """
    Canal Rayleigh selectivo CRÍTICO
    - Delay spread comparable al CP
    """

    delays_ns = np.array([0, 500, 1200, 2000, 3500, 5000])
    powers_db = np.array([0, -1, -3, -6, -10, -15])

    delays_s = delays_ns * 1e-9
    powers_lin = 10**(powers_db / 10)
    powers_lin /= np.sum(powers_lin)

    delays_samples = np.round(delays_s * fs).astype(int)

    h = np.zeros(delays_samples[-1] + 1, dtype=complex)
    g = (np.random.randn(len(delays_samples)) +
         1j*np.random.randn(len(delays_samples))) / np.sqrt(2)

    for i, d in enumerate(delays_samples):
        h[d] += g[i] * np.sqrt(powers_lin[i])

    y = np.convolve(x, h, mode="full")[:len(x)]

    return y, h, delays_ns, powers_db


# ============================================================
# 5) GRÁFICAS DEL CANAL
# ============================================================

def plot_pdp(delays_ns, powers_db):
    plt.figure(figsize=(6,4))
    plt.stem(delays_ns, powers_db, basefmt=" ")
    plt.grid(True)
    plt.xlabel("Delay (ns)")
    plt.ylabel("Potencia promedio (dB)")
    plt.title("PDP – Canal Rayleigh")


def plot_frequency_response(h, fs, Nfft=8192):
    H = np.fft.fftshift(np.fft.fft(h, Nfft))
    f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fs))

    H_db = 20*np.log10(np.abs(H) + 1e-12)
    H_db -= np.max(H_db)

    plt.figure(figsize=(8,4))
    plt.plot(f/1e6, H_db)
    plt.grid(True)
    plt.xlabel("Frecuencia (MHz)")
    plt.ylabel("Magnitud (dB)")
    plt.title("Respuesta en frecuencia del canal")
    plt.ylim([-30, 5])


def plot_time_signals(x, y, fs, Ns=3000):
    Ns = min(Ns, len(x))
    t = np.arange(Ns) / fs * 1e6

    plt.figure(figsize=(12,5))
    plt.plot(t, np.real(x[:Ns]), label="TX")
    plt.plot(t, np.real(y[:Ns]), label="Salida canal", alpha=0.8)
    plt.grid(True)
    plt.xlabel("Tiempo (µs)")
    plt.ylabel("Amplitud")
    plt.title("Señal temporal – antes y después del canal")
    plt.legend()


# ============================================================
# 6) PRUEBA DEL CANAL CON TU TX OFDM / SC-FDMA
# ============================================================

if __name__ == "__main__":

    print("\n===== PRUEBA CANAL + TX =====")

    # --------------------------------------------------------
    # Importar TU TX FINAL
    # --------------------------------------------------------
    from scfdm_tx import (
        OFDMConfig, build_tx, load_rgb_image_to_bits, M
    )

    IMAGE_PATH = "img2.png"
    IMAGE_SIZE = (1024,1024)

    cfg = OFDMConfig()
    fs = cfg.fs

    bits = load_rgb_image_to_bits(IMAGE_PATH, IMAGE_SIZE)

    # ---------------- OFDM ----------------
    X_ofdm, x_ofdm = build_tx(bits, cfg, M, use_dft=False)
    x_tx_ofdm = np.concatenate(x_ofdm)

    # ---------------- SC-FDMA ----------------
    X_scfdm, x_scfdm = build_tx(bits, cfg, M, use_dft=True)
    x_tx_scfdm = np.concatenate(x_scfdm)

    print(f"Fs: {fs/1e6:.2f} MHz")
    print(f"Longitud señal OFDM   : {len(x_tx_ofdm)} muestras")
    print(f"Longitud señal SC-FDMA: {len(x_tx_scfdm)} muestras\n")

    # --------------------------------------------------------
    # AWGN
    # --------------------------------------------------------
    snr_db = 20
    y_awgn = channel_awgn(x_tx_ofdm, snr_db)
    plot_time_signals(x_tx_ofdm, y_awgn, fs)

    # --------------------------------------------------------
    # Rayleigh Pedestrian A
    # --------------------------------------------------------
    y_ray, h, delays_ns, powers_db = channel_rayleigh_pedA(x_tx_ofdm, fs)

    plot_pdp(delays_ns, powers_db)
    plot_frequency_response(h, fs)
    plot_time_signals(x_tx_ofdm, y_ray, fs)

    plt.show()
