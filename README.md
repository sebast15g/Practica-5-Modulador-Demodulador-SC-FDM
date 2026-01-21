# Simulador OFDM / SC-FDMA (DFT-spread OFDM)

Este repositorio contiene la implementación y análisis de un sistema completo de comunicaciones digitales basado en **OFDM** y **SC-FDMA (DFT-spread OFDM)**.  
El proyecto incluye el modelado del transmisor, canal y receptor, simulaciones Monte Carlo para evaluación de desempeño y una interfaz gráfica de usuario (GUI) para visualización interactiva de resultados.

---

## 📌 Descripción general del proyecto

OFDM es una técnica de modulación multiportadora ampliamente utilizada en sistemas de comunicaciones inalámbricas modernos debido a su eficiencia espectral y robustez frente a canales selectivos en frecuencia. Sin embargo, uno de sus principales inconvenientes es el alto **PAPR (Peak-to-Average Power Ratio)**, lo cual afecta la eficiencia de los amplificadores de potencia.

SC-FDMA, también conocido como **DFT-spread OFDM**, surge como una alternativa que mantiene las ventajas de OFDM pero reduce significativamente el PAPR, motivo por el cual es utilizado, por ejemplo, en el enlace de subida de LTE.

En este proyecto se implementan ambos esquemas bajo condiciones idénticas de transmisión, permitiendo una comparación directa en términos de:
- Desempeño en BER
- Comportamiento espectral
- PAPR
- Reconstrucción de información (imagen)
- Impacto del canal y de la ecualización

---

## 🧩 Arquitectura del sistema

El sistema de comunicación implementado está compuesto por tres bloques principales:

### 🔹 Transmisor (TX)
- Fuente de información basada en una imagen RGB
- Conversión de bits y modulación QAM (4-QAM, 16-QAM, 64-QAM)
- Conversión serial a paralelo
- Precoding DFT (solo en SC-FDMA)
- Asignación de subportadoras
- IFFT
- Inserción de prefijo cíclico (CP)

### 🔹 Canal
- Canal ideal
- Canal AWGN
- Canal Rayleigh selectivo en frecuencia (escenario crítico)
- Visualización en dominio del tiempo y frecuencia

### 🔹 Receptor (RX)
- Eliminación del prefijo cíclico
- FFT
- Estimación del canal mediante pilotos
- Ecualización MMSE en frecuencia
- IDFT (solo para SC-FDMA)
- Demodulación QAM
- Reconstrucción de la imagen transmitida

---

## 📊 Análisis y métricas de desempeño

El sistema permite analizar múltiples representaciones y métricas:

- Espectro continuo de la señal transmitida
- Grilla tiempo–frecuencia
- Potencia instantánea y PAPR
- Diagramas de constelación
- Reconstrucción de imagen en RX
- BER vs SNR
- CCDF del PAPR

Estas métricas se obtienen tanto de simulaciones puntuales como de simulaciones estadísticas mediante Monte Carlo.

---

## 📈 Simulaciones Monte Carlo

Se realizaron simulaciones Monte Carlo para evaluar de forma estadística el desempeño de OFDM y SC-FDMA:

- Curvas BER vs SNR para diferentes órdenes de modulación
- CCDF del PAPR para comparar la probabilidad de ocurrencia de picos de potencia

Los resultados muestran que:
- OFDM y SC-FDMA presentan un desempeño similar en BER en canal AWGN
- SC-FDMA reduce significativamente el PAPR frente a OFDM
- La precodificación DFT no degrada la tasa de error, pero mejora la eficiencia energética

---

## 🖥️ Interfaz Gráfica de Usuario (GUI)

El proyecto incluye una GUI desarrollada en PyQt5 que permite:
- Configurar parámetros del sistema
- Visualizar señales del transmisor
- Observar el efecto del canal
- Analizar constelaciones e imágenes en el receptor
- Ejecutar simulaciones Monte Carlo y visualizar BER y PAPR

La GUI organiza el sistema en pestañas funcionales: Configuración, TX, Canal, RX y Análisis.

---

## 📂 Estructura del repositorio

```text
core/
 ├── scfdm_tx.py
 ├── scfdm_rx.py
 └── scfdm_channel.py

gui/
 ├── gui_main.py
 ├── tab_config.py
 ├── tab_tx.py
 ├── tab_channel.py
 ├── tab_rx.py
 └── tab_analysis.py

figs_results/
 ├── espectro_tx.png
 ├── grid_tx.png
 ├── time_papr_tx.png
 ├── BER.png
 ├── PAPR.png
 └── gui_*.png
```
## ▶️ Ejecución del proyecto

1. **Clonar el repositorio desde GitHub**  

2. **Instalar las dependencias necesarias**  
   El proyecto requiere las siguientes librerías de Python:
   - numpy  
   - scipy  
   - matplotlib  
   - pillow  
   - PyQt5  

3. **Ejecutar el archivo principal de la GUI**  
   El archivo principal se encuentra en la carpeta `gui/`.
---

## 🎓 Contexto académico

Este proyecto fue desarrollado con fines académicos dentro del área de **Comunicaciones Digitales**, abordando los siguientes conceptos fundamentales:

- Modulación multiportadora  
- OFDM y SC-FDMA (DFT-spread OFDM)  
- Ecualización en canales selectivos en frecuencia  
- Reducción del PAPR  
- Análisis estadístico mediante simulaciones Monte Carlo  

---

## 👥 Autores

- **Pablo Bermeo**  
- **Sebastián Guazhima**
