import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from tabs.tab_config import ConfigTab
from tabs.tab_tx import TxTab
from tabs.tab_channel import ChannelTab
from tabs.tab_rx import RxTab
from tabs.tab_analysis import AnalysisTab


class OFDMGui(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OFDM / SC-FDM (DFTs OFDM) Visual Simulator")
        self.resize(1200, 800)

        # Estado compartido entre pestañas
        self.state = {}

        tabs = QTabWidget()
        tabs.addTab(ConfigTab(self.state), "Configuración")
        tabs.addTab(TxTab(self.state), "TX")
        tabs.addTab(ChannelTab(self.state), "Canal")
        tabs.addTab(RxTab(self.state), "RX")
        tabs.addTab(AnalysisTab(self.state), "Análisis")

        self.setCentralWidget(tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OFDMGui()
    win.show()
    sys.exit(app.exec_())
