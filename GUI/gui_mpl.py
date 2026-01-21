from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT
)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout

class MplWidget(QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)

        # Creamos la figura vacía (sin subplots fijos iniciales)
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def get_figure(self):
        """Devuelve el objeto Figure para hacer subplots personalizados"""
        return self.fig

    def draw(self):
        """Fuerza el redibujado del canvas"""
        self.fig.tight_layout()
        self.canvas.draw()

    def clear(self):
        """Limpia la figura completa para nuevos gráficos"""
        self.fig.clear()
        self.canvas.draw()