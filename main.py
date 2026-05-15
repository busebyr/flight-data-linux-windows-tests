import pandas as pd
import sys
import os
import re
import math #Tile koordinat hesaplamaları için
import threading #Tile'ları arka planda indirmek için (arayüzün donmaması için)
from io import BytesIO #Ham bayt verisini resme çevirmek için
from scipy.io import savemat
import numpy as np
import json
import locale
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.backend_bases import MouseButton

import requests #Tile URL'Lerine HTTP isteği atmak için
from PIL import Image #İndirilen PNG'yi PyQt6'nın anlayacağı formata çevirmek için

from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QCursor


from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QListWidget, QFileDialog, QLabel,
                             QAbstractItemView, QMessageBox, QLineEdit, QMainWindow,
                             QRadioButton, QButtonGroup, QTreeWidget, QTreeWidgetItem,
                             QGroupBox, QComboBox, QCheckBox, QTableWidget, QTableWidgetItem,
                             QTabWidget, QHeaderView, QMenu, QInputDialog)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from analiz_motoru import dosyalari_yukle, grafikleri_ciz, DosyaHatasi, veriyi_hizala
from error_analyzer import analyze_errors, find_variable_system, check_special_variable

from scipy.integrate import cumulative_trapezoid

def get_app_dir() -> str:
    """Sadece okuma: xlsx ve data dosyaları için (frozen'da _MEIPASS)."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.path.abspath(os.getcwd())

def get_writable_dir() -> str:
    """Yazma: aliases.json gibi kullanıcı verisi için (frozen'da exe'nin yanı)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.path.abspath(os.getcwd())

TILE_SIZE = 256

TILE_URL = {
    "Google Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
}

HEADERS = {
    "User-Agent": "FlightDataAnalyzer/1.0"
}

#Koordinat Dönüşüm Fonksiyonları
def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n) #boylam
    lat_r = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n) #enlem
    return x, y

def title_to_lat_lon(x, y, zoom):
    n = 2 ** zoom
    lon=x/n*360-180
    lat_r = math.atan(math.sinh(math.pi*(1-2*y/n)))
    lat = math.degrees(lat_r)
    return lat, lon
#Bu formüller tüm internet haritalarının kullandığı standart.Dünyayı düz bir kareye yansıtmanın matematiksel yolu.

#tile'ları arka planda thread ile indiren motor.
class TileWorker(QObject):
    tile_ready = pyqtSignal(int, int, int, object) #Sinyal sistemi.Tile indiğinde arayüze haber verir.

    def __init__(self):
        super().__init__()
        self._cache = {} #İndirilen tilelar burada tutulur.
        self._queue = [] #İndirilmesi istenilen tileların sırası.
        self._lock = threading.Lock() #_queue'ya hem ana thread hem indirme thread'i erişiyor. Aynı anda ikisi birden yazmasın diye kilit koyuyoruz
        self._source = list(TILE_URL.values())[0]
        self._cache_dir="tile_cache"
        os.makedirs(self._cache_dir, exist_ok=True)
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def request(self, z,x,y):
        key = (z, x, y)
        if key in self._cache:
            return
        with self._lock:
            if key not in self._queue:
                self._queue.append(key)

    def get(self, z, x, y):
        return self._cache.get((z, x, y))

    def run(self):
        """run içindeki döngü:
        1.Kuyruktan bir tile al
        2.URL'yi oluştur, HTTP isteği at
        3.PNG'yi PIL ile aç, PyQt6'nın anlayacağı QPixmap'e çevir
        4.Cache'e kaydet, sinyal gönder"""
        while True:
            key = None
            with self._lock:
                if self._queue:
                    key = self._queue.pop()
            if key is None:
                threading.Event().wait(0.02)
                continue
            z, x, y = key
            if key in self._cache:
                continue

            #Önce diske bak
            disk_path = os.path.join(self._cache_dir, f"{z}_{x}_{y}.png")
            if os.path.exists(disk_path):
                try:
                    img = Image.open(disk_path).convert("RGBA")
                    data = img.tobytes("raw", "RGBA")
                    qimg = QImage(data, img.width, img.height,
                                  QImage.Format.Format_RGBA8888)
                    pix = QPixmap.fromImage(qimg)
                    self._cache[key] = pix
                    self.tile_ready.emit(z, x, y, pix)
                    continue
                except Exception:
                    pass

            #Diskte yoksa internetten çek
            try:
                url = self._source.format(z=z, x=x, y=y)
                r = requests.get(url, headers=HEADERS, timeout=5)
                if r.status_code == 200:
                    #Diske kaydet
                    with open(disk_path, "wb") as f:
                        f.write(r.content)
                    #Belleğe al
                    img = Image.open(BytesIO(r.content)).convert("RGBA")
                    data = img.tobytes("raw", "RGBA")
                    qimg = QImage(data, img.width, img.height,
                                  QImage.Format.Format_RGBA8888)
                    pix = QPixmap.fromImage(qimg)
                    self._cache[key] = pix
                    self.tile_ready.emit(z, x, y, pix)
            except Exception:
                pass


class MapWidget(QWidget):
    coord_update = pyqtSignal(float, float)
    zaman_update = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.zoom = 12
        self.center_lat = 39.9334
        self.center_lon = 32.8597
        self.dragging = False
        self.last_mouse = QPoint()
        self.worker = TileWorker()
        self.worker.tile_ready.connect(self.on_tile_ready)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.rota_zamanlari = []
        self.rota_koordinatlari = [] #Uçuş rota verisi

    def on_tile_ready(self, z):
        """Arka planda indirme devam ederken zoom yapıldığında
        görsel bozukluk olmaması için o zooma ait veri mi diye kontrol ediliyor"""
        if z==self.zoom:
            self.update()

    def center_pixel(self):
        """Haritanın merkezindeki enlem-boylam koordinatını dünya üzerindeki pixel koordinatına çevirir"""
        n = 2 ** self.zoom
        px = (self.center_lon + 180)/360 * n * TILE_SIZE
        lat_r = math.radians(self.center_lat)
        py = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n * TILE_SIZE
        return px, py

    def pixel_to_latlon(self, px, py):
        """Ekrandaki pixel noktasını enlem-boylam koordinatına çevirir"""
        w, h = self.width(), self.height()
        cx_px, cy_px = self.center_pixel()
        world_x = cx_px + (px - w / 2)
        world_y = cy_px + (py - h / 2)
        n = 2 ** self.zoom
        lon = world_x / (n * TILE_SIZE) * 360 - 180
        lat = math.degrees(math.atan(math.sinh(
            math.pi * (1 - 2 * world_y / (n * TILE_SIZE)))))
        return lat, lon

    def latlon_to_screen(self, lat, lon):
        """Koordinatı ekran pixeline çevirir"""
        """w=width, h=height"""
        w, h = self.width(), self.height()
        cx_px, cy_px = self.center_pixel()
        n = 2 ** self.zoom
        px = (lon + 180) / 360 * n * TILE_SIZE
        lat_r = math.radians(lat)
        py = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n * TILE_SIZE
        screen_x = int(px - cx_px + w / 2)
        screen_y = int(py - cy_px + h / 2)

        return screen_x, screen_y

    def rota_yukle(self, koordinatlar, zamanlar=None):
        """Ana uygulamadan lat/lon ve zaman listesi alır"""
        self.rota_koordinatlari = koordinatlar
        self.rota_zamanlari = zamanlar if zamanlar is not None else[]

        if koordinatlar:
            self.center_lat, self.center_lon = koordinatlar[0]
        self.update()

    def paintEvent(self, event): #Her çizimde ekranda hangi tilelerin görünmesi gerektiğini hesaplar.Cacehde varsa çizer yoksa worker.requests ile sıraya ekler.
        painter = QPainter(self)
        w, h = self.width(), self.height()
        cx_px, cy_px = self.center_pixel()

        tile_x0 = int((cx_px - w/2) / TILE_SIZE) #sol kenar
        tile_y0 = int((cy_px - h/2) / TILE_SIZE)  #üst
        tile_x1 = int((cx_px + w/2) / TILE_SIZE) + 1 #sağ kenar
        tile_y1 = int((cy_px + h/2) / TILE_SIZE) + 1 #alt

        for tx in range(tile_x0, tile_x1 + 1):
            for ty in range(tile_y0, tile_y1 + 1):
                pix = self.worker.get(self.zoom, tx, ty) #Her tile için cache de var mı diye sorar
                draw_x = int(tx * TILE_SIZE - cx_px + w/2)
                draw_y = int(ty * TILE_SIZE - cy_px + h/2)
                if pix:
                    painter.drawPixmap(draw_x, draw_y, pix) #Cache'de varsa Tile'ı çizer.
                else:
                    self.worker.request(self.zoom, tx, ty)  #Yoksa indirme sırasına ekler
                    painter.fillRect(draw_x, draw_y, TILE_SIZE, TILE_SIZE, QColor(30, 30, 30))

        #Uçuş rotasını çiz
        if len(self.rota_koordinatlari) > 1:
            painter.setPen(QPen(QColor(0, 0, 200), 2)) #Mavi 2 pixel kalınlığında
            for i in range(len(self.rota_koordinatlari) - 1):
                x1, y1 = self.latlon_to_screen(*self.rota_koordinatlari[i])
                x2, y2 = self.latlon_to_screen(*self.rota_koordinatlari[i+1])
                painter.drawLine(x1, y1, x2, y2)

            self.igne_ciz(painter, self.rota_koordinatlari[0], QColor(0, 200, 0)) #Yeşil kalkis, 0 (listenin ilk elemanı)
            self.igne_ciz(painter, self.rota_koordinatlari[-1], QColor(200, 0, 0)) #Kirmizi inis, -1 (listenin son elemanı)

    def igne_ciz(self, painter: QPainter, koordinat: tuple, renk: QColor, boyut: int = 8):
        """Haritada nokta çizer."""
        x, y = self.latlon_to_screen(*koordinat)
        painter.setBrush(renk)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawEllipse(x - boyut // 2, y - boyut // 2, boyut, boyut)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_mouse = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.dragging:
            dx = event.pos().x() - self.last_mouse.x()
            dy = event.pos().y() - self.last_mouse.y()
            self.last_mouse = event.pos()
            n = 2 ** self.zoom
            self.center_lon -= dx / TILE_SIZE / n * 360
            lat_r = math.radians(self.center_lat)
            self.center_lat += dy / TILE_SIZE / n * 360 * math.cos(lat_r)
            self.update()
        lat, lon = self.pixel_to_latlon(event.pos().x(), event.pos().y())
        self.coord_update.emit(lat, lon)

        #Mouse en yakın noktanın zamanı
        if self.rota_koordinatlari and self.rota_zamanlari:
            en_yakin_idx = min(
                range(len(self.rota_koordinatlari)),
                key=lambda i: (self.rota_koordinatlari[i][0] - lat) ** 2 +
                              (self.rota_koordinatlari[i][1] - lon) ** 2
            )
            self.zaman_update.emit(self.rota_zamanlari[en_yakin_idx])

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0 and self.zoom < 20:
            self.zoom += 1
        elif delta < 0 and self.zoom > 2:
            self.zoom -= 1
        lat, lon = self.center_lat, self.center_lon
        self.coord_update.emit(lat, lon)

        self.update()


class HaritaPenceresi(QMainWindow):
    def __init__(self, plotted_data, parent_ref= None):
        super().__init__()
        self.parent_ref = parent_ref
        self.setWindowTitle("Flight Map")
        self.resize(1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        ana_layout = QHBoxLayout(central)

        #Sol panel
        sol_panel = QWidget()
        sol_panel.setFixedWidth(200)
        sol_layout = QVBoxLayout(sol_panel)
        sol_layout.setContentsMargins(5, 5, 5, 5)

        #Latitude
        sol_layout.addWidget(QLabel("LATITUDE"))
        self.lat_group = QButtonGroup()
        self.lat_listesi = QListWidget()
        self.lat_listesi.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.lat_listesi.itemClicked.connect(
            lambda item: [b.setChecked(False) for b in self.lat_group.buttons()] or
                         self.lat_group.buttons()[self.lat_listesi.row(item)].setChecked(True)
            if self.lat_group.buttons() else None
        )
        sol_layout.addWidget(self.lat_listesi)

        #Longitude
        sol_layout.addWidget(QLabel("LONGITUDE"))
        self.lon_group = QButtonGroup()
        self.lon_listesi = QListWidget()
        self.lon_listesi.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        sol_layout.addWidget(self.lon_listesi)

        sol_layout.addStretch()

        #Koordinat gösterimi
        self.koordinat_label = QLabel("Lat: —   Lon: — ")
        sol_layout.addWidget(self.koordinat_label)
        zaman_layout = QHBoxLayout()
        self.t_label = QLabel("time(s): —")
        zaman_layout.addWidget(self.t_label)
        sol_layout.addLayout(zaman_layout)
        self.zoom_label = QLabel("Zoom: 12")
        zaman_layout.addWidget(self.zoom_label)
        self.aralik_label = QLabel("Range(s): —")
        sol_layout.addWidget(self.aralik_label)

        #Apply butonu
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self.uygula)
        sol_layout.addWidget(self.btn_apply)

        #Sağ panel
        self.harita = MapWidget()
        self.harita.coord_update.connect(self.koordinat_guncelle)
        self.harita.zaman_update.connect(self.t_guncelle)

        ana_layout.addWidget(sol_panel)
        ana_layout.addWidget(self.harita, 1)

        #Veriyi sakla ve kolonları doldur
        self.plotted_data = plotted_data
        self.kolonlari_doldur()

    def t_guncelle(self, t):
        self.t_label.setText(f"time(s): {t:.2f}s")

    def kolonlari_doldur(self):
        self.lat_listesi.clear()
        self.lon_listesi.clear()

        for label in self.plotted_data.keys():
            if ' | ' in label:
                self.lat_listesi.addItem(label)
                self.lon_listesi.addItem(label)

    def koordinat_guncelle(self, lat, lon):
        self.koordinat_label.setText(f"Lat: {lat:.5f}   Lon: {lon:.5f}")
        self.zoom_label.setText(f"Zoom: {self.harita.zoom}")

    def zaman_araligini_guncelle(self, t_min, t_max):
        lat_item = self.lat_listesi.currentItem()
        lon_item = self.lon_listesi.currentItem()
        if lat_item is None or lon_item is None:
            return

        #currentItem dolu ama seçili değilse çık
        if not self.lat_listesi.selectedItems() or not self.lon_listesi.selectedItems():
            return

        lat_label = lat_item.text()
        lon_label = lon_item.text()

        lat_t, lat_data, lon_t, lon_data = None, None, None, None

        for label, (t, y) in self.plotted_data.items():
            if label == lat_label:
                lat_t = np.asarray(t)
                lat_data = np.asarray(y)
            if label == lon_label:
                lon_t = np.asarray(t)
                lon_data = np.asarray(y)

        if lat_t is None or lat_data is None or lon_t is None or lon_data is None:
            return

        #Farklı frekansa sahip veriler için lon lat'a hizallanıyor
        if not np.array_equal(lat_t, lon_t):
            df_lon = pd.DataFrame({'t': lon_t, 'v': lon_data})
            t_series = pd.Series(lat_t)
            lon_data = veriyi_hizala(df_lon, t_series, 'v', 'nearest').to_numpy(dtype=float)

        t_min = max(t_min, lat_t[0])
        t_max = min(t_max, lat_t[-1])

        mask = (lat_t >= t_min) & (lat_t <= t_max)
        lat_filtered = lat_data[mask]
        lon_filtered = lon_data[mask]
        t_filtered = lat_t[mask]

        if len(lat_filtered) < 2:
            return

        koordinatlar = list(zip(lat_filtered, lon_filtered))
        self.harita.rota_yukle(koordinatlar, list(t_filtered))
        self.aralik_label.setText(f"Range(s): {t_min:.1f}s — {t_max:.1f}s")

    def uygula(self):
        lat_item = self.lat_listesi.currentItem()
        lon_item = self.lon_listesi.currentItem()

        if lat_item is None or lon_item is None:
            QMessageBox.warning(self, "Warning", "Please select the latitude and longitude columns.")
            return

        lat_label = lat_item.text()
        lon_label = lon_item.text()

        lat_t, lat_data, lon_data = None, None, None

        for label, (t, y) in self.plotted_data.items():
            if label == lat_label:
                lat_t = np.asarray(t)
                lat_data = np.asarray(y)
            if label == lon_label:
                lon_data = np.asarray(y)

        if lat_data is None or lon_data is None:
            QMessageBox.warning(self, "Warning", "The selected columns were not found in the data.")
            return

        koordinatlar = list(zip(lat_data, lon_data))
        self.harita.rota_yukle(koordinatlar, list(lat_t))

        #Mevcut grafik zoom aralığını da hemen uygula
        if self.parent_ref is not None:
            ax = self.parent_ref.canvas.figure.axes[0]
            t_min, t_max = ax.get_xlim()
            self.zaman_araligini_guncelle(t_min, t_max)

    def closeEvent(self, event):
        if self.parent_ref is not None:
            self.parent_ref.btn_harita.setText('Open Map')
            self.parent_ref.harita_penceresi = None
        super().closeEvent(event)


class GrafikPenceresi(QMainWindow):

    def __init__(self, fig, figure_no, mode, parent_ref):
        super().__init__()
        self.parent_ref = parent_ref
        self.figure_no = figure_no
        self.mode = mode
        self.setWindowTitle(f'Figure {figure_no} — TIME MODE: {mode.upper()}')
        #self.resize(900, 500)

        central = QWidget()
        layout = QVBoxLayout(central)

        self.canvas = FigureCanvas(fig)
        self.canvas.mouseDoubleClickEvent = self.canvas_double_click
        self.toolbar = NavigationToolbar(self.canvas, self)

        #Grafik etiketlerini gizle
        self.legend_checkbox = QCheckBox("Insert Legend")
        self.legend_checkbox.setChecked(True)
        self.legend_checkbox.stateChanged.connect(self.legend_guncelle)
        self.btn_harita = QPushButton('Open Map')
        self.btn_harita.setFixedWidth(90)
        self.btn_harita.clicked.connect(self.harita_toggle)

        self.lines = []
        self.line_map = {}
        self.pick_connection = None
        self.press_conection = None
        self.scatter_list = []

        #Tab sistemi
        self.tabs = QTabWidget()

        #Grafik sekmesi
        grafik_widget = QWidget()
        grafik_layout = QVBoxLayout(grafik_widget)
        grafik_layout.addWidget(self.canvas)

        self.tabs.insertTab(0, grafik_widget, "Graph")
        self.harita_penceresi = None

        #Rapor sekmesi
        self.rapor_widget = QWidget()
        rapor_layout = QVBoxLayout(self.rapor_widget)
        self.rapor_tablo = QTableWidget()
        self.rapor_tablo.setColumnCount(4)
        self.rapor_tablo.setHorizontalHeaderLabels(["Variable", "Time (s)", "Value", "Error Type"])
        self.rapor_tablo.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.rapor_tablo.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rapor_layout.addWidget(self.rapor_tablo)
        self.tabs.addTab(self.rapor_widget, "Error Report")

        #Matlab Export Sekmesi
        self.matlab_widget = QWidget()
        matlab_layout = QVBoxLayout(self.matlab_widget)
        self.matlab_listesi = QListWidget()
        self.matlab_listesi.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.btn_matlab_export = QPushButton('Export.mat')
        self.btn_matlab_export.clicked.connect(self.matlab_export)
        matlab_layout.addWidget(QLabel('select parameters to export:'))
        matlab_layout.addWidget(self.matlab_listesi)
        matlab_layout.addWidget(self.btn_matlab_export)
        self.tabs.addTab(self.matlab_widget, 'Matlab Export')

        #CSV Export Sekmesi
        self.csv_widget = QWidget()
        csv_layout = QVBoxLayout(self.csv_widget)
        self.csv_listesi = QListWidget()
        self.csv_listesi.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.btn_csv_export = QPushButton('Export.csv')
        self.btn_csv_export.clicked.connect(self.csv_export)
        csv_layout.addWidget(QLabel('select parameters to export:'))
        csv_layout.addWidget(self.csv_listesi)
        csv_layout.addWidget(self.btn_csv_export)
        self.tabs.addTab(self.csv_widget, 'CSV Export')

        #Operations Sekmesi
        self.ops_widget = QWidget()
        ops_layout = QVBoxLayout(self.ops_widget)

        #İşlem tanımlama satırı
        tanim_layout = QHBoxLayout()

        self.sol_operand = QComboBox()
        self.operator_combo = QComboBox()
        self.operator_combo.addItems(['+', '-', '*', '/'])

        self.radio_parametre = QRadioButton('Param')
        self.radio_sabit = QRadioButton('Const')
        self.radio_parametre.setChecked(True)

        self.sag_operand_combo = QComboBox()
        self.sag_operand_sabit = QLineEdit()
        self.sag_operand_sabit.setPlaceholderText('Enter number')
        self.sag_operand_sabit.setVisible(False)
        self.sag_operand_sabit.setMaximumWidth(90)

        self.radio_parametre.toggled.connect(
            lambda checked: (self.sag_operand_combo.setVisible(checked),
                             self.sag_operand_sabit.setVisible(not checked))
        )

        self.btn_add_op = QPushButton('Add Operation')
        self.btn_add_op.clicked.connect(self.operasyon_ekle)

        tanim_layout.addWidget(QLabel('Left:'))
        tanim_layout.addWidget(self.sol_operand)
        tanim_layout.addWidget(self.operator_combo)
        tanim_layout.addWidget(self.radio_parametre)
        tanim_layout.addWidget(self.radio_sabit)
        tanim_layout.addWidget(self.sag_operand_combo)
        tanim_layout.addWidget(self.sag_operand_sabit)
        tanim_layout.addWidget(self.btn_add_op)

        #Serbest ifade alanı
        ifade_layout = QHBoxLayout()
        self.ifade_giris = QLineEdit()
        self.ifade_giris.setPlaceholderText('Örn: (U1*(U2+5))/(2*U3)')
        self.btn_add_expr = QPushButton('Add Expression')
        self.btn_add_expr.clicked.connect(self.ifade_ekle)
        ifade_layout.addWidget(QLabel('Expression:'))
        ifade_layout.addWidget(self.ifade_giris)
        ifade_layout.addWidget(self.btn_add_expr)

        #İşlem listesi
        self.op_listesi = QTableWidget()
        self.op_listesi.setColumnCount(3)
        self.op_listesi.setHorizontalHeaderLabels(['Name', 'Operation', 'Delete'])
        self.op_listesi.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.op_listesi.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.op_listesi.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.op_listesi.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.btn_apply_ops = QPushButton('Apply')
        self.btn_apply_ops.clicked.connect(self.operasyonlari_uygula)

        ops_layout.addLayout(tanim_layout)
        ops_layout.addLayout(ifade_layout)
        ops_layout.addWidget(self.op_listesi)
        ops_layout.addWidget(self.btn_apply_ops)

        self.tabs.addTab(self.ops_widget, 'Operations')

        #Wh/Ah Sekmesi
        self.energy_widget = QWidget()
        energy_layout = QVBoxLayout(self.energy_widget)
        energy_layout.setSpacing(10)

        #Veri Kaynağı
        kaynak_group = QGroupBox('Data Source')
        kaynak_layout = QVBoxLayout(kaynak_group)

        #Dosya seç satırı
        dosya_sec_layout = QHBoxLayout()
        self.btn_energy_dosya = QPushButton('Select File')
        self.btn_energy_dosya.clicked.connect(self.energy_dosya_sec)
        self.energy_dosya_label = QLabel('No file selected — using plotted data')
        self.energy_dosya_label.setStyleSheet('font-style: italic;')
        self.btn_energy_dosya_temizle = QPushButton('✕')
        self.btn_energy_dosya_temizle.setFixedWidth(28)
        self.btn_energy_dosya_temizle.clicked.connect(self.energy_dosya_temizle)
        dosya_sec_layout.addWidget(self.btn_energy_dosya)
        dosya_sec_layout.addWidget(self.energy_dosya_label, 1)
        dosya_sec_layout.addWidget(self.btn_energy_dosya_temizle)
        kaynak_layout.addLayout(dosya_sec_layout)

        #Kolon seçim satırı
        kolon_layout = QHBoxLayout()
        kolon_layout.addWidget(QLabel('Voltage (Vi):'))
        self.energy_vi_combo = QComboBox()
        kolon_layout.addWidget(self.energy_vi_combo)
        kolon_layout.addSpacing(20)
        kolon_layout.addWidget(QLabel('Current (Ii):'))
        self.energy_ii_combo = QComboBox()
        kolon_layout.addWidget(self.energy_ii_combo)
        kolon_layout.addStretch()
        kaynak_layout.addLayout(kolon_layout)

        energy_layout.addWidget(kaynak_group)

        #Zaman aralığı seçimi
        time_group = QGroupBox('Time Range')
        time_group_layout = QVBoxLayout(time_group)

        self.energy_radio_full = QRadioButton('Full Data')
        self.energy_radio_full.setChecked(True)
        self.energy_time_btn_group = QButtonGroup()
        self.energy_time_btn_group.addButton(self.energy_radio_full)
        time_group_layout.addWidget(self.energy_radio_full)

        #Custom Range seçeneği
        custom_layout = QHBoxLayout()
        self.energy_radio_custom = QRadioButton('Custom Range (s):')
        self.energy_time_btn_group.addButton(self.energy_radio_custom)
        self.energy_start = QLineEdit()
        self.energy_start.setPlaceholderText('0.0')
        self.energy_start.setMaximumWidth(80)
        self.energy_end = QLineEdit()
        self.energy_end.setPlaceholderText('e.g. 120.0')
        self.energy_end.setMaximumWidth(80)
        custom_layout.addWidget(self.energy_radio_custom)
        custom_layout.addWidget(QLabel('Start:'))
        custom_layout.addWidget(self.energy_start)
        custom_layout.addWidget(QLabel('End:'))
        custom_layout.addWidget(self.energy_end)
        custom_layout.addStretch()
        time_group_layout.addLayout(custom_layout)

        #Periyodik örnekleme seçenği
        period_layout = QHBoxLayout()
        self.energy_radio_period = QRadioButton('Periodic Sampling:')
        self.energy_time_btn_group.addButton(self.energy_radio_period)
        self.energy_period_ms = QLineEdit()
        self.energy_period_ms.setPlaceholderText('e.g. 10')
        self.energy_period_ms.setMaximumWidth(80)
        period_layout.addWidget(self.energy_radio_period)
        period_layout.addWidget(self.energy_period_ms)
        period_layout.addWidget(QLabel('ms'))
        period_layout.addStretch()
        time_group_layout.addLayout(period_layout)

        energy_layout.addWidget(time_group)

        #Calculate butonu
        self.btn_energy_calc = QPushButton('Calculate')
        self.btn_energy_calc.setFixedHeight(32)
        self.btn_energy_calc.clicked.connect(self.energy_hesapla)
        energy_layout.addWidget(self.btn_energy_calc)

        #Sonuç alanı
        sonuc_group = QGroupBox('Results')
        sonuc_layout = QHBoxLayout(sonuc_group)
        self.energy_wh_label = QLabel('Watt-Hour  :  —')
        self.energy_ah_label = QLabel('Ampere-Hour:  —')
        self.energy_wh_label.setStyleSheet('font-size: 14px; font-weight: bold;')
        self.energy_ah_label.setStyleSheet('font-size: 14px; font-weight: bold;')
        sonuc_layout.addWidget(self.energy_wh_label)
        sonuc_layout.addSpacing(40)
        sonuc_layout.addWidget(self.energy_ah_label)
        sonuc_layout.addStretch()
        energy_layout.addWidget(sonuc_group)

        #Checkbox
        self.energy_graph_checkbox = QCheckBox('Show Graph')
        self.energy_graph_checkbox.setChecked(False)
        self.energy_graph_checkbox.stateChanged.connect(self.energy_graph_checkbox_degisti)
        energy_layout.addWidget(self.energy_graph_checkbox)

        #Son hesplama verisi
        self._energy_son_t = None
        self._energy_son_v = None
        self._energy_son_i = None
        self._energy_pencere = None
        self._energy_df = None
        self._energy_df_sep = None
        self._energy_df_skiprows = 0

        self.tabs.addTab(self.energy_widget, 'Wh / Ah')

        layout.addWidget(self.toolbar)
        legend_harita_layout = QHBoxLayout()
        legend_harita_layout.addWidget(self.legend_checkbox)
        legend_harita_layout.addWidget(self.btn_harita)
        legend_harita_layout.addStretch()
        layout.addLayout(legend_harita_layout)
        layout.addWidget(self.tabs)

        self.setCentralWidget(central)

        self.plotted_data = {}
        self.label_map = {}
        self.ters_label_map = {}
        self.scatter_data = {}

        #Operations veri deposu: {"Op1": (t, v, aciklama, sol, op, sag, sabit_mi)}
        self.op_data = {}
        self._op_sayac = 0

        self.xlim_cid = None

    def harita_toggle(self):
        if self.harita_penceresi is None or not self.harita_penceresi.isVisible():
            self.harita_penceresi = HaritaPenceresi(self.plotted_data, parent_ref=self)
            self.harita_penceresi.show()
            self.btn_harita.setText('Close Map')

            ax = self.canvas.figure.axes[0]
            self.xlim_cid = ax.callbacks.connect('xlim_changed', self.harita_xlim_degisti)
        else:
            ax = self.canvas.figure.axes[0]
            ax.callbacks.disconnect(self.xlim_cid)
            self.xlim_cid = None
            self.harita_penceresi.close()
            self.harita_penceresi = None
            self.btn_harita.setText('Open Map')

    def harita_xlim_degisti(self, ax):
        if self.harita_penceresi is None or not self.harita_penceresi.isVisible():
            return
        t_min, t_max = ax.get_xlim()
        self.harita_penceresi.zaman_araligini_guncelle(t_min, t_max)

    def energy_dropdown_guncelle(self):
        """Energy sekmesindeki Vi/Ii dropdown'larını mevcut U parametreleriyle doldurur."""
        u_isimleri = [v for v in self.label_map.values()]
        self.energy_vi_combo.clear()
        self.energy_ii_combo.clear()
        self.energy_vi_combo.addItems(u_isimleri)
        self.energy_ii_combo.addItems(u_isimleri)

    def energy_dosya_sec(self):
        dosya_yolu, _ = QFileDialog.getOpenFileName(
            self, 'Select Data File', '', 'CSV Files (*.csv);;All Files (*)')
        if not dosya_yolu:
            return
        try:
            with open(dosya_yolu, 'r', encoding='utf-8', errors='replace') as f:
                satirlar = f.readlines()

            if not satirlar:
                QMessageBox.warning(self, 'Warning', 'File is empty.')
                return

            #Ayırıcı ve header satırı tespiti
            sep = None
            header_satir = 0

            ilk = satirlar[0].strip().lower()
            if ilk.startswith('sep='):
                header_satir = 1
                sep_raw = satirlar[0].strip().split('=', 1)[1].strip()
                if sep_raw in ('\\t', '\t') or sep_raw == '':
                    #boş ise gerçek ayırıcıyı referans satırdan tespit et
                    sep = None
                else:
                    sep = sep_raw

            #sep hala None ise otomatik tespit: header'dan sonraki ilk veri satırına bak
            if sep is None:
                referans_idx = header_satir + 1 if len(satirlar) > header_satir + 1 else header_satir
                referans = satirlar[referans_idx] if referans_idx < len(satirlar) else satirlar[-1]
                tab = referans.count('\t')
                semi = referans.count(';')
                comma = referans.count(',')
                if tab >= semi and tab >= comma:
                    sep = '\t'
                elif semi >= comma:
                    sep = ';'
                else:
                    sep = ','

            df = pd.read_csv(dosya_yolu, sep=sep, skiprows=header_satir,
                             encoding='utf-8', encoding_errors='replace', engine='python')

            #ESC dosyalarında son kolonda "Serial Number; V2.1" gibi ekler olabilir
            temiz_kolonlar = [str(k).split(';')[0].strip() for k in df.columns]
            df.columns = temiz_kolonlar
            #Tamamen boş isimli kolonları at
            df = df.loc[:, df.columns.str.strip() != '']

            if df.empty or len(df.columns) == 0:
                QMessageBox.warning(self, 'Warning', 'No usable data found in file.')
                return

            self._energy_df = df
            self._energy_df_sep = sep
            self._energy_df_skiprows = header_satir

            self.energy_dosya_label.setText(os.path.basename(dosya_yolu))
            self.energy_dosya_label.setStyleSheet('')

            kolonlar = list(df.columns)

            #Vi / Ii dropdown
            self.energy_vi_combo.clear()
            self.energy_ii_combo.clear()
            self.energy_vi_combo.addItems(kolonlar)
            self.energy_ii_combo.addItems(kolonlar)

            #Otomatik seçimler
            for i, k in enumerate(kolonlar):
                if 'voltage' in k.lower():
                    self.energy_vi_combo.setCurrentIndex(i)
                    break
            for i, k in enumerate(kolonlar):
                if 'current' in k.lower() and 'phase' not in k.lower():
                    self.energy_ii_combo.setCurrentIndex(i)
                    break
            for i, k in enumerate(kolonlar):
                if any(x in k.lower() for x in ('time', 'timestamp', 'zeit', 'temps', 'zaman')):
                    break

        except Exception as e:
            QMessageBox.warning(self, 'Warning', f'File could not be read:\n{e}')
            self._energy_df = None

    def energy_dosya_temizle(self):
        self._energy_df = None
        self._energy_df_sep = None
        self._energy_df_skiprows = 0
        self.energy_dosya_label.setText('No file selected — using plotted data')
        self.energy_dosya_label.setStyleSheet('font-style: italic;')
        self.energy_dropdown_guncelle()

    def energy_hesapla(self):
        if self._energy_df is not None:
            vi_kolon = self.energy_vi_combo.currentText()
            ii_kolon = self.energy_ii_combo.currentText()

            if vi_kolon == ii_kolon:
                QMessageBox.warning(self, 'Warning', 'Voltage and Current must be different columns.')
                return

            try:
                v_data = self._energy_df[vi_kolon].to_numpy(dtype=float)
                i_data = self._energy_df[ii_kolon].to_numpy(dtype=float)
            except Exception as e:
                QMessageBox.warning(self, 'Warning', f'Column could not be read:\n{e}')
                return

            #Zaman kolonunu otomatik bul
            time_col = next(
                (k for k in self._energy_df.columns
                 if any(x in k.lower() for x in ('time', 'timestamp', 'zeit', 'temps', 'zaman'))),
                None
            )

            if time_col:
                t = self._energy_df[time_col].to_numpy(dtype=float) / 1e6
            else:
                period_text = self.energy_period_ms.text().strip()
                if not period_text:
                    QMessageBox.warning(self, 'Warning',
                                        'This file has no time column. '
                                        'Please enter a Sampling Period (ms) to generate the time axis.')
                    return
                try:
                    period_ms = float(period_text)
                except ValueError:
                    QMessageBox.warning(self, 'Warning', 'Please enter a valid number for sampling period.')
                    return
                if period_ms <= 0:
                    QMessageBox.warning(self, 'Warning', 'Sampling period must be greater than 0.')
                    return
                t = np.arange(len(v_data)) * (period_ms / 1000.0)

            v = v_data
            i = i_data

        else:
            #Plotted data modu
            vi_isim = self.energy_vi_combo.currentText()
            ii_isim = self.energy_ii_combo.currentText()

            if vi_isim == ii_isim:
                QMessageBox.warning(self, 'Warning', 'Voltage and Current must be different channels.')
                return

            t_v, v_data = self.operand_verisini_al(vi_isim)
            t_i, i_data = self.operand_verisini_al(ii_isim)

            if t_v is None or t_i is None:
                QMessageBox.warning(self, 'Warning',
                                    'Could not retrieve data. Please apply operations first if needed.')
                return

            if self.mode == 'realtime':
                t_min = max(t_v[0], t_i[0])
                t_max = min(t_v[-1], t_i[-1])
                if t_min >= t_max:
                    QMessageBox.warning(self, 'Warning', 'Voltage and Current time ranges do not overlap.')
                    return
                dt = min(np.mean(np.diff(t_v)), np.mean(np.diff(t_i)))
                t = np.arange(t_min, t_max, dt)
                t_series = pd.Series(t)
                df_v = pd.DataFrame({'t': t_v, vi_isim: v_data})
                df_i = pd.DataFrame({'t': t_i, ii_isim: i_data})
                v = veriyi_hizala(df_v, t_series, vi_isim, 'nearest').to_numpy(dtype=float)
                i = veriyi_hizala(df_i, t_series, ii_isim, 'nearest').to_numpy(dtype=float)
            else:
                t = t_v
                v = v_data
                i = i_data

        #Zaman kırpma (her iki mod için ortak)
        if self.energy_radio_custom.isChecked():
            try:
                t_start = float(self.energy_start.text())
                t_end = float(self.energy_end.text())
            except ValueError:
                QMessageBox.warning(self, 'Warning', 'Please enter valid Start and End times.')
                return
            if t_start >= t_end:
                QMessageBox.warning(self, 'Warning', 'Start time must be less than End time.')
                return
            t_norm = t - t[0]
            mask = (t_norm >= t_start) & (t_norm <= t_end)
            t = t[mask]
            v = v[mask]
            i = i[mask]
            if len(t) < 2:
                QMessageBox.warning(self, 'Warning', 'Not enough data points in the selected range.')
                return

        elif self.energy_radio_period.isChecked():
            period_text = self.energy_period_ms.text().strip()
            if not period_text:
                QMessageBox.warning(self, 'Warning', 'Please enter a sampling period in ms.')
                return
            try:
                period_ms = float(period_text)
            except ValueError:
                QMessageBox.warning(self, 'Warning', 'Please enter a valid number for sampling period.')
                return
            if period_ms <= 0:
                QMessageBox.warning(self, 'Warning', 'Sampling period must be greater than 0.')
                return
            dt_period = period_ms / 1000.0
            t_norm = t - t[0]
            sure = t_norm[-1]
            t_periodic = np.arange(0, sure, dt_period)
            if len(t_periodic) < 2:
                QMessageBox.warning(self, 'Warning',
                                    'Sampling period is larger than the data duration. '
                                    'Use a smaller period.')
                return
            idx = np.searchsorted(t_norm, t_periodic).clip(0, len(t_norm) - 1)
            t = t_periodic
            v = v[idx]
            i = i[idx]

        #Hesapla
        p = v * i
        wh_toplam = np.trapezoid(p, t) / 3600.0
        ah_toplam = np.trapezoid(i, t) / 3600.0

        self.energy_wh_label.setText(f'Watt-Hour  :  {wh_toplam:.4f} Wh')
        self.energy_ah_label.setText(f'Ampere-Hour:  {ah_toplam:.4f} Ah')

        self._energy_son_t = t
        self._energy_son_v = v
        self._energy_son_i = i

        if self.energy_graph_checkbox.isChecked():
            self.energy_grafik_ac(t, v, i)

    def energy_graph_checkbox_degisti(self, state):
        if state and self._energy_son_t is not None:
            self.energy_grafik_ac(self._energy_son_t, self._energy_son_v, self._energy_son_i)
        elif not state and self._energy_pencere is not None:
            self._energy_pencere.close()
            self._energy_pencere = None

    def energy_grafik_ac(self, t, v, i):
        p = v * i
        wh_kumulatif = cumulative_trapezoid(p, t, initial=0) / 3600.0
        ah_kumulatif = cumulative_trapezoid(i, t, initial=0) / 3600.0
        t_kumulatif = t

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_title('Wh & Ah')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Wh', color='tab:blue')
        line_wh, = ax1.plot(t_kumulatif, wh_kumulatif, color='tab:blue', label='Wh')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Ah', color='tab:orange')
        line_ah, = ax2.plot(t_kumulatif, ah_kumulatif, color='tab:orange', label='Ah')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        ax1.grid(True)

        #Toplam değerleri hesapla
        wh_toplam = np.trapezoid(p, t) / 3600.0
        ah_toplam = np.trapezoid(i, t) / 3600.0

        lines = [line_wh, line_ah]
        labels = [f'Wh: {wh_toplam:.4f} Wh', f'Ah: {ah_toplam:.4f} Ah']

        legend = ax1.legend(lines, labels, loc='upper left')
        legend.set_draggable(True)
        fig.tight_layout()

        #UI ve Pencere İşlemleri
        self._energy_pencere = QMainWindow()
        self._energy_pencere.setWindowTitle(f'Wh & Ah — Figure {self.figure_no}')
        self._energy_pencere.resize(900, 500)

        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self._energy_pencere)

        #Toggle Mekanizması
        def on_click(event):
            if event.inaxes is None: return
            for leg_handle, orig in zip(legend.legend_handles, lines):
                bbox = leg_handle.get_window_extent()
                if (bbox.x0 <= event.x <= bbox.x1 + 20) and (abs(event.y - bbox.y0) < 10):
                    orig.set_visible(not orig.get_visible())
                    canvas.draw_idle()
                    break

        canvas.mpl_connect('button_press_event', on_click)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        self._energy_pencere.setCentralWidget(widget)
        self._energy_pencere.show()

    def legend_guncelle(self):
        if not self.canvas.figure.axes:
            return
        ax = self.canvas.figure.axes[0]
        legend = ax.get_legend()

        if legend:
            legend.set_visible(self.legend_checkbox.isChecked())
            self.canvas.draw_idle()

    def enable_legend_toggle(self, lines, scatter_list=None):
        self.lines = lines
        self.scatter_list = scatter_list or []
        self.line_map = {}

        ax = self.canvas.figure.axes[0]
        tum_nesneler = list(lines) + list(self.scatter_list)
        if not tum_nesneler:
            return

        old_legend = ax.get_legend()
        if old_legend:
            old_legend.remove()

        all_labels = [obj.get_label() for obj in tum_nesneler]
        ax.legend(handles=tum_nesneler, labels=all_labels, loc='upper left')
        legend = ax.get_legend()

        if legend is None:
            return
        legend.set_draggable(True)
        legend.set_visible(self.legend_checkbox.isChecked())

        if hasattr(self, "pick_connection") and self.pick_connection:
            self.canvas.mpl_disconnect(self.pick_connection)
        if hasattr(self, "press_connection") and self.press_connection:
            self.canvas.mpl_disconnect(self.press_connection)

        legend_handles = legend.legend_handles

        for legend_handle, orig_obj in zip(legend_handles, tum_nesneler):
            legend_handle.set_picker(True)
            self.line_map[legend_handle] = orig_obj

        self.pick_connection = self.canvas.mpl_connect("pick_event", self.on_pick)
        self.press_connection = self.canvas.mpl_connect("button_press_event", self.legend_sag_tik)

    #Sol ve sağ operand dropdown'larını mevcut U parametreleri + Op sonuçlarıyla güncelle
    def ops_dropdown_guncelle(self):
        u_isimleri = list(self.label_map.values())
        op_isimleri = list(self.op_data.keys())
        tum_secenekler = u_isimleri + op_isimleri

        self.sol_operand.clear()
        self.sol_operand.addItems(tum_secenekler)

        self.sag_operand_combo.clear()
        self.sag_operand_combo.addItems(tum_secenekler)

        #Energy sekmesi dropdown güncelle
        if hasattr(self, 'energy_vi_combo'):
            self.energy_dropdown_guncelle()

    def operand_verisini_al(self, isim):
        #Verilen isim için (t, v) döndürür op ise op_data'dan U ise plotted_data'dan alır
        if isim in self.op_data:
            kayit = self.op_data[isim]
            if kayit[0] is not None:
                return np.asarray(kayit[0]), np.asarray(kayit[1])
            return None, None

        gercek = self.ters_label_map.get(isim)
        if gercek and gercek in self.plotted_data:
            t, v = self.plotted_data[gercek]
            return np.asarray(t), np.asarray(v)
        return None, None

    def operasyon_ekle(self):
        sol = self.sol_operand.currentText()
        op = self.operator_combo.currentText()

        if self.radio_sabit.isChecked():
            sabit_text = self.sag_operand_sabit.text().strip()
            if not sabit_text:
                QMessageBox.warning(self, "Warning", "Please enter a constant value.")
                return
            try:
                float(sabit_text)
            except ValueError:
                QMessageBox.warning(self, "Warning", "Please enter a valid number.")
                return
            sag = sabit_text
        else:
            sag = self.sag_operand_combo.currentText()

        aciklama = f"{sol} {op} {sag}"
        satir = self.op_listesi.rowCount()
        self._op_sayac += 1
        op_isim = f"Op{self._op_sayac}"
        self.op_listesi.insertRow(satir)
        self.op_listesi.setItem(satir, 0, QTableWidgetItem(op_isim))
        self.op_listesi.setItem(satir, 1, QTableWidgetItem(aciklama))

        btn_sil = QPushButton('🗑️')
        btn_sil.clicked.connect(lambda _, btn=btn_sil: self.op_sil_buton(btn))
        self.op_listesi.setCellWidget(satir, 2, btn_sil)

        sabit_mi = self.radio_sabit.isChecked()
        self.op_data[op_isim] = (None, None, aciklama, sol, op, sag, sabit_mi)
        self.ops_dropdown_guncelle()

    def operasyon_sil(self, satir):
        op_isim_item = self.op_listesi.item(satir, 0)
        if not op_isim_item:
            self.ops_dropdown_guncelle()
            return

        op_isim = op_isim_item.text()

        #Bu Op'a bağımlı diğer Op'ları bul
        bagimli = []
        for diger_op, kayit in self.op_data.items():
            if diger_op == op_isim:
                continue
            aciklama = kayit[2]
            pattern = r'\b' + re.escape(op_isim.lower()) + r'\b'
            if re.search(pattern, aciklama.lower()):
                bagimli.append(diger_op)

        #op_data'dan sil — MATLAB listesinden de temizle
        silinecek_isimler = set(bagimli) | {op_isim}
        for sil_isim in silinecek_isimler:
            if sil_isim in self.op_data:
                sil_label = f"{sil_isim} {self.op_data[sil_isim][2]}"
                items = self.matlab_listesi.findItems(sil_label, Qt.MatchFlag.MatchExactly)
                for item in items:
                    self.matlab_listesi.takeItem(self.matlab_listesi.row(item))

                #CSV listesinden de temizle
                sil_label_csv = f"{sil_isim} {self.op_data[sil_isim][2]}"
                items_csv = self.csv_listesi.findItems(sil_label_csv, Qt.MatchFlag.MatchExactly)
                for item in items_csv:
                    self.csv_listesi.takeItem(self.csv_listesi.row(item))

                del self.op_data[sil_isim]

        satirlar_silinecek = []

        for row in range(self.op_listesi.rowCount()):
            item = self.op_listesi.item(row, 0)
            if item and item.text() in silinecek_isimler:
                satirlar_silinecek.append(row)

        #Sondan başa doğru sil
        for row in sorted(satirlar_silinecek, reverse=True):
            self.op_listesi.removeRow(row)

        if bagimli:
            QMessageBox.information(self, "Info",
                                    f"{op_isim} silindi. Bağımlı Op'lar da kaldırıldı: {', '.join(bagimli)}")
        yeni_op_data = {}
        for satir in range(self.op_listesi.rowCount()):
            item = self.op_listesi.item(satir, 0)
            if not item:
                continue
            eski_isim = item.text()
            yeni_isim = f"Op{satir + 1}"
            item.setText(yeni_isim)

            if eski_isim in self.op_data:
                kayit = self.op_data[eski_isim]
                #aciklama içinde eski isim geçiyorsa güncelle
                yeni_aciklama = kayit[2].replace(eski_isim, yeni_isim) if kayit[2] else kayit[2]
                yeni_op_data[yeni_isim] = (kayit[0], kayit[1], yeni_aciklama,
                                           kayit[3], kayit[4], kayit[5], kayit[6])

        self.op_data = yeni_op_data
        self._op_sayac = self.op_listesi.rowCount()

        self.ops_dropdown_guncelle()
        self.operasyonlari_uygula()

    def ifade_ekle(self):
        ifade = self.ifade_giris.text().strip()
        if not ifade:
            QMessageBox.warning(self, "Warning", "Please enter an expression.")
            return

        satir = self.op_listesi.rowCount()
        self._op_sayac += 1
        op_isim = f"Op{self._op_sayac}"

        self.op_listesi.insertRow(satir)
        self.op_listesi.setItem(satir, 0, QTableWidgetItem(op_isim))
        self.op_listesi.setItem(satir, 1, QTableWidgetItem(ifade))

        btn_sil = QPushButton('🗑️')
        btn_sil.clicked.connect(lambda _, btn=btn_sil: self.op_sil_buton(btn))
        self.op_listesi.setCellWidget(satir, 2, btn_sil)

        self.op_data[op_isim] = (None, None, ifade, None, None, None, False)
        self.ops_dropdown_guncelle()
        self.ifade_giris.clear()

    def ifadeyi_hesapla(self, op_isim, ifade):
        kucuk = ifade

        #U1, U2 referanslarını çöz
        u_refs = re.findall(r'[Uu](\d+)', kucuk)
        degiskenler = {}
        t_ref = None
        for num in dict.fromkeys(u_refs):
            isim = None
            for gercek, label in self.label_map.items():
                if label.startswith(f"U{num} ") or label.startswith(f"u{num} "):
                    isim = label
                    break
            if isim is None:
                QMessageBox.warning(self, "Warning", f"{op_isim}: U{num} not found.")
                return None, None
            t_arr, v_arr = self.operand_verisini_al(isim)
            if t_arr is None:
                return None, None
            degiskenler[f"U{num}"] = v_arr
            degiskenler[f"u{num}"] = v_arr
            if t_ref is None:
                t_ref = t_arr

        #Op1, Op2 referanslarını çöz
        op_refs = re.findall(r'[Oo]p(\d+)', kucuk)
        for num in dict.fromkeys(op_refs):
            op_key = f"Op{num}"
            if op_key not in self.op_data or self.op_data[op_key][0] is None:
                QMessageBox.warning(self, "Warning", f"{op_isim}: {op_key} has not been calculated yet.")
                return None, None
            t_arr, v_arr = self.op_data[op_key][0], self.op_data[op_key][1]
            degiskenler[f"Op{num}"] = v_arr
            degiskenler[f"op{num}"] = v_arr
            if t_ref is None:
                t_ref = t_arr

        if not degiskenler:
            QMessageBox.warning(self, "Warning", f"{op_isim}: No valid reference found in expression.")
            return None, None

        #Uzunlukları eşitle
        min_len = min(len(v) for v in degiskenler.values())
        degiskenler = {k: v[:min_len] for k, v in degiskenler.items()}
        t_ref = t_ref[:min_len]

        #Güvenli eval = sadece matematik fonksiyonlarına izin verir
        guvenli_ortam = {
            "__builtins__": {},
            "__import__": None,
            "np": np,
            "abs": abs, "round": round, "min": min, "max": max,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "sqrt": np.sqrt, "log": np.log, "log10": np.log10,
            "exp": np.exp, "pi": np.pi, "e": np.e,
        }

        guvenli_ortam.update(degiskenler)

        try:
            v_sonuc = eval(ifade, guvenli_ortam)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"{op_isim}: Expression could not be evaluated: {e}")
            return None, None

        return t_ref, np.asarray(v_sonuc, dtype=float)

    def op_sil_buton(self, btn):
        for row in range(self.op_listesi.rowCount()):
            if self.op_listesi.cellWidget(row, 2) == btn:
                self.operasyon_sil(row)
                break

    def operasyonlari_uygula(self):
        if not self.canvas.figure.axes:
            return
        ax = self.canvas.figure.axes[0]

        #Eski Op çizgilerini temizle
        for line in ax.get_lines()[:]:
            if getattr(line, "op_cizgisi", False):
                line.remove()

        #Legend sıfırla
        legend = ax.get_legend()
        if legend:
            legend.remove()

        #Toggle map temizle
        self.line_map = {}

        #csv_listesi ve matlab_listesi'nden eski Op itemlerini tamamen temizle
        for liste in [self.matlab_listesi, self.csv_listesi]:
            silinecek = []
            for i in range(liste.count()):
                item = liste.item(i)
                if item and re.match(r'^Op\d+ ', item.text()):
                    silinecek.append(i)
            for i in reversed(silinecek):
                liste.takeItem(i)

        yeni_op_lines = []

        for satir in range(self.op_listesi.rowCount()):
            op_isim_item = self.op_listesi.item(satir, 0)
            if not op_isim_item:
                continue
            op_isim = op_isim_item.text()
            if op_isim not in self.op_data:
                continue

            kayit = self.op_data[op_isim]
            _t, _v, aciklama, sol, op, sag, sabit_mi = kayit

            #Serbest ifade ise ayrı işle
            if sol is None:
                t_sonuc, v_sonuc = self.ifadeyi_hesapla(op_isim, aciklama)
                if t_sonuc is None:
                    continue
                self.op_data[op_isim] = (t_sonuc, v_sonuc, aciklama, None, None, None, False)
                line, = ax.step(t_sonuc, v_sonuc, where='post',
                                label=f"{op_isim} {aciklama}", linestyle='--')
                line.op_cizgisi = True
                yeni_op_lines.append(line)
                op_label = f"{op_isim} {aciklama}"
                mevcut_matlab = [self.matlab_listesi.item(i).text()
                                 for i in range(self.matlab_listesi.count())]
                if op_label not in mevcut_matlab:
                    self.matlab_listesi.addItem(op_label)
                    self.csv_listesi.addItem(op_label)
                continue

            t_sol, v_sol = self.operand_verisini_al(sol)
            if t_sol is None:
                QMessageBox.warning(self, "Warning", f"{op_isim}: '{sol}' data not found.")
                continue

            if sabit_mi:
                try:
                    sabit = float(sag)
                except ValueError:
                    QMessageBox.warning(self, "Warning", f"{op_isim}: Invalid constant '{sag}'.")
                    continue
                t_sonuc = t_sol
                v_sag = sabit
            else:
                t_sag, v_sag_arr = self.operand_verisini_al(sag)
                if t_sag is None:
                    QMessageBox.warning(self, "Warning", f"{op_isim}: '{sag}' data not found.")
                    continue
                min_len = min(len(t_sol), len(t_sag))
                t_sol = t_sol[:min_len]
                v_sol = v_sol[:min_len]
                v_sag = v_sag_arr[:min_len]
                t_sonuc = t_sol

            try:
                if op == '+':
                    v_sonuc = v_sol + v_sag
                elif op == '-':
                    v_sonuc = v_sol - v_sag
                elif op == '*':
                    v_sonuc = v_sol * v_sag
                elif op == '/':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        v_sonuc = np.where(v_sag != 0, v_sol / v_sag, np.nan)
                else:
                    continue
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"{op_isim} could not be calculated: {e}")
                continue

            self.op_data[op_isim] = (t_sonuc, v_sonuc, aciklama, sol, op, sag, sabit_mi)

            line, = ax.step(t_sonuc, v_sonuc, where='post',
                            label=f"{op_isim} {aciklama}", linestyle='--')
            line.op_cizgisi = True
            yeni_op_lines.append(line)

            mevcut_matlab = [self.matlab_listesi.item(i).text()
                             for i in range(self.matlab_listesi.count())]
            op_label = f"{op_isim} {aciklama}"
            if op_label not in mevcut_matlab:
                self.matlab_listesi.addItem(op_label)
                self.csv_listesi.addItem(op_label)

        u_lines = [l for l in ax.get_lines() if not getattr(l, "op_cizgisi", False)]

        #Legend oluşturmayı enable_legend_toggle'a bırak çift oluşturma olmasın
        self.enable_legend_toggle(u_lines + yeni_op_lines, self.scatter_list)
        ax.relim()
        ax.autoscale_view()
        self.toolbar.update()
        self.canvas.draw_idle()

    def on_pick(self, event):
        artist = event.artist
        if artist in self.line_map:
            orig = self.line_map[artist]
            orig.set_visible(not orig.get_visible())
            self.canvas.draw_idle()

    def canvas_double_click(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toolbar.home()

    def closeEvent(self, event):
        if self in self.parent_ref.acik_grafikler:
            self.parent_ref.acik_grafikler.remove(self)

        #Matplotlib figure'ı RAM'den temizle
        try:
            fig = self.canvas.figure
            fig.clear()
            plt.close(fig)
        except (AttributeError, TypeError):
            pass

        #Büyük veri referanslarını serbest bırak
        self.plotted_data.clear()
        self.scatter_data.clear()
        self.op_data.clear()
        self.label_map.clear()
        self.ters_label_map.clear()
        self.lines.clear()
        self.line_map.clear()
        self.scatter_list.clear()

        super().closeEvent(event)

    def legend_sag_tik(self, event):
        if event.button != 3:
            return

        ax = self.canvas.figure.axes[0]
        legend = ax.get_legend()
        if legend is None:
            return

        hedef_text = None
        for text in legend.get_texts():
            bbox = text.get_window_extent()
            if bbox.contains(event.x, event.y):
                hedef_text = text
                break

        if hedef_text is None:
            return

        #Menüden önce kontrol et
        u_ismi = hedef_text.get_text()
        gercek_label = self.ters_label_map.get(u_ismi)
        if not gercek_label or ' | ' not in gercek_label:
            return

        menu = QMenu(self)
        alias_action = menu.addAction("Edit Alias")
        delete_alias_action = menu.addAction("Delete Alias")
        action = menu.exec(QCursor.pos())

        #Mouse yapışmasını çöz
        release_event = MouseEvent(
            'button_release_event', self.canvas,
            event.x, event.y, button=MouseButton.RIGHT
        )

        self.canvas.callbacks.process('button_release_event', release_event)

        if action == delete_alias_action:
            dosya, kolon = gercek_label.split(' | ', 1)
            alias_map = self.parent_ref.alias_map
            ters_alias = {v: k for k, v in alias_map.items()}
            orijinal_dosya = ters_alias.get(dosya, dosya)
            mevcut_gosterim = alias_map.get(orijinal_dosya, orijinal_dosya)
            if orijinal_dosya in alias_map:
                del alias_map[orijinal_dosya]

                #legend, label_map, ters_label_map, matlab, ops güncelle
                for t in legend.get_texts():
                    if mevcut_gosterim in t.get_text():
                        t.set_text(t.get_text().replace(mevcut_gosterim, orijinal_dosya))
                yeni_ters_map = {}
                for u_label, g_label in self.ters_label_map.items():
                    yeni_u_label = u_label.replace(mevcut_gosterim, orijinal_dosya,
                                                   1) if mevcut_gosterim in u_label else u_label
                    yeni_ters_map[yeni_u_label] = g_label
                self.ters_label_map = yeni_ters_map
                yeni_label_map = {}
                for g_label, u_label in self.label_map.items():
                    yeni_u_label = u_label.replace(mevcut_gosterim, orijinal_dosya,
                                                   1) if mevcut_gosterim in u_label else u_label
                    yeni_label_map[g_label] = yeni_u_label
                self.label_map = yeni_label_map

                for line in ax.get_lines():
                    lbl = str(line.get_label())
                    if mevcut_gosterim in lbl:
                        line.set_label(lbl.replace(mevcut_gosterim, orijinal_dosya, 1))

                for coll in ax.collections:
                    lbl = str(coll.get_label())
                    if mevcut_gosterim in lbl:
                        coll.set_label(lbl.replace(mevcut_gosterim, orijinal_dosya, 1))
                    if hasattr(coll, 'gercek_label') and mevcut_gosterim in str(coll.gercek_label):
                        coll.gercek_label = coll.gercek_label.replace(mevcut_gosterim, orijinal_dosya, 1)
                yeni_op_data = {}

                for op_isim, kayit in self.op_data.items():
                    t, v, aciklama, sol, op_char, sag, sabit_mi = kayit
                    yeni_op_data[op_isim] = (
                        t, v,
                        str(aciklama).replace(mevcut_gosterim, orijinal_dosya, 1) if mevcut_gosterim in str(
                            aciklama or '') else aciklama,
                        str(sol).replace(mevcut_gosterim, orijinal_dosya, 1) if sol and mevcut_gosterim in str(
                            sol) else sol,
                        op_char,
                        str(sag).replace(mevcut_gosterim, orijinal_dosya,
                                         1) if sag and not sabit_mi and mevcut_gosterim in str(sag or '') else sag,
                        sabit_mi
                    )

                self.op_data = yeni_op_data
                self.ops_dropdown_guncelle()

                for i in range(self.parent_ref.parametre_listesi.count()):
                    pitem = self.parent_ref.parametre_listesi.item(i)
                    if pitem and mevcut_gosterim in pitem.text():
                        pitem.setText(pitem.text().replace(mevcut_gosterim, orijinal_dosya, 1))
                self.parent_ref.alias_kaydet(eski_alias=mevcut_gosterim, orijinal=orijinal_dosya)
                self.canvas.figure.canvas.draw()
            return

        if action != alias_action:
            return

        u_ismi = hedef_text.get_text()
        gercek_label = self.ters_label_map.get(u_ismi)
        if not gercek_label or ' | ' not in gercek_label:
            return

        dosya, kolon = gercek_label.split(' | ', 1)
        alias_map = self.parent_ref.alias_map
        ters_alias = {v: k for k, v in alias_map.items()}
        orijinal_dosya = ters_alias.get(dosya, dosya)
        mevcut = alias_map.get(orijinal_dosya, "")
        mevcut_gosterim = alias_map.get(orijinal_dosya, orijinal_dosya)

        yeni, ok = QInputDialog.getText(self, "Edit Alias",
                                        f"Alias for '{orijinal_dosya}':", text=mevcut)
        if not ok or not yeni.strip():
            return

        for d, a in alias_map.items():
            if a == yeni.strip() and d != orijinal_dosya:
                QMessageBox.warning(self, "Warning",
                                    f"'{yeni.strip()}' is already used for '{d}'.")
                return

        yeni_alias = yeni.strip()
        alias_map[orijinal_dosya] = yeni_alias

        kullan = self.parent_ref.radio_use_alias.isChecked()

        #legend text güncelle
        if kullan:
            for t in legend.get_texts():
                if mevcut_gosterim in t.get_text():
                    t.set_text(t.get_text().replace(mevcut_gosterim, yeni_alias))

        #ters_label_map güncelle
        if kullan:
            yeni_ters_map = {}
            for u_label, g_label in self.ters_label_map.items():
                yeni_u_label = u_label.replace(mevcut_gosterim, yeni_alias) if mevcut_gosterim in u_label else u_label
                yeni_ters_map[yeni_u_label] = g_label
            self.ters_label_map = yeni_ters_map

            #label_map güncelle
            yeni_label_map = {}
            for g_label, u_label in self.label_map.items():
                yeni_u_label = u_label.replace(mevcut_gosterim, yeni_alias) if mevcut_gosterim in u_label else u_label
                yeni_label_map[g_label] = yeni_u_label
            self.label_map = yeni_label_map

        #matlab_listesi güncelle
        if kullan:
            for i in range(self.matlab_listesi.count()):
                mitem = self.matlab_listesi.item(i)
                if mitem and mevcut_gosterim in mitem.text():
                    mitem.setText(mitem.text().replace(mevcut_gosterim, yeni_alias))

        #CSV listesi güncelle
        if kullan:
            for i in range(self.csv_listesi.count()):
                citem = self.csv_listesi.item(i)
                if citem and mevcut_gosterim in citem.text():
                    citem.setText(citem.text().replace(mevcut_gosterim, yeni_alias))

        #ops yeniden uygulanınca kaybolmasın
        if kullan:
            for line in ax.get_lines():
                lbl = str(line.get_label())
                if mevcut_gosterim in lbl:
                    line.set_label(lbl.replace(mevcut_gosterim, yeni_alias))

        yeni_op_data = {}
        for op_isim, kayit in self.op_data.items():
            t, v, aciklama, sol, op_char, sag, sabit_mi = kayit
            yeni_aciklama = aciklama.replace(mevcut_gosterim, yeni_alias) if mevcut_gosterim in (
                    aciklama or '') else aciklama
            yeni_sol = sol.replace(mevcut_gosterim, yeni_alias) if sol and mevcut_gosterim in sol else sol
            yeni_sag = sag.replace(mevcut_gosterim, yeni_alias) if sag and not sabit_mi and mevcut_gosterim in (
                    sag or '') else sag
            yeni_op_data[op_isim] = (t, v, yeni_aciklama, yeni_sol, op_char, yeni_sag, sabit_mi)
        self.op_data = yeni_op_data

        #op_listesi UI'ını güncelle
        if kullan:
            for row in range(self.op_listesi.rowCount()):
                aciklama_item = self.op_listesi.item(row, 1)
                if aciklama_item and mevcut_gosterim in aciklama_item.text():
                    aciklama_item.setText(aciklama_item.text().replace(mevcut_gosterim, yeni_alias))

        #ops dropdown güncelle
        self.ops_dropdown_guncelle()

        #parametre_listesi güncelle
        if kullan:
            for i in range(self.parent_ref.parametre_listesi.count()):
                pitem = self.parent_ref.parametre_listesi.item(i)
                if pitem and mevcut_gosterim in pitem.text():
                    pitem.setText(pitem.text().replace(mevcut_gosterim, yeni_alias))

        #ax'taki Line2D nesnelerinin label'larını güncelle
        if kullan:
            for line in ax.get_lines():
                lbl = str(line.get_label())
                if mevcut_gosterim in lbl:
                    line.set_label(lbl.replace(mevcut_gosterim, yeni_alias))

        #scatter nesnelerinin label ve gercek_label'larını güncelle
        if kullan:
            for coll in ax.collections:
                lbl = str(coll.get_label())
                if mevcut_gosterim in lbl:
                    coll.set_label(lbl.replace(mevcut_gosterim, yeni_alias))
                if hasattr(coll, 'gercek_label') and mevcut_gosterim in str(coll.gercek_label):
                    coll.gercek_label = coll.gercek_label.replace(mevcut_gosterim, yeni_alias)

        #scatter_data dict'ini güncelle
        if kullan:
            yeni_scatter_data = {}
            for lbl, deger in self.scatter_data.items():
                yeni_lbl = lbl.replace(mevcut_gosterim, yeni_alias) if mevcut_gosterim in lbl else lbl
                yeni_scatter_data[yeni_lbl] = deger
            self.scatter_data = yeni_scatter_data

        #kaydet ve yenile
        self.parent_ref.alias_kaydet(eski_alias=mevcut_gosterim, orijinal=orijinal_dosya)

        self.canvas.figure.canvas.draw()

    def raporu_guncelle(self, error_results):
        satirlar = []
        for label, data in error_results.items():
            t_arr = np.asarray(data["t"])
            v_arr = np.asarray(data["values"])
            parcalar = label.split(' | ', 1)
            dosya_adi = parcalar[0] if len(parcalar) > 1 else label
            kolon_adi = parcalar[1] if len(parcalar) > 1 else label
            gosterim_adi = f"{dosya_adi} | {kolon_adi}" if len(parcalar) > 1 else label
            for error_type, indices in data["errors"].items():
                for idx in indices:
                    satirlar.append((gosterim_adi, t_arr[idx], v_arr[idx], error_type))

        #Hata tipine göre sırala
        hata_sirasi = {"OVERSHOOT": 0, "SPIKE": 1, "OUT OF RANGE": 2, "CONSTANT OUTPUT": 3, "INVALID SIGNAL": 4}
        satirlar.sort(key=lambda x: hata_sirasi.get(x[3], 99))

        self.rapor_tablo.setSortingEnabled(False)
        self.rapor_tablo.setRowCount(len(satirlar))
        for row, (gosterim_adi, t, v, etype) in enumerate(satirlar):
            self.rapor_tablo.setItem(row, 0, QTableWidgetItem(gosterim_adi))
            self.rapor_tablo.setItem(row, 1, QTableWidgetItem(f"{float(t):.3f}"))
            self.rapor_tablo.setItem(row, 2, QTableWidgetItem(f"{float(v):.6f}"))
            self.rapor_tablo.setItem(row, 3, QTableWidgetItem(etype))
        self.rapor_tablo.setSortingEnabled(True)

    def matlab_export(self):
        secili = [item.text() for item in self.matlab_listesi.selectedItems()]

        if not secili:
            QMessageBox.warning(self, "Warning", "Please select at least one parameter.")
            return

        dosya_yolu, _ = QFileDialog.getSaveFileName(self, "Save .mat file", "", "MATLAB Files (*.mat)")
        if not dosya_yolu:
            return

        mat_dict = {}

        def temiz_isim_uret(isim, mevcut_dict):
            temiz = isim.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_").replace(".",
                                                                                                       "_").replace("-",
                                                                                                                    "_").replace(
                "+", "plus").replace("*", "mul").replace("(", "").replace(")", "").replace(",", "_")
            if len(temiz) > 61:
                temiz = temiz[:61]

            #Çakışma varsa sonuna sayı ekle.Ek sonrası da 61 karakter sınırını koru
            aday = temiz
            sayac = 1
            while f"{aday}_t" in mevcut_dict:
                ek = f"_{sayac}"
                aday = temiz[:61 - len(ek)] + ek
                sayac += 1
            return aday

        for label, (t_arr, v_arr) in self.plotted_data.items():
            u_ismi = self.label_map.get(label, label) if hasattr(self, 'label_map') else label
            if u_ismi in secili:
                temiz_isim = temiz_isim_uret(u_ismi, mat_dict)
                mat_dict[f"{temiz_isim}_t"] = np.asarray(t_arr)
                mat_dict[f"{temiz_isim}_v"] = np.asarray(v_arr)

        if hasattr(self, 'scatter_data'):
            for label, (t_arr, v_arr) in self.scatter_data.items():
                if label in secili:
                    temiz_isim = temiz_isim_uret(label, mat_dict)
                    mat_dict[f"{temiz_isim}_t"] = np.array(t_arr)
                    mat_dict[f"{temiz_isim}_v"] = np.array(v_arr)

        if hasattr(self, 'op_data'):
            for op_isim, kayit in self.op_data.items():
                op_label = f"{op_isim} {kayit[2]}"
                if op_label in secili and kayit[0] is not None:
                    temiz_isim = temiz_isim_uret(op_label, mat_dict)
                    mat_dict[f"{temiz_isim}_t"] = np.asarray(kayit[0])
                    mat_dict[f"{temiz_isim}_v"] = np.asarray(kayit[1])

        mat_dict['labels'] = np.array(secili, dtype=object)
        savemat(dosya_yolu, mat_dict)
        QMessageBox.information(self, "Info", "Export completed.")

    def csv_export(self):
        secili = [item.text() for item in self.csv_listesi.selectedItems()]

        if not secili:
            QMessageBox.warning(self, "Warning", "Please select at least one parameter.")
            return

        dosya_yolu, _ = QFileDialog.getSaveFileName(self, "Save .csv file", "", "CSV Files (*.csv)")
        if not dosya_yolu:
            return
        if not dosya_yolu.endswith('.csv'):
            dosya_yolu += '.csv'

        #Sisteme göre otomatik ayırıcı seç
        try:
            locale.setlocale(locale.LC_ALL, '')
            conv = locale.localeconv()
            decimal_point = conv.get('decimal_point', '.')
            sep = ';' if decimal_point == ',' else ','
        except (locale.Error, KeyError):
            sep = ';'

        parcalar = []

        for label, (t_arr, v_arr) in self.plotted_data.items():
            u_ismi = self.label_map.get(label, label)

            if u_ismi in secili:
                df_par = pd.DataFrame({
                    f"{u_ismi}_t": pd.Series(np.asarray(t_arr)),
                    f"{u_ismi}_v": pd.Series(np.asarray(v_arr))
                })
                parcalar.append(df_par)

        if hasattr(self, 'scatter_data'):
            for label, (t_arr, v_arr) in self.scatter_data.items():
                if label in secili:
                    df_par = pd.DataFrame({
                        f"{label}_t": pd.Series(np.array(t_arr)),
                        f"{label}_v": pd.Series(np.array(v_arr))
                    })
                    parcalar.append(df_par)

        if hasattr(self, 'op_data'):
            for op_isim, kayit in self.op_data.items():
                op_label = f"{op_isim} {kayit[2]}"
                if op_label in secili and kayit[0] is not None:
                    df_par = pd.DataFrame({
                        f"{op_label}_t": pd.Series(np.asarray(kayit[0])),
                        f"{op_label}_v": pd.Series(np.asarray(kayit[1]))
                    })
                    parcalar.append(df_par)

        if parcalar:
            df_final = pd.concat(parcalar, axis=1)

            #İsim çakışması kontrolü
            if os.path.exists(dosya_yolu):
                base, ext = os.path.splitext(dosya_yolu)
                sayac = 1
                while os.path.exists(dosya_yolu):
                    dosya_yolu = f"{base}_{sayac}{ext}"
                    sayac += 1

            df_final.to_csv(
                dosya_yolu,
                index=False,
                sep=sep,
                encoding='utf-8-sig',
            )
            QMessageBox.information(self, "Info", f"Export completed.\nSaved as: {os.path.basename(dosya_yolu)}")


class AnaPencere(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CSV Flight Data Analysis✈️')
        self.resize(1100, 650)
        self.acik_grafikler = []

        self.klasor = None
        self.data_cache = {}
        self.kolon_cache = {}
        self.alias_map = {}

        self.error_results = {}
        self.error_loader = None

        self.error_marker_map = {
            "CONSTANT OUTPUT": "o",
            "OVERSHOOT": "x",
            "SPIKE": "v",
            "OUT OF RANGE": "s"
        }

        hata_butonlari = QHBoxLayout()
        plot_butonlari = QHBoxLayout()
        figure_butonlari = QHBoxLayout()

        left_panel = QVBoxLayout()
        center_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        #Arama kutuları
        self.dosya_Arama = QLineEdit()
        self.dosya_Arama.setPlaceholderText('Search File')
        self.dosya_Arama.textChanged.connect(self.dosya_ara)

        self.parametre_arama = QLineEdit()
        self.parametre_arama.setPlaceholderText('Search Variable')
        self.parametre_arama.textChanged.connect(self.parametre_ara)

        #Dosya alanı
        self.btn_Klasor_sec = QPushButton('🗂️Select CSV File')
        self.dosya_listesi = QListWidget()
        self.btndosya_temizle = QPushButton('🧹Clear File Selection')
        self.parametre_listesi = QListWidget()

        self.dosya_listesi.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.btnsecilenleri_getir = QPushButton('Get Vars')
        self.parametre_listesi.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.alias_radio_group = QButtonGroup()
        self.alias_radio_group.setExclusive(True)
        self.radio_use_alias = QRadioButton('Use ALias')
        self.radio_no_alias = QRadioButton("Dont't Use Alias")
        self.alias_radio_group.addButton(self.radio_use_alias)
        self.alias_radio_group.addButton(self.radio_no_alias)
        self.radio_use_alias.setChecked(True)
        alias_radio_layout = QHBoxLayout()
        alias_radio_layout.addWidget(self.radio_use_alias)
        alias_radio_layout.addWidget(self.radio_no_alias)
        self.btn_parametre_temizle = QPushButton('🧹Clear Variable Selection')

        self.parametre_listesi.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.parametre_listesi.customContextMenuRequested.connect(self.parametre_sag_tik)

        #Figure paneli
        self.figure_listesi = QTreeWidget()
        self.figure_listesi.setHeaderHidden(True)
        self.figure_listesi.setColumnCount(2)
        self.btn_add_to_figure = QPushButton('➡️ Add To Figure')
        self.btn_plot = QPushButton('📈 Plot')
        self.btn_new_figure = QPushButton('➕ New Figure')
        self.btn_analyze_errors = QPushButton("🔎 Analyze Variable For Errors")
        self.btn_error_plot = QPushButton("Error Plot")
        self.btn_delete_figure = QPushButton('🗑️ Delete Figure')

        #Buton bağlantıları
        self.btn_Klasor_sec.clicked.connect(self.klasor_sec)
        self.btndosya_temizle.clicked.connect(self.dosya_secimi_temizle)
        self.btnsecilenleri_getir.clicked.connect(self.secilenleri_getir)
        self.alias_radio_group.buttonClicked.connect(self.alias_modu_degisti)
        self.btn_parametre_temizle.clicked.connect(self.parametre_secimi_temizle)
        self.btn_new_figure.clicked.connect(self.yeni_figur_olustur)
        self.btn_plot.clicked.connect(self.plot_bas)
        self.btn_delete_figure.clicked.connect(self.figure_sil)
        self.figure_listesi.currentItemChanged.connect(
            lambda current, previous: self.figure_degisti(current) if current else None)
        self.btn_add_to_figure.clicked.connect(self.parametreyi_figure_ekle)
        self.btn_analyze_errors.clicked.connect(self.analyze_errors)
        self.btn_error_plot.clicked.connect(self.error_plot)

        #Mode seçici
        self.mode_groupbox = QGroupBox('Time Mode')
        self.radio_realtime = QRadioButton('Realtime')
        self.radio_previous = QRadioButton('Previous')
        self.radio_nearest = QRadioButton('Nearest')
        self.radio_realtime.setChecked(True)

        self.mode_buttons = QButtonGroup()
        self.mode_buttons.addButton(self.radio_realtime)
        self.mode_buttons.addButton(self.radio_previous)
        self.mode_buttons.addButton(self.radio_nearest)
        self.mode_buttons.buttonClicked.connect(self.mode_degisti)

        mode_layout = QVBoxLayout()
        mode_layout.addWidget(self.radio_realtime)
        mode_layout.addWidget(self.radio_previous)
        mode_layout.addWidget(self.radio_nearest)
        self.mode_groupbox.setLayout(mode_layout)

        #Figure sistemi
        self.figures = {1: {'params': [], 'mode': 'realtime', 'units': []}}
        self.current_figure = 1
        self.figure_listesini_guncelle()

        #LEFT PANEL
        left_panel.addWidget(QLabel('CSV File in the Folder'))
        left_panel.addWidget(self.btn_Klasor_sec)
        left_panel.addWidget(QLabel('Search File'))
        left_panel.addWidget(self.dosya_Arama)
        left_panel.addWidget(self.dosya_listesi)
        left_panel.addWidget(self.btndosya_temizle)

        #CENTER PANEL
        center_panel.addWidget(QLabel('Selected Variables'))
        center_panel.addWidget(self.btnsecilenleri_getir)
        center_panel.addLayout(alias_radio_layout)
        center_panel.addWidget(self.parametre_arama)
        center_panel.addWidget(self.parametre_listesi)
        center_panel.addWidget(self.btn_parametre_temizle)

        #RIGHT PANEL
        right_panel.addWidget(self.mode_groupbox)
        right_panel.addWidget(QLabel('List of Figures'))
        right_panel.addWidget(self.figure_listesi)
        plot_butonlari.addWidget(self.btn_add_to_figure)
        plot_butonlari.addWidget(self.btn_plot)
        right_panel.addLayout(plot_butonlari)
        figure_butonlari.addWidget(self.btn_new_figure)
        figure_butonlari.addWidget(self.btn_delete_figure)
        right_panel.addLayout(figure_butonlari)
        hata_butonlari.addWidget(self.btn_analyze_errors)
        hata_butonlari.addWidget(self.btn_error_plot)
        right_panel.addLayout(hata_butonlari)

        #ANA LAYOUT
        mainlayout = QHBoxLayout()
        mainlayout.addLayout(left_panel, 2)
        mainlayout.addLayout(center_panel, 2)
        mainlayout.addLayout(right_panel, 3)
        self.setLayout(mainlayout)

    #FONKSİYONLAR
    def klasor_sec(self):
        klasor = QFileDialog.getExistingDirectory(self, 'Select File')
        if not klasor:
            return

        self.klasor = klasor
        self.data_cache.clear()
        self.kolon_cache.clear()
        self.alias_map.clear()

        for pencere in self.acik_grafikler[:]:
            pencere.close()
        self.error_results.clear()

        try:
            from error_analyzer import ErrorClassLoader
            uygulama_klasoru = get_app_dir()
            xlsx_listesi = [
                f for f in os.listdir(uygulama_klasoru)
                if 'ERROR_CLASS' in f.upper() and f.upper().endswith('.XLSX')
            ]
            if not xlsx_listesi:
                klasordeki_dosyalar = os.listdir(uygulama_klasoru)[:20]
                raise FileNotFoundError(
                    f"No *ERROR_CLASS*.xlsx found in:\n{uygulama_klasoru}\n\n"
                    f"Files there:\n" + "\n".join(klasordeki_dosyalar)
                )
            self.error_loader = ErrorClassLoader(uygulama_klasoru)
        except Exception as e:
            self.error_loader = None
            QMessageBox.warning(self, "Warning", f"Error class files could not be loaded:\n{e}")

        #Alias dosyasını yükle
        uygulama_dizini = get_writable_dir()
        alias_yolu = os.path.join(uygulama_dizini, "data", "aliases.json")
        if os.path.exists(alias_yolu):
            try:
                with open(alias_yolu, 'r', encoding='utf-8') as f:
                    self.alias_map = json.load(f)
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Alias file could not be loaded: {e}")

        self.dosya_listesi.clear()
        for f in os.listdir(self.klasor):
            if f.endswith('.csv'):
                self.dosya_listesi.addItem(f)
        self.dosya_listesi.scrollToTop()

    #DOSYA / ARAMA
    def dosya_secimi_temizle(self):
        self.dosya_listesi.clearSelection()

    def dosya_ara(self, text):
        for i in range(self.dosya_listesi.count()):
            item = self.dosya_listesi.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    #PARAMETRELER
    def secilenleri_getir(self):
        self.parametre_listesi.clear()
        secili_dosyalar = [
            self.dosya_listesi.item(i).text()
            for i in range(self.dosya_listesi.count())
            if self.dosya_listesi.item(i).isSelected()
        ]
        if not secili_dosyalar:
            QMessageBox.warning(self, 'Warning', 'You did not select a file!')
            return
        self.parametre_listesi.scrollToTop()

        yuklenemeyen = []
        for dosya in secili_dosyalar:
            dosya_yolu = os.path.join(self.klasor, dosya)
            try:
                if dosya not in self.kolon_cache:
                    df_cols = pd.read_csv(dosya_yolu, skiprows=1, sep='\t', nrows=0)
                    self.kolon_cache[dosya] = list(df_cols.columns)
                for kolon in self.kolon_cache[dosya]:
                    if self.radio_use_alias.isChecked():
                        dosya_adi = self.alias_map.get(dosya, dosya)
                    else:
                        dosya_adi = dosya
                    self.parametre_listesi.addItem(f'{dosya_adi} | {kolon}')
            except Exception as e:
                yuklenemeyen.append(f"{dosya}: {str(e)}")

        if yuklenemeyen:
            QMessageBox.warning(
                self, 'Partial Load',
                'Some files unavailable:\n' + '\n'.join(yuklenemeyen))

    def alias_modu_degisti(self):
        kullan = self.radio_use_alias.isChecked()
        ters_alias = {v: k for k, v in self.alias_map.items()}

        #Parametre listesini güncelle
        if self.parametre_listesi.count() > 0:
            self.secilenleri_getir()

        #Figure params güncelle
        for fig_no, fig_data in self.figures.items():
            yeni_params = []
            for p in fig_data['params']:
                if ' | ' not in p:
                    yeni_params.append(p)
                    continue
                dosya_adi, kolon = p.split(' | ', 1)
                orijinal = ters_alias.get(dosya_adi, dosya_adi)
                gosterim = self.alias_map.get(orijinal, orijinal) if kullan else orijinal
                yeni_params.append(f'{gosterim} | {kolon}')
            fig_data['params'] = yeni_params
        self.figure_listesini_guncelle()

        #Açık grafik pencerelerini güncelle
        for pencere in self.acik_grafikler:
            ax = pencere.canvas.figure.axes[0]

            def guncelle(metin):
                if not metin:
                    return metin
                for original, alias in self.alias_map.items():
                    if kullan:
                        metin = metin.replace(original, alias)
                    else:
                        metin = metin.replace(alias, original)
                return metin

            #matlab_listesi güncelle
            for i in range(pencere.matlab_listesi.count()):
                item = pencere.matlab_listesi.item(i)
                if item:
                    item.setText(guncelle(item.text()))

            #CSV güncelle
            for i in range(pencere.csv_listesi.count()):
                item = pencere.csv_listesi.item(i)
                if item:
                    item.setText(guncelle(item.text()))

            #label_map ve ters_label_map güncelle
            yeni_label_map = {}
            for g_label, u_label in pencere.label_map.items():
                yeni_label_map[g_label] = guncelle(u_label)
            pencere.label_map = yeni_label_map
            pencere.ters_label_map = {v: k for k, v in yeni_label_map.items()}

            #op_data güncelle
            yeni_op_data = {}
            for op_isim, kayit in pencere.op_data.items():
                t, v, aciklama, sol, op_char, sag, sabit_mi = kayit
                yeni_op_data[op_isim] = (
                    t, v,
                    guncelle(aciklama),
                    guncelle(sol),
                    op_char,
                    guncelle(sag) if not sabit_mi else sag,
                    sabit_mi
                )
            pencere.op_data = yeni_op_data
            pencere.ops_dropdown_guncelle()

            #op_listesi güncelle
            for row in range(pencere.op_listesi.rowCount()):
                item = pencere.op_listesi.item(row, 1)
                if item:
                    item.setText(guncelle(item.text()))

            #Legend ve line label'ları güncelle
            legend = ax.get_legend()
            if legend:
                for t in legend.get_texts():
                    t.set_text(guncelle(t.get_text()))

            for line in ax.get_lines():
                line.set_label(guncelle(str(line.get_label())))

            #scatter güncelle
            for coll in ax.collections:
                coll.set_label(guncelle(str(coll.get_label())))
                if hasattr(coll, 'gercek_label'):
                    coll.gercek_label = guncelle(str(coll.gercek_label))

            #scatter_data güncelle
            yeni_scatter_data = {}
            for lbl, deger in pencere.scatter_data.items():
                yeni_scatter_data[guncelle(lbl)] = deger
            pencere.scatter_data = yeni_scatter_data

            pencere.canvas.draw_idle()

    def parametre_ara(self, text):
        for i in range(self.parametre_listesi.count()):
            item = self.parametre_listesi.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def parametre_secimi_temizle(self):
        self.parametre_listesi.clearSelection()

    def parametre_sag_tik(self, pos):
        item = self.parametre_listesi.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        alias_action = menu.addAction("Edit Alias")
        delete_alias_action = menu.addAction("Delete Alias")
        action = menu.exec(self.parametre_listesi.mapToGlobal(pos))

        if action == alias_action:
            metin = item.text()
            if ' | ' not in metin:
                return
            dosya_adi, kolon = metin.split(' | ', 1)
            ters_alias = {v: k for k, v in self.alias_map.items()}
            orijinal_dosya = ters_alias.get(dosya_adi, dosya_adi)

            mevcut = self.alias_map.get(orijinal_dosya, "")
            yeni, ok = QInputDialog.getText(self, "Add / Edit Alias",
                                            f"Alias for '{orijinal_dosya}':", text=mevcut)
            if not ok or not yeni.strip():
                return

            #Yeni alias başka bir dosyada kullanılıyor mu kontrol et
            for dosya, alias in self.alias_map.items():
                if alias == yeni.strip() and dosya != orijinal_dosya:
                    QMessageBox.warning(self, "Warning",
                                        f"'{yeni.strip()}' is already used for '{dosya}'. Please choose a different alias.")
                    return

            self.alias_map[orijinal_dosya] = yeni.strip()
            self.alias_kaydet(eski_alias=dosya_adi, orijinal=orijinal_dosya)
            self.secilenleri_getir()

        elif action == delete_alias_action:
            metin = item.text()
            if ' | ' not in metin:
                return
            dosya_adi = metin.split(' | ', 1)[0]
            ters_alias = {v: k for k, v in self.alias_map.items()}
            orijinal_dosya = ters_alias.get(dosya_adi, dosya_adi)
            if orijinal_dosya in self.alias_map:
                del self.alias_map[orijinal_dosya]
                self.alias_kaydet(eski_alias=dosya_adi, orijinal=orijinal_dosya)
                self.secilenleri_getir()

    #ALİAS KAYDET
    def alias_kaydet(self, eski_alias=None, orijinal=None):

        if not self.klasor:
            return

        uygulama_dizini = get_writable_dir()
        alias_yolu = os.path.join(uygulama_dizini, "data", "aliases.json")
        os.makedirs(os.path.dirname(alias_yolu), exist_ok=True)
        with open(alias_yolu, 'w', encoding='utf-8') as f:
            json.dump(self.alias_map, f, ensure_ascii=False, indent=2)

        #Figure listesindeki eski adları güncelle
        if eski_alias and orijinal:
            yeni_ad = self.alias_map.get(orijinal, orijinal) if self.radio_use_alias.isChecked() else orijinal

            #Figure params güncelle
            for fig_no, fig_data in self.figures.items():
                fig_data['params'] = [
                    f"{yeni_ad} | {p.split(' | ', 1)[1]}" if p.startswith(f"{eski_alias} | ") else p
                    for p in fig_data['params']
                ]

            #Açık grafik pencerelerini güncelle
            for pencere in self.acik_grafikler:
                ax = pencere.canvas.figure.axes[0]

                #matlab_listesi
                for i in range(pencere.matlab_listesi.count()):
                    mitem = pencere.matlab_listesi.item(i)
                    if mitem and eski_alias in mitem.text():
                        mitem.setText(mitem.text().replace(eski_alias, yeni_ad))

                #CSV güncelle
                for i in range(pencere.csv_listesi.count()):
                    citem = pencere.csv_listesi.item(i)
                    if citem and eski_alias in citem.text():
                        citem.setText(citem.text().replace(eski_alias, yeni_ad))

                #op_listesi
                for row in range(pencere.op_listesi.rowCount()):
                    aciklama_item = pencere.op_listesi.item(row, 1)
                    if aciklama_item and eski_alias in aciklama_item.text():
                        aciklama_item.setText(aciklama_item.text().replace(eski_alias, yeni_ad))

                #op_data
                yeni_op_data = {}
                for op_isim, kayit in pencere.op_data.items():
                    t, v, aciklama, sol, op_char, sag, sabit_mi = kayit
                    yeni_op_data[op_isim] = (
                        t, v,
                        aciklama.replace(eski_alias, yeni_ad) if aciklama and eski_alias in aciklama else aciklama,
                        sol.replace(eski_alias, yeni_ad) if sol and eski_alias in sol else sol,
                        op_char,
                        sag.replace(eski_alias, yeni_ad) if sag and not sabit_mi and eski_alias in (sag or '') else sag,
                        sabit_mi
                    )
                pencere.op_data = yeni_op_data
                pencere.ops_dropdown_guncelle()

                #label_map ve ters_label_map
                yeni_label_map = {}
                for g_label, u_label in pencere.label_map.items():
                    yeni_label_map[g_label] = u_label.replace(eski_alias, yeni_ad) if eski_alias in u_label else u_label
                pencere.label_map = yeni_label_map

                yeni_ters_map = {}
                for u_label, g_label in pencere.ters_label_map.items():
                    yeni_u_label = u_label.replace(eski_alias, yeni_ad) if eski_alias in u_label else u_label
                    yeni_ters_map[yeni_u_label] = g_label
                pencere.ters_label_map = yeni_ters_map

                pencere.ops_dropdown_guncelle()

                #legend ve line label'ları
                legend = ax.get_legend()
                if legend:
                    for t in legend.get_texts():
                        if eski_alias in t.get_text():
                            t.set_text(t.get_text().replace(eski_alias, yeni_ad))
                for line in ax.get_lines():
                    lbl = str(line.get_label())
                    if eski_alias in lbl:
                        line.set_label(lbl.replace(eski_alias, yeni_ad))

                pencere.canvas.figure.canvas.draw()

        self.figure_listesini_guncelle()

    #FIGURE
    def figure_listesini_guncelle(self):
        self.figure_listesi.clear()
        for fig_no, fig_data in self.figures.items():
            fig_item = QTreeWidgetItem([f"Figure {fig_no} ({len(fig_data['params'])} param)"])
            self.figure_listesi.addTopLevelItem(fig_item)

            for i, p in enumerate(fig_data['params']):
                parcalar = p.split(' | ', 1)
                kolon_adi = parcalar[1] if len(parcalar) > 1 else p
                dosya_adi = p.split(' | ', 1)[0] if ' | ' in p else p
                param_item = QTreeWidgetItem([f"U{i + 1}  {dosya_adi} {kolon_adi}"])
                fig_item.addChild(param_item)

                combo = QComboBox()
                combo.addItems([
                    'Default Units', 'deg to rad', 'rad to deg',
                    'm to ft', 'ft to m', 'm/s to kt', 'kt to m/s',
                    'km/h to kt', 'kt to km/h', 'm/s to km/h', 'km/h to m/s'
                ])
                saved_unit = fig_data['units'][i] if i < len(fig_data['units']) else 'Default Units'
                combo.setCurrentText(saved_unit)
                combo.currentTextChanged.connect(
                    lambda text, fn=fig_no, idx=i: self.birim_degisti(fn, idx, text)
                )
                satir_widget = QWidget()
                satir_layout = QHBoxLayout(satir_widget)
                satir_layout.setContentsMargins(0, 0, 0, 0)
                satir_layout.addWidget(combo)

                btn_sil = QPushButton('🗑️')
                btn_sil.setFixedWidth(30)
                btn_sil.clicked.connect(lambda _, fn=fig_no, idx=i: self.parametre_sil(fn, idx))
                satir_layout.addWidget(btn_sil)

                self.figure_listesi.setItemWidget(param_item, 1, satir_widget)

            if fig_no == self.current_figure:
                fig_item.setExpanded(True)
                self.figure_listesi.setCurrentItem(fig_item)

        self.figure_listesi.setColumnWidth(0, 320)
        self.figure_listesi.setColumnWidth(1, 135)

    def birim_degisti(self, fig_no, idx, text):
        self.figures[fig_no]['units'][idx] = text

    def mode_degisti(self):
        if self.radio_realtime.isChecked():
            self.figures[self.current_figure]['mode'] = 'realtime'
        elif self.radio_previous.isChecked():
            self.figures[self.current_figure]['mode'] = 'previous'
        elif self.radio_nearest.isChecked():
            self.figures[self.current_figure]['mode'] = 'nearest'

    def figure_degisti(self, item):
        if item.parent() is not None:
            return

        text = item.text(0)
        self.current_figure = int(text.split()[1])
        mode = self.figures[self.current_figure]['mode']

        if mode == 'realtime':
            self.radio_realtime.setChecked(True)
        elif mode == 'previous':
            self.radio_previous.setChecked(True)
        elif mode == 'nearest':
            self.radio_nearest.setChecked(True)

    def parametre_sil(self, fig_no, idx):
        self.figures[fig_no]['params'].pop(idx)
        self.figures[fig_no]['units'].pop(idx)
        gecici = self.current_figure
        self.current_figure = fig_no
        self.figure_listesini_guncelle()
        self.current_figure = gecici

        for i in range(self.figure_listesi.topLevelItemCount()):
            item = self.figure_listesi.topLevelItem(i)
            try:
                fn = int(item.text(0).split()[1])
                if fn == fig_no:
                    item.setExpanded(True)
            except (ValueError, IndexError):
                pass

    def parametreyi_figure_ekle(self):
        secili_parametreler = [item.text()
                               for item in self.parametre_listesi.selectedItems()]
        if not secili_parametreler:
            QMessageBox.warning(self, 'Warning', "You haven't selected any parameters.")
            return
        for p in secili_parametreler:
            if p not in self.figures[self.current_figure]['params']:
                self.figures[self.current_figure]['params'].append(p)
                self.figures[self.current_figure]['units'].append('Default Units')
        self.figure_listesini_guncelle()

    def yeni_figur_olustur(self):
        yeni_no = max(self.figures.keys()) + 1
        self.figures[yeni_no] = {'params': [], 'mode': 'realtime', 'units': []}
        self.current_figure = yeni_no
        self.figure_listesini_guncelle()

    def figure_sil(self):
        for pencere in self.acik_grafikler[:]:
            if pencere.figure_no == self.current_figure:
                pencere.close()

        del self.figures[self.current_figure]

        if not self.figures:
            self.figures[1] = {'params': [], 'mode': 'realtime', 'units': []}
            self.current_figure = 1
        else:
            self.current_figure = list(self.figures.keys())[0]

        self.figure_listesini_guncelle()

    def plot_bas(self):
        secili_parametreler = self.figures[self.current_figure]['params']
        if not secili_parametreler:
            QMessageBox.warning(self, 'Warning', 'Select Variables.')
            return

        parametre_map = {}
        secili_dosyalar = set()
        pid = 1

        #Alias'tan orijinal dosya adına çevirmek için ters map
        ters_alias = {v: k for k, v in self.alias_map.items()}

        for p in secili_parametreler:
            dosya_adi, kolon = p.split(' | ', 1)
            #Alias ise orijinal ada çevir, değilse olduğu gibi kullan
            dosya = ters_alias.get(dosya_adi, dosya_adi)
            parametre_map[pid] = (dosya, kolon)
            secili_dosyalar.add(dosya)
            pid += 1

        try:
            tum_veriler = {}

            for dosya in secili_dosyalar:
                cache_key = os.path.join(self.klasor, dosya)
                if cache_key in self.data_cache:
                    tum_veriler[dosya] = self.data_cache[cache_key]
                else:
                    yuklenen = dosyalari_yukle(self.klasor, [dosya])
                    self.data_cache[cache_key] = yuklenen[dosya]
                    tum_veriler[dosya] = yuklenen[dosya]

            mode = self.figures[self.current_figure]['mode']
            units = self.figures[self.current_figure]['units']

            aktif_alias = self.alias_map if self.radio_use_alias.isChecked() else {}

            #Aynı figure_no'dan eski pencereyi kapat
            for pencere in self.acik_grafikler[:]:
                if pencere.figure_no == self.current_figure:
                    pencere.close()

            fig, lines, plotted_data, uyarilar = grafikleri_ciz(
                tum_veriler,
                parametre_map,
                mode,
                self.current_figure,
                units=units,
                alias_map=aktif_alias
            )

            if uyarilar:
                QMessageBox.warning(self, "Warning", "\n".join(uyarilar))

            popup_window = GrafikPenceresi(fig, self.current_figure, mode, self)
            popup_window.plotted_data = plotted_data
            popup_window.label_map = {line.gercek_label: line.get_label() for line in lines}
            popup_window.ters_label_map = {v: k for k, v in popup_window.label_map.items()}
            popup_window.enable_legend_toggle(lines)
            popup_window.ops_dropdown_guncelle()
            popup_window.show()

            for gercek, u_ismi in popup_window.label_map.items():
                popup_window.matlab_listesi.addItem(u_ismi)
                popup_window.csv_listesi.addItem(u_ismi)

            self.acik_grafikler.append(popup_window)

        except (RuntimeError, ValueError, KeyError, FileNotFoundError, DosyaHatasi) as e:
            QMessageBox.critical(self, 'Error', str(e))

    def analyze_errors(self):
        popup = None
        for p in reversed(self.acik_grafikler):
            if p.figure_no == self.current_figure:
                popup = p
                break

        if popup is None:
            QMessageBox.warning(self, "Warning", "Please plot the figure first.")
            return

        if not hasattr(popup, "plotted_data") or not popup.plotted_data:
            QMessageBox.warning(self, "Warning", "No plotted data found.")
            return

        plotted_data = popup.plotted_data
        self.error_results[self.current_figure] = {}

        for label, (t_values, y_values) in plotted_data.items():
            if len(y_values) == 0:
                continue

            parcalar = label.split(' | ', 1)
            if len(parcalar) < 2:
                continue
            dosya_adi = parcalar[0].upper()
            kolon_adi = parcalar[1]

            ozel = check_special_variable(kolon_adi, y_values)
            if ozel is not None:
                self.error_results[self.current_figure][label] = {
                    "t": t_values,
                    "values": y_values,
                    "errors": ozel
                }
                continue

            y_values = np.asarray(y_values, dtype=float)
            t_values = np.asarray(t_values, dtype=float)

            #NaN içeren satırları hem t hem y'den eş zamanlı çıkar
            #Böylece analyze_errors'dan dönen indexler t_values ile hizalı kalır
            nan_mask = ~np.isnan(y_values)
            y_clean = y_values[nan_mask]
            t_clean = t_values[nan_mask]

            if len(y_clean) == 0:
                continue

            if self.error_loader is None:
                continue

            dosya_eml_mi = "EML" in dosya_adi and "STANAG" not in dosya_adi

            system = find_variable_system(
                self.error_loader,
                kolon_adi,
                prefer_eml=dosya_eml_mi,
                dosya_adi_upper=dosya_adi
            )

            if system is None:
                continue

            #Tabloda değişken adı farklı case ile kayıtlı olabilir; case-insensitive ara
            sistem_tablosu = self.error_loader.error_tables[system]
            kolon_key = kolon_adi
            if kolon_key not in sistem_tablosu:
                kolon_lower = kolon_adi.lower()
                kolon_key = next(
                    (k for k in sistem_tablosu if k.lower() == kolon_lower), None
                )
                if kolon_key is None:
                    continue

            limits = sistem_tablosu[kolon_key]
            min_val = limits["min"]
            max_val = limits["max"]
            max_change = limits["max_change"]

            eml_10_yok = ["ADC_ERROR_CLASS_EML"]
            if dosya_eml_mi and system not in eml_10_yok:
                max_change = max_change * 10

            ''''MATLAB referansı:
            SMU/AP/RD/RC: sadece 'd' ile başlayan değişkenlerde OVERSHOOT+SPIKE
            VMM: 'd' ile başlayıp 'fixed' içerenlerde OVERSHOOT+SPIKE; diğer 'd' → sadece range; 'd' dışı özel
            EGID/EGIE/EGIS/ADC: tabloda bulunan tüm değişkenler için tam analiz (max_change==0 olanlar hariç)'''

            sistem_adi = system.upper()
            d_ile_basliyor = kolon_adi.lower().startswith('d')
            fixed_iceriyor = 'fixed' in kolon_adi.lower()

            egid_egie_egis_adc_sistemler = (
                    'EGID' in sistem_adi or
                    'EGIE' in sistem_adi or
                    'EGIS' in sistem_adi or
                    'ADC' in sistem_adi
            )

            if egid_egie_egis_adc_sistemler:
                #Bu sistemlerde tüm değişkenler tam analiz alır
                #max_change == 0 ise OVERSHOOT/SPIKE zaten analyze_errors içinde atlanır
                errors = analyze_errors(y_clean, min_val, max_val, max_change)

            elif 'VMM' in dosya_adi:
                if d_ile_basliyor and fixed_iceriyor:
                    #VMM + d+fixed: tüm analizler
                    errors = analyze_errors(y_clean, min_val, max_val, max_change)
                elif d_ile_basliyor and not fixed_iceriyor:
                    #VMM + d (fixed değil): sadece CONSTANT OUTPUT ve OUT OF RANGE
                    co_errors = analyze_errors(y_clean, min_val, max_val, 0)
                    range_flags = []
                    if not (min_val == 0 and max_val == 0):
                        range_flags = np.where((y_clean > max_val) | (y_clean < min_val))[0].tolist()
                    errors = {
                        "CONSTANT OUTPUT": co_errors["CONSTANT OUTPUT"],
                        "OVERSHOOT": [],
                        "SPIKE": [],
                        "OUT OF RANGE": range_flags
                    }
                else:
                    #VMM + d ile başlamıyor (ivehiclemode, ivehiclestate vb.)
                    #check_special_variable zaten yukarıda yakaladı; buraya düşmemeli
                    #Yine de güvenli taraf: sadece range
                    co_errors = analyze_errors(y_clean, min_val, max_val, 0)
                    errors = {
                        "CONSTANT OUTPUT": co_errors["CONSTANT OUTPUT"],
                        "OVERSHOOT": [],
                        "SPIKE": [],
                        "OUT OF RANGE": co_errors["OUT OF RANGE"]
                    }

            elif d_ile_basliyor:
                #SMU / AP / RD / RC sistemleri — d ile başlıyor: tüm analizler
                errors = analyze_errors(y_clean, min_val, max_val, max_change)

            else:
                #SMU / AP / RD / RC — d ile başlamıyor: sadece CONSTANT OUTPUT ve OUT OF RANGE
                co_errors = analyze_errors(y_clean, min_val, max_val, 0)
                errors = {
                    "CONSTANT OUTPUT": co_errors["CONSTANT OUTPUT"],
                    "OVERSHOOT": [],
                    "SPIKE": [],
                    "OUT OF RANGE": co_errors["OUT OF RANGE"]
                }

            self.error_results[self.current_figure][label] = {
                "t": t_clean,
                "values": y_clean,
                "errors": errors
            }
        QMessageBox.information(self, "Info", "Error analysis completed.")

    def error_plot(self):
        if self.current_figure not in self.error_results:
            QMessageBox.warning(self, "Warning", "No error analysis found.")
            return

        popup = None
        for p in reversed(self.acik_grafikler):
            if p.figure_no == self.current_figure:
                popup = p
                break

        if popup is None:
            QMessageBox.warning(self, "Warning", "Graph window not found.")
            return

        #Önceki scatter kayıtlarını matlab listesinden temizle
        current_labels = [popup.matlab_listesi.item(i).text()
                          for i in range(popup.matlab_listesi.count())]
        for label in current_labels:
            if ' - ' in label:
                items = popup.matlab_listesi.findItems(label, Qt.MatchFlag.MatchExactly)
                for item in items:
                    popup.matlab_listesi.takeItem(popup.matlab_listesi.row(item))

        ax = popup.canvas.figure.axes[0]
        figure_errors = self.error_results[self.current_figure]
        scatter_list = []

        for coll in ax.collections[:]:
            coll.remove()

        for label, data in figure_errors.items():
            t_arr = np.asarray(data["t"])
            v_arr = np.asarray(data["values"])
            errors = data["errors"]
            u_ismi = popup.label_map.get(label, label)

            for error_type, indices in errors.items():
                idx = np.asarray(indices)
                if len(idx) == 0:
                    continue

                marker = self.error_marker_map.get(error_type, "o")
                unfilled = marker in ["x", "+", "|", "_"]

                if unfilled:
                    sc = ax.scatter(
                        t_arr[idx], v_arr[idx],
                        marker=marker, color="red",
                        zorder=5, label=f"{u_ismi} - {error_type}"
                    )
                else:
                    sc = ax.scatter(
                        t_arr[idx], v_arr[idx],
                        marker=marker, facecolors="none", edgecolors="red",
                        zorder=5, label=f"{u_ismi} - {error_type}"
                    )

                sc.gercek_label = f"{label} - {error_type}"
                scatter_list.append(sc)

        for sc in scatter_list:
            popup.matlab_listesi.addItem(sc.get_label())
            popup.csv_listesi.addItem(sc.get_label())

        #Op çizgileri dahil tüm mevcut line'ları al — legend enable_legend_toggle içinde oluştur
        guncel_lines = [l for l in ax.get_lines() if l.get_label() and not l.get_label().startswith('_')]
        popup.enable_legend_toggle(guncel_lines, scatter_list)
        popup.raporu_guncelle(figure_errors)
        popup.canvas.draw_idle()
        popup.raise_()
        popup.activateWindow()

        popup.scatter_data = {}
        for sc in scatter_list:
            offsets = np.asarray(sc.get_offsets())
            popup.scatter_data[sc.get_label()] = (offsets[:, 0], offsets[:, 1])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnaPencere()
    window.show()
    sys.exit(app.exec())
