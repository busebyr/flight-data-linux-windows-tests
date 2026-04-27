import os
import sys
import tempfile

if sys.platform != 'win32':
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
import numpy as np
import pandas as pd

from analiz_motoru import birim_donustur,ortak_zaman_ekseni, temiz_veriye_donustur, veriyi_hizala
from error_analyzer import analyze_errors, find_variable_system, check_special_variable

from unittest.mock import MagicMock, mock_open
from main import GrafikPenceresi
from main import AnaPencere
import matplotlib.pyplot as plt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtCore import QPoint

import io
from unittest.mock import patch
from analiz_motoru import dosyalari_yukle, DosyaHatasi
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QTableWidgetItem, QPushButton)
from analiz_motoru import grafikleri_ciz


app = QApplication.instance() or QApplication(sys.argv)

class TestBirimDonustur(unittest.TestCase):

    def test_default_units_degismemeli(self):
        veri = np.array([1.0, 2.0, 3.0, -1.0, 0.0])
        sonuc = birim_donustur(veri, 'Default Units')
        np.testing.assert_array_equal(sonuc, veri)

    def test_deg_to_rad(self):
        sonuc = birim_donustur(np.array([0.0, 90.0, 180.0, 270.0, 360.0]), 'deg to rad')
        self.assertAlmostEqual(sonuc[0], 0.0, places=5)
        self.assertAlmostEqual(sonuc[1], np.pi / 2, places=5)
        self.assertAlmostEqual(sonuc[2], np.pi, places=5)
        self.assertAlmostEqual(sonuc[3], 3 * np.pi / 2, places=5)
        self.assertAlmostEqual(sonuc[4], 2 * np.pi, places=5)

    def test_rad_to_deg(self):
        sonuc = birim_donustur(np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]), 'rad to deg')
        self.assertAlmostEqual(sonuc[0], 0.0, places=5)
        self.assertAlmostEqual(sonuc[1], 90.0, places=5)
        self.assertAlmostEqual(sonuc[2], 180.0, places=5)
        self.assertAlmostEqual(sonuc[3], 270.0, places=5)
        self.assertAlmostEqual(sonuc[4], 360.0, places=5)


    def test_m_to_ft(self):
        sonuc = birim_donustur(np.array([1.0, 2.0, 3.0, -1.0, 0.0]), 'm to ft')
        self.assertAlmostEqual(sonuc[0], 3.28084, places=4)
        self.assertAlmostEqual(sonuc[1], 6.56168, places=4)
        self.assertAlmostEqual(sonuc[2], 9.84252, places=4)
        self.assertAlmostEqual(sonuc[3], -3.28084, places=4)
        self.assertAlmostEqual(sonuc[4], 0.0, places=4)

    def test_ft_to_m(self):
        sonuc = birim_donustur(np.array([1.0, 2.0, 3.0, -1.0, 0.0]), 'ft to m')
        self.assertAlmostEqual(sonuc[0], 0.3048, places=4)
        self.assertAlmostEqual(sonuc[1], 0.6096, places=4)
        self.assertAlmostEqual(sonuc[2], 0.9144, places=4)
        self.assertAlmostEqual(sonuc[3], -0.3048, places=4)
        self.assertAlmostEqual(sonuc[4], 0.0, places=4)

    def test_kt_to_ms(self):
        sonuc = birim_donustur(np.array([1.0, 2.0, 10.0, -1.0, 0.0]), 'kt to m/s')
        self.assertAlmostEqual(sonuc[0], 0.514444, places=4)
        self.assertAlmostEqual(sonuc[1], 1.028888, places=4)
        self.assertAlmostEqual(sonuc[2], 5.14444, places=4)
        self.assertAlmostEqual(sonuc[3], -0.514444, places=4)
        self.assertAlmostEqual(sonuc[4], 0.0, places=4)

    def test_ms_to_kt(self):
        sonuc = birim_donustur(np.array([1.0, 2.0, 10.0, -1.0, 0.0]), 'm/s to kt')
        self.assertAlmostEqual(sonuc[0], 1.94384, places=4)
        self.assertAlmostEqual(sonuc[1], 3.88768, places=4)
        self.assertAlmostEqual(sonuc[2], 19.4384, places=4)
        self.assertAlmostEqual(sonuc[3], -1.94384, places=4)
        self.assertAlmostEqual(sonuc[4], 0.0, places=4)

    def test_kmh_to_kt(self):
        sonuc = birim_donustur(np.array([1.0, 2.0, 10.0, -1.0, 0.0]), 'km/h to kt')
        self.assertAlmostEqual(sonuc[0], 0.539957, places=4)
        self.assertAlmostEqual(sonuc[1], 1.079914, places=4)
        self.assertAlmostEqual(sonuc[2], 5.39957, places=4)
        self.assertAlmostEqual(sonuc[3], -0.539957, places=4)
        self.assertAlmostEqual(sonuc[4], 0.0, places=4)

    def test_kt_to_kmh(self):
        sonuc = birim_donustur(np.array([1.0, 2.0, 10.0, -1.0, 0.0]), 'kt to km/h')
        self.assertAlmostEqual(sonuc[0], 1.0 / 0.539957, places=4)
        self.assertAlmostEqual(sonuc[1], 2.0 / 0.539957, places=4)
        self.assertAlmostEqual(sonuc[2], 10.0 / 0.539957, places=4)
        self.assertAlmostEqual(sonuc[3], -1.0 / 0.539957, places=4)
        self.assertAlmostEqual(sonuc[4], 0.0, places=4)

    def test_ms_to_kmh(self):
        sonuc = birim_donustur(np.array([1.0, 2.0, 10.0, -1.0, 0.0]), 'm/s to km/h')
        self.assertAlmostEqual(sonuc[0], 3.6, places=4)
        self.assertAlmostEqual(sonuc[1], 7.2, places=4)
        self.assertAlmostEqual(sonuc[2], 36.0, places=4)
        self.assertAlmostEqual(sonuc[3], -3.6, places=4)
        self.assertAlmostEqual(sonuc[4], 0.0, places=4)

    def test_kmh_to_ms(self):
        sonuc = birim_donustur(np.array([1.0, 2.0, 10.0, -1.0, 0.0]), 'km/h to m/s')
        self.assertAlmostEqual(sonuc[0], 1.0 / 3.6, places=4)
        self.assertAlmostEqual(sonuc[1], 2.0 / 3.6, places=4)
        self.assertAlmostEqual(sonuc[2], 10.0 / 3.6, places=4)
        self.assertAlmostEqual(sonuc[3], -1.0 / 3.6, places=4)
        self.assertAlmostEqual(sonuc[4], 0.0, places=4)

    def test_invalid_input(self):
        sonuc = birim_donustur(np.array([1.0]), 'km/hs to m/s')
        self.assertIsNone(sonuc, "sonuc none degil")

    def test_sifir_ve_negatif_deger(self):
        sonuc = birim_donustur(np.array([0.0, -1.0, -2.0, -10.0, -100.0]), 'm to ft')
        self.assertAlmostEqual(sonuc[0], 0.0, places=5)
        self.assertAlmostEqual(sonuc[1], -3.28084, places=4)
        self.assertAlmostEqual(sonuc[2], -6.56168, places=4)
        self.assertAlmostEqual(sonuc[3], -32.8084, places=4)
        self.assertAlmostEqual(sonuc[4], -328.084, places=4)


class TestOrtakZamanEkseni(unittest.TestCase):

    def test_normal_veri_bos_olmayan_series_dondurulmesli(self):
        df= pd.DataFrame({'time':[0.0,1.0,2.0,3.0], 'value':[1,2,3,4]})
        veriler={'dosya.csv':df}
        sonuc =ortak_zaman_ekseni(veriler)

        self.assertGreater(len(sonuc), 0)
        self.assertEqual(sonuc.dtype, float)
        self.assertGreaterEqual(sonuc.iloc[0], 0.0)
        self.assertLessEqual(sonuc.iloc[-1], 3.0)

    def test_bos_veri_bos_series_dondurmeli(self):
        sonuc = ortak_zaman_ekseni({})
        self.assertEqual(len(sonuc), 0)

    def test_tek_satirli_df_atlanmali(self):
        df = pd.DataFrame({'time': [0.0], 'value': [1]})
        veriler = {'dosya.csv': df}
        sonuc = ortak_zaman_ekseni(veriler)
        self.assertEqual(len(sonuc), 0)

    def test_yuksek_frekansta_dt_10ms_secilmeli(self):
        times = [i * 0.010 for i in range(10)]
        df = pd.DataFrame({'time': times, 'value': range(10)})
        veriler = {'dosya.csv': df}
        sonuc = ortak_zaman_ekseni(veriler)
        adim = round(np.diff(sonuc.to_numpy()).mean(), 5)
        self.assertAlmostEqual(adim, 0.010, places=3)


    def test_dusuk_frekansta_dt_100ms_secilmeli(self):
        times = [i * 0.100 for i in range(10)]
        df = pd.DataFrame({'time': times, 'value': range(10)})
        veriler = {'dosya.csv': df}
        sonuc = ortak_zaman_ekseni(veriler)
        adim = round(np.diff(sonuc.to_numpy()).mean(), 5)
        self.assertAlmostEqual(adim, 0.100, places=3)

    def test_coklu_dosya_en_yuksek_cozunurluk_secilmeli(self):
        df1 = pd.DataFrame({'time': [i * 0.010 for i in range(5)], 'value': range(5)})
        df2 = pd.DataFrame({'time': [i * 0.100 for i in range(5)], 'value': range(5)})
        veriler = {'hizli.csv': df1, 'yavas.csv': df2}
        sonuc = ortak_zaman_ekseni(veriler)
        adim = round(np.diff(sonuc.to_numpy()).mean(), 5)
        self.assertAlmostEqual(adim, 0.010, places=3)

    def test_coklu_dosya_tmax_dosyalar_arasi_boslugu_saymamali(self):
        df1 = pd.DataFrame({'time': [0.0, 1.0, 2.0], 'value': [1, 2, 3]})
        df2 = pd.DataFrame({'time': [10.0, 12.0, 15.0], 'value': [4, 5, 6]})
        veriler = {'dosya1.csv': df1, 'dosya2.csv': df2}
        sonuc = ortak_zaman_ekseni(veriler)
        self.assertAlmostEqual(sonuc.iloc[-1], 5.0, places=1)


class TestTemizVeriyeDonustur(unittest.TestCase):

    def test_normal_float_degismemeli(self):
        seri = pd.Series([1.0, 2.5, 3.14])
        sonuc = temiz_veriye_donustur(seri)
        pd.testing.assert_series_equal(sonuc, pd.Series([1.0, 2.5, 3.14]))

    def test_virgullu_sayi_noktaya_donusturulmeli(self):
        seri = pd.Series(["1,5", "2,75"])
        sonuc = temiz_veriye_donustur(seri)
        pd.testing.assert_series_equal(sonuc, pd.Series([1.5, 2.75]))

    def test_bosluklu_sayi_temizlemesi(self):
        seri = pd.Series([' 3.14 '])
        sonuc = temiz_veriye_donustur(seri)
        pd.testing.assert_series_equal(sonuc, pd.Series([3.14]))

    def test_gecersiz_deger(self):
        seri= pd.Series(['abc'])
        sonuc = temiz_veriye_donustur(seri)
        self.assertTrue(sonuc.isna().all())

    def test_bos_seri(self):
        seri = pd.Series([], dtype=object)
        sonuc = temiz_veriye_donustur(seri)
        self.assertEqual(len(sonuc), 0)

    def test_karisik_tipli_seri_temizlenmeli(self):
        seri = pd.Series([1.0, 'abc', '2,5', None, ' 3.14 ', '0', -1.0, '1,000', 'xyz', 0.0])
        sonuc = temiz_veriye_donustur(seri)
        self.assertAlmostEqual(sonuc.iloc[0], 1.0)
        self.assertTrue(pd.isna(sonuc.iloc[1]))
        self.assertAlmostEqual(sonuc.iloc[2], 2.5)
        self.assertTrue(pd.isna(sonuc.iloc[3]))
        self.assertAlmostEqual(sonuc.iloc[4], 3.14)
        self.assertAlmostEqual(sonuc.iloc[5], 0.0)
        self.assertAlmostEqual(sonuc.iloc[6], -1.0)
        self.assertAlmostEqual(sonuc.iloc[7], 1.0)
        self.assertTrue(pd.isna(sonuc.iloc[8]))
        self.assertAlmostEqual(sonuc.iloc[9], 0.0)


class TestVeriyiHizala(unittest.TestCase):

    def test_previous_onceki_degeri_almali(self):
        df = pd.DataFrame({'time': [0.0, 1.0, 2.0], 'value': [10, 20, 30]})
        timeline=pd.Series([0.5, 1.5])
        sonuc = veriyi_hizala(df, timeline, 'value', 'previous')
        self.assertEqual(sonuc.iloc[0], 10.0)  # 0.5 anında önceki değer 10.0
        self.assertEqual(sonuc.iloc[1], 20.0)  # 1.5 anında önceki değer 20.0

    def test_nearest_yakin_degeri_almali(self):
        df = pd.DataFrame({'time':[0.0, 1.0, 2.0], 'value': [10, 20, 30]})
        timline=pd.Series([0.4, 1.6])
        sonuc=veriyi_hizala(df, timline, 'value', 'nearest')
        self.assertEqual(sonuc.iloc[0],10)
        self.assertEqual(sonuc.iloc[1],30)

    def test_bos_kaynak_veri(self):
        df = pd.DataFrame({'time': [], 'value': []})
        timline = pd.Series([0.0, 1.0, 2.0])
        sonuc = veriyi_hizala(df, timline, 'value', 'previous')
        self.assertTrue(sonuc.isna().all())

    def test_sirasiz_veri(self):
        df=pd.DataFrame({'time':[2.0, 0.0, 1.0], 'value':[30.0, 10.0, 20.0]})
        timeline=pd.Series([0.0, 1.0, 2.0])
        sonuc = veriyi_hizala(df, timeline, 'value', 'nearest')
        self.assertEqual(sonuc.iloc[0],10)
        self.assertEqual(sonuc.iloc[1],20)
        self.assertEqual(sonuc.iloc[2],30)


class TestAnalyzeErrors(unittest.TestCase):

    def test_constant_output_tespit_edilmeli(self):
        sonuc = analyze_errors([5, 5, 5, 5, 5], min_val=0, max_val=10, max_change=1)
        self.assertEqual(sonuc['CONSTANT OUTPUT'], [0])

    def test_constant_output_pencere_altinda_tespit_edilmemeli(self):
        sonuc = analyze_errors([3, 3, 3, 3], min_val=0, max_val=10, max_change=1)
        self.assertEqual(sonuc['CONSTANT OUTPUT'], [])

    def test_overshoot_esigi_asan_degisim(self):
        # [0, 0, 15, 20] — index 2'de diff=15 > max_change=10, geri dönüş yok → spike değil, overshoot
        sonuc = analyze_errors([0, 0, 15, 20], min_val=0, max_val=200, max_change=10)
        self.assertIn(2, sonuc['OVERSHOOT'])
        self.assertNotIn(2, sonuc['SPIKE'])

    def test_max_change_sifir_overshoot_spike_hesaplanmaz(self):
        sonuc = analyze_errors([0, 100, 0], min_val=0, max_val=200, max_change=0)
        self.assertEqual(sonuc['OVERSHOOT'], [])
        self.assertEqual(sonuc['SPIKE'], [])

    def test_spike_overshoot_degil(self):
        sonuc = analyze_errors([0, 0, 50, 0, 0], min_val=0, max_val=100, max_change=10)
        self.assertIn(2, sonuc['SPIKE'])
        self.assertNotIn(2, sonuc['OVERSHOOT'])

    def test_out_of_range_max_ustu(self):
        sonuc = analyze_errors([5, 15, 5], min_val=0, max_val=10, max_change=0)
        self.assertIn(1, sonuc['OUT OF RANGE'])

    def test_out_of_range_sinir_tanimsiz(self):
        sonuc = analyze_errors([999], min_val=0, max_val=0, max_change=0)
        self.assertEqual(sonuc['OUT OF RANGE'], [])

    def test_constant_gecis_overshoot_sayilmaz(self):
        sonuc = analyze_errors([0, 50, 50, 50, 50, 50, 0], min_val=0, max_val=100, max_change=10)
        self.assertEqual(sonuc['OVERSHOOT'], [])

    def test_spike_komsulari_overshoot_disinda(self):
        sonuc = analyze_errors([0, 0, 50, 0, 0], min_val=0, max_val=100, max_change=10)
        for komsu in [1, 2, 3]:
            self.assertNotIn(komsu, sonuc['OVERSHOOT'])

    def test_bos_dizi_tum_listeler_bos(self):
        sonuc = analyze_errors([], min_val=0, max_val=10, max_change=1)
        self.assertEqual(sonuc['CONSTANT OUTPUT'], [])
        self.assertEqual(sonuc['OVERSHOOT'], [])
        self.assertEqual(sonuc['SPIKE'], [])
        self.assertEqual(sonuc['OUT OF RANGE'], [])

    def test_donus_yapisi_4_anahtar_liste_tipinde(self):
        sonuc = analyze_errors([1, 2, 3], min_val=0, max_val=10, max_change=1)
        for anahtar in ['CONSTANT OUTPUT', 'OVERSHOOT', 'SPIKE', 'OUT OF RANGE']:
            self.assertIn(anahtar, sonuc)
            self.assertIsInstance(sonuc[anahtar], list)

    def test_nan_iceren_dizi_hata_firlatmamali(self):
        sonuc = analyze_errors([1.0, float('nan'), 3.0], min_val=0, max_val=10, max_change=1)
        for anahtar in ['CONSTANT OUTPUT', 'OVERSHOOT', 'SPIKE', 'OUT OF RANGE']:
            self.assertIsInstance(sonuc[anahtar], list)

    def test_tek_elemanli_dizi_hata_firlatmamali(self):
        sonuc = analyze_errors([5.0], min_val=0, max_val=10, max_change=1)
        for anahtar in ['CONSTANT OUTPUT', 'OVERSHOOT', 'SPIKE', 'OUT OF RANGE']:
            self.assertIsInstance(sonuc[anahtar], list)

    def test_cok_uzun_dizi_hata_firlatmamali(self):
        veri = list(range(1000))
        sonuc = analyze_errors(veri, min_val=0, max_val=999, max_change=10)
        for anahtar in ['CONSTANT OUTPUT', 'OVERSHOOT', 'SPIKE', 'OUT OF RANGE']:
            self.assertIsInstance(sonuc[anahtar], list)

    def test_min_siniri_altinda_out_of_range(self):
        sonuc = analyze_errors([-5.0, 0.0, 5.0, -10.0, 3.0], min_val=0, max_val=10, max_change=0)
        self.assertIn(0, sonuc['OUT OF RANGE'])
        self.assertIn(3, sonuc['OUT OF RANGE'])
        self.assertNotIn(1, sonuc['OUT OF RANGE'])

    def test_spike_ve_out_of_range_ayni_indeks(self):
        sonuc = analyze_errors([0, 0, 150, 0, 0], min_val=0, max_val=100, max_change=10)
        self.assertIn(2, sonuc['SPIKE'])
        self.assertIn(2, sonuc['OUT OF RANGE'])

    def test_tum_degerler_max_ustunde_hepsi_out_of_range(self):
        sonuc = analyze_errors([20.0, 30.0, 40.0, 50.0, 60.0], min_val=0, max_val=10, max_change=0)
        self.assertEqual(len(sonuc['OUT OF RANGE']), 5)
        for i in range(5):
            self.assertIn(i, sonuc['OUT OF RANGE'])


class DummyLoader:
    """
    Tablo yapısı — 7 find_variable_system senaryosunu karşılar:

    gyro_x   → EGID_RS422 (normal) + EGID_RS422_EML (eml) + EGIS_RS232 (farklı sistem)
    dPressure → sadece SMU_ERROR_CLASS (tek eşleşme)
    sensor_a  → sadece EGIE_RS485 (EML versiyonu yok)
    voltage   → ADC_ERROR_CLASS (normal) + ADC_ERROR_CLASS_EML (eml)
    """

    def __init__(self):
        entry = {'min': 0, 'max': 10, 'max_change': 1, 'is_eml': False}
        self.error_tables = {
            'EGID_RS422': {'gyro_x': entry},
            'EGID_RS422_EML': {'gyro_x': {**entry, 'is_eml': True}},
            'EGIS_RS232': {'gyro_x': entry},
            'SMU_ERROR_CLASS': {'dPressure': entry},
            'EGIE_RS485': {'sensor_a': entry},
            'ADC_ERROR_CLASS': {'voltage': entry},
            'ADC_ERROR_CLASS_EML': {'voltage': {**entry, 'is_eml': True}},
        }

class TestFindVariableSystem(unittest.TestCase):

    def test_bulunamayan_degisken_none_doner(self):
        loader = DummyLoader()
        self.assertIsNone(find_variable_system(loader, 'OLMAYAN_VAR'))

    def test_tek_esleme_o_sistem_doner(self):
        loader= DummyLoader()
        sonuc=find_variable_system(loader, 'dPressure')
        self.assertEqual(sonuc, 'SMU_ERROR_CLASS')

    def test_dosya_adi_ortuseni_tercih_eder(self):
        loader=DummyLoader()
        sonuc=find_variable_system(loader,'gyro_x',dosya_adi_upper='EML_EGIDRX422')
        self.assertIn('EGID',sonuc)
        self.assertNotIn('EGIS',sonuc)

    def test_dosya_adi_prefer_eml_ama_eml_sistemi_yok(self):
        loader = DummyLoader()
        sonuc = find_variable_system(loader, 'sensor_a', prefer_eml=True, dosya_adi_upper='EML_EGIERS485')
        self.assertEqual(sonuc, 'EGIE_RS485')
        self.assertNotIn('EML', sonuc)

    def test_dosya_adi_yok_prefer_eml_false_normal_secer(self):
        loader = DummyLoader()
        sonuc = find_variable_system(loader, 'voltage', prefer_eml=False)
        self.assertEqual(sonuc, 'ADC_ERROR_CLASS')
        self.assertNotIn('EML', sonuc)

    def test_dosya_adi_yok_prefer_eml_true_eml_secer(self):
        loader = DummyLoader()
        sonuc = find_variable_system(loader, 'voltage', prefer_eml=True)
        self.assertIn('EML', sonuc)


class TestCheckSpecialVariable(unittest.TestCase):

    def test_ivehiclemode_gecerli_degerler(self):
        sonuc = check_special_variable('ivehiclemode', [0, 7, 15])
        self.assertEqual(sonuc['INVALID SIGNAL'], [])

    def test_ivehiclemode_gecersiz_deger(self):
        sonuc = check_special_variable('ivehiclemode', [1, 16, 5])
        self.assertIn(1, sonuc['INVALID SIGNAL'])

    def test_ivehiclestate_gecerli_deger(self):
        sonuc = check_special_variable('ivehiclestate', [0, 100, 200])
        self.assertEqual(sonuc['INVALID SIGNAL'], [])

    def test_ivehiclestate_gecersiz_deger(self):
        sonuc = check_special_variable('ivehiclestate', [100, 99, 200])
        self.assertIn(1, sonuc['INVALID SIGNAL'])

    def test_imcnav_type_gecerli_degerler(self):
        sonuc = check_special_variable('imcnav_type', [1, 2, 3])
        self.assertEqual(sonuc['INVALID SIGNAL'], [])

    def test_imcnav_type_gecersiz_deger(self):
        sonuc = check_special_variable('imcnav_type', [1, 4, 2])
        self.assertIn(1, sonuc['INVALID SIGNAL'])

    def test_binary_degisken_temiz_veri(self):
        sonuc = check_special_variable('bStatus', [0, 1, 0, 1])
        self.assertEqual(sonuc['INVALID SIGNAL'], [])

    def test_binary_degisken_gecersiz_deger(self):
        sonuc = check_special_variable('bStatus', [0, 1, 2])
        self.assertIsNone(sonuc)

    def test_binary_flgs_icerenler_bypass(self):
        sonuc = check_special_variable('bFlgs', [0, 2])
        self.assertIsNone(sonuc)

    def test_binary_uzun_isim_bypass(self):
        uzun_isim = 'b' + 'x' * 20  # 21 karakter
        sonuc = check_special_variable(uzun_isim, [0, 2])
        self.assertIsNone(sonuc)

    def test_istatusreport_gecersiz_deger(self):
        sonuc = check_special_variable('istatusreport', [1, 0, 1])
        self.assertIn(1, sonuc['INVALID SIGNAL'])

    def test_istatusreport_buyuk_harf(self):
        sonuc = check_special_variable('iStatusReport', [1, 0])
        self.assertIn(1, sonuc['INVALID SIGNAL'])

    def test_red_flag_gecersiz_deger(self):
        sonuc = check_special_variable('red_flag', [0, 1, 0])
        self.assertIn(1, sonuc['INVALID SIGNAL'])

    def test_red_flag_buyuk_harf(self):
        sonuc = check_special_variable('RED_FLAG', [0, 1])
        self.assertIn(1, sonuc['INVALID SIGNAL'])

    def test_bilinmeyen_degisken_none_doner(self):
        sonuc = check_special_variable('unknown_var', [1, 2, 3])
        self.assertIsNone(sonuc)


class TestCloseEvent(unittest.TestCase):

    def setUp(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.acik_grafikler = []  #gerçek liste, tip sorunu çözülür

    def make_pencere(self):
        fig, ax = plt.subplots()
        parent = MagicMock()
        parent.acik_grafikler = self.acik_grafikler
        pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)
        return pencere

    def test_listede_olan_pencere_kapatilinca_cikarilmali(self):
        event = QCloseEvent()
        pencere = self.make_pencere()
        self.acik_grafikler.append(pencere)

        pencere.closeEvent(event)

        self.assertNotIn(pencere, self.acik_grafikler)

    def test_listede_olmayan_pencere_kapatilinca_hata_firlatmamali(self):
        event = QCloseEvent()
        pencere = self.make_pencere()

        try:
            pencere.closeEvent(event)
        except Exception as e:
            self.fail(f"Beklenmedik hata: {e}")

        self.assertEqual(self.acik_grafikler, [])

    def test_sadece_kapanan_pencere_cikarilmali_digeri_kalmali(self):
        event = QCloseEvent()
        pencere_a = self.make_pencere()
        pencere_b = self.make_pencere()
        self.acik_grafikler.extend([pencere_a, pencere_b])

        pencere_a.closeEvent(event)

        self.assertNotIn(pencere_a, self.acik_grafikler)
        self.assertIn(pencere_b, self.acik_grafikler)
        self.assertEqual(len(self.acik_grafikler), 1)

    def test_super_close_event_her_durumda_cagirilmali(self):
        event = QCloseEvent()
        pencere = self.make_pencere()
        self.acik_grafikler.append(pencere)

        try:
            pencere.closeEvent(event)
        except Exception as e:
            self.fail(f"super().closeEvent() hata fırlattı: {e}")

class TestDosyalariYukle(unittest.TestCase):

    def sahte_csv_olustur(self):
        # skiprows=1 olduğu için ilk satır atlanır, ikinci satır header olur
        icerik = "meta_satir\nTimestamp\tvalue\n1000000\t5.0\n2000000\t10.0\n"
        return pd.read_csv(io.StringIO(icerik), skiprows=1, sep='\t')

    def test_normal_dosya_dict_e_girmeli(self):
        sahte_df = self.sahte_csv_olustur()
        with patch('analiz_motoru.pd.read_csv', return_value=sahte_df):
            sonuc = dosyalari_yukle('klasor', ['test.csv'])
        self.assertIn('test.csv', sonuc)

    def test_zaman_kolonu_1e6_ya_bolunmeli(self):
        sahte_df = self.sahte_csv_olustur()
        with patch('analiz_motoru.pd.read_csv', return_value=sahte_df):
            sonuc = dosyalari_yukle('klasor', ['test.csv'])
        zaman_kolon = sonuc['test.csv'].columns[0]
        self.assertAlmostEqual(sonuc['test.csv'][zaman_kolon].iloc[0], 1.0, places=5)
        self.assertAlmostEqual(sonuc['test.csv'][zaman_kolon].iloc[1], 2.0, places=5)

    def test_olmayan_dosya_dosyahatasi_firlatmali(self):
        with self.assertRaises(DosyaHatasi):
            dosyalari_yukle('klasor', ['olmayan.csv'])

    def test_coklu_dosya_hepsi_dict_e_girmeli(self):
        sahte_df = self.sahte_csv_olustur()
        with patch('analiz_motoru.pd.read_csv', return_value=sahte_df):
            sonuc = dosyalari_yukle('klasor', ['a.csv', 'b.csv'])
        self.assertIn('a.csv', sonuc)
        self.assertIn('b.csv', sonuc)
        self.assertEqual(len(sonuc), 2)

    def test_bos_liste_bos_dict_donmeli(self):
        sonuc = dosyalari_yukle('klasor', [])
        self.assertEqual(sonuc, {})


class TestLegendGuncelle(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='test')
        ax.legend()
        parent = MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)

    def test_checkbox_isaretli_legend_gorunur_olmali(self):
        self.pencere.legend_checkbox.setChecked(True)
        self.pencere.legend_guncelle()
        ax = self.pencere.canvas.figure.axes[0]
        self.assertTrue(ax.get_legend().get_visible())

    def test_checkbox_isaretsiz_legend_gizlenmeli(self):
        self.pencere.legend_checkbox.setChecked(False)
        self.pencere.legend_guncelle()
        ax = self.pencere.canvas.figure.axes[0]
        self.assertFalse(ax.get_legend().get_visible())

    def test_legend_yoksa_hata_firlatmamali(self):
        ax = self.pencere.canvas.figure.axes[0]
        ax.get_legend().remove()
        try:
            self.pencere.legend_guncelle()
        except Exception as e:
            self.fail(f"Beklenmedik Hata: {e}")

    def test_axes_yoksa_hata_firlatmamali(self):
        self.pencere.canvas.figure.clf()
        try:
            self.pencere.legend_guncelle()
        except Exception as e:
            self.fail(f"Beklenmedik hata: {e}")


class TestRaporuGuncelle(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent = MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)

    def hata_verisi_olustur(self, t, values, errors):
        return {'t':t, 'values':values, 'errors':errors}

    def test_normal_veri_satir_sayisi_eslesemeli(self):
        error_results = {
            "dosya.csv | kolon1": self.hata_verisi_olustur(
                [0.0, 1.0, 2.0], [10.0, 20.0, 30.0],
                {"OVERSHOOT": [1], "SPIKE": [], "OUT OF RANGE": [], "CONSTANT OUTPUT": [], "INVALID SIGNAL": []}
            )
        }
        self.pencere.raporu_guncelle(error_results)
        self.assertEqual(self.pencere.rapor_tablo.rowCount(), 1)

    def test_bos_error_results(self):
        self.pencere.raporu_guncelle({})
        self.assertEqual(self.pencere.rapor_tablo.rowCount(), 0)

    def test_siralama_dogru_olmali(self):
        error_results = {
            "dosya.csv | kolon1": self.hata_verisi_olustur(
                [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0],
                {"OVERSHOOT": [2], "SPIKE": [1], "OUT OF RANGE": [3],
                 "CONSTANT OUTPUT": [0], "INVALID SIGNAL": [4]}
            )
        }
        self.pencere.raporu_guncelle(error_results)
        sira = [self.pencere.rapor_tablo.item(i, 3).text()
                for i in range(self.pencere.rapor_tablo.rowCount())]
        self.assertEqual(sira, ["OVERSHOOT", "SPIKE", "OUT OF RANGE", "CONSTANT OUTPUT", "INVALID SIGNAL"])

    def test_hucre_icerikleri_dogru_olmali(self):
        error_results = {
            "dosya.csv | kolon1": self.hata_verisi_olustur(
                [1.5], [42.123456], {"OVERSHOOT": [0], "SPIKE": [],
                                     "OUT OF RANGE": [], "CONSTANT OUTPUT": [], "INVALID SIGNAL": []}
            )
        }
        self.pencere.raporu_guncelle(error_results)
        self.assertEqual(self.pencere.rapor_tablo.item(0, 0).text(), "dosya.csv | kolon1")
        self.assertEqual(self.pencere.rapor_tablo.item(0, 1).text(), "1.500")
        self.assertEqual(self.pencere.rapor_tablo.item(0, 2).text(), "42.123456")
        self.assertEqual(self.pencere.rapor_tablo.item(0, 3).text(), "OVERSHOOT")

    def test_label_ayrac_yoksa_gosterim_adi_label_olmali(self):
        error_results = {
            "sadece_label": self.hata_verisi_olustur(
                [0.0], [1.0], {"OVERSHOOT": [0], "SPIKE": [],
                               "OUT OF RANGE": [], "CONSTANT OUTPUT": [], "INVALID SIGNAL": []}
            )
        }
        self.pencere.raporu_guncelle(error_results)
        self.assertEqual(self.pencere.rapor_tablo.item(0, 0).text(), "sadece_label")

    def test_coklu_degisken_toplam_satir_dogru_olmali(self):
        error_results = {
            "dosya.csv | kolon1": self.hata_verisi_olustur(
                [0.0, 1.0], [10.0, 20.0],
                {"OVERSHOOT": [0], "SPIKE": [1], "OUT OF RANGE": [],
                 "CONSTANT OUTPUT": [], "INVALID SIGNAL": []}
            ),
            "dosya.csv | kolon2": self.hata_verisi_olustur(
                [0.0, 1.0], [5.0, 6.0],
                {"OVERSHOOT": [], "SPIKE": [], "OUT OF RANGE": [0],
                 "CONSTANT OUTPUT": [], "INVALID SIGNAL": []}
            )
        }
        self.pencere.raporu_guncelle(error_results)
        self.assertEqual(self.pencere.rapor_tablo.rowCount(), 3)

    def test_ayni_index_farkli_hata_tipleri_ayri_satir_olmali(self):
        error_results = {
            'dosya.csv|kolon1': self.hata_verisi_olustur(
                [0.0],[999.0],
                {'OVERSHOOT': [0], 'SPIKE':[], 'OUT OF RANGE':[0],
                 'CONSTANT OUTPUT':[], 'INVALID SIGNAL':[]}
            )
        }
        self.pencere.raporu_guncelle(error_results)
        self.assertEqual(self.pencere.rapor_tablo.rowCount(),2)

    def test_zaman_ve_deger_format_dogru_olmali(self):
        error_results = {
            "dosya.csv | kolon1": self.hata_verisi_olustur(
                [1.123456789], [3.14159265],
                {"OVERSHOOT": [0], "SPIKE": [], "OUT OF RANGE": [],
                 "CONSTANT OUTPUT": [], "INVALID SIGNAL": []}
            )
        }
        self.pencere.raporu_guncelle(error_results)
        self.assertEqual(self.pencere.rapor_tablo.item(0, 1).text(), "1.123")
        self.assertEqual(self.pencere.rapor_tablo.item(0, 2).text(), "3.141593")


class TestOpsDropdownGuncelle(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent = MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)

    def test_her_iki_dropdown_dogru_icerik_olmali(self):
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1', 'dosya.csv | kolon2': 'U2'}
        self.pencere.op_data = {'Op1': MagicMock(), 'Op2': MagicMock()}
        self.pencere.ops_dropdown_guncelle()
        sol_items = [self.pencere.sol_operand.itemText(i) for i in range(self.pencere.sol_operand.count())]
        sag_items = [self.pencere.sag_operand_combo.itemText(i) for i in range(self.pencere.sag_operand_combo.count())]
        self.assertEqual(sol_items, sag_items)
        for item in ['U1', 'U2', 'Op1', 'Op2']:
            self.assertIn(item, sol_items)

    def test_label_map_bos_sadece_op_isimleri(self):
        self.pencere.label_map = {}
        self.pencere.op_data = {'Op1': MagicMock()}
        self.pencere.ops_dropdown_guncelle()
        sol_items = [self.pencere.sol_operand.itemText(i) for i in range(self.pencere.sol_operand.count())]
        self.assertEqual(sol_items, ['Op1'])

    def test_op_data_bos_sadece_u_parametreleri_olmali(self):
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1'}
        self.pencere.op_data = {}
        self.pencere.ops_dropdown_guncelle()
        sol_items = [self.pencere.sol_operand.itemText(i) for i in range(self.pencere.sol_operand.count())]
        self.assertEqual(sol_items, ['U1'])

    def test_ikisi_de_bos_dropdown_bos_olmali(self):
        self.pencere.label_map={}
        self.pencere.op_data = {}
        self.pencere.ops_dropdown_guncelle()
        self.assertEqual(self.pencere.sol_operand.count(), 0)
        self.assertEqual(self.pencere.sag_operand_combo.count(),0)

    def test_siralama_u_parametreleri_once_op_sonra_olmali(self):
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1'}
        self.pencere.op_data = {'Op1': MagicMock()}
        self.pencere.ops_dropdown_guncelle()
        sol_items = [self.pencere.sol_operand.itemText(i) for i in range(self.pencere.sol_operand.count())]
        self.assertEqual(sol_items.index('U1'), 0)
        self.assertEqual(sol_items.index('Op1'), 1)

    def test_onceki_cagri_ogeleri_kalmamali(self):
        self.pencere.label_map = {'dosya.csv|kolon1': 'U1','dosya.csv|kolon2': 'U2'}
        self.pencere.op_data = {}
        self.pencere.ops_dropdown_guncelle()
        self.pencere.label_map = {'dosya.csv|kolon1': 'U1'}
        self.pencere.ops_dropdown_guncelle()
        self.assertEqual(self.pencere.sol_operand.count(), 1)

    def test_toplam_eleman_sayisi_eslesemeli(self):
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1', 'dosya.csv | kolon2': 'U2'}
        self.pencere.op_data = {'Op1': MagicMock(), 'Op2': MagicMock(), 'Op3': MagicMock()}
        self.pencere.ops_dropdown_guncelle()
        self.assertEqual(self.pencere.sol_operand.count(), 5)
        self.assertEqual(self.pencere.sag_operand_combo.count(), 5)


class TestEnableLegendToggle(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        ax.plot([1,2],[3,4], label = 'test')
        ax.legend()
        parent = MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)

    def line_olustur(self, label='line1'):
        ax = self.pencere.canvas.figure.axes[0]
        line, = ax.plot([1,2],[3,4], label=label)
        return line

    def scatter_olustur(self, label='scatter1'):
        ax = self.pencere.canvas.figure.axes[0]
        sc = ax.scatter([1,2], [3,4], label=label)
        return sc

    def test_lines_ve_scatter_legend_a_girmeli(self):
        line = self.line_olustur('L1')
        sc = self.scatter_olustur('S1')
        self.pencere.enable_legend_toggle([line], [sc])
        ax = self.pencere.canvas.figure.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn('L1', legend_labels)
        self.assertIn('S1', legend_labels)

    def test_sadece_lines_legend_a_girmeli(self):
        line = self.line_olustur('L1')
        self.pencere.enable_legend_toggle([line])
        ax = self.pencere.canvas.figure.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn('L1', legend_labels)
        self.assertEqual(len(legend_labels), 1)

    def test_sadece_scatter_legend_a_girmeli(self):
        sc = self.scatter_olustur('S1')
        self.pencere.enable_legend_toggle([], [sc])
        ax = self.pencere.canvas.figure.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn('S1', legend_labels)

    def test_ikisi_de_bos_legend_olusturulmamali(self):
        ax = self.pencere.canvas.figure.axes[0]
        if ax.get_legend():
            ax.get_legend().remove()
        self.pencere.enable_legend_toggle([],[])
        self.assertIsNone(ax.get_legend())

    def test_line_map_dogru_eslesmeli(self):
        line = self.line_olustur('L1')
        self.pencere.enable_legend_toggle([line])
        values = list(self.pencere.line_map.values())
        self.assertIn(line, values)

    def test_legend_draggable_olmali(self):
        line = self.line_olustur('L1')
        self.pencere.enable_legend_toggle([line])
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        self.assertTrue(legend.get_draggable())

    def test_pick_connection_baglanmali(self):
        line = self.line_olustur('L1')
        self.pencere.pick_connection = None
        self.pencere.enable_legend_toggle([line])
        self.assertIsNotNone(self.pencere.pick_connection)

    def test_onceki_pick_connection_disconnect_edilmeli(self):
        line = self.line_olustur('L1')
        self.pencere.enable_legend_toggle([line])
        ilk_connection = self.pencere.pick_connection
        line2 = self.line_olustur('L2')
        self.pencere.enable_legend_toggle([line2])
        self.assertNotEqual(self.pencere.pick_connection, ilk_connection)

    def test_all_labels_dogru_olmali(self):
        line1 = self.line_olustur('L1')
        line2 = self.line_olustur('L2')
        self.pencere.enable_legend_toggle([line1, line2])
        ax = self.pencere.canvas.figure.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(legend_labels, ['L1','L2'])

    def test_scatter_list_none_gecilirse_hata_olmali(self):
        line = self.line_olustur('L1')
        try:
            self.pencere.enable_legend_toggle([line], None)
        except Exception as e:
            self.fail(f'Beklenmedik hata: {e}')

    def test_lines_bos_scatter_varsa_calismali(self):
        sc = self.scatter_olustur('S1')
        try:
            self.pencere.enable_legend_toggle([], [sc])
        except Exception as e:
            self.fail(f'Beklemedik hata: {e}')
        ax = self.pencere.canvas.figure.axes[0]
        self.assertIsNotNone(ax.get_legend())


class TestMatlabExport(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent=MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)
        self.pencere.plotted_data = {}
        self.pencere.scatter_data = {}
        self.pencere.op_data = {}
        self.pencere.label_map = {}

    def test_secim_yoksa_warning_gosterilmeli(self):
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.matlab_export()
            mock_warning.assert_called_once()

    def test_dosya_yolu_secilmezse_savemat_cagrilmamali_2(self):
        self.pencere.matlab_listesi.addItem('U1')
        self.pencere.matlab_listesi.selectAll()
        with patch('main.QFileDialog.getSaveFileName', return_value=('','')):
            with patch('main.savemat') as mock_savemat:
                self.pencere.matlab_export()
                mock_savemat.assert_not_called()

    def test_plotted_data_mat_dict_e_girmeli(self):
        self.pencere.plotted_data = {'dosya.csv | kolon1': (np.array([0.0, 1.0]), np.array([10.0, 20.0]))}
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1'}
        self.pencere.matlab_listesi.addItem('U1')
        self.pencere.matlab_listesi.selectAll()
        with patch('main.QFileDialog.getSaveFileName', return_value=('test.mat', '')):
            with patch('main.savemat') as mock_savemat:
                with patch('main.QMessageBox.information'):
                    self.pencere.matlab_export()
                    args = mock_savemat.call_args[0]
                    mat_dict = args[1]
                    self.assertIn('U1_t', mat_dict)
                    self.assertIn('U1_v', mat_dict)

    def test_scatter_data_mat_dict_e_girmeli(self):
        self.pencere.scatter_data = {'U1 - OVERSHOOT': (np.array([1.0]), np.array([5.0]))}
        self.pencere.matlab_listesi.addItem('U1 - OVERSHOOT')
        self.pencere.matlab_listesi.selectAll()
        with patch('main.QFileDialog.getSaveFileName', return_value=('test.mat', '')):
            with patch('main.savemat') as mock_savemat:
                with patch('main.QMessageBox.information'):
                    self.pencere.matlab_export()
                    args = mock_savemat.call_args[0]
                    mat_dict = args[1]
                    self.assertTrue(any('_t' in k for k in mat_dict))

    def test_op_data_mat_dict_e_girmeli(self):
        self.pencere.op_data = {'Op1': (np.array([0.0, 1.0]), np.array([1.0, 2.0]), 'U1+U2', None, None, None, None)}
        self.pencere.matlab_listesi.addItem('Op1 U1+U2')
        self.pencere.matlab_listesi.selectAll()
        with patch('main.QFileDialog.getSaveFileName', return_value=('test.mat', '')):
            with patch('main.savemat') as mock_savemat:
                with patch('main.QMessageBox.information'):
                    self.pencere.matlab_export()
                    args = mock_savemat.call_args[0]
                    mat_dict = args[1]
                    self.assertTrue(any('_t' in k for k in mat_dict))

    def test_secilemeyen_parametre_mat_dict_e_girmemeli(self):
        self.pencere.plotted_data = {
            'dosya.csv | kolon1': (np.array([0.0]), np.array([10.0])),
            'dosya.csv | kolon2': (np.array([0.0]), np.array([20.0]))
        }
        self.pencere.label_map = {
            'dosya.csv | kolon1':'U1',
            'dosya.csv | kolon2':'U2'
        }
        self.pencere.matlab_listesi.addItem('U1')
        self.pencere.matlab_listesi.addItem('U2')
        self.pencere.matlab_listesi.item(0).setSelected(True)
        self.pencere.matlab_listesi.item(1).setSelected(False)
        with patch('main.QFileDialog.getSaveFileName', return_value=('test.mat','')):
            with patch('main.savemat') as mock_savemat:
                with patch('main.QMessageBox.information'):
                    self.pencere.matlab_export()
                    args = mock_savemat.call_args[0]
                    mat_dict = args[1]
                    self.assertIn('U1_t',mat_dict)
                    self.assertNotIn('U2_t', mat_dict)

    def test_ozel_karakterler_temizlenmeli(self):
        self.pencere.plotted_data = {'dosya.csv | kolon1': (np.array([0.0]), np.array([1.0]))}
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1 [m/s]'}
        self.pencere.matlab_listesi.addItem('U1 [m/s]')
        self.pencere.matlab_listesi.selectAll()
        with patch('main.QFileDialog.getSaveFileName', return_value=('test.mat', '')):
            with patch('main.savemat') as mock_savemat:
                with patch('main.QMessageBox.information'):
                    self.pencere.matlab_export()
                    args = mock_savemat.call_args[0]
                    mat_dict = args[1]
                    self.assertFalse(any(' ' in k or '[' in k or '/' in k for k in mat_dict if k != 'labels'))

    def test_uzun_isim_kisaltilmali(self):
        uzun_isim = 'U' + 'x' * 70
        self.pencere.plotted_data = {'dosya.csv | kolon1': (np.array([0.0]), np.array([1.0]))}
        self.pencere.matlab_listesi.addItem(uzun_isim)
        self.pencere.matlab_listesi.selectAll()
        with patch('main.QFileDialog.getSaveFileName', return_value=('test.mat', '')):
            with patch('main.savemat') as mock_savemat:
                with patch('main.QMessageBox.information'):
                    self.pencere.matlab_export()
                    args = mock_savemat.call_args[0]
                    mat_dict = args[1]
                    for k in mat_dict:
                        if k != 'labels':
                            self.assertLessEqual(len(k),63)

    def test_cakisan_isimde_sayi_eklemeli(self):
        self.pencere.plotted_data = {
            'dosya.csv | kolon1': (np.array([0.0]), np.array([1.0])),
            'dosya.csv | kolon2': (np.array([0.0]), np.array([2.0]))
        }
        self.pencere.label_map = {
            'dosya.csv | kolon1': 'U1',
            'dosya.csv | kolon2': 'U2'
        }
        self.pencere.matlab_listesi.addItem('U1')
        self.pencere.matlab_listesi.selectAll()
        with patch('main.QFileDialog.getSaveFileName', return_value=('test.mat', '')):
            with patch('main.savemat') as mock_savemat:
                with patch('main.QMessageBox.information'):
                    self.pencere.matlab_export()
                    args = mock_savemat.call_args[0]
                    mat_dict = args[1]
                    t_keys = [k for k in mat_dict if k.endswith('_t')]
                    self.assertGreater(len( t_keys),0)

    def test_labels_anahtari_her_zaman_girmeli(self):
        self.pencere.plotted_data = {'dosya.csv|kolon1': (np.array([0.0]), np.array([1.0]))}
        self.pencere.label_map={'dosya.csv|kolon1':'U1'}
        self.pencere.matlab_listesi.addItem('U1')
        self.pencere.matlab_listesi.selectAll()
        with patch('main.QFileDialog.getSaveFileName', return_value=('test.mat','')):
            with patch('main.savemat') as mock_savemat:
                with patch('main.QMessageBox.information'):
                    self.pencere.matlab_export()
                    args = mock_savemat.call_args[0]
                    mat_dict = args[1]
                    self.assertIn('labels', mat_dict)

    def test_savemat_basariliysa_information_gosterilmeli(self):
        self.pencere.plotted_data = {'dosya.csv | kolon1': (np.array([0.0]), np.array([1.0]))}
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1'}
        self.pencere.matlab_listesi.addItem('U1')
        self.pencere.matlab_listesi.selectAll()
        with patch('main.QFileDialog.getSaveFileName', return_value=('test.mat', '')):
            with patch('main.savemat'):
                with patch('main.QMessageBox.information') as mock_info:
                    self.pencere.matlab_export()
                    mock_info.assert_called_once()


class TestOnePick(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent = MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)

    def test_artist_gorunurluk_true_dan_false(self):
        ax = self.pencere.canvas.figure.axes[0]
        line, = ax.plot([1, 2], [3,4], label='L1')
        line.set_visible(True)
        legend_handle = MagicMock()
        self.pencere.line_map = {legend_handle: line}
        event = MagicMock()
        event.artist = legend_handle
        self.pencere.on_pick(event)
        self.assertFalse(line.get_visible())

    def test_artist_gorunurluk_false_dan_true(self):
        ax = self.pencere.canvas.figure.axes[0]
        line, = ax.plot([1,2], [3,4], label = 'L1')
        line.set_visible(False)
        legend_handle = MagicMock()
        self.pencere.line_map = {legend_handle: line}
        event = MagicMock()
        event.artist = legend_handle
        self.pencere.on_pick(event)
        self.assertTrue(line.get_visible())

    def test_line_mapte_olmayan_artist_hata_firlatmamali(self):
        self.pencere.line_map = {}
        event = MagicMock()
        event.artist = MagicMock()
        try:
            self.pencere.on_pick(event)
        except Exception as e:
            self.fail(f'Beklenmedik hata: {e}')

    def test_toggle_sonrasi_draw_idle_cagrilmali(self):
        ax = self.pencere.canvas.figure.axes[0]
        line, = ax.plot([1,2],[3,4], label ='L1')
        legend_handle = MagicMock()
        self.pencere.line_map = {legend_handle: line}
        event = MagicMock()
        event.artist = legend_handle
        self.pencere.canvas.draw_idle = MagicMock()
        self.pencere.on_pick(event)
        self.pencere.canvas.draw_idle.assert_called_once()

    def test_line_mapte_olmayan_Artist_draw_dile_cagrilmamali(self):
        self.pencere.line_map =  {}
        event = MagicMock()
        event.artist = MagicMock()
        self.pencere.canvas.draw_idle = MagicMock()
        self.pencere.on_pick(event)
        self.pencere.canvas.draw_idle.assert_not_called()


class TestCanvasDoubleClick(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent = MagicMock()
        parent.acik_grafiler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)
        self.pencere.toolbar.home=MagicMock()

    def test_sol_tiklamada_toolbar_home_cagrilmali(self):
        event = MagicMock()
        event.button.return_value = Qt.MouseButton.LeftButton
        self.pencere.canvas_double_click(event)
        self.pencere.toolbar.home.assert_called_once()

    def test_sag_tiklamada_toolbar_home_cagrilmamali(self):
        event = MagicMock()
        event.button.return_value = Qt.MouseButton.RightButton
        self.pencere.canvas_double_click(event)
        self.pencere.toolbar.home.assert_not_called()

    def test_orta_tiklamada_tooblar_home_cagrilmamali(self):
        event = MagicMock()
        event.button.return_value = Qt.MouseButton.MiddleButton
        self.pencere.canvas_double_click(event)
        self.pencere.toolbar.home.assert_not_called()


class TestOperandVerisiAl(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent = MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)
        self.pencere.op_data={}
        self.pencere.ters_label_map = {}
        self.pencere.plotted_data = {}

    def test_op_data_dolu_t_v_donmeli(self):
        t = np.array([0.0, 1.0])
        v = np.array([10.0, 20.0])
        self.pencere.op_data = {'Op1': (t,v, 'aciklama', None, None, None, None)}
        t_sonuc, v_sonuc = self.pencere._operand_verisini_al('Op1')
        np.testing.assert_array_equal(t_sonuc, t)
        np.testing.assert_array_equal(v_sonuc, v)

    def test_op_data_da_olan_isim_none_donmeli(self):
        self.pencere.op_data = {'Op1':(None, None, 'aciklama', None, None, None, None)}
        t_sonuc, v_sonuc = self.pencere._operand_verisini_al('Op1')
        self.assertIsNone(t_sonuc)
        self.assertIsNone(v_sonuc)

    def test_ters_label_map_plotted_data_eslesmeli(self):
        t = np.array([0.0, 1.0])
        v = np.array([5.0, 6.0])
        self.pencere.ters_label_map = {'U1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {'dosya.csv | kolon1': (t,v)}
        t_sonuc, v_sonuc = self.pencere._operand_verisini_al('U1')
        np.testing.assert_array_equal(t_sonuc, t)
        np.testing.assert_array_equal(v_sonuc, v)

    def test_ters_label_var_op_data_yok_none_donmeli(self):
        self.pencere.ters_label_map = {'U1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {}
        t_sonuc, v_sonuc = self.pencere._operand_verisini_al('U1')
        self.assertIsNone(t_sonuc)
        self.assertIsNone(v_sonuc)

    def test_hicbir_yerde_yok_none_donmeli(self):
        t_sonuc, v_sonuc = self.pencere._operand_verisini_al('Olmayan')
        self.assertIsNone(t_sonuc)
        self.assertIsNone(v_sonuc)


class TestIfadeyiHesapla(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent = MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)
        self.pencere.op_data = {}
        self.pencere.ters_label_map = {}
        self.pencere.plotted_data = {}
        self.pencere.label_map = {}

    def test_gecerli_u_referansi_t_v_donmeli(self):
        t = np.array([0.0, 1.0, 2.0])
        v = np.array([1.0, 2.0, 3.0])
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1 kolon1'}
        self.pencere.ters_label_map = {'U1 kolon1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {'dosya.csv | kolon1': (t, v)}
        t_sonuc, v_sonuc = self.pencere._ifadeyi_hesapla('Op1', 'U1')
        np.testing.assert_array_equal(t_sonuc, t)
        np.testing.assert_array_equal(v_sonuc, v)

    def test_iki_u_referasni_toplama(self):
        t = np.array([0.0, 1.0, 2.0])
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        self.pencere.label_map = {
            'dosya.csv | kolon1': 'U1 kolon1',
            'dosya.csv | kolon2': 'U2 kolon2'
        }
        self.pencere.ters_label_map = {
            'U1 kolon1': 'dosya.csv | kolon1',
            'U2 kolon2': 'dosya.csv | kolon2'
        }
        self.pencere.plotted_data = {
            'dosya.csv | kolon1': (t, v1),
            'dosya.csv | kolon2': (t, v2)
        }
        t_sonuc, v_sonuc = self.pencere._ifadeyi_hesapla('Op1', 'U1+U2')
        np.testing.assert_array_equal(v_sonuc, v1+v2)

    def test_u_referansi_label_mapte_yoksa_none_donmeli(self):
        self.pencere.label_map = {}
        with patch('main.QMessageBox.warning') as mock_warning:
            t_sonuc, v_sonuc = self.pencere._ifadeyi_hesapla('Op1', 'U1')
            self.assertIsNone(t_sonuc)
            self.assertIsNone(v_sonuc)
            mock_warning.assert_called_once()

    def test_op_ref_op_datada_varsa_dogru_hesaplanmali(self):
        t = np.array([0.0, 1.0, 2.0])
        v = np.array([2.0, 4.0, 6.0])
        self.pencere.op_data = {'Op1': (t, v, 'aciklama', None, None, None)}
        t_sonuc, v_sonuc = self.pencere._ifadeyi_hesapla('Op2', 'Op1*2')
        np.testing.assert_array_equal(v_sonuc, v*2)

    def test_op_referansi_op_data_da_yoksa_none_donmeli(self):
        self.pencere.op_data = {}
        with patch('main.QMessageBox.warning') as mock_warning:
            t_sonuc, v_sonuc = self.pencere._ifadeyi_hesapla('Op2', 'Op1')
            self.assertIsNone(t_sonuc)
            self.assertIsNone(v_sonuc)
            mock_warning.assert_called_once()

    def test_hic_referans_yoksa_none_donmeli(self):
        with patch('main.QMessageBox.warning') as mock_warning:
            t_sonuc, v_sonuc = self.pencere._ifadeyi_hesapla('Op1', '42')
            self.assertIsNone(t_sonuc)
            self.assertIsNone(v_sonuc)
            mock_warning.assert_called_once()

    def test_gecersiz_ifade_yoksa_none_donmeli(self):
        t = np.array([0.0, 1.0])
        v = np.array([1.0, 2.0])
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1 kolon1'}
        self.pencere.ters_label_map = {'U1 kolon1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {'dosya.csv | kolon1': (t, v)}
        with patch('main.QMessageBox.warning') as mock_warning:
            t_sonuc, v_sonuc = self.pencere._ifadeyi_hesapla('Op1', 'U1 +++')
            self.assertIsNone(t_sonuc)
            self.assertIsNone(v_sonuc)
            mock_warning.assert_called_once()

    def test_farkli_uzunlukta_diziler_min_len_e_gore_kisaltilmali(self):
        t1 = np.array([0.0, 1.0, 2.0])
        v1 = np.array([1.0, 2.0, 3.0])
        t2 = np.array([0.0, 1.0])
        v2 = np.array([4.0, 5.0])
        self.pencere.label_map = {
            'dosya.csv | kolon1': 'U1 kolon1',
            'dosya.csv | kolon2': 'U2 kolon2'
        }
        self.pencere.ters_label_map = {
            'U1 kolon1': 'dosya.csv | kolon1',
            'U2 kolon2': 'dosya.csv | kolon2'
        }
        self.pencere.plotted_data = {
            'dosya.csv | kolon1': (t1, v1),
            'dosya.csv | kolon2': (t2, v2)
        }
        t_sonuc, v_sonuc = self.pencere._ifadeyi_hesapla('Op1', 'U1+U2')
        self.assertEqual(len(v_sonuc), 2)

    def test_matematik_fonksiyon_dogru_calisimali(self):
        t = np.array([0.0, 1.0, 2.0])
        v = np.array([0.0, np.pi / 2, np.pi])
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1 kolon1'}
        self.pencere.ters_label_map = {'U1 kolon1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {'dosya.csv | kolon1': (t, v)}
        t_sonuc, v_sonuc = self.pencere._ifadeyi_hesapla('Op1', 'sin(U1)')
        np.testing.assert_array_almost_equal(v_sonuc, np.sin(v))

    def test_tehlikeli_ifade_none_donmeli(self):
        t = np.array([0.0, 1.0])
        v = np.array([1.0, 2.0])
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1 kolon1'}
        self.pencere.ters_label_map = {'U1 kolon1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {'dosya.csv | kolon1': (t, v)}
        with patch('main.QMessageBox.warning') as mock_warning:
            t_sonuc, v_sonuc = self.pencere._ifadeyi_hesapla('Op1', '__import__("os")')
            self.assertIsNone(t_sonuc)
            self.assertIsNone(v_sonuc)
            mock_warning.assert_called_once()


class TestOperasyonEkle(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent = MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)
        self.pencere.op_data = {}
        self.pencere.ters_label_map = {}
        self.pencere.plotted_data = {}
        self.pencere.label_map = {}

    def test_parametre_modunda_satir_ekle(self):
        self.pencere.radio_parametre.setChecked(True)
        self.pencere.sol_operand.addItem('U1')
        self.pencere.sol_operand.setCurrentText('U1')
        self.pencere.sag_operand_combo.addItem('U2')
        self.pencere.sag_operand_combo.setCurrentText('U2')
        self.pencere.operator_combo.setCurrentText('+')
        self.pencere.operasyon_ekle()
        self.assertEqual(self.pencere.op_listesi.rowCount(),1)

    def test_parametre_modunda_op_data_dogru_girmeli(self):
        self.pencere.radio_parametre.setChecked(True)
        self.pencere.sol_operand.addItem('U1')
        self.pencere.sol_operand.setCurrentText('U1')
        self.pencere.sag_operand_combo.addItem('U2')
        self.pencere.sag_operand_combo.setCurrentText('U2')
        self.pencere.operator_combo.setCurrentText('+')
        self.pencere.operasyon_ekle()
        self.assertIn('Op1', self.pencere.op_data)
        kayit = self.pencere.op_data['Op1']
        self.assertEqual(kayit[2],'U1 + U2')
        self.assertIsNone(kayit[0])
        self.assertIsNone(kayit[1])

    def test_sabit_mod_gecerli_sayi_satir_ekle(self):
        self.pencere.radio_sabit.setChecked(True)
        self.pencere.sol_operand.addItem('U1')
        self.pencere.sol_operand.setCurrentText('U1')
        self.pencere.sag_operand_sabit.setText('3.14')
        self.pencere.operator_combo.setCurrentText('*')
        self.pencere.operasyon_ekle()
        self.assertEqual(self.pencere.op_listesi.rowCount(), 1)

    def test_sabit_modunda_bos_input_warning_vermeli(self):
        self.pencere.radio_sabit.setChecked(True)
        self.pencere.sag_operand_sabit.setText('')
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.operasyon_ekle()
            mock_warning.assert_called_once()
        self.assertEqual(self.pencere.op_listesi.rowCount(), 0)

    def test_sabit_modunda_gecersiz_sayi_warning_vermeli(self):
        self.pencere.radio_sabit.setChecked(True)
        self.pencere.sag_operand_sabit.setText('abc')
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.operasyon_ekle()
            mock_warning.assert_called_once()
        self.assertEqual(self.pencere.op_listesi.rowCount(),0)

    def test_op_sayac_her_eklemede_artmali(self):
        self.pencere.radio_parametre.setChecked(True)
        self.pencere.sol_operand.addItem('U1')
        self.pencere.sag_operand_combo.addItem('U2')
        self.pencere.operasyon_ekle()
        self.pencere.operasyon_ekle()
        self.assertIn('Op1', self.pencere.op_data)
        self.assertIn('Op2', self.pencere.op_data)

    def test_ops_dropdown_guncelle_cagrilmali(self):
        self.pencere.radio_parametre.setChecked(True)
        self.pencere.sol_operand.addItem('U1')
        self.pencere.sag_operand_combo.addItem('U2')
        self.pencere.ops_dropdown_guncelle = MagicMock()
        self.pencere.operasyon_ekle()
        self.pencere.ops_dropdown_guncelle.assert_called_once()


class TestOperasyonSil(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent = MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)
        self.pencere.op_data = {}
        self.pencere.label_map = {}
        self.pencere.ters_label_map = {}
        self.pencere.plotted_data = {}

    def satir_ekle(self, op_isim, aciklama):
        satir = self.pencere.op_listesi.rowCount()
        self.pencere.op_listesi.insertRow(satir)
        self.pencere.op_listesi.setItem(satir, 0, QTableWidgetItem(op_isim))
        self.pencere.op_listesi.setItem(satir, 1, QTableWidgetItem(aciklama))
        self.pencere.op_data[op_isim] = (None, None, aciklama, None, None, None, False)

    def test_satir_op_listesinden_kaldirilmali(self):
        self.satir_ekle('Op1', 'U1 + U2')
        with patch('main.QMessageBox.information'):
            with patch.object(self.pencere, 'operasyonlari_uygula'):
                self.pencere.operasyon_sil(0)
        self.assertEqual(self.pencere.op_listesi.rowCount(), 0)

    def test_satir_op_data_dan_kaldirilmali(self):
        self.satir_ekle('Op1', 'U1 + U2')
        with patch('main.QMessageBox.information'):
            with patch.object(self.pencere, 'operasyonlari_uygula'):
                self.pencere.operasyon_sil(0)
        self.assertNotIn('Op1', self.pencere.op_data)

    def test_bagimli_op_da_silinmeli(self):
        self.satir_ekle('Op1', 'U1 + U2')
        self.satir_ekle('Op2', 'Op1 * 2')
        with patch('main.QMessageBox.information'):
            with patch.object(self.pencere, 'operasyonlari_uygula'):
                self.pencere.operasyon_sil(0)
        self.assertNotIn('Op1', self.pencere.op_data)
        self.assertNotIn('Op2', self.pencere.op_data)
        self.assertEqual(self.pencere.op_listesi.rowCount(), 0)

    def test_op_isim_item_none_ise_erken_return(self):
        self.pencere.op_listesi.insertRow(0)
        self.pencere.ops_dropdown_guncelle = MagicMock()
        self.pencere.operasyon_sil(0)
        self.pencere.ops_dropdown_guncelle.assert_called_once()

    def test_ops_dropdown_guncelle_her_durumda_cagirilmali(self):
        self.satir_ekle('Op1', 'U1 + U2')
        self.pencere.ops_dropdown_guncelle = MagicMock()
        with patch('main.QMessageBox.information'):
            with patch.object(self.pencere, 'operasyonlari_uygula'):
                self.pencere.operasyon_sil(0)
        self.pencere.ops_dropdown_guncelle.assert_called_once()


class TestIfadeEkle(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent = MagicMock
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)
        self.pencere.label_map = {}
        self.pencere.ters_label_map = {}
        self.pencere.plotted_data = {}

    def test_bos_ifade_warning_vermeli(self):
        self.pencere.ifade_giris.setText('')
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.ifade_ekle()
            mock_warning.assert_called_once()
        self.assertEqual(self.pencere.op_listesi.rowCount(), 0)

    def test_bosluktan_olusan_ifade_warning(self):
        self.pencere.ifade_giris.setText('   ')
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.ifade_ekle()
            mock_warning.assert_called_once()
        self.assertEqual(self.pencere.op_listesi.rowCount(), 0)

    def test_gecerli_ifade_girilirse_satir_ekle(self):
        self.pencere.ifade_giris.setText('U1+U2')
        self.pencere.ifade_ekle()
        self.assertEqual(self.pencere.op_listesi.rowCount(),1)

    def test_gecerli_ifade_girilince_op_data_ya_kayit_girmeli(self):
        self.pencere.ifade_giris.setText('U1+U2')
        self.pencere.ifade_ekle()
        self.assertIn('Op1', self.pencere.op_data)
        kayit=self.pencere.op_data['Op1']
        self.assertEqual(kayit, (None, None, 'U1+U2', None, None, None, False),1)

    def test_op_ismi_op_sayac_formatinda_olmali_sayac_artmali(self):
        self.pencere.ifade_giris.setText('U1+U2')
        self.pencere.ifade_ekle()
        self.pencere.ifade_giris.setText('U1*U2')
        self.pencere.ifade_ekle()
        self.assertIn('Op1', self.pencere.op_data)
        self.assertIn('Op2', self.pencere.op_data)

    def test_ops_dropdown_guncelle_cagrilmali(self):
        self.pencere.ifade_giris.setText('U1+U2')
        self.pencere.ops_dropdown_guncelle = MagicMock()
        self.pencere.ifade_ekle()
        self.pencere.ops_dropdown_guncelle.assert_called_once()

    def test_operasyon_sonrasi_ifade_giris_temzilenmeli(self):
        self.pencere.ifade_giris.setText('U1+U2')
        self.pencere.ifade_ekle()
        self.assertEqual(self.pencere.ifade_giris.text(),'')

    def test_silme_butonu_widget_olarak_eklenmeli(self):
        self.pencere.ifade_giris.setText('U1+U2')
        self.pencere.ifade_ekle()
        widget = self.pencere.op_listesi.cellWidget(0, 2)
        self.assertIsNotNone(widget)
        self.assertIsInstance(widget, QPushButton)


class TestOpSilButon(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent = MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)
        self.pencere.label_map = {}
        self.pencere.ters_label_map = {}
        self.pencere.plotted_data = {}

    def satir_ekle(self, op_isim, aciklama):
        satir = self.pencere.op_listesi.rowCount()
        self.pencere.op_listesi.insertRow(satir)
        self.pencere.op_listesi.setItem(satir, 0, QTableWidgetItem(op_isim) )
        self.pencere.op_listesi.setItem(satir, 1, QTableWidgetItem(aciklama))
        btn = QPushButton('🗑️')
        self.pencere.op_listesi.setCellWidget(satir, 2, btn)
        self.pencere.op_data[op_isim] = (None, None, aciklama, None, None, None, False)
        return btn

    def test_dogru_indx_ile_operasyon_sil_cagirmali(self):
        btn = self.satir_ekle('Op1', 'U1+U2')
        with patch.object(self.pencere, 'operasyon_sil') as mock_sil:
            self.pencere._op_sil_buton(btn)
            mock_sil.assert_called_once_with(0)

    def test_olmayan_butonda_operasyon_sil_cagirilmali(self):
        self.satir_ekle('Op1', 'U1+U2')
        yabanci_btn = QPushButton('🗑️')
        with patch.object(self.pencere, 'operasyon_sil') as mock_sil:
            self.pencere._op_sil_buton(yabanci_btn)
            mock_sil.assert_not_called()

    def test_coklu_satirda_dogru_satir_silinmeli(self):
        self.satir_ekle('Op1', 'U1 + U2')
        btn2 = self.satir_ekle('Op2', 'U1 * 2')
        with patch.object(self.pencere, 'operasyon_sil') as mock_sil:
            self.pencere._op_sil_buton(btn2)
            mock_sil.assert_called_once_with(1)


class TestOperasyonlariUygula(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        parent = MagicMock()
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        plt.close(fig)
        self.pencere.op_data = {}
        self.pencere.label_map = {}
        self.pencere.ters_label_map = {}
        self.pencere.plotted_data = {}
        self.pencere.scatter_list = []

    def op_satir_ekle(self, op_isim, aciklama, sol=None, op=None, sag=None, sabit_mi=False):
        satir = self.pencere.op_listesi.rowCount()
        self.pencere.op_listesi.insertRow(satir)
        self.pencere.op_listesi.setItem(satir, 0, QTableWidgetItem(op_isim))
        self.pencere.op_listesi.setItem(satir, 1, QTableWidgetItem(aciklama))
        self.pencere.op_data[op_isim] = (None, None, aciklama, sol, op, sag, sabit_mi)

    def test_axes_bos_erken_return(self):
        self.pencere.canvas.figure.clf()
        self.pencere.enable_legend_toggle = MagicMock()
        self.pencere.operasyonlari_uygula()
        self.pencere.enable_legend_toggle.assert_not_called()

    def test_eski_op_cizgileri_kaldirilamli(self):
        ax = self.pencere.canvas.figure.axes[0]
        line, = ax.plot([0,1],[0,1])
        line.op_cizgisi = True
        self.pencere.operasyonlari_uygula()
        self.assertNotIn(line, ax.get_lines())

    def test_op_cizgisi_olmayanlar_line_dan_kaldirilmali(self):
        ax = self.pencere.canvas.figure.axes[0]
        line, = ax.plot([0,1],[0,1], label='U1')
        self.pencere.operasyonlari_uygula()
        self.assertIn(line, ax.get_lines())

    def test_serbest_ifade_line_eklenmeli_op_data_guncellenmeli(self): #?
        t = np.array([0.0, 1.0])
        v = np.array([1.0, 2.0])
        self.pencere.label_map = {'dosya.csv | kolon1': 'U1 kolon1'}
        self.pencere.ters_label_map = {'U1 kolon1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {'dosya.csv | kolon1': (t, v)}
        self.op_satir_ekle('Op1', 'U1')
        ax = self.pencere.canvas.figure.axes[0]
        onceki_line_sayisi = len(ax.get_lines())
        self.pencere.operasyonlari_uygula()
        self.assertGreater(len(ax.get_lines()), onceki_line_sayisi)
        self.assertIsNotNone(self.pencere.op_data['Op1'][0])

    def test_serbest_ifade_none_donerse_line_eklenmemeli(self):
        self.op_satir_ekle('Op1', 'OLMAYAN')
        ax = self.pencere.canvas.figure.axes[0]
        onceki_line_sayisi = len(ax.get_lines())
        with patch('main.QMessageBox.warning'):
            self.pencere.operasyonlari_uygula()
        self.assertEqual(len(ax.get_lines()), onceki_line_sayisi)

    def test_sol_operand_bulunmazsa_warning_verilmeli(self):
        self.op_satir_ekle('Op1', 'U1+U2', sol='U1', op='+', sag='U2', sabit_mi=False)
        ax = self.pencere.canvas.figure.axes[0]
        onceki_line_sayisi = len(ax.get_lines())
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.operasyonlari_uygula()
            mock_warning.assert_called_once()
        self.assertEqual(len(ax.get_lines()), onceki_line_sayisi)

    def test_sag_operand_verisi_bulunmazsa_warning_verilmeli(self):
        t = np.array([0.0, 1.0])
        v = np.array([1.0, 2.0])
        self.pencere.ters_label_map = {'U1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {'dosya.csv | kolon1': (t, v)}
        self.op_satir_ekle('Op1', 'U1 + U2', sol='U1', op='+', sag='U2', sabit_mi=False)
        ax = self.pencere.canvas.figure.axes[0]
        onceki_line_sayisi = len(ax.get_lines())
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.operasyonlari_uygula()
            mock_warning.assert_called_once()
        self.assertEqual(len(ax.get_lines()), onceki_line_sayisi)

    def test_sabit_modunda_toplam_dogru_sonuc(self):
        t = np.array([0.0, 1.0])
        v = np.array([1.0, 2.0])
        self.pencere.ters_label_map = {'U1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {'dosya.csv | kolon1': (t, v)}
        self.op_satir_ekle('Op1', 'U1 + 3.0', sol='U1', op='+', sag='3.0', sabit_mi=True)
        self.pencere.operasyonlari_uygula()
        v_sonuc = self.pencere.op_data['Op1'][1]
        np.testing.assert_array_almost_equal(v_sonuc, np.array([4.0, 5.0]))

    def test_sabit_modunda_gecersiz_sabit_warning_vermeli(self):
        t = np.array([0.0, 1.0])
        v = np.array([1.0, 2.0])
        self.pencere.ters_label_map = {'U1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {'dosya.csv | kolon1': (t, v)}
        self.op_satir_ekle('Op1', 'U1 + abc', sol ='U1', op='+', sag='abc', sabit_mi=True)
        ax = self.pencere.canvas.figure.axes[0]
        onceki_line_sayisi = len(ax.get_lines())
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.operasyonlari_uygula()
            mock_warning.assert_called_once()
        self.assertEqual(len(ax.get_lines()), onceki_line_sayisi)

    def test_parametre_modunda_toplam_dogru_sonuc(self):
        t = np.array([0.0, 1.0])
        v1 = np.array([1.0, 2.0])
        v2 = np.array([3.0, 4.0])
        self.pencere.ters_label_map = {
            'U1': 'dosya.csv | kolon1',
            'U2': 'dosya.csv | kolon2'
        }
        self.pencere.plotted_data = {
            'dosya.csv | kolon1': (t, v1),
            'dosya.csv | kolon2': (t, v2)
        }
        self.op_satir_ekle('Op1', 'U1+U2', sol='U1', op='+', sag='U2', sabit_mi=False)
        self.pencere.operasyonlari_uygula()
        v_sonuc = self.pencere.op_data['Op1'][1]
        np.testing.assert_array_almost_equal(v_sonuc, np.array([4.0, 6.0]))

    def test_sifira_bolme_nan_uretmeli(self):
        t = np.array([0.0, 1.0])
        v1 = np.array([1.0, 2.0])
        v2 = np.array([0.0, 2.0])
        self.pencere.ters_label_map = {
            'U1': 'dosya.csv | kolon1',
            'U2': 'dosya.csv | kolon2'
        }
        self.pencere.plotted_data = {
            'dosya.csv | kolon1': (t, v1),
            'dosya.csv | kolon2': (t, v2)
        }
        self.op_satir_ekle('Op1', 'U1 / U2', sol='U1', op='/', sag='U2', sabit_mi=False)
        try:
            self.pencere.operasyonlari_uygula()
        except Exception as e:
            self.fail(f'Beklenmedik hata: {e}')
        v_sonuc = self.pencere.op_data['Op1'][1]
        self.assertTrue(np.isnan(v_sonuc[0]))
        self.assertAlmostEqual(v_sonuc[1], 1.0)

    def test_farkli_uzunlukta_diziler_min_len_e_gore_kisaltilmali(self):
        t1 = np.array([0.0, 1.0, 2.0])
        v1 = np.array([1.0, 2.0, 3.0])
        t2 = np.array([0.0, 1.0])
        v2 = np.array([4.0, 5.0])
        self.pencere.ters_label_map = {
            'U1': 'dosya.csv | kolon1',
            'U2': 'dosya.csv | kolon2'
        }
        self.pencere.plotted_data = {
            'dosya.csv | kolon1': (t1, v1),
            'dosya.csv | kolon2': (t2, v2)
        }
        self.op_satir_ekle('Op1', 'U1 + U2', sol='U1', op='+', sag='U2', sabit_mi=False)
        self.pencere.operasyonlari_uygula()
        v_sonuc = self.pencere.op_data['Op1'][1]
        self.assertEqual(len(v_sonuc), 2)

    def test_yeni_op_label_matlab_listesine_eklenmeli(self):
        t = np.array([0.0, 1.0])
        v = np.array([1.0, 2.0])
        self.pencere.ters_label_map = {'U1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {'dosya.csv | kolon1': (t, v)}
        self.op_satir_ekle('Op1', 'U1 + 1.0', sol='U1', op='+', sag='1.0', sabit_mi=True)
        self.pencere.operasyonlari_uygula()
        matlab_items = [self.pencere.matlab_listesi.item(i).text()
                        for i in range(self.pencere.matlab_listesi.count())]
        self.assertIn('Op1 U1 + 1.0', matlab_items)

    def test_mmevcut_op_label_tekrar_eklenmemeli(self):
        t = np.array([0.0, 1.0])
        v = np.array([1.0, 2.0])
        self.pencere.ters_label_map = {'U1': 'dosya.csv | kolon1'}
        self.pencere.plotted_data = {'dosya.csv | kolon1': (t, v)}
        self.op_satir_ekle('Op1', 'U1 + 1.0', sol='U1', op='+', sag='1.0', sabit_mi=True)
        self.pencere.operasyonlari_uygula()
        matlab_items = [self.pencere.matlab_listesi.item(i).text()
                        for i in range(self.pencere.matlab_listesi.count())]
        self.assertEqual(matlab_items.count('Op1 U1 + 1.0'), 1)

    def test_enable_legend_toggle_cagrilmali(self):
        self.pencere.enable_legend_toggle = MagicMock()
        self.pencere.operasyonlari_uygula()
        self.pencere.enable_legend_toggle.assert_called_once()

    def test_canvas_draw_idle_cagrilmali(self):
        self.pencere.canvas.draw_idle = MagicMock()
        self.pencere.operasyonlari_uygula()
        self.pencere.canvas.draw_idle.assert_called_once()


class AnaPencereTestBase(unittest.TestCase):

    def setUp(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.pencere = AnaPencere()
        self.pencere.figures = {1: {'params': [], 'mode': 'realtime', 'units': []}}
        self.pencere.current_figure = 1


class TestDosyaAra(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.dosya_listesi.addItem('test.csv')
        self.pencere.dosya_listesi.addItem('flight_data.csv')
        self.pencere.dosya_listesi.addItem('sensor_log.csv')

    def test_eslesen_dosya_gorunur_olmali(self):
        self.pencere.dosya_ara('test')
        self.assertFalse(self.pencere.dosya_listesi.item(0).isHidden())

    def test_hicbir_dosya_eslesmiyorsa_gizlenmeli(self):
        self.pencere.dosya_ara('xyz123')
        for i in range(self.pencere.dosya_listesi.count()):
            self.assertTrue(self.pencere.dosya_listesi.item(i).isHidden())

    def test_buyuk_kucuk_harf_duyarsiz_arama(self):
        self.pencere.dosya_ara('TEST')
        self.assertFalse(self.pencere.dosya_listesi.item(0).isHidden())

    def test_bos_arama_tum_dosyalar_gorunur(self):
        self.pencere.dosya_ara('xyz123')
        self.pencere.dosya_ara('')
        for i in range(self.pencere.dosya_listesi.count()):
            self.assertFalse(self.pencere.dosya_listesi.item(i).isHidden())

    def test_kismi_eslesmede_dogru_filtreleme(self):
        self.pencere.dosya_ara('flight')
        self.assertTrue(self.pencere.dosya_listesi.item(0).isHidden())
        self.assertFalse(self.pencere.dosya_listesi.item(1).isHidden())
        self.assertTrue(self.pencere.dosya_listesi.item(2).isHidden())


class TestDosyaSecimiTemizle(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.dosya_listesi.addItem('test.csv')
        self.pencere.dosya_listesi.addItem('flight_data.csv')
        self.pencere.dosya_listesi.addItem('sensor_log.csv')

    def test_secili_dosyalar_varken_secim_temizlenmeli(self):
        self.pencere.dosya_listesi.item(0).setSelected(True)
        self.pencere.dosya_secimi_temizle()
        self.assertEqual(len(self.pencere.dosya_listesi.selectedItems()), 0)

    def test_secim_yokken_hata_firlatmamali(self):
        try:
            self.pencere.dosya_secimi_temizle()
        except Exception as e:
            self.fail(f'Beklenmedik hata: {e}')


class TestParametreAra(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.parametre_listesi.clear()
        self.pencere.parametre_listesi.addItem('dosya.csv | gyro_x')
        self.pencere.parametre_listesi.addItem('dosya.csv | velocity')
        self.pencere.parametre_listesi.addItem('dosya.csv | altitude')

    def test_eslesen_parametre_gorunur_olmali(self):
        self.pencere.parametre_ara('gyro')
        self.assertFalse(self.pencere.parametre_listesi.item(0).isHidden())

    def test_hicbir_parametre_eslesmiyorsa_tum_gizlenmeli(self):
        self.pencere.parametre_ara('xyz123')
        for i in range(self.pencere.parametre_listesi.count()):
            self.assertTrue(self.pencere.parametre_listesi.item(i).isHidden())

    def test_buyuk_kucuk_harf_duyarsiz_arama(self):
        self.pencere.parametre_ara('GYRO')
        self.assertFalse(self.pencere.parametre_listesi.item(0).isHidden())

    def test_bos_arama_tum_parametreler_gorunur(self):
        self.pencere.parametre_ara('xyz123')
        self.pencere.parametre_ara('')
        for i in range(self.pencere.parametre_listesi.count()):
            self.assertFalse(self.pencere.parametre_listesi.item(i).isHidden())

    def test_kismi_eslesmede_dogru_filtreleme(self):
        self.pencere.parametre_ara('velocity')
        self.assertTrue(self.pencere.parametre_listesi.item(0).isHidden())
        self.assertFalse(self.pencere.parametre_listesi.item(1).isHidden())
        self.assertTrue(self.pencere.parametre_listesi.item(2).isHidden())


class TestParametreSecimiTemizle(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.parametre_listesi.clear()
        self.pencere.parametre_listesi.addItem('dosya.csv | gyro_x')
        self.pencere.parametre_listesi.addItem('dosya.csv | velocity')

    def test_secili_parametre_varken_cagrilinca_temizlenmeli(self):
        self.pencere.parametre_listesi.item(0).setSelected(True)
        self.pencere.parametre_secimi_temizle()
        self.assertEqual(len(self.pencere.parametre_listesi.selectedItems()), 0)

    def test_secim_yokken_hata_firlatmamali(self):
        try:
            self.pencere.parametre_secimi_temizle()
        except Exception as e:
            self.fail(f'Beklenmedik hata: {e}')


class TestBirimDegisti(AnaPencereTestBase):

    def test_birim_guncellemeli(self):
        self.pencere.figures[1]['units'] = ['Default Units']
        self.pencere.birim_degisti(1, 0, 'm to ft')
        self.assertEqual(self.pencere.figures[1]['units'][0], 'm to ft')

    def test_farkli_idx_ler_ayri_kaydedilmeli(self):
        self.pencere.figures[1]['units'] = ['Default Units', 'Default Units']
        self.pencere.birim_degisti(1, 0, 'm to ft')
        self.pencere.birim_degisti(1, 1, 'deg to rad')
        self.assertEqual(self.pencere.figures[1]['units'][0], 'm to ft')
        self.assertEqual(self.pencere.figures[1]['units'][1], 'deg to rad')


class TestModeDegisti(AnaPencereTestBase):

    def test_realtime_modu_kaydedilmeli(self):
        self.pencere.radio_realtime.setChecked(True)
        self.pencere.mode_degisti()
        self.assertEqual(self.pencere.figures[self.pencere.current_figure]['mode'], 'realtime')

    def test_previous_modu_kaydedilmeli(self):
        self.pencere.radio_previous.setChecked(True)
        self.pencere.mode_degisti()
        self.assertEqual(self.pencere.figures[self.pencere.current_figure]['mode'], 'previous')

    def test_nearest_modu_kaydedilmeli(self):
        self.pencere.radio_nearest.setChecked(True)
        self.pencere.mode_degisti()
        self.assertEqual(self.pencere.figures[self.pencere.current_figure]['mode'], 'nearest')


class TestFigureDegisti(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.figures[2] = {'params': [], 'mode': 'nearest', 'units': []}
        self.pencere.figure_listesini_guncelle()

    def test_ust_seviye_item_current_figure_guncellenmeli(self):
        item = self.pencere.figure_listesi.topLevelItem(1)
        self.pencere.figure_degisti(item)
        self.assertEqual(self.pencere.current_figure, 2)

    def test_alt_item_current_figure_degismemeli(self):
        self.pencere.figures[1]['params'] = ['dosya.csv | gyro_x']
        self.pencere.figures[1]['units'] = ['Default Units']
        self.pencere.figure_listesini_guncelle()
        parent = self.pencere.figure_listesi.topLevelItem(0)
        child = parent.child(0)
        onceki = self.pencere.current_figure
        self.pencere.figure_degisti(child)
        self.assertEqual(self.pencere.current_figure, onceki)

    def test_previous_modu_radio_previous_secili_olmali(self):
        self.pencere.figures[1]['mode'] = 'previous'
        item = self.pencere.figure_listesi.topLevelItem(0)
        self.pencere.figure_degisti(item)
        self.assertTrue(self.pencere.radio_previous.isChecked())

    def test_realtime_modu_radio_realtime_seicli_olmali(self):
        self.pencere.figures[1]['mode'] = 'realtime'
        item = self.pencere.figure_listesi.topLevelItem(0)
        self.pencere.figure_degisti(item)
        self.assertTrue(self.pencere.radio_realtime.isChecked())

    def test_nearest_modu_radio_nearest_secili_olmali(self):
        self.pencere.figures[1]['mode'] = 'nearest'
        item = self.pencere.figure_listesi.topLevelItem(0)
        self.pencere.figure_degisti(item)
        self.assertTrue(self.pencere.radio_nearest.isChecked())


class TestParametreSil(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.figures[1]['params'] = ['dosya.csv | gyro_x', 'dosya.csv | velocity']
        self.pencere.figures[1]['units'] = ['Default Units', 'm to ft']
        self.pencere.figure_listesini_guncelle()

    def test_gecerli_idx_silinince_param_listesinden_cikarilmali(self):
        self.pencere.parametre_sil(1, 0)
        self.assertNotIn('dosya.csv | gyro_x', self.pencere.figures[1]['params'])

    def test_gecerli_idx_silinince_unit_listesinden_cikrailmali(self):
        self.pencere.parametre_sil(1, 0)
        self.assertNotIn('dosya.csv | gyro_x', self.pencere.figures[1]['units'])

    def test_figure_listesini_guncelle_cagrilmali(self):
        self.pencere.figure_listesini_guncelle = MagicMock()
        self.pencere.parametre_sil(1, 0)
        self.pencere.figure_listesini_guncelle.assert_called()

    def test_current_figure_degismemeli(self):
        onceki = self.pencere.current_figure
        self.pencere.parametre_sil(1, 0)
        self.assertEqual(self.pencere.current_figure, onceki)


class TestParametreyiFigureEkle(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.parametre_listesi.clear()
        self.pencere.parametre_listesi.addItem('dosya.csv | gyro_x')
        self.pencere.parametre_listesi.addItem('dosya.csv | velocity')

    def test_secili_parametre_yoksa_warning_verilmeli(self):
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.parametreyi_figure_ekle()
            mock_warning.assert_called_once()
        self.assertEqual(len(self.pencere.figures[1]['params']), 0)

    def test_secili_parametre_eklenmeli(self):
        self.pencere.parametre_listesi.item(0).setSelected(True)
        self.pencere.parametreyi_figure_ekle()
        self.assertIn('dosya.csv | gyro_x', self.pencere.figures[1]['params'])

    def test_default_units_eklenmeli(self):
        self.pencere.parametre_listesi.item(0).setSelected(True)
        self.pencere.parametreyi_figure_ekle()
        self.assertEqual(self.pencere.figures[1]['units'][0], 'Default Units')

    def test_ayni_parametre_tekrar_eklenmemeli(self):
        self.pencere.parametre_listesi.item(0).setSelected(True)
        self.pencere.parametreyi_figure_ekle()
        self.pencere.parametreyi_figure_ekle()
        self.assertEqual(self.pencere.figures[1]['params'].count('dosya.csv | gyro_x'), 1)

    def test_figure_listesini_guncelle_cagrilmali(self):
        self.pencere.parametre_listesi.item(0).setSelected(True)
        self.pencere.figure_listesini_guncelle = MagicMock()
        self.pencere.parametreyi_figure_ekle()
        self.pencere.figure_listesini_guncelle.assert_called_once()


class TestYeniFigureOlustur(AnaPencereTestBase):

    def test_yeni_figure_dict_e_eklenmeli(self):
        onceki_sayi = len(self.pencere.figures)
        self.pencere.yeni_figur_olustur()
        self.assertEqual(len(self.pencere.figures), onceki_sayi + 1)

    def test_yeni_figure_numarasi_dogru_olmali(self):
        maks = max(self.pencere.figures.keys())
        self.pencere.yeni_figur_olustur()
        self.assertIn(maks + 1, self.pencere.figures)

    def test_current_figure_guncellenmeli(self):
        maks = max(self.pencere.figures.keys())
        self.pencere.yeni_figur_olustur()
        self.assertEqual(self.pencere.current_figure, maks + 1)

    def test_yeni_figure_dogru_yapi_ile_olusturulmali(self):
        self.pencere.yeni_figur_olustur()
        yeni_no = max(self.pencere.figures.keys())
        self.assertEqual(self.pencere.figures[yeni_no]['params'], [])
        self.assertEqual(self.pencere.figures[yeni_no]['mode'], 'realtime')
        self.assertEqual(self.pencere.figures[yeni_no]['units'], [])

    def test_figure_listesini_guncelle_cagrilmali(self):
        self.pencere.figure_listesini_guncelle = MagicMock()
        self.pencere.yeni_figur_olustur()
        self.pencere.figure_listesini_guncelle.assert_called_once()


class TestFigureSil(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.figures[2] = {'params': [], 'mode': 'realtime', 'units': []}
        self.pencere.current_figure = 2
        self.pencere.figure_listesini_guncelle()

    def test_silinen_figure_dict_ten_kaldirilmali(self):
        self.pencere.figure_sil()
        self.assertNotIn(2, self.pencere.figures)

    def test_current_figure_kalan_figure_a_guncellenmeli(self):
        self.pencere.figure_sil()
        self.assertIn(self.pencere.current_figure, self.pencere.figures)

    def test_tum_figureler_silinince_varsayilan_olusturulmali(self):
        del self.pencere.figures[1]
        self.pencere.figure_sil()
        self.assertIn(1, self.pencere.figures)
        self.assertEqual(self.pencere.current_figure, 1)

    def test_acik_grafikler_kapatilmali(self):
        mock_pencere = MagicMock()
        mock_pencere.figure_no = 2
        self.pencere.acik_grafikler = [mock_pencere]
        self.pencere.figure_sil()
        mock_pencere.close.assert_called_once()

    def test_figure_listesini_guncelle_cagrilmali(self):
        self.pencere.figure_listesini_guncelle = MagicMock()
        self.pencere.figure_sil()
        self.pencere.figure_listesini_guncelle.assert_called_once()

class TestSecilenleriGetir(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.klasor = '/klasor'
        self.pencere.kolon_cache = {}
        self.pencere.alias_map = {}
        self.pencere.dosya_listesi.addItem('test.csv')
        self.pencere.dosya_listesi.addItem('flight.csv')

    def test_dosya_secili_degilse_warning_verillmeli(self):
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.secilenleri_getir()
            mock_warning.assert_called_once()
        self.assertEqual(self.pencere.parametre_listesi.count(), 0)

    def test_secili_dosya_kolonlari_listelenmeli(self):
        self.pencere.dosya_listesi.item(0).setSelected(True)
        self.pencere.kolon_cache['test.csv'] = ['gyro_x', 'gyro_y']
        self.pencere.secilenleri_getir()
        items = [self.pencere.parametre_listesi.item(i).text()
                 for i in range(self.pencere.parametre_listesi.count())]
        self.assertIn('test.csv | gyro_x', items),
        self.assertIn('test.csv | gyro_y', items)

    def test_cache_varsa_read_csv_cagrilmamali(self):
        self.pencere.dosya_listesi.item(0).setSelected(True)
        self.pencere.kolon_cache['test.csv'] = ['gyro_x']
        with patch('main.pd.read_csv') as mock_read:
               self.pencere.secilenleri_getir()
               mock_read.assert_not_called()

    def test_alias_map_teki_dosya_alias_ile_listelenmeli(self):
        self.pencere.dosya_listesi.item(0).setSelected(True)
        self.pencere.kolon_cache['test.csv'] = ['gyro_x']
        self.pencere.alias_map['test.csv'] = 'IMU'
        self.pencere.secilenleri_getir()
        items = [self.pencere.parametre_listesi.item(i).text()
                for i in range(self.pencere.parametre_listesi.count())]
        self.assertIn('IMU | gyro_x', items)
        self.assertNotIn('test.csv | gyro_x', items)

    def test_okunamayan_dosya_warning_verilmeli(self):
            self.pencere.dosya_listesi.item(0).setSelected(True)
            with patch('main.pd.read_csv', side_effect=Exception('dosya bulunamadı')):
                with patch('main.QMessageBox.warning') as mock_warning:
                    self.pencere.secilenleri_getir()
                    mock_warning.assert_called_once()

    def test_coklu_dosya_seciliyse_hepsi_listelenmeli(self):
          self.pencere.dosya_listesi.item(0).setSelected(True)
          self.pencere.dosya_listesi.item(1).setSelected(True)
          self.pencere.kolon_cache['test.csv'] = ['gyro_x']
          self.pencere.kolon_cache['flight.csv'] = ['velocity']
          self.pencere.secilenleri_getir()
          items = [self.pencere.parametre_listesi.item(i).text()
                for i in range(self.pencere.parametre_listesi.count())]
          self.assertIn('test.csv | gyro_x', items)
          self.assertIn('flight.csv | velocity', items)


class TestParametreSagTik(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.klasor = '/klasor'
        self.pencere.alias_map = {}
        self.pencere.kolon_cache = {}
        self.pencere.parametre_listesi.addItem('test.csv | gyro_x')
        self.pos = QPoint(0, 0)

    def test_item_yoksa_erken_return(self):
        self.pencere.parametre_listesi.itemAt = MagicMock(return_value=None)
        with patch('main.QMenu') as mock_menu:
            self.pencere.parametre_sag_tik(self.pos)
            mock_menu.assert_not_called()

    def test_ayrac_yoksa_alias_kaydedilmemeli(self):
        self.pencere.parametre_listesi.clear()
        self.pencere.parametre_listesi.addItem('sadece_label')
        item = self.pencere.parametre_listesi.item(0)
        self.pencere.parametre_listesi.itemAt = MagicMock(return_value=item)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText') as mock_input:
                self.pencere.parametre_sag_tik(self.pos)
                mock_input.assert_not_called()

    def test_iptal_olursa_alias_kaydedilmemeli(self):
        item = self.pencere.parametre_listesi.item(0)
        self.pencere.parametre_listesi.itemAt = MagicMock(return_value=item)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('yeni_alias', False)):
                self.pencere.parametre_sag_tik(self.pos)
        self.assertNotIn('test.csv', self.pencere.alias_map)

    def test_bos_alias_kaydedilmemeli(self):
        item = self.pencere.parametre_listesi.item(0)
        self.pencere.parametre_listesi.itemAt = MagicMock(return_value=item)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('', True )):
                self.pencere.parametre_sag_tik(self.pos)
        self.assertNotIn('test.csv', self.pencere.alias_map)

    def test_gecerli_alias_kaydedilmeli(self):
        item = self.pencere.parametre_listesi.item(0)
        self.pencere.parametre_listesi.itemAt = MagicMock(return_value=item)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('IMU', True)):
                    with patch.object(self.pencere, 'alias_kaydet'):
                        with patch.object(self.pencere, 'secilenleri_getir'):
                            self.pencere.parametre_sag_tik(self.pos)
        self.assertEqual(self.pencere.alias_map.get('test.csv'), 'IMU')

    def test_kullanilan_alias_warning_verilmeli(self):
        self.pencere.alias_map = {'baska.csv': 'IMU'}
        item = self.pencere.parametre_listesi.item(0)
        self.pencere.parametre_listesi.itemAt = MagicMock(return_value=item)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('IMU', True)):
                with patch('main.QMessageBox.warning') as mock_warning:
                    self.pencere.parametre_sag_tik(self.pos)
                    mock_warning.assert_called_once()
        self.assertNotIn('test.csv', self.pencere.alias_map)

    def test_gecerli_alias_alias_kaydet_cagrilmali(self):
        item = self.pencere.parametre_listesi.item(0)
        self.pencere.parametre_listesi.itemAt = MagicMock(return_value=item)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('IMU', True)):
                with patch.object(self.pencere, 'alias_kaydet') as mock_kaydet:
                    with patch.object(self.pencere, 'secilenleri_getir'):
                        self.pencere.parametre_sag_tik(self.pos)
                        mock_kaydet.assert_called_once()

    def test_gecerli_alias_secilenleri_getir_cagrilmali(self):
        item = self.pencere.parametre_listesi.item(0)
        self.pencere.parametre_listesi.itemAt = MagicMock(return_value=item)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('IMU', True)):
                with patch.object(self.pencere, 'alias_kaydet'):
                    with patch.object(self.pencere, 'secilenleri_getir') as mock_getir:
                        self.pencere.parametre_sag_tik(self.pos)
                        mock_getir.assert_called_once()


class TestAliasKaydet(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.klasor = '/klasor'
        self.pencere.alias_map = {'test.csv': 'IMU'}
        self.pencere.figures = {
            1: {'params': ['IMU | gyro_x', 'diger.csv | velocity'], 'mode': 'realtime', 'units':[]}
        }

    def test_klasor_bossa_erken_return(self):
        self.pencere.klasor = ''
        with patch('main.json.dump') as mock_json:
            self.pencere.alias_kaydet()
            mock_json.assert_not_called()

    def test_alias_map_csv_ye_yazilmali(self):
        with patch('builtins.open', mock_open()):
            with patch('main.json.dump') as mock_json:
                with patch('main.os.makedirs'):
                    with patch.object(self.pencere, 'figure_listesini_guncelle'):
                        self.pencere.alias_kaydet()
                        mock_json.assert_called_once()

    def test_eski_alias_figure_larda_guncellenmeli(self):
        with patch('main.pd.DataFrame.to_csv'):
            with patch.object(self.pencere, 'figure_listesini_guncelle'):
                self.pencere.alias_kaydet(eski_alias='IMU', orijinal='test.csv')
        self.assertIn('IMU | gyro_x', self.pencere.figures[1]['params'])

    def test_diger_parametreler_degismemeli(self):
        with patch('main.pd.DataFrame.to_csv'):
            with patch.object(self.pencere, 'figure_listesini_guncelle'):
                self.pencere.alias_kaydet(eski_alias='IMU', orijinal='test.csv')
        self.assertIn('diger.csv | velocity', self.pencere.figures[1]['params'])

    def test_figure_listesini_guncelle_her_durumda_cagrilmali(self):
        with patch('builtins.open', mock_open()):
            with patch('main.json.dump'):
                with patch('main.os.makedirs'):
                    with patch.object(self.pencere, 'figure_listesini_guncelle') as mock_guncelle:
                        self.pencere.alias_kaydet()
                        mock_guncelle.assert_called_once()


class TestPlotBas(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.klasor = '/klasor'
        self.pencere.alias_map = {}
        self.pencere.data_cache = {}
        self.pencere.acik_grafikler = []
        self.pencere.figures = {
            1: {'params': ['test.csv | gyro_x'], 'mode': 'previous', 'units': ['Default Units']}
        }
        self.pencere.current_figure = 1

        self.sahte_fig, self.sahte_ax = plt.subplots()
        self.sahte_line, = self.sahte_ax.plot([0, 1], [0, 1], label='U1 gyro_x')
        self.sahte_line.gercek_label = 'test.csv | gyro_x'
        self.sahte_plotted_data = {'test.csv | gyro_x': (np.array([0.0, 1.0]), np.array([1.0, 2.0]))}

        self.grafikleri_ciz_return = (
            self.sahte_fig,
            [self.sahte_line],
            self.sahte_plotted_data,
            []
        )

    def tearDown(self):
        plt.close('all')

    def test_secili_parametre_yoksa_warning_verilmeli(self):
        self.pencere.figures[1]['params'] = []
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.plot_bas()
            mock_warning.assert_called_once()

    def test_cache_varsa_dosyalari_yukle_cagrilmamali(self):
        sahte_df = pd.DataFrame({'time': [0.0, 1.0], 'gyro_x':[1.0, 2.0]})
        self.pencere.data_cache[os.path.join('/klasor', 'test.csv')] = sahte_df
        with patch('main.grafikleri_ciz', return_value=self.grafikleri_ciz_return):
            with patch('main.dosyalari_yukle') as mock_yukle:
                with patch('main.GrafikPenceresi'):
                    self.pencere.plot_bas()
                    mock_yukle.assert_not_called()

    def test_cache_yoksa_dosyalari_yukle_cagrilmali(self):
        sahte_df = pd.DataFrame({'time': [0.0, 1.0], 'gyro_x': [1.0, 2.0]})
        with patch('main.grafikleri_ciz', return_value=self.grafikleri_ciz_return):
            with patch('main.dosyalari_yukle', return_value={'test.csv': sahte_df}) as mock_yukle:
                with patch('main.GrafikPenceresi'):
                    self.pencere.plot_bas()
                    mock_yukle.assert_called_once()
        self.assertIn(os.path.join('/klasor', 'test.csv'), self.pencere.data_cache)

    def test_grafikleri_ciz_dogru_parametrelerle_cagrilmali(self):
        sahte_df = pd.DataFrame({'time': [0.0, 1.0], 'gyro_x': [1.0, 2.0]})
        self.pencere.data_cache['/klasor/test.csv'] = sahte_df
        with patch('main.dosyalari_yukle'):
            with patch('main.grafikleri_ciz', return_value=self.grafikleri_ciz_return) as mock_ciz:
                with patch('main.GrafikPenceresi'):
                    self.pencere.plot_bas()
                    args, kwargs = mock_ciz.call_args
                    self.assertEqual(kwargs.get('alias_map'), self.pencere.alias_map)
                    self.assertEqual(args[2], 'previous')
                    self.assertEqual(args[3], 1)

    def test_grafikleri_ciz_uyari_dondururse_warning_verilmeli(self):
        sahte_df = pd.DataFrame({'time': [0.0, 1.0], 'gyro_x': [1.0, 2.0]})
        self.pencere.data_cache['/klasor/test.csv'] = sahte_df
        uyarili_return = (self.sahte_fig, [self.sahte_line], self.sahte_plotted_data, ['uyarı mesajı'])
        with patch('main.dosyalari_yukle'):
            with patch('main.grafikleri_ciz', return_value=uyarili_return):
                with patch('main.GrafikPenceresi'):
                    with patch('main.QMessageBox.warning') as mock_warning:
                        self.pencere.plot_bas()
                        mock_warning.assert_called_once()

    def test_eski_pencere_kapatilmali(self):
        sahte_df = pd.DataFrame({'time':[0.0, 1.0], 'gyro_x': [1.0, 2.0]})
        self.pencere.data_cache['/klasor/test.csv'] = sahte_df
        eski_pencere = MagicMock()
        eski_pencere.figure_no = 1
        self.pencere.acik_grafikler = [eski_pencere]
        with patch('main.dosyalari_yukle'):
            with patch('main.grafikleri_ciz', return_value=self.grafikleri_ciz_return):
                with patch('main.GrafikPenceresi'):
                    self.pencere.plot_bas()
                    eski_pencere.close.assert_called_once()

    def test_yeni_pencere_acik_grafiklere_eklenmeli(self):
        sahte_df = pd.DataFrame({'time': [0.0, 1.0], 'gyro_x': [1.0, 2.0]})
        self.pencere.data_cache['/klasor/test.csv'] = sahte_df
        with patch('main.dosyalari_yukle'):
            with patch('main.grafikleri_ciz', return_value=self.grafikleri_ciz_return):
                with patch('main.GrafikPenceresi') as mock_pencere_cls:
                    self.pencere.plot_bas()
                    self.assertIn(mock_pencere_cls.return_value, self.pencere.acik_grafikler)

    def test_hata_firlatilinca_critical_gostermeli(self):
        with patch('main.dosyalari_yukle', side_effect=DosyaHatasi('hata')):
            with patch('main.QMessageBox.critical') as mock_critical:
                self.pencere.plot_bas()
                mock_critical.assert_called_once()


class TestAnalyzeErrors2(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.current_figure = 1
        self.pencere.error_results = {}
        self.pencere.acik_grafikler = []

        #Sahte error_loader
        self.pencere.error_loader = MagicMock()
        self.pencere.error_loader.error_tables = {
            'EGID_RS422': {
                'gyro_x': {'min': -10, 'max': 10, 'max_change': 1, 'is_eml': False}
            },
            'SMU_ERROR_CLASS': {
                'dPressure': {'min': 0, 'max': 100, 'max_change': 5, 'is_eml': False},
                'pressure': {'min': 0, 'max': 100, 'max_change': 5, 'is_eml': False}
            },
            'VMM_CLASS': {
                'dSpeed': {'min': 0, 'max': 100, 'max_change': 5, 'is_eml': False},
                'dSpeedFixed': {'min': 0, 'max': 100, 'max_change': 5, 'is_eml': False},
                'status': {'min': 0, 'max': 1, 'max_change': 0, 'is_eml': False}
            },
            'ADC_ERROR_CLASS_EML': {
                'gyro_x': {'min': -10, 'max': 10, 'max_change': 1, 'is_eml': True}
            }
        }

        #Sahte popup pencere
        self.popup = MagicMock()
        self.popup.figure_no = 1
        self.popup.plotted_data = {
            'test.csv | gyro_x': (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        }
        self.pencere.acik_grafikler.append(self.popup)

    def test_pencere_yoksa_warning_verilmeli(self):
        self.pencere.acik_grafikler = []
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.analyze_errors()
            mock_warning.assert_called_once()

    def test_plotted_data_bossa_warning_verilmeli(self):
        self.popup.plotted_data = {}
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.analyze_errors()
            mock_warning.assert_called_once()

    def test_ayrac_olmayan_label_atlanmali(self):
        self.popup.plotted_data = {
            'sadece_label': (np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        }
        with patch('main.QMessageBox.information'):
            self.pencere.analyze_errors()
        self.assertEqual(self.pencere.error_results[1], {})

    def test_special_none_ise_find_variable_system_cagrilmali(self):
        self.popup.plotted_data = {
            'test.csv | gyro_x': (np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        }
        with patch('main.check_special_variable', return_value=None):
            with patch('main.find_variable_system', return_value=None) as mock_find:
                with patch('main.QMessageBox.information'):
                    self.pencere.analyze_errors()
                    mock_find.assert_called_once()

    def test_special_none_degilse_error_results_a_girmeli(self):
        self.popup.plotted_data = {
            'test.csv | gyro_x': (np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        }
        ozel_sonuc = {'INVALID SIGNAL': [0]}
        with patch('main.check_special_variable', return_value=ozel_sonuc):
            with patch('main.QMessageBox.information'):
                self.pencere.analyze_errors()
        self.assertIn('test.csv | gyro_x', self.pencere.error_results[1])
        self.assertEqual(
            self.pencere.error_results[1]['test.csv | gyro_x']['errors'],
            ozel_sonuc
        )

    def test_error_loader_none_ise_analiz_yapilmamali(self):
        self.pencere.error_loader = None
        with patch('main.check_special_variable', return_value=None):
            with patch('main.QMessageBox.information'):
                self.pencere.analyze_errors()
        self.assertEqual(self.pencere.error_results[1], {})

    def test_find_variable_system_none_ise_label_atlanmali(self):
        with patch('main.check_special_variable', return_value=None):
            with patch('main.find_variable_system', return_value=None):
                with patch('main.QMessageBox.information'):
                    self.pencere.analyze_errors()
        self.assertEqual(self.pencere.error_results[1], {})

    def test_egid_sisteminde_tam_analiz_yapilmali(self):
        self.popup.plotted_data = {
            'test.csv | gyro_x': (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        }
        with patch('main.check_special_variable', return_value=None):
            with patch('main.find_variable_system', return_value='EGID_RS422'):
                with patch('main.analyze_errors', return_value={
                    'CONSTANT OUTPUT': [], 'OVERSHOOT': [], 'SPIKE': [], 'OUT OF RANGE': []
                }) as mock_analyze:
                    with patch('main.QMessageBox.information'):
                        self.pencere.analyze_errors()
                        args = mock_analyze.call_args[0]
                        self.assertNotEqual(args[3], 0)

    def test_eml_dosyasinda_max_change_10_kati_olmali(self):
        self.popup.plotted_data = {
            'EML_TEST.csv | gyro_x': (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        }
        with patch('main.check_special_variable', return_value=None):
            with patch('main.find_variable_system', return_value='EGID_RS422'):
                with patch('main.analyze_errors', return_value={
                    'CONSTANT OUTPUT': [], 'OVERSHOOT':[], 'SPIKE': [], 'OUT OF RANGE': []
                }) as mock_analyze:
                    with patch('main.QMessageBox.information'):
                        self.pencere.analyze_errors()
                        args = mock_analyze.call_args[0]
                        self.assertEqual(args[3], 10)

    def test_vmm_d_fixed_tam_analiz(self):
        self.popup.plotted_data = {
            'VMM_TEST.csv | dSpeedFixed': (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        }
        with patch('main.check_special_variable', return_value=None):
            with patch('main.find_variable_system', return_value='VMM_CLASS'):
                with patch('main.analyze_errors', return_value={
                    'CONSTANT OUTPUT': [], 'OVERSHOOT':[1], 'SPIKE': [], 'OUT OF RANGE': []
                })  as mock_analyze:
                    with patch('main.QMessageBox.information'):
                        self.pencere.analyze_errors()
                        mock_analyze.assert_called_once()

    def test_vmm_d_fixed_olmayan_overshoot_spike_bas(self):
        self.popup.plotted_data = {
            'VMM.test | dSpeed': (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        }
        with patch('main.check_special_variable', return_value=None):
            with patch('main.find_variable_system',return_value='VMM_CLASS'):
                with patch('main.QMessageBox.information'):
                    self.pencere.analyze_errors()
        label = 'VMM_TEST.csv | dSpeed'
        if label in self.pencere.error_results[1]:
            errors = self.pencere.error_results[1][label]['errors']
            self.assertEqual(errors['OVERSHOOT'], [])
            self.assertEqual(errors['SPIKE'], [])

    def test_smu_d_ile_baslamayan_overshoot_spike_bos(self):
        self.popup.plotted_data = {
            'SMU_TEST.csv | pressure': (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        }
        with patch('main.check_special_variable', return_value=None):
            with patch('main.find_variable_system', return_value='SMU_ERROR_CLASS'):
                with patch('main.QMessageBox.information'):
                    self.pencere.analyze_errors()
        label = 'SMU_TEST.csv | pressure'
        if label in self.pencere.error_results[1]:
            errors = self.pencere.error_results[1][label]['errors']
            self.assertEqual(errors['OVERSHOOT'], [])
            self.assertEqual(errors['SPIKE'], [])

    def test_analiz_tamamlaninca_information_gosterilmeli(self):
        with patch('main.check_special_variable', return_value=None):
            with patch('main.find_variable_system', return_value=None):
                with patch('main.QMessageBox.information') as mock_info:
                    self.pencere.analyze_errors()
                    mock_info.assert_called_once()


class TestErrorPlot(AnaPencereTestBase):

    def setUp(self):
        super().setUp()
        self.pencere.current_figure = 1
        self.pencere.error_results = {}
        self.pencere.acik_grafikler = []
        self.pencere.error_marker_map = {
            'OVERSHOOT': 'o',
            'SPIKE': 'x',
            'OUT OF RANGE': '^',
            'CONSTANT OUTPUT': 's',
            'INVALID SIGNAL': '+'
        }

        fig, ax = plt.subplots()
        self.popup = MagicMock()
        self.popup.figure_no = 1
        self.popup.canvas.figure = fig
        self.popup.label_map = {'test.csv | gyro_x': 'U1 gyro_x'}
        self.popup.matlab_listesi.count.return_value = 0
        self.popup.matlab_listesi.item = MagicMock(return_value=None)
        self.pencere.acik_grafikler.append(self.popup)

        self.pencere.error_results[1] = {
            'test.csv | gyro_x': {
                't': np.array([0.0, 1.0, 2.0]),
                'values': np.array([1.0, 2.0, 3.0]),
                'errors': {
                    'OVERSHOOT': [1],
                    'SPIKE': [],
                    'OUT OF RANGE': [],
                    'CONSTANT OUTPUT': [],
                    'INVALID SIGNAL': []
                }
            }
        }

    def tearDown(self):
        plt.close('all')

    def test_error_results_yoksa_warnig_verilmeli(self):
        self.pencere.error_results = {}
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.error_plot()
            mock_warning.assert_called_once()

    def test_pencere_yoksa_warning_verilmeli(self):
        self.pencere.acik_grafikler = []
        with patch('main.QMessageBox.warning') as mock_warning:
            self.pencere.error_plot()
            mock_warning.assert_called_once()

    def test_onceki_scatter_itemlari_temizlenmeli(self):
        item1 = MagicMock()
        item1.text.return_value = 'U1 - OVERSHOOT'
        item2 = MagicMock()
        item2.text.return_value = 'U1 gyro_x'

        self.popup.matlab_listesi.count.return_value = 2
        self.popup.matlab_listesi.item.side_effect = lambda i: [item1, item2][i]
        self.popup.matlab_listesi.findItems.return_value = [item1]

        self.pencere.error_plot()

        self.popup.matlab_listesi.takeItem.assert_called()

    def test_bos_hata_iindeksi_scatter_eklenmemeli(self):
        self.pencere.error_results[1] = {
            'test.csv | gyro_x':{
                't': np.array([0.0, 1.0, 2.0]),
                'values': np.array([1.0, 2.0, 3.0]),
                'errors':{
                    'OVERSHOOT': [],
                    'SPIKE': [],
                    'OUT OF RANGE': [],
                    'CONSTANT OUTPUT': [],
                    'INVALID SIGNAL': []
                }
            }
        }
        self.pencere.error_plot()
        ax = self.popup.canvas.figure.axes[0]
        self.assertEqual(len(ax.collections), 0)

    def test_dolu_hata_indeksi_scatter_eklenmeli(self):
        self.pencere.error_plot()
        ax = self.popup.canvas.figure.axes[0]
        self.assertGreater(len(ax.collections),0)

    def test_scatter_matlab_listesine_eklenmeli(self):
        self.pencere.error_plot()
        self.popup.matlab_listesi.addItem.assert_called()

    def test_enable_legend_toggle_cagrilmali(self):
        self.pencere.error_plot()
        self.popup.enable_legend_toggle.assert_called_once()

    def test_raporu_guncelle_cagrilmali(self):
        self.pencere.error_plot()
        self.popup.raporu_guncelle.assert_called_once_with(
            self.pencere.error_results[1]
        )

    def test_canvas_draw_idle_cagrilmali(self):
        self.pencere.error_plot()
        self.popup.canvas.draw_idle.assert_called_once()

    def test_scatter_data_dogru_doldurulmali(self):
        self.pencere.error_plot()
        self.assertIsNotNone(self.popup.scatter_data)
        self.popup.__setattr__('scatter_data', self.popup.scatter_data)


class TestGrafikleriCiz(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({
            'Timestamp': [1.0, 2.0, 3.0],
            'gyro_x': [1.0, 2.0, 3.0]
        })
        self.df2 = pd.DataFrame({
            'Timestamp': [1.0, 2.0, 3.0],
            'velocity': [10.0, 20.0, 30.0]
        })
        self.tum_veriler = {
            'test.csv': self.df1,
            'flight.csv': self.df2
        }
        self.parametre_map = {
            1: ('test.csv', 'gyro_x'),
            2: ('flight.csv', 'velocity')
        }

    def tearDown(self):
        plt.close('all')

    def test_realtime_modunda_t_ekseni_sifirlanmali(self):
        fig, lines, plotted_data, uyarilar = grafikleri_ciz(
            self.tum_veriler, {1: ('test.csv', 'gyro_x')}, 'realtime', 1
        )
        t_plot = plotted_data['test.csv | gyro_x'][0]
        self.assertAlmostEqual(t_plot[0], 0.0, places=5)

    def test_previous_modunda_t_ekseni_sifirlanmali(self):
        fig, lines, plotted_data, uyarilar = grafikleri_ciz(
            self.tum_veriler, {1: ('test.csv', 'gyro_x')}, 'previous', 1
        )
        t_plot = plotted_data['test.csv | gyro_x'][0]
        self.assertAlmostEqual(t_plot[0], 0.0, places=5)

    def test_nearest_modunda_t_ekseni_sifirlanmali(self):
        fig, lines, plotted_data, uyarilar = grafikleri_ciz(
            self.tum_veriler, {1: ('test.csv', 'gyro_x')}, 'nearest',1)
        t_plot = plotted_data['test.csv | gyro_x'][0]
        self.assertAlmostEqual(t_plot[0], 0.0, places=5)

    def test_dosya_yoksa_parametre_atlanmali(self):
        fig, lines, plotted_data, uyarilar = grafikleri_ciz(
            self.tum_veriler, {1: ('olmayan.csv', 'gyro_x')}, 'realtime', 1
        )
        self.assertEqual(len(lines), 0)
        self.assertEqual(len(plotted_data), 0)

    def test_kolon_yoksa_parametre_atlanmali(self):
        fig, lines, plotted_data, uyarilar = grafikleri_ciz(
            self.tum_veriler, {1: ('test.csv', 'olmayan_kolon')}, 'realtime', 1
        )
        self.assertEqual(len(lines), 0)

    def test_alias_map_verilmisse_label_alias_olmali(self):
        alias_map = {'test.csv': 'IMU'}
        fig, lines, plotted_data, uyarilar = grafikleri_ciz(
            self.tum_veriler, {1: ('test.csv', 'gyro_x')}, 'realtime', 1,
            alias_map=alias_map
        )
        self.assertTrue(any('IMU' in l.get_label() for l in lines))

    def test_unit_verilmisse_label_da_gosterilmeli(self):
        fig, lines, plotted_data, uyarilar = grafikleri_ciz(
            self.tum_veriler, {1: ('test.csv', 'gyro_x')}, 'realtime', 1,
            units=['deg to rad']
        )
        self.assertTrue(any('[deg to rad]' in l.get_label() for l in lines))

    def test_uyarilar_bos_liste_donmeli(self):
        fig, lines, plotted_data, uyarilar = grafikleri_ciz(
            self.tum_veriler, {1: ('test.csv', 'gyro_x')}, 'previous',1
        )
        self.assertIsInstance(uyarilar, list)
        self.assertEqual(len(uyarilar), 0)

    def test_plotted_data_da_gercek_label_olmali(self):
        fig, lines, plotted_data, uyarilar = grafikleri_ciz(
            self.tum_veriler, {1: ('test.csv', 'gyro_x')}, 'realtime', 1
        )
        self.assertIn('test.csv | gyro_x', plotted_data)

    def test_lines_gercek_label_attribute_olmali(self):
        fig, lines, plotted_data, uyarilar = grafikleri_ciz(
            self.tum_veriler, {1: ('test.csv', 'gyro_x')}, 'realtime', 1
        )
        for line in lines:
            self.assertTrue(hasattr(line, 'gercek_label'))
            self.assertEqual(line.gercek_label, 'test.csv | gyro_x')


class TestLegendSagTik(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='U1 test.csv gyro_x')
        ax.legend()
        parent = MagicMock()
        parent.alias_map = {}
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        self.pencere.ters_label_map = {'U1 test.csv gyro_x': 'test.csv | gyro_x'}
        plt.close(fig)

    def tearDown(self):
        plt.close('all')

    def event_olustur(self, button=3, x=0, y=0):
        event = MagicMock()
        event.button = button
        event.x = x
        event.y = y
        return event

    def test_sag_tik_degilse_erken_return(self):
        event = self.event_olustur(button=1)
        with patch('main.QMenu') as mock_menu:
            self.pencere.legend_sag_tik(event)
            mock_menu.assert_not_called()

    def test_legend_yoksa_erken_return(self):
        ax = self.pencere.canvas.figure.axes[0]
        ax.get_legend().remove()
        event = self.event_olustur(button=3)
        with patch('main.QMenu') as mock_menu:
            self.pencere.legend_sag_tik(event)
            mock_menu.assert_not_called()

    def test_legend_metnine_tiklanmadiysa_islem_yapilmamali(self):
        event = self.event_olustur(button=3, x=-9999, y=-9999)
        with patch('main.QMenu') as mock_menu:
            self.pencere.legend_sag_tik(event)
            mock_menu.assert_not_called()

    def test_ters_label_map_eslesmiyorsa_islem_yapilmamali(self):
        self.pencere.ters_label_map = {}
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        text = legend.get_texts()[0]
        bbox = text.get_window_extent()
        event = self.event_olustur(button=3, x=bbox.x0 + 1, y=bbox.y0 + 1)
        with patch('main.QMenu') as mock_menu:
            self.pencere.legend_sag_tik(event)
            mock_menu.assert_not_called()

    def test_iptal_edilirse_alias_kaydedilmemeli(self):
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        text = legend.get_texts()[0]
        bbox = text.get_window_extent()
        event = self.event_olustur(button=3, x=bbox.x0 + 1, y=bbox.y0 + 1)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('yeni', False)):
                self.pencere.legend_sag_tik(event)
        self.assertNotIn('test.csv', self.pencere.parent_ref.alias_map)

    def test_bos_alias_kaydedilmemeli(self):
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        text = legend.get_texts()[0]
        bbox = text.get_window_extent()
        event = self.event_olustur(button=3, x=bbox.x0 + 1, y=bbox.y0 + 1)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('   ', True)):
                self.pencere.legend_sag_tik(event)
        self.assertNotIn('test.csv', self.pencere.parent_ref.alias_map)

    def test_kullanilan_alias_warning_verilmeli(self):
        self.pencere.parent_ref.alias_map = {'baska.csv': 'IMU'}
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        text = legend.get_texts()[0]
        bbox = text.get_window_extent()
        event = self.event_olustur(button=3, x=bbox.x0 + 1, y=bbox.y0 + 1)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('IMU', True)):
                with patch('main.QMessageBox.warning') as mock_warning:
                    self.pencere.legend_sag_tik(event)
                    mock_warning.assert_called_once()

    def test_gecerli_alias_alias_map_e_kaydedilmeli(self):
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        text = legend.get_texts()[0]
        bbox = text.get_window_extent()
        event = self.event_olustur(button=3, x=bbox.x0 + 1, y=bbox.y0 + 1)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('IMU', True)):
                with patch.object(self.pencere.parent_ref, 'alias_kaydet'):
                    with patch.object(self.pencere.parent_ref, 'secilenleri_getir'):
                        self.pencere.legend_sag_tik(event)
        self.assertEqual(self.pencere.parent_ref.alias_map.get('test.csv'), 'IMU')

    def test_gecerli_alias_alias_kaydet_cagrilmali(self):
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        text = legend.get_texts()[0]
        bbox = text.get_window_extent()
        event = self.event_olustur(button=3, x=bbox.x0 + 1, y=bbox.y0 + 1)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('IMU', True)):
                with patch.object(self.pencere.parent_ref, 'alias_kaydet') as mock_kaydet:
                    with patch.object(self.pencere.parent_ref, 'secilenleri_getir'):
                        self.pencere.legend_sag_tik(event)
                        mock_kaydet.assert_called_once()

    def test_gecerli_alias_legend_metinleri_guncellenmeli(self):
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        text = legend.get_texts()[0]
        bbox = text.get_window_extent()
        event = self.event_olustur(button=3, x=bbox.x0 + 1, y=bbox.y0 + 1)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=('IMU', True)):
                with patch.object(self.pencere.parent_ref, 'alias_kaydet'):
                    with patch.object(self.pencere.parent_ref, 'secilenleri_getir'):
                        self.pencere.legend_sag_tik(event)
        legend_metinleri = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertTrue(any('IMU' in m for m in legend_metinleri))

class TestDeleteAliasLegend(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='U1 IMU gyro_x')
        ax.legend()
        parent = MagicMock()
        parent.alias_map = {'test.csv': 'IMU'}
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1,mode='previous', parent_ref=parent)
        self.pencere.ters_label_map = {'U1 IMU gyro_x': 'test.csv | gyro_x '}
        self.pencere.label_map = {'test.csv | gyro_x': 'U1 IMU gyro_x'}
        plt.close(fig)

    def tearDown(self):
        plt.close('all')

    def _event_olustur(self, button=3, x=0, y=0):
        event = MagicMock()
        event.button = button
        event.x = x
        event.y = y
        return event

    def delete_alias_action_calistir(self):
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        text = legend.get_texts()[0]
        bbox = text.get_window_extent()
        event = self._event_olustur(button=3, x=bbox.x0 + 1, y=bbox.y0 + 1)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = delete_alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch.object(self.pencere.parent_ref, 'alias_kaydet') as mock_kaydet:
                self.pencere.legend_sag_tik(event)
                return mock_kaydet

    def test_delete_alias_alias_map_ten_silinmeli(self):
        self.delete_alias_action_calistir()
        self.assertNotIn('test.csv', self.pencere.parent_ref.alias_map)

    def test_delete_alias_alias_kaydet_bir_kez_cagirilmali(self):
        mock_kaydet = self.delete_alias_action_calistir()
        mock_kaydet.assert_called_once()

    def test_delete_alias_legend_metni_orijinale_donmeli(self):
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        text = legend.get_texts()[0]
        bbox = text.get_window_extent()
        event = self._event_olustur(button=3, x=bbox.x0 + 1, y=bbox.y0 + 1)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = delete_alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch.object(self.pencere.parent_ref, 'alias_kaydet'):
                self.pencere.legend_sag_tik(event)
        legend_metinleri = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertTrue(any('test.csv' in m for m in legend_metinleri))

    def test_delete_alias_ters_label_map_guncellenmeli(self):
        self.delete_alias_action_calistir()
        self.assertTrue(any('test.csv' in k for k in self.pencere.ters_label_map))

    def test_delete_alias_label_map_guncellenmeli(self):
        self.delete_alias_action_calistir()
        self.assertTrue(any('test.csv' in v for v in self.pencere.label_map.values()))


class TestDeleteAliasParametreListesi(unittest.TestCase):

    def setUp(self):
        self.pencere = AnaPencere()
        self.pencere.alias_map = {'test.csv': 'IMU'}
        self.pencere.parametre_listesi.addItem('IMU | gyro_x')

    def tearDown(self):
        plt.close('all')

    def delete_alias_calistir(self):
        item = self.pencere.parametre_listesi.item(0)
        pos = QPoint(0, 0)
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = delete_alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch.object(self.pencere.parametre_listesi, 'itemAt', return_value=item):
                with patch.object(self.pencere, 'alias_kaydet') as mock_kaydet:
                    with patch.object(self.pencere, 'secilenleri_getir'):
                        self.pencere.parametre_sag_tik(pos)
                        return mock_kaydet

    def test_delete_alias_alias_map_ten_silinmeli(self):
        self.delete_alias_calistir()
        self.assertNotIn('test.csv', self.pencere.alias_map)

    def test_delete_alias_alias_kaydet_cagrilmali(self):
        mock_kaydet = self.delete_alias_calistir()
        mock_kaydet.assert_called_once()

    def test_delete_olmayan_alias_icin_islem_yapilmamali(self):
        self.pencere.alias_map = {}
        mock_kaydet = self.delete_alias_calistir()
        mock_kaydet.assert_not_called()


class TestAliasKayitYeri(unittest.TestCase):

    def setUp(self):
        self.pencere = AnaPencere()
        self.test_klasor = tempfile.mkdtemp()
        self.pencere.klasor = self.test_klasor
        self.pencere.alias_map = {'test.csv': 'IMU'}

    def tearDown(self):
        plt.close('all')

    def test_frozen_modda_executable_yanina_kaydedilmeli(self):
        with patch('sys.frozen', True, create=True):
            frozen_exe = r'C:\app\analiz.exe' if sys.platform == 'win32' else '/app/analiz.exe'
            with patch('sys.executable', frozen_exe):
                with patch('os.makedirs'):
                    with patch('builtins.open', mock_open()) as mock_dosya:
                        self.pencere.alias_kaydet()
                        cagri_yolu = mock_dosya.call_args[0][0]
                        self.assertIn('app', cagri_yolu)
                        self.assertNotIn(self.test_klasor, cagri_yolu)

    def test_normal_modda_dosya_yanina_kaydedilmeli(self):
        with patch('os.makedirs'):
            with patch('builtins.open', mock_open()) as mock_dosya:
                self.pencere.alias_kaydet()
                cagri_yolu = mock_dosya.call_args[0][0]
                self.assertIn('aliases.json', cagri_yolu)

class TestScatterAliasGuncelleme(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='U1 IMU gyro_x')
        sc = ax.scatter([1], [3], label='U1 IMU gyro_x - SPIKE')
        sc.gercek_label = 'test.csv | gyro_x - SPIKE'
        ax.legend()
        parent = MagicMock()
        parent.alias_map = {'test.csv': 'IMU'}
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        self.pencere.ters_label_map = {'U1 IMU gyro_x': 'test.csv | gyro_x'}
        self.pencere.label_map = {'test.csv | gyro_x': 'U1 IMU gyro_x'}
        self.pencere.scatter_data = {'U1 IMU gyro_x - SPIKE': (np.array([1.0]), np.array([3.0]))}
        plt.close(fig)

    def tearDown(self):
        plt.close('all')

    def alias_degistir(self, yeni_alias):
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        text = legend.get_texts()[0]
        bbox = text.get_window_extent()
        event = MagicMock()
        event.button = 3
        event.x = bbox.x0 + 1
        event.y = bbox.y0 + 1
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=(yeni_alias, True)):
                with patch.object(self.pencere.parent_ref, 'alias_kaydet'):
                    with patch.object(self.pencere.parent_ref, 'secilenleri_getir'):
                        self.pencere.legend_sag_tik(event)

    def test_alias_degisince_scatter_label_guncellenmeli(self):
        self.alias_degistir('AHRS')
        ax = self.pencere.canvas.figure.axes[0]
        scatter_labels = [str(coll.get_label()) for coll in ax.collections]
        self.assertTrue(any('AHRS' in lbl for lbl in scatter_labels))

    def test_alias_degisince_gercel_label_degismemeli(self):
        self.alias_degistir('AHRS')
        ax = self.pencere.canvas.figure.axes[0]
        for coll in ax.collections:
            if hasattr(coll, 'gercek_label'):
                self.assertIn('test.csv', coll.gercek_label)

    def test_alias_degisince_scatter_data_guncellenmelli(self):
        self.alias_degistir('AHRS')
        self.assertTrue(any('AHRS' in k for k in self.pencere.scatter_data))

    def test_alias_degisince_eski_scatter_data_kalkmali(self):
        self.alias_degistir('AHRS')
        self.assertFalse(any('IMU' in k for k in self.pencere.scatter_data))


class TestOpDataAliasGuncelleme(unittest.TestCase):

    def setUp(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label='U1 IMU gyro_x')
        ax.legend()
        parent = MagicMock()
        parent.alias_map = {'test.csv': 'IMU'}
        parent.acik_grafikler = []
        self.pencere = GrafikPenceresi(fig, figure_no=1, mode='previous', parent_ref=parent)
        self.pencere.ters_label_map = {'U1 IMU gyro_x': 'test.csv | gyro_x'}
        self.pencere.label_map = {'test.csv | gyro_x': 'U1 IMU gyro_x'}
        self.pencere.op_data = {
            'Op1': (
                np.array([1.0, 2.0]),
                np.array([3.0, 4.0]),
                'U1 IMU gyro_x + 1',
                'U1 IMU gyro_x',
                '+',
                '1',
                True
            )
        }
        plt.close(fig)

    def tearDown(self):
        plt.close('all')

    def alias_degistir(self, yeni_alias):
        ax = self.pencere.canvas.figure.axes[0]
        legend = ax.get_legend()
        text = legend.get_texts()[0]
        bbox = text.get_window_extent()
        event = MagicMock()
        event.button = 3
        event.x = bbox.x0 + 1
        event.y = bbox.y0 + 1
        alias_action = MagicMock()
        delete_alias_action = MagicMock()
        menu_mock = MagicMock()
        menu_mock.addAction.side_effect = [alias_action, delete_alias_action]
        menu_mock.exec.return_value = alias_action
        with patch('main.QMenu', return_value=menu_mock):
            with patch('main.QInputDialog.getText', return_value=(yeni_alias, True)):
                with patch.object(self.pencere.parent_ref, 'alias_kaydet'):
                    with patch.object(self.pencere.parent_ref, 'secilenleri_getir'):
                        self.pencere.legend_sag_tik(event)

    def test_alias_degisince_op_data_sol_guncellenmeli(self):
        self.alias_degistir('AHRS')
        kayit = self.pencere.op_data['Op1']
        self.assertIn('AHRS', kayit[3])

    def test_alias_degisince_op_data_aciklama_guncellenmeli(self):
        self.alias_degistir('AHRS')
        kayit = self.pencere.op_data['Op1']
        self.assertIn('AHRS', kayit[2])

    def test_alias_degisince_eski_isim_op_data_da_kalmamali(self):
        self.alias_degistir('AHRS')
        kayit = self.pencere.op_data['Op1']
        self.assertNotIn('IMU', kayit[2])
        self.assertNotIn('IMU', str(kayit[3]))


ALIAS_MAP = {'flight_001.csv': 'Uçuş 1', 'flight_002.csv': 'Uçuş 2'}

def guncelle_fn(alias_map, kullan, metin):
    if not metin:
        return metin
    for original, alias in alias_map.items():
        if kullan:
            metin = metin.replace(original, alias)
        else:
            metin = metin.replace(alias, original)
    return metin

def figures_guncelle(figures, alias_map, kullan):
    ters_alias = {v: k for k, v in alias_map.items()}
    for fig_data in figures.values():
        yeni = []
        for p in fig_data['params']:
            if ' | ' not in p:
                yeni.append(p)
                continue
            dosya_adi, kolon = p.split(' | ', 1)
            orijinal = ters_alias.get(dosya_adi, dosya_adi)
            gosterim = alias_map.get(orijinal, orijinal) if kullan else orijinal
            yeni.append(f'{gosterim} | {kolon}')
        fig_data['params'] = yeni

class TestGuncelleFn(unittest.TestCase):

    def test_kullan_true_orijinal_alias_olur(self):
        sonuc = guncelle_fn(ALIAS_MAP, True, 'flight_001.csv altitude')
        self.assertEqual(sonuc, 'Uçuş 1 altitude')

    def test_kullan_false_alias_orijinal_olur(self):
        sonuc = guncelle_fn(ALIAS_MAP, False, 'Uçuş 2 speed')
        self.assertEqual(sonuc, 'flight_002.csv speed')

    def test_bos_string_degismez(self):
        self.assertEqual(guncelle_fn(ALIAS_MAP, True, ''), '')

    def test_none_none_doner(self):
        self.assertIsNone(guncelle_fn(ALIAS_MAP, True, None))

    def test_eslesmeyene_dokunmaz(self):
        sonuc = guncelle_fn(ALIAS_MAP, True, 'other_file.csv')
        self.assertEqual(sonuc, 'other_file.csv')

class TestFiguresGuncelle(unittest.TestCase):

    def test_kullan_true_dosya_adi_alias_olur_kolon_korunur(self):
        figures = {1: {'params': ['flight_001.csv | altitude']}}
        figures_guncelle(figures, ALIAS_MAP, True)
        self.assertEqual(figures[1]['params'], ['Uçuş 1 | altitude'])

    def test_kullan_false_alias_orijinal_olur_kolon_korunur(self):
        figures = {1: {'params': ['Uçuş 1 | altitude']}}
        figures_guncelle(figures, ALIAS_MAP, False)
        self.assertEqual(figures[1]['params'], ['flight_001.csv | altitude'])

    def test_op_data_sabit_mi_true_sag_degismez(self):
        t, v = np.array([0.0]), np.array([1.0])
        op_data = {'op1': (t, v, 'flight_001.csv desc', 'flight_001.csv', '+', '5.0', True)}
        yeni_op_data = {}
        for op_isim, kayit in op_data.items():
            t_, v_, aciklama, sol, op_char, sag, sabit_mi = kayit
            yeni_op_data[op_isim] = (
                t_, v_,
                guncelle_fn(ALIAS_MAP, True, aciklama),
                guncelle_fn(ALIAS_MAP, True, sol),
                op_char,
                guncelle_fn(ALIAS_MAP, True, sag) if not sabit_mi else sag,
                sabit_mi
            )
        self.assertEqual(yeni_op_data['op1'][5], '5.0')  # sag değişmemeli

    def test_op_data_sabit_mi_false_sag_donusur(self):
        t, v = np.array([0.0]), np.array([1.0])
        op_data = {'op1': (t, v, 'desc', 'flight_001.csv', '+', 'flight_002.csv', False)}
        yeni_op_data = {}
        for op_isim, kayit in op_data.items():
            t_, v_, aciklama, sol, op_char, sag, sabit_mi = kayit
            yeni_op_data[op_isim] = (
                t_, v_,
                guncelle_fn(ALIAS_MAP, True, aciklama),
                guncelle_fn(ALIAS_MAP, True, sol),
                op_char,
                guncelle_fn(ALIAS_MAP, True, sag) if not sabit_mi else sag,
                sabit_mi
            )
        self.assertEqual(yeni_op_data['op1'][5], 'Uçuş 2')  # sag dönüşmeli


def csv_export_logic(plotted_data, label_map, scatter_data, op_data, secili, dosya_yolu):
    parcalar = []
    for label, (t_arr, v_arr) in plotted_data.items():
        u_ismi = label_map.get(label, label)
        if u_ismi in secili:
            parcalar.append(pd.DataFrame({
                f"{u_ismi}_t": pd.Series(np.asarray(t_arr)),
                f"{u_ismi}_v": pd.Series(np.asarray(v_arr))
            }))
    for label, (t_arr, v_arr) in scatter_data.items():
        if label in secili:
            parcalar.append(pd.DataFrame({
                f"{label}_t": pd.Series(np.array(t_arr)),
                f"{label}_v": pd.Series(np.array(v_arr))
            }))
    for op_isim, kayit in op_data.items():
        op_label = f"{op_isim} {kayit[2]}"
        if op_label in secili and kayit[0] is not None:
            parcalar.append(pd.DataFrame({
                f"{op_label}_t": pd.Series(np.asarray(kayit[0])),
                f"{op_label}_v": pd.Series(np.asarray(kayit[1]))
            }))
    if parcalar:
        pd.concat(parcalar, axis=1).to_csv(dosya_yolu, index=False, sep=',', encoding='utf-8-sig')
        return True
    return False

class TestCsvExport(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.tmp.close()
        self.dosya = self.tmp.name

    def tearDown(self):
        if os.path.exists(self.dosya):
            os.remove(self.dosya)

    def test_plotted_data_yazilir(self):
        csv_export_logic(
            {'ch1': (np.array([0.0, 1.0]), np.array([10.0, 20.0]))},
            {'ch1': 'altitude'}, {}, {}, ['altitude'], self.dosya
        )
        df = pd.read_csv(self.dosya)
        self.assertIn('altitude_t', df.columns)
        self.assertIn('altitude_v', df.columns)

    def test_scatter_data_yazilir(self):
        csv_export_logic(
            {}, {}, {'scatter_ch': (np.array([0.0, 1.0]), np.array([1.0, 2.0]))},
            {}, ['scatter_ch'], self.dosya
        )
        df = pd.read_csv(self.dosya)
        self.assertIn('scatter_ch_t', df.columns)

    def test_op_data_yazilir(self):
        t, v = np.array([0.0, 1.0]), np.array([5.0, 6.0])
        csv_export_logic(
            {}, {}, {},
            {'op1': (t, v, 'alt+spd', 'alt', '+', 'spd', False)},
            ['op1 alt+spd'], self.dosya
        )
        df = pd.read_csv(self.dosya)
        self.assertIn('op1 alt+spd_t', df.columns)

    def test_secili_disindaki_yazilmaz(self):
        ret = csv_export_logic(
            {'ch1': (np.array([0.0]), np.array([1.0]))},
            {'ch1': 'altitude'}, {}, {}, ['speed'], self.dosya
        )
        self.assertFalse(ret)

    def test_bos_secili_false_doner(self):
        ret = csv_export_logic({}, {}, {}, {}, [], self.dosya)
        self.assertFalse(ret)

    def test_label_map_alias_sutun_adi_olur(self):
        csv_export_logic(
            {'raw_key': (np.array([0.0]), np.array([1.0]))},
            {'raw_key': 'Alias İsim'}, {}, {}, ['Alias İsim'], self.dosya
        )
        df = pd.read_csv(self.dosya)
        self.assertIn('Alias İsim_t', df.columns)
        self.assertNotIn('raw_key_t', df.columns)

    def test_farkli_uzunluk_nan_ile_pad(self):
        csv_export_logic(
            {'ch1': (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
             'ch2': (np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0, 5.0]))},
            {'ch1': 'ch1', 'ch2': 'ch2'}, {}, {}, ['ch1', 'ch2'], self.dosya
        )
        df = pd.read_csv(self.dosya)
        self.assertEqual(len(df), 3)
        self.assertTrue(df['ch1_v'].isna().any())

if __name__ == '__main__':
    unittest.main(verbosity=2)