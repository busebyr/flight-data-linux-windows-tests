import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from typing import Literal

class AnalizMotoruError(Exception):
    pass

class DosyaHatasi(AnalizMotoruError):
    pass

def ortak_zaman_ekseni(veriler ):#uyari_callback=None
    t_max_listesi = []
    dt_listesi = []

    for df in veriler.values():
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().to_numpy(dtype=float)
        if len(t) < 2:
            continue
        t_norm = t - t[0]
        t_max_listesi.append(t_norm[-1])

        # MATLAB mantığı: ilk 5 adımın ortalaması
        n_check = min(5, len(t))
        ort_diff_ms = np.mean(np.diff(t[:n_check])) * 1000  # saniye → ms
        dt = 0.010 if ort_diff_ms < 50 else 0.100

        dt_listesi.append(dt)

    if not t_max_listesi or not dt_listesi:
        return pd.Series([], dtype=float)

    dt = min(dt_listesi)       # en yüksek çözünürlük
    t_max = max(t_max_listesi)
    ortak = np.arange(0, t_max + dt, dt)
    return pd.Series(ortak, dtype=float)

#VERİYİ HİZALA
def veriyi_hizala(df, timeline, kolon, mode):
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    y = pd.to_numeric(df[kolon], errors='coerce')

    kaynak = pd.DataFrame({'t': t, 'y': y}).dropna(subset=['t', 'y'])
    kaynak = kaynak.sort_values('t').reset_index(drop=True)

    if len(kaynak) == 0:
        return pd.Series(np.nan, index=timeline.index, dtype=float)

    # Normalizasyon sadece burada — ortak_zaman_ekseni'nde değil
    kaynak['t'] = kaynak['t'] - kaynak['t'].iloc[0]

    # MATLAB'daki unique() — tekrar eden timestamp'leri temizle
    kaynak = kaynak.drop_duplicates(subset='t', keep='first').reset_index(drop=True)

    direction: Literal['backward', 'nearest'] = 'backward' if mode == 'previous' else 'nearest'
    hedef = pd.DataFrame({'t': timeline}).reset_index()
    hedef_s = hedef.sort_values('t').copy()
    hedef_s['t'] = hedef_s['t'].astype(float)
    kaynak['t'] = kaynak['t'].astype(float)

    birlesik = pd.merge_asof(hedef_s, kaynak, on='t', direction=direction)

    if mode == 'previous':
        birlesik['y'] = birlesik['y'].ffill()

    return birlesik.sort_values('index')['y'].reset_index(drop=True)

#BİRİM DÖNÜŞÜMÜ
def birim_donustur(veri, donusum):
    try:
        if donusum == 'Default Units':
            return veri
        elif donusum == 'deg to rad':
            return veri * np.pi / 180
        elif donusum == 'rad to deg':
            return veri * 180 / np.pi
        elif donusum == 'm to ft':
            return veri * 3.28084
        elif donusum == 'ft to m':
            return veri * 0.3048
        elif donusum == 'kt to m/s':
            return veri * 0.514444
        elif donusum == 'm/s to kt':
            return veri * 1.94384
        elif donusum == 'km/h to kt':
            return veri * 0.539957
        elif donusum == 'kt to km/h':
            return veri / 0.539957
        elif donusum == 'm/s to km/h':
            return veri * 3.6
        elif donusum == 'km/h to m/s':
            return veri / 3.6
        return None
    except (TypeError, ValueError):
        return None

#DOSYA YÜKLEME
def dosyalari_yukle(klasor: str, secili_dosyalar):
    tum_veriler = {}

    for secili_dosya in secili_dosyalar:
        dosya_yolu = os.path.join(klasor, secili_dosya)

        try:
            df = pd.read_csv(dosya_yolu, skiprows=1, sep='\t', low_memory=False)
        except Exception as e:
            raise DosyaHatasi(f"{secili_dosya} yüklenemedi: {e}")

        zaman_kolon = df.columns[0]
        zaman = pd.to_numeric(df[zaman_kolon], errors='coerce').astype('float64')
        df = df.copy()

        df[zaman_kolon] = zaman/1e6
        tum_veriler[secili_dosya] = df

    return tum_veriler

def temiz_veriye_donustur(seri):
    return pd.to_numeric(
        seri.astype(str).str.replace(',', '.', regex=False).str.strip(),
        errors='coerce'
    )

#GRAFİK ÇİZİMİ
def grafikleri_ciz(tum_veriler, parametre_map, mode, figure_no, units=None, alias_map=None):
    if units is None:
        units = ['Default Units'] * len(parametre_map)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'Figure {figure_no} - Mode: {mode.upper()}')

    plotted_data = {}
    lines = []
    uyarilar = []

    if mode in ['previous', 'nearest']:
        ortak_t = ortak_zaman_ekseni(tum_veriler)
        referans_timeline = ortak_t if len(ortak_t) > 0 else None
    else:
        referans_timeline = None

    for pid, (dosya, kolon) in parametre_map.items():

        unit = units[pid - 1] if pid - 1 < len(units) else 'Default Units'

        if dosya not in tum_veriler:
            continue

        df = tum_veriler[dosya]

        if kolon not in df.columns:
            continue

        dosya_adi = alias_map.get(dosya, dosya) if alias_map else dosya
        label = f'U{pid} {dosya_adi} {kolon}' if unit == 'Default Units' else f'U{pid} {dosya_adi} {kolon} [{unit}]'
        gercek_label = f'{dosya} | {kolon}'

        if mode == 'realtime':
            zaman = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            veri = temiz_veriye_donustur(df[kolon])

            temp_df = pd.DataFrame({'t': zaman, 'v': veri}).dropna(subset=['t', 'v'])
            if temp_df.empty:
                continue

            temp_df = temp_df.sort_values('t')
            t_plot = temp_df['t'].to_numpy() - temp_df['t'].iloc[0]
            v_plot = birim_donustur(temp_df['v'].values, unit)

        else:
            if referans_timeline is None or len(referans_timeline) == 0:
                continue

            v_hizali = veriyi_hizala(df, referans_timeline, kolon, mode)
            v_plot = np.array(birim_donustur(v_hizali, unit), dtype=float)
            t_plot = referans_timeline.to_numpy()
            t_plot = t_plot - t_plot[0]  #sıfırla normalize et

        if v_plot is None:
            raise ValueError(
                f"Unit conversion failed: '{dosya} | {kolon}'\n"
                f"Selected conversion '{unit}' could not be applied to this data."
            )

        plotted_data[gercek_label] = (t_plot, v_plot)  #tam veri

        line, = ax.step(t_plot, v_plot, where='post', label=label)
        line.gercek_label = gercek_label
        lines.append(line)

    #Eksen etiketleri
    ax.set_ylabel('Value')
    ax.set_xlabel('Time (s)')

    #Y ekseni
    ymin, ymax = ax.get_ylim()
    if ymin == ymax:
        ax.set_ylim(ymin - 1, ymax + 1)

    if ax.get_lines():
        legend = ax.legend(loc='upper left')
    else:
        legend = None

    if legend is not None:
        legend.set_draggable(True)
        for legline in legend.get_lines():
            legline.set_linewidth(6)
            legline.set_markersize(10)
        for handle in legend.legend_handles:
            if hasattr(handle, 'set_sizes'):
                handle.set_sizes([80])

    ax.grid(True)

    return fig, lines, plotted_data, uyarilar