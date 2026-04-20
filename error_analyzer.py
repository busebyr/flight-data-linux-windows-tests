import numpy as np
import pandas as pd
from pathlib import Path

class ErrorClassLoader:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.error_tables = {}
        self.load_all_error_classes()

    def load_all_error_classes(self):

        for file in self.folder_path.glob("*ERROR_CLASS*.xlsx"):
            df = pd.read_excel(file)
            df.columns = df.columns.str.strip()

            system_name = file.stem.upper()
            is_eml = 'EML' in system_name

            system_dict = {}

            for _, row in df.iterrows():
                variable = str(row.iloc[0]).strip()

                if variable == "" or variable.lower() == "nan":
                    continue
                system_dict[variable] = {
                    "min":        row.iloc[1],
                    "max":        row.iloc[2],
                    "max_change": row.iloc[3],
                    "is_eml":     is_eml
                }

            self.error_tables[system_name] = system_dict


# ERROR ANALYSIS
def analyze_errors(values, min_val, max_val, max_change):

    values = np.array(values, dtype=float)
    n = len(values)

    # CONSTANT OUTPUT
    # Sadece her sabit bloğun İLK indexini kaydet (çakışan pencereleri tekrarlama).
    # Önceki yaklaşım her kayan pencere başlangıcını ekliyordu → aynı noktaya
    window = 5
    constant_flags = []
    if n >= window:
        in_block = False
        for i in range(n - window + 1):
            pencere = values[i:i + window]
            if np.all(np.isclose(pencere, pencere[0], equal_nan=False)):
                if not in_block:
                    constant_flags.append(i)   # bloğun yalnızca ilk indexi
                    in_block = True
            else:
                in_block = False

    # OVERSHOOT ve SPIKE
    # max_change == 0 ise hesaplama yapma
    if max_change != 0 and n >= 3:
        diff_prev = values[1:-1] - values[:-2]
        diff_next = values[2:]   - values[1:-1]

        overshoot_mask = np.abs(diff_prev) > max_change
        overshoot_flags = (np.where(overshoot_mask)[0] + 1).tolist()

        spike_mask = (
                (np.abs(diff_prev) > 0.5 * max_change) &
                (np.abs(diff_next) > 0.5 * max_change) &
                (np.sign(diff_prev) != np.sign(diff_next))
        )
        spike_flags = (np.where(spike_mask)[0] + 1).tolist()

        # Spike olan noktaları ve onların komşularını OVERSHOOT listesinden çıkar.
        # SPIKE noktasına giriş (idx-1) ve çıkış (idx+1) geçişleri de OVERSHOOT değil.
        spike_ve_komsu = set()
        for s in spike_flags:
            spike_ve_komsu.update([s - 1, s, s + 1])
        overshoot_flags = [i for i in overshoot_flags if i not in spike_ve_komsu]

        # CONSTANT OUTPUT bloğu içindeki ve geçiş noktalarını OVERSHOOT/SPIKE'tan çıkar.
        # Sabit bloğa giriş ve çıkış geçişleri hata değil, beklenen davranıştır.
        if constant_flags:
            constant_indices = set()
            for start in constant_flags:
                j = start
                while j < n and np.isclose(values[j], values[start]):
                    constant_indices.add(j)
                    j += 1
                constant_indices.add(j)

            gecis_indices = set()
            for idx in constant_indices:
                if idx > 0:
                    gecis_indices.add(idx)
                if idx + 1 < n:
                    gecis_indices.add(idx + 1)

            masum = constant_indices | gecis_indices
            overshoot_flags = [i for i in overshoot_flags if i not in masum]
            spike_flags     = [i for i in spike_flags     if i not in masum]
    else:
        overshoot_flags = []
        spike_flags     = []

    #OUT OF RANGE
    # min ve max ikisi de 0 ise sınır tanımlı değil -> atla.
    if not (min_val == 0 and max_val == 0):
        range_mask  = (values > max_val) | (values < min_val)
        range_flags = np.where(range_mask)[0].tolist()
    else:
        range_flags = []

    return {
        "CONSTANT OUTPUT": constant_flags,
        "OVERSHOOT":       overshoot_flags,
        "SPIKE":           spike_flags,
        "OUT OF RANGE":    range_flags
    }


# VARIABLE SYSTEM FINDER
# dosya_adi_upper: dosya adının büyük harf hali — önce buna uyan sistemi seç
def find_variable_system(loader, variable_name, prefer_eml=False, dosya_adi_upper=""):
    variable_lower = variable_name.lower()
    bulunan = []
    for system, variables in loader.error_tables.items():
        for var in variables:
            if var.lower() == variable_lower:
                bulunan.append(system)
                break

    if not bulunan:
        return None

    # Öncelik sırası:
    # 1. Dosya adıyla örtüşen sistem
    # 2. EML tercih ediliyorsa EML tablosu
    # 3. Normal tablo
    # 4. İlk bulunan

    if dosya_adi_upper:
        # Dosya adında geçen anahtar kelimelerle sistem adını eşleştir
        sistem_anahtar = {
            'EGID': 'EGID',
            'EGIE': 'EGIE',
            'EGIS': 'EGIS',
            'ADC':  'ADC',
            'SMU':  'SMU',
            'VMM':  'VMM',
            'AP':   'AP_',
            'RD':   'RD_',
            'RC':   'RC_',
        }
        for anahtar, sistem_prefix in sistem_anahtar.items():
            if anahtar in dosya_adi_upper:
                # Bu dosyayla örtüşen sistemleri filtrele
                eslesenler = [s for s in bulunan if sistem_prefix in s]
                if eslesenler:
                    # EML dosyasıysa EML tablosunu, değilse normal tabloyu tercih et
                    if prefer_eml:
                        eml = [s for s in eslesenler if '_EML' in s]
                        if eml:
                            return eml[0]
                    normal = [s for s in eslesenler if '_EML' not in s]
                    return normal[0] if normal else eslesenler[0]

    if prefer_eml:
        eml_sistemler = [s for s in bulunan if s.endswith('_EML')]
        if eml_sistemler:
            return eml_sistemler[0]

    normal_sistemler = [s for s in bulunan if not s.endswith('_EML')]
    return normal_sistemler[0] if normal_sistemler else bulunan[0]


def check_special_variable(name, values):

    values = np.array(values, dtype=float)
    unique_vals = np.unique(values[~np.isnan(values)])

    # VMM özel değişkenler
    if name.lower() == "ivehiclemode":
        valid = set(range(0, 16))  # 0-15
        error_idx = np.where([v not in valid for v in values])[0]
        return {"INVALID SIGNAL": error_idx.tolist()}

    if name.lower() == "ivehiclestate":
        valid = {0, 100, 101, 200, 201, 202, 300, 301, 400, 500, 501, 502,
                 503, 600, 700, 701, 702, 703, 704, 800, 801, 802, 803,
                 804, 805, 806, 807, 808, 809, 900, 1000, 1100, 1500}
        error_idx = np.where([v not in valid for v in values])[0]
        return {"INVALID SIGNAL": error_idx.tolist()}

    if name.lower() == "imcnav_type":
        valid = {1, 2, 3}
        error_idx = np.where([v not in valid for v in values])[0]
        return {"INVALID SIGNAL": error_idx.tolist()}

    if name.lower().startswith("b") and len(name) <= 20 and "flgs" not in name.lower():
        is_binary = np.all((unique_vals == 0) | (unique_vals == 1))
        if is_binary:
            error_idx = np.where((values != 0) & (values != 1))[0]
            return {"INVALID SIGNAL": error_idx.tolist()}

    if "istatusreport" in name.lower():
        error_idx = np.where(values != 1)[0]
        return {"INVALID SIGNAL": error_idx.tolist()}

    if "red_flag" in name.lower():
        error_idx = np.where(values != 0)[0]
        return {"INVALID SIGNAL": error_idx.tolist()}

    return None