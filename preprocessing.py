import pandas as pd


def load_excel_as_transactions(path):
    """
    Membaca file Excel, menormalkan kolom, dan mengubah menjadi list transaksi.
    Kolom wajib: NO FAKTUR, NAMA BARANG.

    Return:
    - df_raw: DataFrame asli setelah normalisasi header
    - transactions: list[list[str]] transaksi per faktur
    - tabulation: DataFrame biner 0/1 (index=faktur, columns=item)
    - faktur_col: nama kolom faktur yang terdeteksi
    - item_col: nama kolom item yang terdeteksi
    """

    df = pd.read_excel(path)

    # Normalisasi header (uppercase + trim spasi)
    df.columns = df.columns.str.upper().str.strip()

    # Deteksi kolom item: utamakan NO BARANG jika ada, fallback ke NAMA BARANG
    item_col = None
    for col in df.columns:
        if "NO" in col and "BARANG" in col:
            item_col = col
            break
    if item_col is None:
        for col in df.columns:
            if "NAMA" in col and "BARANG" in col:
                item_col = col
                break
    if item_col is None:
        raise KeyError("Kolom item tidak ditemukan. Harus ada 'NO BARANG' atau 'NAMA BARANG'.")

    # Deteksi kolom FAKTUR
    faktur_col = None
    for col in df.columns:
        if "FAKTUR" in col:
            faktur_col = col
            break
    if faktur_col is None:
        raise KeyError("Kolom 'NO FAKTUR' tidak ditemukan pada file Excel.")

    working = df[[faktur_col, item_col]].copy()
    working[faktur_col] = working[faktur_col].astype(str).str.strip()
    working[item_col] = working[item_col].astype(str).str.strip()
    working = working[(working[faktur_col] != "") & (working[item_col] != "")]

    # Group transaksi per faktur
    grouped = working.groupby(faktur_col)[item_col].apply(lambda s: list(dict.fromkeys(s)))
    transactions = grouped.tolist()

    # Tabulasi 0/1 seperti pivot Excel (faktur x item)
    tabulation = pd.crosstab(working[faktur_col], working[item_col])
    tabulation = (tabulation > 0).astype(int)

    return df, transactions, tabulation, faktur_col, item_col
