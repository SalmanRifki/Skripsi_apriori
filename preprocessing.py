import pandas as pd


def load_excel_as_transactions(path):
    """
    Membaca file Excel, menormalkan kolom, dan mengubah menjadi list transaksi.
    Kolom wajib: NO FAKTUR, NAMA BARANG.
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

    # Pastikan nama barang string
    df[item_col] = df[item_col].astype(str)

    # Group transaksi per faktur
    grouped = df.groupby(faktur_col)[item_col].apply(list)

    return df, grouped.tolist()
