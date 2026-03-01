import io
from datetime import date
import streamlit as st
import pandas as pd

from preprocessing import load_excel_as_transactions
from apriori_service import get_frequent_itemsets, get_association_rules

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


st.set_page_config(
    page_title="Apriori Bundle Produk",
    layout="wide",
    page_icon=":package:",
)

st.title("Sistem Rekomendasi Bundle Produk (Apriori)")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0.15rem; }
    [data-testid="stSidebar"] .stSlider { margin-top: -0.75rem; }
    [data-testid="stSidebar"] .stFileUploader { margin-top: -0.75rem; }
    [data-testid="stSidebar"] .stFileUploader label { margin-bottom: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Pengaturan Analisis")

    uploaded_file = st.file_uploader("Upload file transaksi (.xlsx)", type=["xlsx"])

    st.write("### Minimum Support")
    st.caption("🛈 Seberapa sering kombinasi produk muncul dalam seluruh transaksi.")

    min_support_pct = st.slider(
        "",
        min_value=0.05,
        max_value=2.0,
        value=0.10,
        step=0.01,
        format="%.2f%%",
        key="min_support_pct"
    )

    st.write("### Minimum Confidence")
    st.caption("🛈︎ Peluang pelanggan membeli produk B setelah membeli produk A.")

    min_conf_pct = st.slider(
        "",
        min_value=10.0,
        max_value=50.0,
        value=30.0,
        step=1.0,
        format="%.0f%%",
        key="min_conf_pct"
    )

    st.write("### Minimum Lift")
    st.caption("🛈︎ Mengukur kekuatan hubungan antar produk. Lift > 1 berarti hubungan kuat.")

    min_lift = st.number_input(
        "",
        min_value=1.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key="min_lift"
    )
    max_rules = st.slider("Jumlah Bundle ditampilkan:", 3, 20, 10)

    run = st.button("Jalankan Analisis")

    min_support = min_support_pct / 100.0
    min_conf = min_conf_pct / 100.0

if "analysis_requested" not in st.session_state:
    st.session_state["analysis_requested"] = False
if "analysis_data" not in st.session_state:
    st.session_state["analysis_data"] = None


@st.cache_data(show_spinner=False)
def tabulation_to_csv_bytes(tabulation_df: pd.DataFrame) -> bytes:
    return tabulation_df.to_csv(index=True).encode("utf-8-sig")


def run_analysis(file_obj):
    """Proses lengkap analisis Apriori, hasil disimpan di session_state."""
    df, transactions, tabulation, faktur_col, item_col = load_excel_as_transactions(file_obj)
    frequent, _ = get_frequent_itemsets(transactions, min_support)

    if len(frequent) == 0:
        st.warning("Tidak ada frequent itemset ditemukan. Coba kecilkan support.")
        st.stop()

    rules = get_association_rules(frequent, min_conf, min_lift)

    if len(rules) == 0:
        st.warning("Tidak ada rules ditemukan. Coba kecilkan confidence atau lift.")
        st.stop()

    simple_rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    simple_rules["antecedents"] = simple_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    simple_rules["consequents"] = simple_rules["consequents"].apply(lambda x: ", ".join(list(x)))
    simple_rules = simple_rules.sort_values("confidence", ascending=False).reset_index(drop=True)

    st.session_state["analysis_data"] = {
        "transactions_len": len(transactions),
        "rules_len": len(rules),
        "simple_rules": simple_rules,
        "tabulation": tabulation,
        "faktur_col": faktur_col,
        "item_col": item_col,
        "source_df": df,
        "source_name": getattr(file_obj, "name", "-"),
    }
    st.session_state["analysis_requested"] = True


if uploaded_file and run:
    with st.spinner("Memproses data..."):
        try:
            run_analysis(uploaded_file)
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")
            st.stop()


if st.session_state["analysis_requested"] and st.session_state["analysis_data"]:

    data = st.session_state["analysis_data"]
    simple_rules = data["simple_rules"]

    st.success(f"Data berhasil dibaca. Total transaksi (faktur): {data['transactions_len']}")

    st.subheader("Tabulasi Transaksi (0/1)")
    tabulation_df = data["tabulation"].copy()
    tabulation_df.index.name = data["faktur_col"]
    st.caption(
        f"Ukuran tabulasi: **{tabulation_df.shape[0]:,} faktur x {tabulation_df.shape[1]:,} item**."
    )

    preview_rows = min(200, len(tabulation_df))
    st.dataframe(tabulation_df.head(preview_rows), use_container_width=True, height=380)
    st.download_button(
        "Unduh Semua Tabulasi (CSV)",
        tabulation_to_csv_bytes(tabulation_df),
        file_name="tabulasi_transaksi_01.csv",
        mime="text/csv",
    )

    st.subheader("Aturan Asosiasi")
    st.write(f"Rules ditemukan: **{data['rules_len']}**")

    display_rules = simple_rules.head(max_rules)
    display_table = simple_rules.copy()
    display_table.index = range(1, len(display_table) + 1)
    display_table["support"] = (display_table["support"] * 100).round(2)
    display_table["confidence"] = (display_table["confidence"] * 100).round(2)
    display_table["lift"] = display_table["lift"].round(2)

    row_height = 35
    visible_rows = max(1, len(display_rules))
    table_height = max(140, (visible_rows + 1) * row_height)
    st.dataframe(
        display_table,
        use_container_width=True,
        height=table_height,
        column_config={
            "support": st.column_config.NumberColumn("support", format="%.2f%%"),
            "confidence": st.column_config.NumberColumn("confidence", format="%.2f%%"),
            "lift": st.column_config.NumberColumn("lift", format="%.2f"),
        },
    )

    st.subheader("Rekomendasi Bundle Produk")

    for idx, (_, row) in enumerate(display_rules.iterrows()):

        A = row["antecedents"]
        B = row["consequents"]

        conf_pct = round(row["confidence"] * 100, 1)  # persentase
        expander_label = (
            f"**{A} + {B}** "
            f"(conf: `{conf_pct}%`, lift: `{row['lift']:.2f}`)"
        )
        with st.expander(expander_label, expanded=False):
            st.markdown(
                f"Jika pelanggan membeli **{A}**, "
                f"ada **{conf_pct}%** peluang mereka juga membeli **{B}**."
            )

    st.subheader("Unduh Laporan")
    report_df = simple_rules.copy()
    report_df["support"] = (report_df["support"] * 100).round(2).astype(str) + "%"
    report_df["confidence"] = (report_df["confidence"] * 100).round(2).astype(str) + "%"
    report_df["lift"] = report_df["lift"].round(2).astype(str)

    if REPORTLAB_AVAILABLE:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            title="Laporan Hasil Analisis",
            leftMargin=36,
            rightMargin=36,
            topMargin=72,
            bottomMargin=48,
        )
        styles = getSampleStyleSheet()
        elements = []

        source_name = data.get("source_name", "-")
        report_date = date.today().strftime("%Y-%m-%d")
        elements.append(Paragraph(f"Total transaksi: {data['transactions_len']}", styles["Normal"]))
        elements.append(Paragraph(f"Rules ditemukan: {data['rules_len']}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Aturan Asosiasi", styles["Heading2"]))
        table_data = [report_df.columns.tolist()]
        body_style = styles["BodyText"]
        body_style.fontSize = 8
        body_style.leading = 10
        for row in report_df.values.tolist():
            row = list(row)
            row[0] = Paragraph(str(row[0]), body_style)
            row[1] = Paragraph(str(row[1]), body_style)
            table_data.append(row)
        table = Table(
            table_data,
            repeatRows=1,
            colWidths=[200, 200, 55, 65, 45],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF3")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#222222")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B5BCC4")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Rekomendasi Bundle Produk", styles["Heading2"]))
        for i, (_, row) in enumerate(simple_rules.iterrows(), start=1):
            A = row["antecedents"]
            B = row["consequents"]
            conf_pct = round(row["confidence"] * 100, 1)
            lift = f"{row['lift']:.2f}"
            text = (
                f"<b>{i}. {A} + {B}</b> (conf: {conf_pct}%, lift: {lift})<br/>"
                f"Jika pelanggan membeli <b>{A}</b>, ada <b>{conf_pct}%</b> "
                f"peluang mereka juga membeli <b>{B}</b>."
            )
            elements.append(Paragraph(text, styles["BodyText"]))
            elements.append(Spacer(1, 6))

        def on_page(canvas, doc_obj):
            width, height = A4
            canvas.setStrokeColor(colors.HexColor("#1F77B4"))
            canvas.setLineWidth(2)
            canvas.line(36, height - 52, width - 36, height - 52)

            canvas.setFillColor(colors.HexColor("#111111"))
            canvas.setFont("Helvetica-Bold", 11)
            canvas.drawString(36, height - 45, "Laporan Hasil Analisis")

            canvas.setFont("Helvetica", 9)
            canvas.drawRightString(width - 36, height - 45, f"Sumber: {source_name}")

            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(colors.HexColor("#444444"))
            canvas.drawString(36, 28, f"Tanggal laporan: {report_date}")
            canvas.drawRightString(width - 36, 28, f"Halaman {canvas.getPageNumber()}")

        doc.build(elements, onFirstPage=on_page, onLaterPages=on_page)
        pdf_bytes = buffer.getvalue()
        st.download_button(
            "Unduh Laporan (PDF)",
            pdf_bytes,
            file_name="Association_Rules_Report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("Untuk export PDF, install package `reportlab` terlebih dulu.")


else:
    st.info("Silakan upload file dan klik *Jalankan Analisis* di sidebar.")
