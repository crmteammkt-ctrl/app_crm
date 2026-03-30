import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import traceback

st.set_page_config(page_title="CRM & Pareto", layout="wide")

DEFAULT_PARQUET_PATH = "data/crm_cohort.parquet"
PREVIEW_ROWS = 300


# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.sidebar.file_uploader(
    "📂 Tải file parquet (nếu không tải sẽ dùng file mặc định)",
    type=["parquet"]
)


# =====================================================
# HELPERS
# =====================================================
def fmt_int(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return ""


def fmt_pct(x, decimals=2):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):,.{decimals}f}%"
    except Exception:
        return ""


def to_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return output.getvalue()


def show_preview(df_show: pd.DataFrame, title: str | None = None, preview_rows: int = PREVIEW_ROWS):
    if title:
        st.subheader(title)

    total_rows = len(df_show)
    preview_df = df_show.head(preview_rows).copy()

    st.dataframe(preview_df, use_container_width=True, hide_index=True)

    if total_rows > preview_rows:
        st.caption(f"Đang hiển thị {preview_rows:,} dòng đầu trên tổng {total_rows:,} dòng.")


def safe_multiselect_all(
    key: str,
    label: str,
    options,
    all_label: str = "All",
    default_all: bool = True,
    normalize: bool = True,
):
    opts = pd.Series(list(options)).dropna().astype(str)
    if normalize:
        opts = opts.str.strip()

    opts = opts[opts != ""]
    opts = sorted(opts.unique().tolist())

    ui_opts = [all_label] + opts

    if key not in st.session_state:
        st.session_state[key] = [all_label] if default_all else (opts[:1] if opts else [all_label])

    cur = st.session_state.get(key, [])
    cur = [str(x).strip() for x in cur if str(x).strip() in ui_opts]
    if not cur:
        cur = [all_label] if default_all else (opts[:1] if opts else [all_label])
        st.session_state[key] = cur

    selected = st.multiselect(label, options=ui_opts, key=key)

    if (not selected) or (all_label in selected):
        return opts
    return [x for x in selected if x in opts]


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    source = uploaded_file if uploaded_file is not None else DEFAULT_PARQUET_PATH
    df = pd.read_parquet(source)

    if df is None or df.empty:
        return pd.DataFrame()

    if "Ngày" in df.columns:
        df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
        df = df.dropna(subset=["Ngày"])

    for c in ["Tổng_Gross", "Tổng_Net"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in [
        "LoaiCT", "Brand", "Region", "Điểm_mua_hàng",
        "Kiểm_tra_tên", "Trạng_thái_số_điện_thoại"
    ]:
        if c in df.columns:
            try:
                df[c] = df[c].astype("string").fillna("")
            except Exception:
                pass

    if "tên_KH" not in df.columns:
        df["tên_KH"] = ""

    if "Số_điện_thoại" in df.columns:
        df["Số_điện_thoại"] = (
            df["Số_điện_thoại"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.strip()
        )

    if "Số_CT" in df.columns:
        df["Số_CT"] = df["Số_CT"].astype(str).str.strip()

    return df


def base_date_filter(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    mask = (df["Ngày"] >= pd.to_datetime(start_date)) & (df["Ngày"] <= pd.to_datetime(end_date))
    return df.loc[mask].copy()


def apply_filters(df: pd.DataFrame, loaiCT, brand, region, store) -> pd.DataFrame:
    work = df

    if "LoaiCT" in work.columns:
        work = work[work["LoaiCT"].isin(loaiCT if loaiCT else [])]
    if "Brand" in work.columns:
        work = work[work["Brand"].isin(brand if brand else [])]
    if "Region" in work.columns:
        work = work[work["Region"].isin(region if region else [])]
    if "Điểm_mua_hàng" in work.columns:
        work = work[work["Điểm_mua_hàng"].isin(store if store else [])]

    return work.copy()


# =====================================================
# BUILD CRM
# =====================================================
def build_crm(df_f: pd.DataFrame, group_cols, group_all_customer: bool):
    work = df_f.copy()

    if "tên_KH" not in work.columns:
        work["tên_KH"] = ""
    if "Kiểm_tra_tên" not in work.columns:
        work["Kiểm_tra_tên"] = ""
    if "Trạng_thái_số_điện_thoại" not in work.columns:
        work["Trạng_thái_số_điện_thoại"] = ""
    if "Số_CT" not in work.columns:
        work["Số_CT"] = ""
    if "Tổng_Gross" not in work.columns:
        work["Tổng_Gross"] = 0
    if "Tổng_Net" not in work.columns:
        work["Tổng_Net"] = 0

    # ==================================================
    # CASE 1: KHÔNG gộp tất cả giao dịch của 1 KH
    # => mỗi đơn = 1 dòng
    # ==================================================
    if not group_all_customer:
        order_keys = ["Số_điện_thoại"]
        if "Điểm_mua_hàng" in work.columns:
            order_keys.append("Điểm_mua_hàng")
        order_keys += ["Ngày", "Số_CT"]

        d = (
            work.groupby(order_keys, observed=True, dropna=False)
            .agg(
                Name=("tên_KH", "first"),
                Name_Check=("Kiểm_tra_tên", "first"),
                Gross=("Tổng_Gross", "sum"),
                Net=("Tổng_Net", "sum"),
                Check_SDT=("Trạng_thái_số_điện_thoại", "first"),
            )
            .reset_index()
        )

        d["Orders"] = 1
        d["First_Order"] = d["Ngày"]
        d["Last_Order"] = d["Ngày"]
        d["Last_Số_CT"] = d["Số_CT"]
        return d

    # ==================================================
    # CASE 2: CÓ gộp tất cả giao dịch của 1 KH
    # ==================================================
    required_cols = [c for c in group_cols if c in work.columns]

    d = (
        work.groupby(required_cols, observed=True, dropna=False)
        .agg(
            Name=("tên_KH", "first"),
            Name_Check=("Kiểm_tra_tên", "first"),
            Gross=("Tổng_Gross", "sum"),
            Net=("Tổng_Net", "sum"),
            Orders=("Số_CT", "nunique"),
            First_Order=("Ngày", "min"),
            Last_Order=("Ngày", "max"),
            Check_SDT=("Trạng_thái_số_điện_thoại", "first"),
        )
        .reset_index()
    )

    latest_tx = (
        work.sort_values(["Ngày", "Số_CT"])
        .groupby(required_cols, observed=True, dropna=False)
        .tail(1)
        .copy()
    )

    latest_tx = latest_tx[required_cols + ["Số_CT"]].rename(columns={"Số_CT": "Last_Số_CT"})
    d = d.merge(latest_tx, on=required_cols, how="left")
    return d


# =====================================================
# PARETO
# =====================================================
def pareto_customer_by_store(df: pd.DataFrame, percent=20, top=True) -> pd.DataFrame:
    rows = []

    if "Điểm_mua_hàng" not in df.columns or df.empty:
        return pd.DataFrame()

    for store, d in df.groupby("Điểm_mua_hàng", observed=True):
        g = (
            d.groupby("Số_điện_thoại", observed=True)
            .agg(
                Gross=("Tổng_Gross", "sum"),
                Net=("Tổng_Net", "sum"),
                Orders=("Số_CT", "nunique"),
            )
            .reset_index()
            .sort_values("Net", ascending=False)
        )

        if g.empty:
            continue

        g["CK_%"] = np.where(
            g["Gross"] > 0,
            ((g["Gross"] - g["Net"]) / g["Gross"] * 100).round(2),
            0,
        )

        total_net = g["Net"].sum()
        g["Contribution_%"] = (g["Net"] / total_net * 100).round(2) if total_net != 0 else 0
        g["Cum_%"] = g["Contribution_%"].cumsum().round(2)

        n = max(1, int(len(g) * percent / 100))
        g_sel = g.head(n) if top else g.tail(n)

        g_sel = g_sel.copy()
        g_sel["Điểm_mua_hàng"] = store
        rows.append(g_sel)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


# =====================================================
# PAGE
# =====================================================
st.title("📤 CRM & Pareto")

try:
    df = load_data(uploaded_file)

    if uploaded_file is not None:
        st.sidebar.success("✅ Đang dùng file parquet bạn tải lên")
    else:
        st.sidebar.info(f"📦 Đang dùng file mặc định: {DEFAULT_PARQUET_PATH}")

    if df.empty:
        st.warning("⚠ Không có dữ liệu để phân tích. Kiểm tra lại file parquet.")
        st.stop()

    required_base = ["Ngày", "Số_điện_thoại"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        st.error(f"❌ File thiếu cột bắt buộc: {missing_base}")
        st.write("Các cột hiện có:", list(df.columns))
        st.stop()

    st.sidebar.caption(f"RAM df ~ {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # =====================================================
    # FILTER: DATE FIRST
    # =====================================================
    with st.sidebar:
        st.header("🎛️ Bộ lọc dữ liệu (CRM)")
        start_date = st.date_input("Từ ngày", df["Ngày"].min().date())
        end_date = st.date_input("Đến ngày", df["Ngày"].max().date())

    df_base = base_date_filter(df, start_date, end_date)

    if df_base.empty:
        st.warning("⚠ Không có dữ liệu sau khi lọc ngày.")
        st.stop()

    with st.sidebar:
        loaiCT_filter = safe_multiselect_all(
            key="loaiCT_filter",
            label="Loại CT",
            options=df_base["LoaiCT"].unique() if "LoaiCT" in df_base.columns else [],
            all_label="All",
            default_all=True,
        )

        brand_filter = safe_multiselect_all(
            key="brand_filter",
            label="Brand",
            options=df_base["Brand"].unique() if "Brand" in df_base.columns else [],
            all_label="All",
            default_all=True,
        )

    df_b = df_base.copy()
    if "LoaiCT" in df_b.columns:
        df_b = df_b[df_b["LoaiCT"].isin(loaiCT_filter)]
    if "Brand" in df_b.columns:
        df_b = df_b[df_b["Brand"].isin(brand_filter)]

    with st.sidebar:
        region_filter = safe_multiselect_all(
            key="region_filter",
            label="Region",
            options=df_b["Region"].unique() if "Region" in df_b.columns else [],
            all_label="All",
            default_all=True,
        )

    df_br = df_b.copy()
    if "Region" in df_br.columns:
        df_br = df_br[df_br["Region"].isin(region_filter)]

    with st.sidebar:
        store_filter = safe_multiselect_all(
            key="store_filter",
            label="Cửa hàng",
            options=df_br["Điểm_mua_hàng"].unique() if "Điểm_mua_hàng" in df_br.columns else [],
            all_label="All",
            default_all=True,
        )

    df_f = apply_filters(df_base, loaiCT_filter, brand_filter, region_filter, store_filter)

    if df_f.empty:
        st.warning("⚠ Không có dữ liệu sau khi áp bộ lọc.")
        st.stop()

    today = df_f["Ngày"].max()

    # =====================================================
    # CRM PARAMS
    # =====================================================
    st.sidebar.header("📤 Xuất KH")

    INACTIVE_DAYS = st.sidebar.slider("Inactive ≥ bao nhiêu ngày", 30, 365, 90, 15)

    VIP_NET_THRESHOLD = st.sidebar.number_input(
        "Net tối thiểu để vào VIP",
        min_value=0,
        value=300_000_000,
        step=10_000_000,
    )

    GROUP_BY_CUSTOMER = st.sidebar.checkbox("Gộp tất cả giao dịch của 1 KH", value=False)
    min_net = st.sidebar.number_input("Net tối thiểu (lọc)", 0, value=0)

    group_cols = ["Số_điện_thoại"]
    if not GROUP_BY_CUSTOMER and "Điểm_mua_hàng" in df_f.columns:
        group_cols.append("Điểm_mua_hàng")

    # =====================================================
    # BUILD CRM TABLE
    # =====================================================
    df_export = build_crm(df_f, group_cols, GROUP_BY_CUSTOMER)

    df_export["CK_%"] = np.where(
        df_export["Gross"] > 0,
        (df_export["Gross"] - df_export["Net"]) / df_export["Gross"] * 100,
        0,
    ).round(2)

    if GROUP_BY_CUSTOMER:
        df_export["Days_Inactive"] = (today - df_export["Last_Order"]).dt.days

        df_export["KH_tag"] = np.select(
            [
                df_export["Days_Inactive"] >= INACTIVE_DAYS,
                df_export["Net"] >= VIP_NET_THRESHOLD,
            ],
            ["KH Inactive", "KH VIP"],
            default="Khách hàng",
        )

        df_export["Bao_lâu_không_mua"] = np.where(
            df_export["KH_tag"] == "KH Inactive",
            df_export["Days_Inactive"],
            np.nan,
        ).astype("float")
    else:
        df_export["KH_tag"] = ""
        df_export["Bao_lâu_không_mua"] = np.nan

    df_export = df_export[df_export["Net"] >= min_net].copy()

    if GROUP_BY_CUSTOMER:
        display_cols = [
            "Số_điện_thoại",
            "Name",
            "KH_tag",
            "Gross",
            "Net",
            "CK_%",
            "Orders",
            "Bao_lâu_không_mua",
            "Last_Order",
            "Last_Số_CT",
        ]
    else:
        display_cols = [
            "Số_điện_thoại",
            "Name",
            "Ngày",
            "Số_CT",
            "Gross",
            "Net",
            "CK_%",
        ]
        if "Điểm_mua_hàng" in df_export.columns:
            display_cols.insert(1, "Điểm_mua_hàng")

    # =====================================================
    # FILTER ON CRM TABLE
    # =====================================================
    st.subheader("📄 Danh sách KH xuất CRM")
    st.markdown("### 🔎 Lọc nhanh trên bảng")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        show_inactive = st.checkbox("Chỉ KH Inactive", value=False)
    with col2:
        show_vip = st.checkbox("Chỉ KH VIP", value=False)
    with col3:
        show_customer = st.checkbox("Khách hàng thường", value=True)
    with col4:
        kiem_tra_ten_filter = safe_multiselect_all(
            key="kiem_tra_ten_filter",
            label="Kiểm tra tên KH",
            options=df_f["Kiểm_tra_tên"].unique() if "Kiểm_tra_tên" in df_f.columns else [],
            all_label="All",
            default_all=True,
        )
    with col5:
        check_sdt_filter = safe_multiselect_all(
            key="check_sdt_filter",
            label="Check SĐT",
            options=df_export["Check_SDT"].unique() if "Check_SDT" in df_export.columns else [],
            all_label="All",
            default_all=True,
        )

    if GROUP_BY_CUSTOMER:
        selected_tags = []
        if show_inactive:
            selected_tags.append("KH Inactive")
        if show_vip:
            selected_tags.append("KH VIP")
        if show_customer:
            selected_tags.append("Khách hàng")
        if selected_tags:
            df_export = df_export[df_export["KH_tag"].isin(selected_tags)]

    if check_sdt_filter and "Check_SDT" in df_export.columns:
        df_export = df_export[df_export["Check_SDT"].isin(check_sdt_filter)]

    if kiem_tra_ten_filter and "Name_Check" in df_export.columns:
        df_export = df_export[df_export["Name_Check"].isin(kiem_tra_ten_filter)]

    sort_col = st.selectbox(
        "Sắp xếp theo",
        options=df_export.columns.tolist(),
        index=df_export.columns.tolist().index("Net") if "Net" in df_export.columns else 0,
    )
    sort_order = st.radio("Thứ tự", ["Giảm dần", "Tăng dần"], horizontal=True)
    df_export = df_export.sort_values(sort_col, ascending=(sort_order == "Tăng dần"))

    total_kh_filtered = df_export["Số_điện_thoại"].nunique() if "Số_điện_thoại" in df_export.columns else 0
    st.info(f"👥 Tổng số KH theo bộ lọc hiện tại: **{total_kh_filtered:,}** khách hàng")

    total_row = {}
    for col in df_export.columns:
        if col in ["Gross", "Net", "Orders"]:
            total_row[col] = df_export[col].sum()
        elif col == "CK_%":
            total_row[col] = df_export[col].mean()
        elif col in ["Last_Order", "Ngày"]:
            total_row[col] = pd.NaT
        elif col == "Số_điện_thoại":
            total_row[col] = "TỔNG"
        elif col == "Bao_lâu_không_mua":
            total_row[col] = np.nan
        else:
            total_row[col] = ""

    df_export_with_total = pd.concat([df_export, pd.DataFrame([total_row])], ignore_index=True)

    display_cols = [c for c in display_cols if c in df_export_with_total.columns]
    df_export_display = df_export_with_total[display_cols].copy()

    if not GROUP_BY_CUSTOMER and "Ngày" in df_export_display.columns:
        df_export_display = df_export_display.rename(columns={"Ngày": "Ngày_mua"})

    for c in ["Gross", "Net", "Orders"]:
        if c in df_export_display.columns:
            df_export_display[c] = df_export_display[c].apply(fmt_int)

    if "CK_%" in df_export_display.columns:
        df_export_display["CK_%"] = df_export_display["CK_%"].apply(lambda v: fmt_pct(v, 2))

    if "Bao_lâu_không_mua" in df_export_display.columns:
        df_export_display["Bao_lâu_không_mua"] = df_export_display["Bao_lâu_không_mua"].apply(
            lambda v: "" if pd.isna(v) else fmt_int(v)
        )

    for dt_col in ["Last_Order", "Ngày_mua", "Ngày"]:
        if dt_col in df_export_display.columns:
            df_export_display[dt_col] = pd.to_datetime(df_export_display[dt_col], errors="coerce").dt.strftime("%Y-%m-%d")

    show_preview(df_export_display, preview_rows=PREVIEW_ROWS)

    st.download_button(
        "📥 Tải danh sách KH (Excel)",
        data=to_excel(df_export_display),
        file_name="customer_marketing.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # =====================================================
    # PARETO
    # =====================================================
    st.sidebar.header("🏆 Pareto KH theo Cửa hàng")
    run_pareto = st.sidebar.checkbox("Chạy Pareto", value=False)

    if run_pareto:
        pareto_percent = st.sidebar.slider("Chọn % KH Pareto", 5, 50, 20)
        pareto_type = st.sidebar.radio("Loại Pareto", ["Top", "Bottom"])

        store_options = sorted(df_f["Điểm_mua_hàng"].dropna().astype(str).unique()) if "Điểm_mua_hàng" in df_f.columns else []
        default_stores = store_options[:5] if len(store_options) > 5 else store_options

        store_filter_pareto = st.sidebar.multiselect(
            "Chọn Cửa hàng (Pareto)",
            store_options,
            default=default_stores,
        )

        df_pareto_base = df_f.copy()
        if store_filter_pareto and "Điểm_mua_hàng" in df_pareto_base.columns:
            df_pareto_base = df_pareto_base[df_pareto_base["Điểm_mua_hàng"].astype(str).isin(store_filter_pareto)]

        df_pareto = pareto_customer_by_store(
            df_pareto_base,
            percent=pareto_percent,
            top=(pareto_type == "Top"),
        )

        st.subheader(f"🏆 {pareto_type} {pareto_percent}% KH theo từng Cửa hàng (Pareto)")

        if not df_pareto.empty:
            df_pareto_show = df_pareto[
                ["Điểm_mua_hàng", "Số_điện_thoại", "Gross", "Net", "CK_%", "Orders", "Contribution_%", "Cum_%"]
            ].copy()

            for c in ["Gross", "Net", "Orders"]:
                if c in df_pareto_show.columns:
                    df_pareto_show[c] = df_pareto_show[c].apply(fmt_int)
            if "CK_%" in df_pareto_show.columns:
                df_pareto_show["CK_%"] = df_pareto_show["CK_%"].apply(lambda v: fmt_pct(v, 2))
            if "Contribution_%" in df_pareto_show.columns:
                df_pareto_show["Contribution_%"] = df_pareto_show["Contribution_%"].apply(lambda v: fmt_pct(v, 2))
            if "Cum_%" in df_pareto_show.columns:
                df_pareto_show["Cum_%"] = df_pareto_show["Cum_%"].apply(lambda v: fmt_pct(v, 2))

            show_preview(df_pareto_show, preview_rows=PREVIEW_ROWS)
        else:
            st.info("Không có dữ liệu phù hợp cho Pareto.")

    # =====================================================
    # DEBUG
    # =====================================================
    with st.expander("🔍 Xem danh sách cột trong file hiện tại"):
        st.write(list(df.columns))

    # =====================================================
    # RESET
    # =====================================================
    with st.sidebar:
        if st.button("🔄 Reset filters"):
            for k in [
                "loaiCT_filter",
                "brand_filter",
                "region_filter",
                "store_filter",
                "kiem_tra_ten_filter",
                "check_sdt_filter",
            ]:
                st.session_state.pop(k, None)
            st.rerun()

        if st.button("♻️ Reset cache"):
            st.cache_data.clear()
            st.rerun()

except Exception as e:
    st.error("❌ App đang lỗi khi khởi động.")
    st.exception(e)

    with st.expander("Xem traceback chi tiết"):
        st.code(traceback.format_exc())
