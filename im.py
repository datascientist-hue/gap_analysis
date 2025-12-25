import streamlit as st
import pandas as pd
import numpy as np
import ftplib
import io
import toml

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Distributor Pareto Analysis Dashboard", layout="wide")

def load_config():
    """Accesses secrets directly from Streamlit's internal secrets management"""
    try:
        # On Streamlit Cloud, st.secrets behaves like a dictionary
        return st.secrets
    except Exception as e:
        st.error("Secrets not found! Please add them to the Streamlit App Settings.")
        return None
# --- 2. INDIAN CURRENCY FORMATTER ---
def format_inr(number):
    """Formats numbers into Indian numbering system (K, L, Cr)"""
    if number is None or pd.isna(number):
        return "₹0"
    val = abs(number)
    sign = "-" if number < 0 else ""
    if val >= 10000000:  # 1 Crore
        return f"{sign}₹{val / 10000000:.2f} Cr"
    elif val >= 100000:   # 1 Lakh
        return f"{sign}₹{val / 100000:.2f} L"
    elif val >= 1000:     # 1 Thousand
        return f"{sign}₹{val / 1000:.2f} K"
    else:
        return f"{sign}₹{val:.2f}"

# --- 3. FTP DATA LOADER ---
@st.cache_data(ttl=300)
def get_ftp_parquet(file_path, config):
    try:
        ftp_cfg =  st.secrets['ftp']
        ftp = ftplib.FTP(ftp_cfg['host'])
        ftp.login(ftp_cfg['user'], ftp_cfg['password'])
        flo = io.BytesIO()
        ftp.retrbinary(f"RETR {file_path}", flo.write)
        flo.seek(0)
        ftp.quit()
        return pd.read_parquet(flo, engine='pyarrow')
    except Exception as e:
        st.error(f"Failed to load: {file_path}. Error: {e}")
        return pd.DataFrame()

def main():
    st.title("🚀 Distributor 80/20 Gap & Action Dashboard")
    st.markdown("### Strategic Performance Analysis (Tamil Nadu Pricing)")

    # Check if secrets exist
    if "paths" not in st.secrets:
        st.error("Secrets not found! Please add them to the Streamlit App Settings.")
        return

    paths = st.secrets['paths']

    # Fetch Files - REMOVED 'config' from the calls below
    with st.spinner('Syncing data from FTP...'):
        df_aop = get_ftp_parquet(paths['aop'])
        df_sales = get_ftp_parquet(paths['primary_sales'])
        df_cheque = get_ftp_parquet(paths['cheque_status'])
        df_master = get_ftp_parquet(paths['db_master'])
        df_price = get_ftp_parquet(paths['price_list'])
    if df_aop.empty or df_sales.empty or df_price.empty:
        st.error("Essential files are missing. Check FTP paths.")
        return

    # --- 4. DATA CLEANING & STANDARDIZATION ---
    for df in [df_aop, df_sales, df_cheque, df_master, df_price]:
        df.columns = df.columns.str.strip()

    # Apply Price List Rules: Tamil Nadu + Valid Only
    df_price = df_price[
        (df_price['Price Validity'] == 'Price List Valid Still') & 
        (df_price['State'] == 'Tamil Nadu')
    ]

    # Numeric Enforcement
    df_aop['total_unit'] = pd.to_numeric(df_aop['total_unit'], errors='coerce').fillna(0).astype(float)
    df_aop['Cases'] = pd.to_numeric(df_aop['Cases'], errors='coerce').fillna(0).astype(float)
    df_aop['target_in_qty(vol)'] = pd.to_numeric(df_aop['target_in_qty(vol)'], errors='coerce').fillna(0).astype(float)
    
    df_sales['PrimaryQtyinNos'] = pd.to_numeric(df_sales['PrimaryQtyinNos'], errors='coerce').fillna(0).astype(float)
    df_sales['PrimaryQtyinCases/Bags'] = pd.to_numeric(df_sales['PrimaryQtyinCases/Bags'], errors='coerce').fillna(0).astype(float)
    
    df_price['CFA Inv'] = pd.to_numeric(df_price['CFA Inv'], errors='coerce').fillna(0).astype(float)
    df_master['Balance'] = pd.to_numeric(df_master['Balance'], errors='coerce').fillna(0).astype(float)

    # Key Standardization
    df_aop['DB Code'] = df_aop['DB Code'].astype(str).str.strip()
    df_aop['New SKU'] = df_aop['New SKU'].astype(str).str.strip()
    df_sales['BP Code'] = df_sales['BP Code'].astype(str).str.strip()
    df_sales['ProductGroup'] = df_sales['ProductGroup'].astype(str).str.strip()
    df_sales['JCPeriodNum'] = df_sales['JCPeriodNum'].astype(str).str.strip().str.zfill(2)

    # --- 5. SIDEBAR FILTERS ---
    st.sidebar.header("Data Selection")
    unique_jcs = sorted(df_aop['jc'].dropna().unique()) if 'jc' in df_aop.columns else []
    selected_jc = st.sidebar.selectbox("Select JC Period", options=unique_jcs)
    df_aop = df_aop[df_aop['jc'] == selected_jc]

    if 'Kurukshetra' in df_aop.columns:
        kurukshetra_options = sorted(df_aop['Kurukshetra'].dropna().unique())
        selected_k = st.sidebar.multiselect("Kurukshetra Category", options=kurukshetra_options, default=kurukshetra_options)
        df_aop = df_aop[df_aop['Kurukshetra'].isin(selected_k)]

    df_sales = df_sales[(df_sales['DocumentType'] == 'SalesInvoice') & (df_sales['CustomerClass'] == 'ARJUNAN')]
    clean_jc_num = ''.join(filter(str.isdigit, str(selected_jc))).zfill(2)
    df_sales = df_sales[df_sales['JCPeriodNum'].str.contains(clean_jc_num, na=False)]

    # --- 6. MERGING & GAP LOGIC ---
    sales_grouped = df_sales.groupby(['BP Code', 'ProductGroup'], as_index=False).agg({
        'PrimaryQtyinNos': 'sum',
        'PrimaryQtyinCases/Bags': 'sum'
    })

    merged = pd.merge(df_aop, sales_grouped, left_on=['DB Code', 'New SKU'], right_on=['BP Code', 'ProductGroup'], how='left').fillna(0)

    # Performance Status Logic
    conditions = [
        (merged['target_in_qty(vol)'] == 0),
        (merged['PrimaryQtyinNos'] == merged['total_unit']),
        (merged['PrimaryQtyinNos'] > merged['total_unit'])
    ]
    choices = ["aop zero", "Target Achieved", "Greater than target"]
    merged['Performance_Status'] = np.select(conditions, choices, default="Less than target")

    # Financial Math
    price_map = df_price[['Name', 'CFA Inv']].drop_duplicates('Name')
    merged = pd.merge(merged, price_map, left_on='New SKU', right_on='Name', how='left').fillna({'CFA Inv': 0})
    merged['Unit Gap'] = np.where(merged['Performance_Status'] == "Less than target", merged['total_unit'] - merged['PrimaryQtyinNos'], 0)
    merged['Case Gap'] = np.where(merged['Performance_Status'] == "Less than target", merged['Cases'] - merged['PrimaryQtyinCases/Bags'], 0)
    merged['Value Loss Raw'] = merged['Unit Gap'] * merged['CFA Inv']

    # Master enrichment
    cheque_set = set(df_cheque['U_BPCode'].astype(str).unique())
    merged['Cheque Status'] = merged['DB Code'].apply(lambda x: "cheque available" if x in cheque_set else "cheque not available")
    merged = pd.merge(merged, df_master[['CardCode', 'Balance']], left_on='DB Code', right_on='CardCode', how='left')
    merged['Balance Raw'] = merged['Balance'].fillna(0).astype(float)
    merged['Payment Status'] = np.where(merged['Balance Raw'] > 0, "overdue", np.where(merged['Balance Raw'] < 0, "advance", "no due"))

    # --- 7. PARETO CALCULATION (Distributor Level) ---
    db_loss_summary = merged.groupby(['Region', 'DB Code', 'DB Name'])['Value Loss Raw'].sum().reset_index()
    db_loss_summary = db_loss_summary.sort_values(by='Value Loss Raw', ascending=False)
    db_loss_summary['Cumulative Loss'] = db_loss_summary['Value Loss Raw'].cumsum()
    total_market_loss = db_loss_summary['Value Loss Raw'].sum()
    db_loss_summary['Cumulative %'] = (db_loss_summary['Cumulative Loss'] / total_market_loss * 100) if total_market_loss > 0 else 0
    
    # Define the "Vital Few"
    priority_db_codes = db_loss_summary[db_loss_summary['Cumulative %'] <= 81]['DB Code'].tolist()
    
    # Filter for Unbilled SKUs for those priority DBs
    unbilled_focus_df = merged[
        (merged['DB Code'].isin(priority_db_codes)) & 
        (merged['PrimaryQtyinNos'] == 0) & 
        (merged['total_unit'] > 0)
    ].copy()

    # --- 8. UI RENDERING & DYNAMIC FILTERS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Global View Controls")
    
    # Region Filter
    sel_reg = st.sidebar.multiselect("Filter by Region", sorted(merged['Region'].unique()) if 'Region' in merged.columns else [])
    
    # NEW: Performance Status Filter
    perf_status_options = sorted(merged['Performance_Status'].unique())
    sel_perf = st.sidebar.multiselect("Filter by Performance Status", options=perf_status_options, default=perf_status_options)
    
    final_df = merged.copy()
    
    # Apply Filters
    if sel_reg:
        final_df = final_df[final_df['Region'].isin(sel_reg)]
        db_loss_summary = db_loss_summary[db_loss_summary['Region'].isin(sel_reg)]
        unbilled_focus_df = unbilled_focus_df[unbilled_focus_df['Region'].isin(sel_reg)]

    if sel_perf:
        final_df = final_df[final_df['Performance_Status'].isin(sel_perf)]

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current JC", selected_jc)
    m2.metric("Total Money Loss", format_inr(final_df['Value Loss Raw'].sum()))
    m3.metric("Vital Few DBs (80%)", f"{len(priority_db_codes)}")
    m4.metric("AOP Zero SKUs", final_df[final_df['Performance_Status'] == "aop zero"]['New SKU'].nunique())

    # TABLE 1: PARETO DISTRIBUTOR SUMMARY
    st.subheader("🎯 Table 1: Top Distributors causing 80% of Loss")
    st.markdown("Focus on these Distributors to capture the bulk of the missing business.")
    st.dataframe(db_loss_summary.style.format({
        'Value Loss Raw': lambda x: format_inr(x),
        'Cumulative Loss': lambda x: format_inr(x),
        'Cumulative %': '{:.2f}%'
    }), use_container_width=True)

    # TABLE 2: UNBILLED ACTION LIST
    st.markdown("---")
    st.subheader("📝 Table 2: Action List - Unbilled SKUs for Priority Distributors")
    st.info("These are SKUs where these priority distributors have a target but 0 actual billing.")
    st.dataframe(unbilled_focus_df[['Region', 'DB Code', 'DB Name', 'New SKU', 'total_unit','Cases' ,'Value Loss Raw', 'Cheque Status', 'Payment Status']].rename(columns={
        'total_unit': 'Target Units', 'Value Loss Raw': 'Potential Recovery','Cases':'Target Cases'
    }).sort_values('Potential Recovery', ascending=False).style.format({
        'Target Units': '{:.0f}', 'Potential Recovery': lambda x: format_inr(x), 'Target Cases': '{:.0f}'
    }), use_container_width=True)

    # TABLE 3: FULL REPORT
    st.markdown("---")
    st.subheader("📋 Table 3: Full Distributor Detail Report")
    st.dataframe(final_df.rename(columns={
        'total_unit': 'Target(U)', 'PrimaryQtyinNos': 'Actual(U)', 'Cases': 'Target(C)', 'PrimaryQtyinCases/Bags': 'Actual(C)', 'Value Loss Raw': 'Loss(INR)'
    }).style.format({
        'Target(U)': '{:.0f}', 'Actual(U)': '{:.0f}', 'Unit Gap': '{:.0f}',
        'Target(C)': '{:.0f}', 'Actual(C)': '{:.0f}', 'Case Gap': '{:.0f}',
        'Loss(INR)': lambda x: format_inr(x), 'Balance Raw': lambda x: format_inr(x),'target_in_qty(vol)':'{:.0f}','CFA Inv':'{:.2f}'
    }), use_container_width=True)

    st.download_button("📥 Download Action List", unbilled_focus_df.to_csv(index=False), "unbilled_action_list.csv")

if __name__ == "__main__":
    main()