import streamlit as st
import pandas as pd
import numpy as np
import ftplib
import io
import os
import re

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Distributor Pareto Analysis Dashboard", layout="wide")

# --- 2. INDIAN CURRENCY FORMATTER ---
def format_inr(number, decimals=2):
    """Formats numbers into Indian numbering system (K, L, Cr)"""
    if number is None or pd.isna(number):
        return "₹0"
    val = abs(number)
    sign = "-" if number < 0 else ""
    
    if val >= 10000000:  # 1 Crore
        res = f"{sign}₹{val / 10000000:.{decimals}f} Cr"
    elif val >= 100000:   # 1 Lakh
        res = f"{sign}₹{val / 100000:.{decimals}f} L"
    elif val >= 1000:     # 1 Thousand
        res = f"{sign}₹{val / 1000:.{decimals}f} K"
    else:
        res = f"{sign}₹{val:.{decimals}f}"
    return res

def display_fixed_totals(df, numeric_cols, format_mapping):
    """Displays a non-scrollable summary row below the main table"""
    if df.empty:
        return
    
    # Calculate totals
    totals = df[numeric_cols].sum()
    
    # Create a summary dataframe
    summary_df = pd.DataFrame([totals])
    summary_df.insert(0, 'Summary', 'GRAND TOTAL')
    
    # Display as a static table (no scrollbars)
    st.markdown("---")
    st.write("📊 **Grand Totals (All Rows):**")
    st.dataframe(
        summary_df.style.format(format_mapping), 
        use_container_width=True,
        hide_index=True
    )
# Helper to extract digits from JC strings (e.g., "JC01" -> 1)
def extract_jc_number(jc_str):
    if pd.isna(jc_str): return 0
    nums = re.findall(r'\d+', str(jc_str))
    return int(nums[0]) if nums else 0

# --- 3. DYNAMIC DATA LOADER (FTP or LOCAL) ---
@st.cache_data(ttl=600)
def load_data(file_key, source_mode="FTP"):
    if "paths" not in st.secrets:
        st.error("Secrets file missing or 'paths' not defined.")
        return pd.DataFrame()
    
    ftp_path = st.secrets['paths'][file_key]
    
    if source_mode == "Local":
        filename = os.path.basename(ftp_path)
        local_path = os.path.join("data", filename)
        try:
            return pd.read_parquet(local_path, engine='pyarrow')
        except:
            st.error(f"Local file {local_path} not found in /data folder.")
            return pd.DataFrame()
    else:
        try:
            ftp_cfg = st.secrets['ftp']
            ftp = ftplib.FTP(ftp_cfg['host'])
            ftp.login(ftp_cfg['user'], ftp_cfg['password'])
            flo = io.BytesIO()
            ftp.retrbinary(f"RETR {ftp_path}", flo.write)
            flo.seek(0)
            ftp.quit()
            return pd.read_parquet(flo, engine='pyarrow')
        except Exception as e:
            st.error(f"FTP Error: {e}")
            return pd.DataFrame()

def main():
    st.title("🚀 Distributor 80/20 Gap & Action Dashboard")

    # --- 4. SIDEBAR CONFIGURATION ---
    st.sidebar.header("🔧 Settings")
    source_mode = st.sidebar.radio("Data Source Mode", ["FTP"])

    with st.spinner('Syncing data...'):
        df_aop = load_data('aop', source_mode)
        df_sales = load_data('primary_sales', source_mode)
        df_cheque = load_data('cheque_status', source_mode)
        df_master = load_data('db_master', source_mode)
        df_price = load_data('price_list', source_mode)

    if df_aop.empty or df_sales.empty:
        st.warning("Critical data missing. App execution stopped.")
        st.stop()

    # --- 5. DATA CLEANING & STANDARDIZATION ---
    for df in [df_aop, df_sales, df_cheque, df_master, df_price]:
        df.columns = df.columns.str.strip()

    # Numeric Enforcement
    df_aop['total_unit'] = pd.to_numeric(df_aop['total_unit'], errors='coerce').fillna(0)
    df_aop['Cases'] = pd.to_numeric(df_aop['Cases'], errors='coerce').fillna(0)
    
    df_sales['PrimaryQtyinNos'] = pd.to_numeric(df_sales['PrimaryQtyinNos'], errors='coerce').fillna(0)
    df_sales['PrimaryQtyinCases/Bags'] = pd.to_numeric(df_sales['PrimaryQtyinCases/Bags'], errors='coerce').fillna(0)
    df_sales['PrimaryQtyInLtrs/Kgs'] = pd.to_numeric(df_sales['PrimaryQtyInLtrs/Kgs']/1000, errors='coerce').fillna(0)
    
    df_price['Taxable Value'] = pd.to_numeric(df_price['Taxable Value'], errors='coerce').fillna(0)
    df_master['Balance'] = pd.to_numeric(df_master['Balance'], errors='coerce').fillna(0)

    # --- 6. PRIMARY FILTERS (JC & CUSTOMER CLASS) ---
    st.sidebar.markdown("---")
    st.sidebar.header("Global Filters")
    unique_jcs = sorted(df_aop['jc'].dropna().unique())
    selected_jc = st.sidebar.selectbox("Select JC Period", options=unique_jcs)

    #unique_Prod_ctg=sorted(df_aop['Updated Category'].dropna().unique())
    #selected_Prod_ctg=st.sidebar.selectbox('Select_Prod_Ctg',options=unique_Prod_ctg)
    
    # Filter AOP
    df_aop_jc = df_aop[df_aop['jc'] == selected_jc].copy()
    #df_aop_Prod = df_aop[df_aop['Updated Category'] == selected_Prod_ctg].copy()
    
    # Filter Sales for 'ARJUNAN' and the selected JC
    clean_jc_num = ''.join(filter(str.isdigit, str(selected_jc))).zfill(2)
    df_sales_jc = df_sales[
        (df_sales['DocumentType'] == 'SalesInvoice') & 
        (df_sales['CustomerClass'] == 'ARJUNAN') & 
        (df_sales['JCPeriodNum'].str.contains(clean_jc_num, na=False))
    ].copy()

    # --- 7. BILLCUTS LOGIC (Based on InvNum) ---
    if 'InvNum' in df_sales_jc.columns:
        db_billcuts = df_sales_jc.groupby('BP Code')['InvNum'].nunique().reset_index()
        db_billcuts.columns = ['BP Code', 'Billcuts']
    else:
        db_billcuts = pd.DataFrame(columns=['BP Code', 'Billcuts'])

    # --- 8. MERGING & CALCULATIONS ---
    # SKU Level Aggregation
    sales_sku = df_sales_jc.groupby(['BP Code', 'ProductGroup'], as_index=False).agg({
        'PrimaryQtyinNos': 'sum',
        'PrimaryQtyinCases/Bags': 'sum',
        'PrimaryQtyInLtrs/Kgs': 'sum'
    })
    
    # Merge AOP + Sales + Billcuts
    merged = pd.merge(df_aop_jc, sales_sku, left_on=['DB Code', 'New SKU'], right_on=['BP Code', 'ProductGroup'], how='left').fillna(0)
    merged = pd.merge(merged, db_billcuts, left_on='DB Code', right_on='BP Code', how='left').fillna({'Billcuts': 0})

    # --- START NEW PRICE LIST LOGIC ---
    # 1. Convert selected JC and Price List JC ranges to integers for numeric comparison
    curr_jc_val = extract_jc_number(selected_jc)
    df_price['jc_from_int'] = df_price['U_JCFrom'].apply(extract_jc_number)
    df_price['jc_to_int'] = df_price['U_JCTo'].apply(extract_jc_number)

    # 2. Filter Price List: Selected JC must be between U_JCFrom and U_JCTo
    # 3. Match by State (from AOP) and SKU Name (from Price List)
    price_map = df_price[
        (curr_jc_val >= df_price['jc_from_int']) & 
        (curr_jc_val <= df_price['jc_to_int'])
    ][['State', 'Name', 'Taxable Value']].drop_duplicates(['State', 'Name'])

    # Merge on State and SKU Name
    merged = pd.merge(
        merged, 
        price_map, 
        left_on=['State', 'New SKU'], 
        right_on=['State', 'Name'], 
        how='left'
    ).fillna({'Taxable Value': 0})
    # --- END NEW PRICE LIST LOGIC ---
    
    # Loss & Gap Math
    merged['Unit Gap'] = np.where(merged['PrimaryQtyinNos'] < merged['total_unit'], merged['total_unit'] - merged['PrimaryQtyinNos'], 0)
    # Case Gap logic for KPIs
    merged['Case Gap'] = np.where(merged['PrimaryQtyinCases/Bags'] < merged['Cases'], merged['Cases'] - merged['PrimaryQtyinCases/Bags'], 0)
    merged['volume gap']= np.where(merged['PrimaryQtyInLtrs/Kgs'] < merged['target_in_qty(vol)'], merged['target_in_qty(vol)'] - merged['PrimaryQtyInLtrs/Kgs'], 0)
    
    merged['Sales Loss'] = merged['Unit Gap'] * merged['Taxable Value']
    merged['Actual Value'] = merged['PrimaryQtyinNos'] * merged['Taxable Value']

    # Enrichment (Cheques & Payment Status)
    cheque_set = set(df_cheque['U_BPCode'].astype(str).unique())
    merged['Cheque Status'] = merged['DB Code'].apply(lambda x: "Available" if x in cheque_set else "Not Available")
    merged = pd.merge(merged, df_master[['CardCode', 'Balance']], left_on='DB Code', right_on='CardCode', how='left')
    merged['Balance Raw'] = merged['Balance'].fillna(0).astype(float)
    merged['Payment Status'] = np.where(merged['Balance Raw'] > 0, "Overdue", np.where(merged['Balance Raw'] < 0, "Advance", "No Due"))

    # --- 9. DISTRIBUTOR RANKING & SIDEBAR FILTER ---
    # --- 9. DISTRIBUTOR RANKING & SIDEBAR FILTER ---
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Target Ranking")
    db_summary = merged.groupby(['Region', 'ASE', 'DB Code', 'DB Name', 'Billcuts', 'Cheque Status', 'Payment Status']).agg({
    'Sales Loss': 'sum',
    'target_in_qty(vol)': 'sum',
    'volume gap': 'sum'  # This is the "Volume Loss" logic you requested
    }).reset_index()

# Rename columns for cleaner use in the display
    db_summary = db_summary.rename(columns={
    'volume gap': 'Volume Loss',
    'target_in_qty(vol)': 'Target Volume'})

    db_summary = db_summary.sort_values(by='Sales Loss', ascending=False)
    db_summary['Rank'] = range(1, len(db_summary) + 1)

    rank_opts = ["All", "Top 20", "Top 40", "Top 60", "Top 80", "Top 100", "Greater than 100"]
    sel_rank = st.sidebar.selectbox("Filter by Sales Loss Rank", rank_opts)

    if sel_rank == "Top 20": db_summary = db_summary[db_summary['Rank'] <= 20]
    elif sel_rank == "Top 40": db_summary = db_summary[db_summary['Rank'] <= 40]
    elif sel_rank == "Top 60": db_summary = db_summary[db_summary['Rank'] <= 60]
    elif sel_rank == "Top 80": db_summary = db_summary[db_summary['Rank'] <= 80]
    elif sel_rank == "Top 100": db_summary = db_summary[db_summary['Rank'] <= 100]
    elif sel_rank == "Greater than 100": db_summary = db_summary[db_summary['Rank'] > 100]

    all_regs = sorted(db_summary['Region'].unique())
    sel_reg = st.sidebar.multiselect("Filter by Region", all_regs, default=all_regs)
    db_summary = db_summary[db_summary['Region'].isin(sel_reg)]
    
    allowed_dbs = db_summary['DB Code'].tolist()
    final_df = merged[merged['DB Code'].isin(allowed_dbs)].copy()

    # --- 10. KPI SECTION ---
    st.markdown("### 📈 Key Performance Indicators")
    
    # Calculations for KPIs
    total_sales_loss_val = final_df['Sales Loss'].sum()
    total_sales_loss_cases = final_df['volume gap'].sum()
    
    lt_target_df = final_df[(final_df['Unit Gap'] > 0) & (final_df['PrimaryQtyinCases/Bags'] > 0)]
    lt_val = lt_target_df['Sales Loss'].sum()
    lt_cases = lt_target_df['volume gap'].sum()
    
    unbilled_df = final_df[(final_df['PrimaryQtyinNos'] == 0) & (final_df['total_unit'] > 0)]
    unbilled_val = unbilled_df['Sales Loss'].sum()
    unbilled_cases = unbilled_df['volume gap'].sum()
    
    gt_target_df = final_df[final_df['PrimaryQtyinNos'] >= final_df['total_unit']]
    gt_val = gt_target_df['Actual Value'].sum()
    gt_cases = gt_target_df['PrimaryQtyInLtrs/Kgs'].sum()
    
    total_target_val = (final_df['total_unit'] * final_df['Taxable Value']).sum()
    total_achieved_val = final_df['Actual Value'].sum()

    total_target_mt = final_df['target_in_qty(vol)'].sum()
    total_achieved_mt = final_df['PrimaryQtyInLtrs/Kgs'].sum()

    r1c1, r1c2, r1c3 = st.columns(3)
    
    with r1c1:
        st.metric("Total Target ", f"{format_inr(total_target_val)}")
        st.caption(f"{total_target_mt:.1f} MT")

    with r1c2:
        st.metric("Actual Sales", format_inr(total_achieved_val))
        st.caption(f"{total_achieved_mt:.0f} MT")

    with r1c3:
        st.metric("Total Sales Loss", format_inr(total_sales_loss_val))
        st.caption(f"{int(total_sales_loss_cases)} MT")
        


        


    st.write("") # Adds a small vertical space between rows

    # --- ROW 2 ---
    r2c1, r2c2, r2c3 = st.columns(3) # Using 3 columns so the size matches Row 1
    

    with r2c1:
        st.metric("Performance Gap", format_inr(lt_val))
        st.caption(f"{int(lt_cases)} MT")
    with r2c2:
        st.metric("Unbilled Action Loss", format_inr(unbilled_val))
        st.caption(f"{int(unbilled_cases)} MT")    
    with r2c3:
        st.metric("Greater than Target Sales", format_inr(gt_val))
        st.caption(f"{int(gt_cases)} MT")
        


    # --- 11. UI RENDERING: RADIO BUTTON VIEWS ---
# --- 11. UI RENDERING: RADIO BUTTON VIEWS ---
# --- 11. UI RENDERING: RADIO BUTTON VIEWS ---
    st.markdown("---")
    view = st.radio(
        "📊 Select Dashboard View:",
        ["🎯 Pareto Summary", "📉 Performance Gap", "📝 Unbilled Action List", "✅ Target Achieved", "📋 Full Master Report"],
        horizontal=True
    )

    # Naming & Formatting rules
    col_map = {
        'total_unit': 'Target Units', 'PrimaryQtyinNos': 'Actual Units', 
        'Cases': 'Target Cases', 'PrimaryQtyinCases/Bags': 'Actual Cases',
        'Sales Loss': 'Sales Loss', 'Billcuts': 'DB Billcuts', 'Actual Value': 'Actual Value',
        'target_in_qty(vol)': 'Target volume', 'PrimaryQtyInLtrs/Kgs': 'Actual volume',
        'Volume Loss': 'Volume Loss (MT)', 'Target Volume': 'Target Volume (MT)'
    }

    # Formatters
    fmt_std = {
        'Target Units': '{:.0f}', 'Actual Units': '{:.0f}', 
        'Target Cases': '{:.0f}', 'Actual Cases': '{:.0f}',
        'Sales Loss': lambda x: format_inr(x), 'Actual Value': lambda x: format_inr(x),
        'DB Billcuts': '{:.0f}', 'Target volume': '{:.2f}', 'Actual volume': '{:.2f}',
        'Target Volume (MT)': '{:.2f}', 'Volume Loss (MT)': '{:.2f}'
    }

    if view == "🎯 Pareto Summary":
        st.subheader("High-Loss Distributor Analysis")
        # Ensure db_summary is prepped as per previous step (agg Sales Loss, Target Volume, Volume Loss)
        display_df = db_summary.rename(columns=col_map)
        
        # 1. Main Table (Scrollable)
        st.dataframe(display_df.style.format(fmt_std), use_container_width=True)
        
        # 2. Fixed Totals (No Scroll)
        numeric_to_sum = ['Sales Loss', 'Target Volume (MT)', 'Volume Loss (MT)', 'DB Billcuts']
        display_fixed_totals(display_df, numeric_to_sum, fmt_std)

    elif view == "📉 Performance Gap":
        st.subheader("Gap Analysis: Partially Billed SKUs below Target")
        gap_data = final_df[(final_df['Unit Gap'] > 0) & (final_df['PrimaryQtyinCases/Bags'] > 0)].copy()
        display_cols = ['Region', 'DB Name', 'New SKU', 'Cases', 'PrimaryQtyinCases/Bags', 'target_in_qty(vol)', 'PrimaryQtyInLtrs/Kgs', 'Sales Loss', 'Billcuts']
        
        res_df = gap_data[display_cols].rename(columns=col_map).sort_values('Sales Loss', ascending=False)
        st.dataframe(res_df.style.format(fmt_std), use_container_width=True)
        
        numeric_to_sum = ['Target Cases', 'Actual Cases', 'Target volume', 'Actual volume', 'Sales Loss', 'DB Billcuts']
        display_fixed_totals(res_df, numeric_to_sum, fmt_std)

    elif view == "📝 Unbilled Action List":
        st.subheader("Urgent Action: Targeted SKUs with 0 Billing")
        unbilled = final_df[(final_df['PrimaryQtyinNos'] == 0) & (final_df['total_unit'] > 0)].copy()
        display_cols = ['Region', 'DB Name', 'New SKU', 'total_unit', 'Cases', 'target_in_qty(vol)', 'Sales Loss', 'Billcuts']
        
        res_df = unbilled[display_cols].rename(columns=col_map).sort_values('Sales Loss', ascending=False)
        st.dataframe(res_df.style.format(fmt_std), use_container_width=True)
        
        numeric_to_sum = ['Target Units', 'Target Cases', 'Target volume', 'Sales Loss', 'DB Billcuts']
        display_fixed_totals(res_df, numeric_to_sum, fmt_std)

    elif view == "✅ Target Achieved":
        st.subheader("Achievement Tracking: Performance >= Target")
        billed = final_df[final_df['PrimaryQtyinNos'] >= final_df['total_unit']].copy()
        display_cols = ['Region', 'DB Name', 'New SKU', 'total_unit', 'PrimaryQtyinNos', 'Cases', 'PrimaryQtyinCases/Bags', 'target_in_qty(vol)', 'PrimaryQtyInLtrs/Kgs', 'Actual Value', 'Billcuts']
        
        res_df = billed[display_cols].rename(columns=col_map).sort_values('Actual Value', ascending=False)
        st.dataframe(res_df.style.format(fmt_std), use_container_width=True)
        
        numeric_to_sum = ['Target Units', 'Actual Units', 'Target Cases', 'Actual Cases', 'Target volume', 'Actual volume', 'Actual Value', 'DB Billcuts']
        display_fixed_totals(res_df, numeric_to_sum, fmt_std)

    elif view == "📋 Full Master Report":
        st.subheader("Complete Data Matrix")
        master_display = final_df.rename(columns=col_map)
        st.dataframe(master_display.style.format(fmt_std), use_container_width=True)
        
        # Auto-detect numeric columns for master summary
        num_cols = master_display.select_dtypes(include=[np.number]).columns.tolist()
        display_fixed_totals(master_display, num_cols, fmt_std)

    # --- 12. EXPORT ---
    st.sidebar.markdown("---")
    st.sidebar.download_button("📥 Download Report (CSV)", final_df.to_csv(index=False), f"analysis_{selected_jc}.csv")

if __name__ == "__main__":

    main()
