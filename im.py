import streamlit as st
import pandas as pd
import numpy as np
import ftplib
import io
import os
import re

# --- 0. PANDAS CONFIGURATION ---
# Fix for the "styler.render.max_elements" error for large master reports
pd.set_option("styler.render.max_elements", 2000000)

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
    
    # Filter numeric_cols to only those actually present in the dataframe
    existing_cols = [c for c in numeric_cols if c in df.columns]
    if not existing_cols:
        return

    # Calculate totals
    totals = df[existing_cols].sum()
    
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
@st.cache_data(ttl=100)
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
    

    # --- 4. SIDEBAR CONFIGURATION ---
    #st.sidebar.header("🔧 Settings")
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
    df_sales['PrimaryLineTotalBeforeTax'] = pd.to_numeric(df_sales['PrimaryLineTotalBeforeTax'], errors='coerce').fillna(0)
    
    df_price['Taxable Value'] = pd.to_numeric(df_price['Taxable Value'], errors='coerce').fillna(0)
    df_master['Balance'] = pd.to_numeric(df_master['Balance'], errors='coerce').fillna(0)

    # --- 6. GLOBAL FILTERS (JC & CUSTOMER CLASS) ---
    st.sidebar.markdown("---")
    st.sidebar.header("Global Filters")
    
    # 6a. JC Filter
    unique_jcs = sorted(df_aop['jc'].dropna().unique())
    selected_jc = st.sidebar.selectbox("Select JC Period", options=unique_jcs)
    
    # 6b. Customer Class Filter
    # Extract unique classes from sales data
    unique_classes = sorted(df_sales['CustomerClass'].dropna().unique())
    # Default to 'ARJUNAN' if it exists in the list, otherwise use the first class
    default_classes = ['ARJUNAN'] if 'ARJUNAN' in unique_classes else [unique_classes[0]] if unique_classes else []
    selected_classes = st.sidebar.multiselect("Select Customer Class", options=unique_classes, default=default_classes)

    # Filter AOP based on JC
    df_aop_jc = df_aop[df_aop['jc'] == selected_jc].copy()
    
    # Filter Sales based on JC and Customer Class
    clean_jc_num = ''.join(filter(str.isdigit, str(selected_jc))).zfill(2)
    df_sales_jc = df_sales[
        (df_sales['JCPeriodNum'].str.contains(clean_jc_num, na=False)) &
        (df_sales['CustomerClass'].isin(selected_classes))
    ].copy()

     # --- 3. DYNAMIC TITLE ---
    # Create a string for the selected classes (e.g., "ARJUNAN, CLASS2")
    class_str = ", ".join(selected_classes) if selected_classes else "None Selected"
    
    # Display the updated title
    st.title(f"🚀 Distributor 80/20 Gap & Action Dashboard")
    st.markdown(f"#### 📅 JC: **{selected_jc}** | 👤 Class: **{class_str}**")
    st.markdown("---")

    # --- 7. BILLCUTS LOGIC ---
    if 'InvNum' in df_sales_jc.columns:
        db_billcuts = df_sales_jc.groupby('BP Code')['InvNum'].nunique().reset_index()
        db_billcuts.columns = ['BP Code', 'Billcuts']
    else:
        db_billcuts = pd.DataFrame(columns=['BP Code', 'Billcuts'])

    # --- 8. MERGING & CALCULATIONS (OUTER JOIN) ---
    # Aggregating sales including the requested 'PrimaryLineTotalBeforeTax'
    sales_sku = df_sales_jc.groupby(['BP Code', 'ProductGroup'], as_index=False).agg({
        'PrimaryQtyinNos': 'sum',
        'PrimaryQtyinCases/Bags': 'sum',
        'PrimaryQtyInLtrs/Kgs': 'sum',
        'PrimaryLineTotalBeforeTax': 'sum'
    })
    
    # Outer join to capture AOP Targets and Non-AOP sales
    merged = pd.merge(
        df_aop_jc, 
        sales_sku, 
        left_on=['DB Code', 'New SKU'], 
        right_on=['BP Code', 'ProductGroup'], 
        how='outer'
    ).fillna(0)

    # Fix IDs for rows that exist only in Sales
    merged['DB Code'] = np.where(merged['DB Code'] == 0, merged['BP Code'], merged['DB Code'])
    merged['New SKU'] = np.where(merged['New SKU'] == 0, merged['ProductGroup'], merged['New SKU'])

    # METADATA ENRICHMENT: Fetch missing Names/Regions for non-AOP rows from Master
    # Headers: CardCode, CardName, U_Zone (Region), U_ASM (ASE)
    db_meta = df_master[['CardCode', 'CardName', 'U_DSM', 'U_ASM', 'State']].drop_duplicates('CardCode')
    merged = pd.merge(merged, db_meta, left_on='DB Code', right_on='CardCode', how='left', suffixes=('', '_m'))
    
    # Patch Names/Regions for Non-AOP rows
    merged['DB Name'] = merged['DB Name'].replace(0, np.nan).fillna(merged['CardName'])
    merged['Region'] = merged['Region'].replace(0, np.nan).fillna(merged['U_DSM'])
    merged['ASE'] = merged['ASE'].replace(0, np.nan).fillna(merged['U_ASM'])
    merged['State'] = merged['State'].replace(0, np.nan).fillna(merged['State_m'])

    # Merge Billcuts
    merged = pd.merge(merged, db_billcuts, left_on='DB Code', right_on='BP Code', how='left').fillna({'Billcuts': 0})

    # --- PRICE LIST LOGIC ---
    curr_jc_val = extract_jc_number(selected_jc)
    df_price['jc_from_int'] = df_price['U_JCFrom'].apply(extract_jc_number)
    df_price['jc_to_int'] = df_price['U_JCTo'].apply(extract_jc_number)

    price_map = df_price[
        (curr_jc_val >= df_price['jc_from_int']) & 
        (curr_jc_val <= df_price['jc_to_int'])
    ][['State', 'Name', 'Taxable Value']].drop_duplicates(['State', 'Name'])

    merged = pd.merge(
        merged, 
        price_map, 
        left_on=['State', 'New SKU'], 
        right_on=['State', 'Name'], 
        how='left'
    ).fillna({'Taxable Value': 0})
    
    # Math Calculations
    merged['Unit Gap'] = np.where(merged['PrimaryQtyinNos'] < merged['total_unit'], merged['total_unit'] - merged['PrimaryQtyinNos'], 0)
    merged['volume gap'] = np.where(merged['PrimaryQtyInLtrs/Kgs'] < merged['target_in_qty(vol)'], merged['target_in_qty(vol)'] - merged['PrimaryQtyInLtrs/Kgs'], 0)
    
    merged['Sales Loss'] = merged['Unit Gap'] * merged['Taxable Value']
    # Achievement value is taken from the invoice report column
    merged['Actual Value'] = merged['PrimaryLineTotalBeforeTax']

    # Enrichment (Cheques & Payment Status)
    cheque_set = set(df_cheque['U_BPCode'].astype(str).unique())
    merged['Cheque Status'] = merged['DB Code'].apply(lambda x: "Available" if x in cheque_set else "Not Available")
    
    merged = pd.merge(merged, df_master[['CardCode', 'Balance']], left_on='DB Code', right_on='CardCode', how='left', suffixes=('', '_bal'))
    merged['Balance Raw'] = merged['Balance'].fillna(0).astype(float)
    merged['Payment Status'] = np.where(merged['Balance Raw'] > 0, "Overdue", np.where(merged['Balance Raw'] < 0, "Advance", "No Due"))

    # --- 9. DISTRIBUTOR RANKING & SIDEBAR FILTER ---
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Target Ranking")
    
    db_summary = merged.groupby(['Region', 'ASE', 'DB Code', 'DB Name', 'Billcuts', 'Cheque Status', 'Payment Status']).agg({
        'Sales Loss': 'sum',
        'target_in_qty(vol)': 'sum',
        'volume gap': 'sum',
        'Actual Value': 'sum'
    }).reset_index()

    db_summary = db_summary.rename(columns={'volume gap': 'Volume Loss', 'target_in_qty(vol)': 'Target Volume'})
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

    all_regs = sorted([str(x) for x in db_summary['Region'].unique() if pd.notna(x) and x != 0])
    sel_reg = st.sidebar.multiselect("Filter by DSM", all_regs, default=[])
    if sel_reg:
        db_summary = db_summary[db_summary['Region'].isin(sel_reg)]

    all_ASM = sorted([str(x) for x in db_summary['ASE'].unique() if pd.notna(x) and x != 0])
    sel_ASE = st.sidebar.multiselect("Filter by ASE", all_ASM, default=[])
    if sel_ASE:
        db_summary = db_summary[db_summary['ASE'].isin(sel_ASE)]


    
    allowed_dbs = db_summary['DB Code'].tolist()
    final_df = merged[merged['DB Code'].isin(allowed_dbs)].copy()

# --- 10. KPI SECTION ---
    st.markdown("### 📈 Key Performance Indicators")
    
    # Pre-calculations for Logic
    # 1. AOP Sales: Sales value where a target was actually set (total_unit > 0)
    aop_sales_df = final_df[(final_df['total_unit'] > 0) & (final_df['PrimaryQtyinNos'] > 0)]
    aop_sales_val = aop_sales_df['Actual Value'].sum()
    aop_sales_mt = aop_sales_df['PrimaryQtyInLtrs/Kgs'].sum()

    # 2. Non-AOP Sales: Sales value where no target was set (total_unit == 0)
    non_aop_df = final_df[(final_df['total_unit'] == 0) & (final_df['PrimaryQtyinNos'] > 0)]
    non_aop_val = non_aop_df['Actual Value'].sum()
    non_aop_mt = non_aop_df['PrimaryQtyInLtrs/Kgs'].sum()

    # 3. Gaps & Losses
    total_sales_loss_val = final_df['Sales Loss'].sum()
    total_sales_loss_mt = final_df['volume gap'].sum()
    
    lt_target_df = final_df[(final_df['Unit Gap'] > 0) & (final_df['PrimaryQtyinNos'] > 0)]
    lt_val = lt_target_df['Sales Loss'].sum()
    lt_mt = lt_target_df['volume gap'].sum()
    
    unbilled_df = final_df[(final_df['PrimaryQtyinNos'] == 0) & (final_df['total_unit'] > 0)]
    unbilled_val = unbilled_df['Sales Loss'].sum()
    unbilled_mt = unbilled_df['volume gap'].sum()
    
    gt_target_df = final_df[(final_df['PrimaryQtyinNos'] >= final_df['total_unit']) & (final_df['total_unit'] > 0)]
    gt_val = gt_target_df['Actual Value'].sum()
    gt_mt = gt_target_df['PrimaryQtyInLtrs/Kgs'].sum()

    # 4. Global Totals
    total_target_val = (final_df['total_unit'] * final_df['Taxable Value']).sum()
    total_target_mt = final_df['target_in_qty(vol)'].sum()
    total_achieved_val = final_df['Actual Value'].sum()
    total_achieved_mt = final_df['PrimaryQtyInLtrs/Kgs'].sum()

    # --- ROW 1: SALES PERFORMANCE ---
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        st.metric("Total Target", format_inr(total_target_val))
        st.caption(f"{total_target_mt:.1f} MT")
    with r1c2:
        st.metric("Actual Sales (Inv)", format_inr(total_achieved_val))
        st.caption(f"{total_achieved_mt:.1f} MT")
    with r1c3:
        st.metric("AOP Sales", format_inr(aop_sales_val))
        st.caption(f"{aop_sales_mt:.1f} MT")
    with r1c4:
        st.metric("Non-AOP Sales", format_inr(non_aop_val))
        st.caption(f"{non_aop_mt:.1f} MT")

    st.write("") # Spacer

    # --- ROW 2: LOSS & GAP ANALYSIS ---
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        st.metric("Total Sales Loss", format_inr(total_sales_loss_val), delta_color="inverse")
        st.caption(f"{int(total_sales_loss_mt)} MT")
    with r2c2:
        st.metric("Performance Gap", format_inr(lt_val))
        st.caption(f"{int(lt_mt)} MT")
    with r2c3:
        st.metric("Unbilled Action Loss", format_inr(unbilled_val))
        st.caption(f"{int(unbilled_mt)} MT")    
    with r2c4:
        st.metric("Greater than Target Sales", format_inr(gt_val))
        st.caption(f"{int(gt_mt)} MT")
    # --- 11. UI RENDERING: RADIO BUTTON VIEWS ---
    st.markdown("---")
    view = st.radio(
        "📊 Select Dashboard View:",
        ["🎯 Pareto Summary", '👥 ASE Wise Summary','👥 SO Wise Summary',"📉 Performance Gap", "📝 Unbilled Action List", "✅ Target Achieved", "🎁 Non-AOP Sales", "📋 Full Master Report"],
        horizontal=True
    )

    col_map = {
        'total_unit': 'Target Units', 'PrimaryQtyinNos': 'Actual Units', 
        'Cases': 'Target Cases', 'PrimaryQtyinCases/Bags': 'Actual Cases',
        'Sales Loss': 'Sales Loss', 'Billcuts': 'DB Billcuts', 
        'Actual Value': 'Actual Value (Inv)', 'PrimaryLineTotalBeforeTax': 'Inv Value Before Tax',
        'target_in_qty(vol)': 'Target volume', 'PrimaryQtyInLtrs/Kgs': 'Actual volume',
        'Volume Loss': 'Volume Loss (MT)', 'Target Volume': 'Target Volume (MT)'
    }

    fmt_std = {
        'Target Units': '{:.0f}', 'Actual Units': '{:.0f}', 
        'Target Cases': '{:.0f}', 'Actual Cases': '{:.0f}',
        'Sales Loss': lambda x: format_inr(x), 'Actual Value (Inv)': lambda x: format_inr(x),
        'Inv Value Before Tax': lambda x: format_inr(x),
        'DB Billcuts': '{:.0f}', 'Target volume': '{:.2f}', 'Actual volume': '{:.2f}',
        'Target Volume (MT)': '{:.2f}', 'Volume Loss (MT)': '{:.2f}'
    }

    if view == "🎯 Pareto Summary":
        st.subheader("High-Loss Distributor Analysis")
        display_df = db_summary.rename(columns=col_map)
        st.dataframe(display_df.style.format(fmt_std), use_container_width=True)
        numeric_to_sum = ['Sales Loss', 'Target Volume (MT)', 'Volume Loss (MT)', 'DB Billcuts', 'Actual Value (Inv)']
        display_fixed_totals(display_df, numeric_to_sum, fmt_std)

    elif view == "👥 ASE Wise Summary":
        st.subheader("ASE Performance Breakdown")
        ase_summary = final_df.groupby(['Region', 'ASE']).agg({
            'target_in_qty(vol)': 'sum',
            'PrimaryQtyInLtrs/Kgs': 'sum',
            'Sales Loss': 'sum',
            'Actual Value': 'sum',
            'DB Code': 'nunique'
        }).reset_index().rename(columns={'DB Code': 'Distributors'})
        
        
        # Calculate Achievement %
        ase_summary['Ach % (Vol)'] = (ase_summary['PrimaryQtyInLtrs/Kgs'] / ase_summary['target_in_qty(vol)'] * 100).fillna(0)
        
        ase_display = ase_summary.rename(columns=col_map)
        fmt_ase = {**fmt_std, 'Ach % (Vol)': '{:.1f}%', 'Distributors': '{:.0f}'}
        st.dataframe(ase_display.sort_values('Sales Loss', ascending=False).style.format(fmt_ase), use_container_width=True)
        display_fixed_totals(ase_display, ['Target Units', 'Actual Units', 'Target vol', 'Actual vol', 'Sales Loss', 'Actual Value (Inv)'], fmt_ase)

    elif view == "👥 SO Wise Summary":
        st.subheader("SO Performance Breakdown")
        so_summary = final_df.groupby(['Region', 'SO']).agg({
            'target_in_qty(vol)': 'sum',
            'PrimaryQtyInLtrs/Kgs': 'sum',
            ''
            'Sales Loss': 'sum',
            'Actual Value': 'sum',
            'DB Code': 'nunique'
        }).reset_index().rename(columns={'DB Code': 'Distributors'})
        
        
        # Calculate Achievement %
        so_summary['Ach % (Vol)'] = (so_summary['PrimaryQtyInLtrs/Kgs'] / so_summary['target_in_qty(vol)'] * 100).fillna(0)
        
        so_display = so_summary.rename(columns=col_map)
        fmt_so = {**fmt_std, 'Ach % (Vol)': '{:.1f}%', 'Distributors': '{:.0f}'}
        st.dataframe(so_display.sort_values('Sales Loss', ascending=False).style.format(fmt_so), use_container_width=True)
        display_fixed_totals(so_display, ['Target Units', 'Actual Units', 'Target vol', 'Actual vol', 'Sales Loss', 'Actual Value (Inv)'], fmt_so)

    elif view == "📉 Performance Gap":
        st.subheader("Gap Analysis: Partially Billed SKUs below Target")
        gap_data = final_df[(final_df['Unit Gap'] > 0) & (final_df['PrimaryQtyinNos'] > 0)].copy()
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
        billed = final_df[(final_df['PrimaryQtyinNos'] >= final_df['total_unit']) & (final_df['total_unit'] > 0)].copy()
        display_cols = ['Region', 'DB Name', 'New SKU', 'total_unit', 'PrimaryQtyinNos', 'target_in_qty(vol)', 'PrimaryQtyInLtrs/Kgs', 'Actual Value', 'Billcuts']
        res_df = billed[display_cols].rename(columns=col_map).sort_values('Actual Value (Inv)', ascending=False)
        st.dataframe(res_df.style.format(fmt_std), use_container_width=True)
        numeric_to_sum = ['Target Units', 'Actual Units', 'Target volume', 'Actual volume', 'Actual Value (Inv)', 'DB Billcuts']
        display_fixed_totals(res_df, numeric_to_sum, fmt_std)

    elif view == "🎁 Non-AOP Sales":
        st.subheader("Incremental Sales: SKUs Billed without AOP Targets")
        display_cols = ['Region', 'DB Name', 'New SKU', 'PrimaryQtyinNos', 'PrimaryQtyinCases/Bags', 'PrimaryQtyInLtrs/Kgs', 'PrimaryLineTotalBeforeTax', 'Billcuts']
        res_df = non_aop_df[display_cols].rename(columns=col_map).sort_values('Inv Value Before Tax', ascending=False)
        st.dataframe(res_df.style.format(fmt_std), use_container_width=True)
        numeric_to_sum = ['Actual Units', 'Actual Cases', 'Actual volume', 'Inv Value Before Tax', 'DB Billcuts']
        display_fixed_totals(res_df, numeric_to_sum, fmt_std)

    

    elif view == "📋 Full Master Report":
        st.subheader("Complete Data Matrix")
        master_display = final_df.rename(columns=col_map)
        st.dataframe(master_display.style.format(fmt_std), use_container_width=True)
        num_cols = master_display.select_dtypes(include=[np.number]).columns.tolist()
        display_fixed_totals(master_display, num_cols, fmt_std)

    # --- 12. EXPORT ---
    st.sidebar.markdown("---")
    st.sidebar.download_button("📥 Download Report (CSV)", final_df.to_csv(index=False), f"analysis_{selected_jc}.csv")

if __name__ == "__main__":
    main()