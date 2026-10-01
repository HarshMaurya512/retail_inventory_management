import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RetailIQ - Intelligent Inventory Management",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD CSS
# =========================================================

css_path = Path(__file__).parent / "style.css"

if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


# =========================================================
# FILE PATHS
# =========================================================

DATA_PATH = (
    "data/raw/"
    "retail_inventory_management_dataset.csv"
)

MODEL_PATH = "models/demand_model.pkl"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    data["Date"] = pd.to_datetime(
    data["Date"],
    format="mixed",
    errors="coerce"
)
    
    data = data.dropna(subset=["Date"])
    return data


df = load_data()


# =========================================================
# LOAD ML MODEL
# =========================================================




# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:28px;
            font-weight:700;
            color:white;
            margin-bottom:5px;
        ">
        📦 RetailIQ
        </div>

        <div style="
            color:#9ca3af;
            font-size:13px;
            margin-bottom:25px;
        ">
        Intelligent Inventory Management
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📦 Products",
            "🛒 Record Sale",
            "📥 Restock Inventory",
            "📊 Analytics",
            "🤖 Demand Prediction",
            "⚠️ Inventory Alerts",
            "➕ Add Product"
        ]
    )

    st.divider()

    st.markdown(
        """
        <div style="color:#6b7280;font-size:12px;">
        RetailIQ v1.0<br>
        ML-powered inventory system
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("Good morning 👋")

    st.write(
        "Here's what's happening with your inventory today."
    )

    st.divider()

    total_units = int(
        df["Units_Sold"].sum()
    )

    total_revenue = float(
        df["Revenue"].sum()
    )

    average_stock = int(
        df["Stock_Level"].mean()
    )

    low_stock_count = int(
        (
            df["Stock_Level"]
            <= df["Reorder_Level"]
        ).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Units Sold",
        f"{total_units:,}"
    )

    c2.metric(
        "Total Revenue",
        f"₹{total_revenue:,.0f}"
    )

    c3.metric(
        "Average Stock",
        f"{average_stock:,}"
    )

    c4.metric(
        "Low Stock Items",
        f"{low_stock_count:,}"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # SALES PERFORMANCE
    # =====================================================

    left, right = st.columns([2, 1])

    with left:

        st.markdown(
            "### 📈 Sales Performance"
        )

        daily_sales = (
            df.groupby(
                "Date",
                as_index=False
            )["Units_Sold"]
            .sum()
        )

        fig = px.area(
            daily_sales,
            x="Date",
            y="Units_Sold"
        )

        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # CATEGORY
    # =====================================================

    with right:

        st.markdown(
            "### 🏷️ Revenue by Category"
        )

        category = (
            df.groupby(
                "Product_Category",
                as_index=False
            )["Revenue"]
            .sum()
        )

        fig2 = px.pie(
            category,
            names="Product_Category",
            values="Revenue",
            hole=0.55
        )

        fig2.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # =====================================================
    # LOW STOCK
    # =====================================================

    st.markdown(
        "### ⚠️ Inventory Requiring Attention"
    )

    low_stock = df[
        df["Stock_Level"]
        <= df["Reorder_Level"]
    ]

    st.dataframe(
        low_stock[
            [
                "Product_Name",
                "City",
                "Stock_Level",
                "Reorder_Level",
                "Units_Sold"
            ]
        ].head(10),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PRODUCTS
# =========================================================

elif page == "📦 Products":

    st.title("📦 Product Management")

    search = st.text_input(
        "🔎 Search Product"
    )

    filtered = df.copy()

    if search:

        filtered = filtered[
            filtered["Product_Name"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.write(
        f"Showing {len(filtered)} records"
    )

    st.dataframe(
        filtered[
            [
                "Product_ID",
                "Product_Name",
                "Product_Category",
                "Supplier",
                "Unit_Price",
                "Stock_Level",
                "Reorder_Level"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# RECORD SALE
# =========================================================

elif page == "🛒 Record Sale":

    st.title("🛒 Record Sale")

    st.write(
        "Record a customer sale and automatically "
        "update the inventory."
    )

    st.divider()

    products = sorted(
        df["Product_Name"]
        .dropna()
        .unique()
        .tolist()
    )

    with st.form("record_sale_form"):

        selected_product = st.selectbox(
            "Select Product",
            products
        )

        quantity_sold = st.number_input(
            "Quantity Sold",
            min_value=1,
            value=1,
            step=1
        )

        sale_date = st.date_input(
            "Sale Date",
            value=pd.Timestamp.today()
        )

        submitted = st.form_submit_button(
            "🛒 Record Sale",
            use_container_width=True
        )

    if submitted:

        # Get all records for selected product
        product_records = df[
            df["Product_Name"]
            == selected_product
        ].copy()

        if product_records.empty:

            st.error(
                "Product not found."
            )

        else:

            # Latest inventory record
            product_records = (
                product_records
                .sort_values("Date")
            )

            latest = product_records.iloc[-1]

            current_stock = int(
                latest["Stock_Level"]
            )

            unit_price = float(
                latest["Unit_Price"]
            )

            reorder_level = int(
                latest["Reorder_Level"]
            )

            # Check inventory
            if quantity_sold > current_stock:

                st.error(
                    f"❌ Insufficient stock. "
                    f"Available stock: "
                    f"{current_stock} units."
                )

            else:

                new_stock = (
                    current_stock
                    - quantity_sold
                )

                revenue = (
                    quantity_sold
                    * unit_price
                )

                low_stock = (
                    new_stock
                    <= reorder_level
                )

                # Create transaction record
                new_sale = latest.copy()

                new_sale["Date"] = pd.Timestamp(
                    sale_date
                )

                new_sale["Units_Sold"] = (
                    quantity_sold
                )

                new_sale["Revenue"] = (
                    revenue
                )

                new_sale["Stock_Level"] = (
                    new_stock
                )

                new_sale["Low_Stock"] = (
                    low_stock
                )

                # Add transaction
                updated_df = pd.concat(
                    [
                        df,
                        pd.DataFrame([new_sale])
                    ],
                    ignore_index=True
                )

                # Save
                updated_df.to_csv(
                    DATA_PATH,
                    index=False
                )

                # Clear cached data
                st.cache_data.clear()

                st.success(
                    "✅ Sale recorded successfully!"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Quantity Sold",
                    f"{quantity_sold} units"
                )

                c2.metric(
                    "Remaining Stock",
                    f"{new_stock} units"
                )

                c3.metric(
                    "Sale Revenue",
                    f"₹{revenue:,.2f}"
                )

                if low_stock:

                    st.warning(
                        "⚠️ Stock has reached "
                        "the reorder level."
                    )

                else:

                    st.success(
                        "✅ Inventory level is healthy."
                    )


# =========================================================
# RESTOCK INVENTORY
# =========================================================

elif page == "📥 Restock Inventory":

    st.title("📥 Restock Inventory")

    st.write(
        "Add incoming stock to your inventory."
    )

    st.divider()

    products = sorted(
        df["Product_Name"]
        .dropna()
        .unique()
        .tolist()
    )

    with st.form("restock_form"):

        selected_product = st.selectbox(
            "Select Product",
            products,
            key="restock_product"
        )

        quantity_added = st.number_input(
            "Quantity to Add",
            min_value=1,
            value=10,
            step=1
        )

        restock_date = st.date_input(
            "Restock Date",
            value=pd.Timestamp.today()
        )

        submitted = st.form_submit_button(
            "📥 Add Stock",
            use_container_width=True
        )

    if submitted:

        product_records = df[
            df["Product_Name"]
            == selected_product
        ].copy()

        if product_records.empty:

            st.error(
                "Product not found."
            )

        else:

            product_records = (
                product_records
                .sort_values("Date")
            )

            latest = product_records.iloc[-1]

            current_stock = int(
                latest["Stock_Level"]
            )

            reorder_level = int(
                latest["Reorder_Level"]
            )

            new_stock = (
                current_stock
                + quantity_added
            )

            low_stock = (
                new_stock
                <= reorder_level
            )

            # Create restock record
            new_restock = latest.copy()

            new_restock["Date"] = pd.Timestamp(
                restock_date
            )

            # No sale during restocking
            new_restock["Units_Sold"] = 0

            new_restock["Revenue"] = 0

            new_restock["Stock_Level"] = (
                new_stock
            )

            new_restock["Low_Stock"] = (
                low_stock
            )

            # Add restock record
            updated_df = pd.concat(
                [
                    df,
                    pd.DataFrame([new_restock])
                ],
                ignore_index=True
            )

            # Save
            updated_df.to_csv(
                DATA_PATH,
                index=False
            )

            st.cache_data.clear()

            st.success(
                f"✅ {quantity_added} units "
                f"added successfully!"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Stock Added",
                f"{quantity_added} units"
            )

            c2.metric(
                "Previous Stock",
                f"{current_stock} units"
            )

            c3.metric(
                "Updated Stock",
                f"{new_stock} units"
            )

            if new_stock > reorder_level:

                st.success(
                    "✅ Inventory is now above "
                    "the reorder level."
                )

            else:

                st.warning(
                    "⚠️ Inventory is still below "
                    "the reorder level."
                )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.title("📊 Business Analytics")

    # =====================================================
    # TOP PRODUCTS
    # =====================================================

    top_products = (
        df.groupby(
            "Product_Name"
        )["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_products,
        x="Revenue",
        y="Product_Name",
        orientation="h",
        title="Top 10 Products by Revenue"
    )

    fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # CITY
    # =====================================================

    city_sales = (
        df.groupby(
            "City",
            as_index=False
        )["Revenue"]
        .sum()
        .sort_values(
            "Revenue",
            ascending=False
        )
    )

    fig2 = px.bar(
        city_sales,
        x="City",
        y="Revenue",
        title="Revenue by City"
    )

    fig2.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# =========================================================
# DEMAND PREDICTION
# =========================================================

elif page == "🤖 Demand Prediction":

    st.title("🤖 AI Demand Prediction")

    st.markdown(
        "Predict future product demand using historical sales "
        "and inventory information."
    )

    # ---------------------------------------------------------
    # PRODUCT SELECTION
    # ---------------------------------------------------------

    product_list = sorted(
        df["Product_Name"]
        .dropna()
        .unique()
    )

    if len(product_list) == 0:

        st.warning(
            "No products are available for demand prediction."
        )

    else:

        product = st.selectbox(
            "Select Product",
            product_list
        )

        # ---------------------------------------------------------
        # GET PRODUCT HISTORY
        # ---------------------------------------------------------

        history = df[
            df["Product_Name"] == product
        ].copy()

        history = history.sort_values(
            "Date"
        )

        # Make sure Units_Sold is numeric
        history["Units_Sold"] = pd.to_numeric(
            history["Units_Sold"],
            errors="coerce"
        )

        history = history.dropna(
            subset=["Units_Sold"]
        )

        # ---------------------------------------------------------
        # CREATE DEMAND FEATURES
        # ---------------------------------------------------------

        history["Previous_Day_Sales"] = (
            history["Units_Sold"].shift(1)
        )

        history["Rolling_7_Day_Sales"] = (
            history["Units_Sold"]
            .shift(1)
            .rolling(7)
            .mean()
        )

        history["Rolling_30_Day_Sales"] = (
            history["Units_Sold"]
            .shift(1)
            .rolling(30)
            .mean()
        )

        # ---------------------------------------------------------
        # CHECK DATA
        # ---------------------------------------------------------

        valid = history.dropna(
            subset=[
                "Previous_Day_Sales",
                "Rolling_7_Day_Sales"
            ]
        )

        if len(valid) == 0:

            st.warning(
                "Not enough historical sales data "
                "for this product."
            )

        else:

            # -----------------------------------------------------
            # LATEST PRODUCT DATA
            # -----------------------------------------------------

            latest = valid.iloc[-1]

            previous_day_sales = float(
                latest["Previous_Day_Sales"]
            )

            rolling_7_day_sales = float(
                latest["Rolling_7_Day_Sales"]
            )

            # If 30-day average is unavailable,
            # use 7-day average
            if pd.isna(
                latest["Rolling_30_Day_Sales"]
            ):

                rolling_30_day_sales = (
                    rolling_7_day_sales
                )

            else:

                rolling_30_day_sales = float(
                    latest["Rolling_30_Day_Sales"]
                )

            # -----------------------------------------------------
            # DEMAND ESTIMATION
            # -----------------------------------------------------

            # Weighted demand estimate
            #
            # 50% recent 7-day demand
            # 30% 30-day demand
            # 20% previous day demand

            predicted_demand = (
                0.50 * rolling_7_day_sales
                + 0.30 * rolling_30_day_sales
                + 0.20 * previous_day_sales
            )

            predicted_demand = max(
                0,
                round(predicted_demand)
            )

            # -----------------------------------------------------
            # CURRENT INVENTORY
            # -----------------------------------------------------

            current_stock = int(
                pd.to_numeric(
                    latest["Stock_Level"],
                    errors="coerce"
                )
            )

            reorder_level = int(
                pd.to_numeric(
                    latest["Reorder_Level"],
                    errors="coerce"
                )
            )

            # -----------------------------------------------------
            # SAFETY STOCK
            # -----------------------------------------------------

            safety_stock = max(
                1,
                round(
                    rolling_7_day_sales * 0.20
                )
            )

            # -----------------------------------------------------
            # REQUIRED STOCK
            # -----------------------------------------------------

            required_stock = (
                predicted_demand
                + safety_stock
            )

            # -----------------------------------------------------
            # RECOMMENDED ORDER
            # -----------------------------------------------------

            recommended_order = max(
                required_stock
                - current_stock,
                0
            )

            # -----------------------------------------------------
            # DISPLAY RESULT
            # -----------------------------------------------------

            st.markdown(
                "### 📊 Prediction Result"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Predicted Demand",
                f"{predicted_demand} units"
            )

            c2.metric(
                "Current Stock",
                f"{current_stock} units"
            )

            c3.metric(
                "Safety Stock",
                f"{safety_stock} units"
            )

            c4.metric(
                "Recommended Order",
                f"{recommended_order} units"
            )

            # -----------------------------------------------------
            # INVENTORY STATUS
            # -----------------------------------------------------

            if current_stock <= reorder_level:

                st.error(
                    "🚨 REORDER REQUIRED"
                )

                st.write(
                    f"Current stock ({current_stock}) "
                    f"is at or below the reorder level "
                    f"({reorder_level})."
                )

            elif current_stock < required_stock:

                st.warning(
                    "⚠️ STOCK LEVEL LOW"
                )

                st.write(
                    f"Recommended additional stock: "
                    f"{recommended_order} units."
                )

            else:

                st.success(
                    "✅ STOCK SUFFICIENT"
                )

                st.write(
                    "Current inventory is sufficient "
                    "for the estimated demand."
                )

            # -----------------------------------------------------
            # DEMAND INFORMATION
            # -----------------------------------------------------

            st.markdown(
                "### 📈 Demand Analysis"
            )

            d1, d2, d3 = st.columns(3)

            d1.metric(
                "Previous Day Sales",
                f"{round(previous_day_sales)} units"
            )

            d2.metric(
                "7-Day Average",
                f"{round(rolling_7_day_sales)} units"
            )

            d3.metric(
                "30-Day Average",
                f"{round(rolling_30_day_sales)} units"
            )

            # -----------------------------------------------------
            # SALES TREND CHART
            # -----------------------------------------------------

            chart_data = history[
                [
                    "Date",
                    "Units_Sold"
                ]
            ].tail(30)

            if not chart_data.empty:

                fig = px.line(
                    chart_data,
                    x="Date",
                    y="Units_Sold",
                    markers=True,
                    title=f"Sales Trend - {product}"
                )

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Units Sold"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # -----------------------------------------------------
            # RECOMMENDATION
            # -----------------------------------------------------

            st.markdown(
                "### 💡 Inventory Recommendation"
            )

            if recommended_order > 0:

                st.info(
                    f"For **{product}**, the estimated demand "
                    f"is **{predicted_demand} units**. "
                    f"Considering safety stock, approximately "
                    f"**{recommended_order} additional units** "
                    f"should be considered for replenishment."
                )

            else:

                st.success(
                    f"No additional order is currently "
                    f"required for **{product}**."
                )


# =========================================================
# INVENTORY ALERTS
# =========================================================

elif page == "⚠️ Inventory Alerts":

    st.title(
        "⚠️ Inventory Alerts"
    )

    alerts = df[
        df["Stock_Level"]
        <= df["Reorder_Level"]
    ].copy()

    st.metric(
        "Products Requiring Reorder",
        len(alerts)
    )

    st.dataframe(
        alerts[
            [
                "Product_ID",
                "Product_Name",
                "City",
                "Stock_Level",
                "Reorder_Level",
                "Units_Sold"
            ]
        ].sort_values(
            "Stock_Level"
        ),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ADD PRODUCT
# =========================================================

elif page == "➕ Add Product":

    st.title(
        "➕ Add New Product"
    )

    st.write(
        "Add a new product to your inventory."
    )

    st.divider()

    with st.form("add_product"):

        col1, col2 = st.columns(2)

        with col1:

            product_name = st.text_input(
                "Product Name"
            )

            category = st.selectbox(
                "Product Category",
                sorted(
                    df["Product_Category"]
                    .dropna()
                    .unique()
                )
            )

            supplier = st.text_input(
                "Supplier",
                "Supplier A"
            )

            price = st.number_input(
                "Unit Price",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

            discount = st.number_input(
                "Discount (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0
            )

        with col2:

            store_id = st.selectbox(
                "Store",
                sorted(
                    df["Store_ID"]
                    .dropna()
                    .unique()
                )
            )

            city = st.selectbox(
                "City",
                sorted(
                    df["City"]
                    .dropna()
                    .unique()
                )
            )

            stock = st.number_input(
                "Initial Stock",
                min_value=0,
                value=100
            )

            reorder = st.number_input(
                "Reorder Level",
                min_value=0,
                value=30
            )

            lead_time = st.number_input(
                "Lead Time (Days)",
                min_value=1,
                value=7
            )

        submitted = st.form_submit_button(
            "➕ Add Product",
            use_container_width=True
        )

    if submitted:

        if not product_name.strip():

            st.error(
                "Please enter a product name."
            )

        else:

            # Check duplicate
            duplicate = df[
                df["Product_Name"]
                .str.lower()
                == product_name.strip().lower()
            ]

            if not duplicate.empty:

                st.warning(
                    "This product already exists."
                )

            else:

                product_numbers = (
                    df["Product_ID"]
                    .astype(str)
                    .str.extract(
                        r"(\d+)",
                        expand=False
                    )
                )

                product_numbers = pd.to_numeric(
                    product_numbers,
                    errors="coerce"
                )

                next_number = int(
                    product_numbers.max()
                ) + 1

                new_id = (
                    f"P{next_number:03d}"
                )

                new_product = {

                    "Date":
                        pd.Timestamp.today(),

                    "Store_ID":
                        store_id,

                    "City":
                        city,

                    "Product_ID":
                        new_id,

                    "Product_Name":
                        product_name.strip(),

                    "Product_Category":
                        category,

                    "Supplier":
                        supplier,

                    "Unit_Price":
                        price,

                    "Discount_Percent":
                        discount,

                    "Units_Sold":
                        0,

                    "Revenue":
                        0,

                    "Stock_Level":
                        stock,

                    "Reorder_Level":
                        reorder,

                    "Lead_Time_Days":
                        lead_time,

                    "Low_Stock":
                        stock <= reorder
                }

                updated_df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [new_product]
                        )
                    ],
                    ignore_index=True
                )

                updated_df.to_csv(
                    DATA_PATH,
                    index=False
                )

                st.cache_data.clear()

                st.success(
                    f"✅ {product_name} "
                    f"added successfully!"
                )

                st.info(
                    f"Product ID: {new_id}"
                )