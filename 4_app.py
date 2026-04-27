import streamlit as st
import pandas as pd
import os
import altair as alt

st.set_page_config(page_title="Live Meme Trends", page_icon="📈", layout="wide")
st.title("📈 Bluesky Live Meme Trend Detector")

@st.cache_data(ttl=300) # Cache clears every 5 mins to load fresh data
def load_data():
    if not os.path.exists('clustered_dataset.pkl'):
        return None, None, None, None
        
    df = pd.read_pickle('clustered_dataset.pkl')
    
    # Use ACTUAL timestamps from Bluesky
    # Use ACTUAL timestamps from Bluesky
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
    df['date'] = df['timestamp'].dt.date
    
    # Calculate Matrices
    daily_counts = df.groupby(['date', 'cluster_id']).size().reset_index(name='volume')
    volume_matrix = daily_counts.pivot(index='date', columns='cluster_id', values='volume').fillna(0)
    velocity_matrix = volume_matrix.diff().fillna(0)
    acceleration_matrix = velocity_matrix.diff().fillna(0)
    
    return df, volume_matrix, velocity_matrix, acceleration_matrix

df, volume_matrix, velocity_matrix, acceleration_matrix = load_data()

if df is None:
    st.error("No data found! Run steps 1, 2, and 3 first.")
    st.stop()

# ==========================================
# TRENDING LEADERBOARD
# ==========================================
st.header("🔥 Hot Right Now")

latest_date = volume_matrix.index[-1]
latest_volume = volume_matrix.loc[latest_date]
latest_velocity = velocity_matrix.loc[latest_date]

trending = []
for cid in volume_matrix.columns:
    if latest_volume[cid] > 0:
        trending.append((cid, latest_volume[cid], latest_velocity[cid]))

# Sort by highest velocity (fastest growing)
trending = sorted(trending, key=lambda x: x[2], reverse=True)[:3]

cols = st.columns(3)
for i, col in enumerate(cols):
    if i < len(trending):
        cid, vol, vel = trending[i]
        with col:
            st.metric(label=f"Template {cid}", value=f"Vol: {int(vol)}", delta=f"{int(vel)} Velocity")

st.divider()

# ==========================================
# TEMPLATE EXPLORER
# ==========================================
st.header("📊 Explore Templates")
col1, col2 = st.columns([1, 2])

with col1:
    selected_cluster = st.selectbox("Select a Template:", volume_matrix.columns)
    
    preview_folder = os.path.join('clustered_preview', f'Template_{selected_cluster}')
    if os.path.exists(preview_folder):
        images = [f for f in os.listdir(preview_folder) if f.endswith(('.png', '.jpg'))]
        if images:
            st.image(os.path.join(preview_folder, images[0]), use_container_width=True)

with col2:
    st.subheader(f"Lifespan of Template {selected_cluster}")
    
    # Ensure dates are proper datetime objects for accurate chart spacing
    chart_data = pd.DataFrame({
        'Date': pd.to_datetime(volume_matrix.index),
        'Posting Volume': volume_matrix[selected_cluster].values
    })
    
    # Define a clean X-axis format (e.g., "Apr 25") to be reused
    clean_x_axis = alt.X('Date:T', title='Date', axis=alt.Axis(format='%b %d', labelAngle=-45, tickCount='day'))
    
    # Check if we have more than one day of data
    if len(chart_data) > 1:
        
        # --- PREDICTION MATH ---
        last_date = chart_data['Date'].iloc[-1]
        last_vol = chart_data['Posting Volume'].iloc[-1]
        
        current_vel = velocity_matrix[selected_cluster].iloc[-1]
        current_acc = acceleration_matrix[selected_cluster].iloc[-1]
        
        # Project Day 1
        pred_date_1 = last_date + pd.Timedelta(days=1)
        pred_vel_1 = current_vel + current_acc
        pred_vol_1 = max(0, last_vol + pred_vel_1) # Cannot go below 0 volume
        
        # Project Day 2
        pred_date_2 = pred_date_1 + pd.Timedelta(days=1)
        pred_vel_2 = pred_vel_1 + current_acc
        pred_vol_2 = max(0, pred_vol_1 + pred_vel_2)

        # Create connection points for the chart layers
        pred1_data = pd.DataFrame({'Date': [last_date, pred_date_1], 'Posting Volume': [last_vol, pred_vol_1]})
        pred2_data = pd.DataFrame({'Date': [pred_date_1, pred_date_2], 'Posting Volume': [pred_vol_1, pred_vol_2]})

        # --- DRAW MULTI-LAYER CHART ---
        # 1. Actual Historical Line (Solid)
        base_chart = alt.Chart(chart_data).mark_line(point=True, color='#0085ff', strokeWidth=3).encode(
            x=clean_x_axis,
            y=alt.Y('Posting Volume:Q', title='Posting Volume'),
            tooltip=[alt.Tooltip('Date:T', title='Actual Date', format='%b %d, %Y'), alt.Tooltip('Posting Volume:Q', title='Volume')]
        )
        
        # 2. Prediction Day 1 (Dotted)
        pred1_chart = alt.Chart(pred1_data).mark_line(point=True, color='#0085ff', strokeWidth=3, strokeDash=[5, 5]).encode(
            x=clean_x_axis,
            y=alt.Y('Posting Volume:Q', title='Posting Volume'),
            tooltip=[alt.Tooltip('Date:T', title='Predicted (Day 1)', format='%b %d, %Y'), alt.Tooltip('Posting Volume:Q', title='Predicted Volume')]
        )
        
        # 3. Prediction Day 2 (Lighter, Thinner, Dotted)
        pred2_chart = alt.Chart(pred2_data).mark_line(point=True, color='#80c2ff', strokeWidth=2, strokeDash=[2, 4]).encode(
            x=clean_x_axis,
            y=alt.Y('Posting Volume:Q', title='Posting Volume'),
            tooltip=[alt.Tooltip('Date:T', title='Predicted (Day 2)', format='%b %d, %Y'), alt.Tooltip('Posting Volume:Q', title='Predicted Volume')]
        )
        
        # Combine all three layers and enable horizontal-only zooming
        chart = (base_chart + pred1_chart + pred2_chart).interactive(bind_y=False)
        
    else:
        # Draw a bar chart if we only have one day of data
        chart = alt.Chart(chart_data).mark_bar(color='#0085ff').encode(
            x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%b %d', labelAngle=0)),
            y=alt.Y('Posting Volume:Q', title='Posting Volume'),
            tooltip=[alt.Tooltip('Date:T', title='Actual Date', format='%b %d, %Y'), alt.Tooltip('Posting Volume:Q', title='Volume')]
        ).interactive(bind_y=False)
    
    st.altair_chart(chart, use_container_width=True)