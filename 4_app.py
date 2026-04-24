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
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
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
    
    chart_data = pd.DataFrame({
        'Date': volume_matrix.index,
        'Posting Volume': volume_matrix[selected_cluster].values
    })
    
    # Check if we have more than one day of data
    if len(chart_data) > 1:
        # Draw a line chart if we have a timeline
        chart = alt.Chart(chart_data).mark_line(point=True, color='#0085ff').encode(
            x='Date:T',
            y='Posting Volume:Q',
            tooltip=['Date:T', 'Posting Volume:Q']
        ).interactive()
    else:
        # Draw a bar chart if we only have one day of data
        chart = alt.Chart(chart_data).mark_bar(color='#0085ff').encode(
            x='Date:T',
            y='Posting Volume:Q',
            tooltip=['Date:T', 'Posting Volume:Q']
        )
    
    st.altair_chart(chart, use_container_width=True)