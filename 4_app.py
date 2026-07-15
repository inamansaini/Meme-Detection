import streamlit as st
import pandas as pd
import os
import altair as alt

st.set_page_config(page_title="Live Meme Trends", page_icon="📈", layout="wide")
st.title("MEME TREND PREDICTION SYSTEM")

@st.cache_data
def get_data(path):
    if not os.path.exists(path):
        return None, None, None, None
        
    data = pd.read_pickle(path)
    
    # fix timezones
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='ISO8601', utc=True)
    data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Kolkata')
    data['date'] = data['timestamp'].dt.date
    
    # math stuff
    counts = data.groupby(['date', 'cluster_id']).size().reset_index(name='volume')
    vol = counts.pivot(index='date', columns='cluster_id', values='volume').fillna(0)
    vel = vol.diff().fillna(0)
    accel = vel.diff().fillna(0)
    
    return data, vol, vel, accel

raw_df, vol_df, vel_df, accel_df = get_data('clustered_dataset.pkl')

if raw_df is None:
    st.error("missing data. run the other scripts first.")
    st.stop()

st.header("🔥 Hot Right Now")

last_d = vol_df.index[-1]
last_v = vol_df.loc[last_d]
last_vel = vel_df.loc[last_d]

hot_list = []
for c in vol_df.columns:
    if last_v[c] > 0:
        hot_list.append((c, last_v[c], last_vel[c]))

# sort by speed
hot_list = sorted(hot_list, key=lambda x: x[2], reverse=True)[:3]

cols = st.columns(3)
for idx, c in enumerate(cols):
    if idx < len(hot_list):
        c_id, v, speed = hot_list[idx]
        with c:
            st.metric(label=f"Template {c_id}", value=f"Vol: {int(v)}", delta=f"{int(speed)} Velocity")

st.divider()

st.header("📊 Explore Templates")
left, right = st.columns([1, 2])

with left:
    choice = st.selectbox("Select a Template:", vol_df.columns)
    
    p_dir = os.path.join('clustered_preview', f'Template_{choice}')
    if os.path.exists(p_dir):
        imgs = [img for img in os.listdir(p_dir) if img.endswith(('.png', '.jpg'))]
        if imgs:
            st.image(os.path.join(p_dir, imgs[0]), use_container_width=True)

with right:
    st.subheader(f"Lifespan of Template {choice}")
    
    plot_df = pd.DataFrame({
        'Date': pd.to_datetime(vol_df.index),
        'Posting Volume': vol_df[choice].values
    })
    
    x_fmt = alt.X('Date:T', title='Date', axis=alt.Axis(format='%b %d', labelAngle=-45, tickCount='day'))
    
    if len(plot_df) > 1:
        # calc predictions
        d_last = plot_df['Date'].iloc[-1]
        v_last = plot_df['Posting Volume'].iloc[-1]
        
        cur_vel = vel_df[choice].iloc[-1]
        cur_acc = accel_df[choice].iloc[-1]
        
        # next day
        d1 = d_last + pd.Timedelta(days=1)
        v1_vel = cur_vel + cur_acc
        v1 = max(0, v_last + v1_vel)
        
        # day after
        d2 = d1 + pd.Timedelta(days=1)
        v2_vel = v1_vel + cur_acc
        v2 = max(0, v1 + v2_vel)

        p1_df = pd.DataFrame({'Date': [d_last, d1], 'Posting Volume': [v_last, v1]})
        p2_df = pd.DataFrame({'Date': [d1, d2], 'Posting Volume': [v1, v2]})

        # base line
        base = alt.Chart(plot_df).mark_line(point=True, color='#0085ff', strokeWidth=3).encode(
            x=x_fmt,
            y=alt.Y('Posting Volume:Q', title='Posting Volume'),
            tooltip=[alt.Tooltip('Date:T', title='Actual Date', format='%b %d, %Y'), alt.Tooltip('Posting Volume:Q', title='Volume')]
        )
        
        # pred 1
        pred1 = alt.Chart(p1_df).mark_line(point=True, color='#0085ff', strokeWidth=3, strokeDash=[5, 5]).encode(
            x=x_fmt,
            y=alt.Y('Posting Volume:Q', title='Posting Volume'),
            tooltip=[alt.Tooltip('Date:T', title='Predicted (Day 1)', format='%b %d, %Y'), alt.Tooltip('Posting Volume:Q', title='Predicted Volume')]
        )
        
        # pred 2
        pred2 = alt.Chart(p2_df).mark_line(point=True, color='#80c2ff', strokeWidth=2, strokeDash=[2, 4]).encode(
            x=x_fmt,
            y=alt.Y('Posting Volume:Q', title='Posting Volume'),
            tooltip=[alt.Tooltip('Date:T', title='Predicted (Day 2)', format='%b %d, %Y'), alt.Tooltip('Posting Volume:Q', title='Predicted Volume')]
        )
        
        final_chart = (base + pred1 + pred2).interactive(bind_y=False)
        
    else:
        final_chart = alt.Chart(plot_df).mark_bar(color='#0085ff').encode(
            x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%b %d', labelAngle=0)),
            y=alt.Y('Posting Volume:Q', title='Posting Volume'),
            tooltip=[alt.Tooltip('Date:T', title='Actual Date', format='%b %d, %Y'), alt.Tooltip('Posting Volume:Q', title='Volume')]
        ).interactive(bind_y=False)
    
    st.altair_chart(final_chart, use_container_width=True)