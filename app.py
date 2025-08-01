import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from keras.models import load_model
import pickle
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga CPO",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk memuat model dan data
@st.cache_resource
def load_model_and_data():
    try:
        # Load model
        model = load_model('cpo_lstm_model.h5')
        
        # Load scaler (coba dua metode)
        scaler = None
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        except Exception as e:
            try:
                scaler = joblib.load('scaler.joblib')
            except Exception as e2:
                st.error(f"Gagal memuat scaler: {e} | {e2}")
                return None, None, None, None, None
        
        # Load konfigurasi
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        # Load data terakhir untuk prediksi
        last_data = np.load('last_data_for_prediction.npy')
        
        # Load dataset historis
        df = pd.read_csv('processed_cpo_data.csv')
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        
        return model, scaler, config, last_data, df
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None

# Fungsi prediksi
def predict_future_prices(model, scaler, last_data, days_ahead=30):
    predictions = []
    current_data = last_data.copy()
    
    for _ in range(days_ahead):
        # Prediksi satu hari ke depan
        pred = model.predict(current_data, verbose=0)
        predictions.append(pred[0, 0])
        
        # Update data untuk prediksi berikutnya
        new_data = np.append(current_data[0, 1:, 0], pred[0, 0])
        current_data = new_data.reshape(1, -1, 1)
    
    # Denormalisasi prediksi
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_inv = scaler.inverse_transform(predictions)
    
    return predictions_inv.flatten()

# Main App
def main():
    st.title("🌴 Prediksi Harga CPO (Crude Palm Oil)")
    st.markdown("---")
    
    # Load model dan data
    model, scaler, config, last_data, df = load_model_and_data()
    
    if model is None:
        st.error("Gagal memuat model. Pastikan semua file model tersedia.")
        st.info("File yang diperlukan:")
        st.code("""
        - cpo_lstm_model.h5
        - scaler.pkl  
        - model_config.json
        - last_data_for_prediction.npy
        - processed_cpo_data.csv
        """)
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Pengaturan Prediksi")
    
    # Input jumlah hari prediksi
    days_ahead = st.sidebar.slider(
        "Jumlah Hari Prediksi ke Depan:",
        min_value=1,
        max_value=90,
        value=30,
        help="Pilih berapa hari ke depan yang ingin diprediksi"
    )
    
    # Tombol prediksi
    predict_button = st.sidebar.button("🔮 Buat Prediksi", type="primary")
    
    # Informasi model
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Informasi Model")
    st.sidebar.metric("RMSE", f"{config['rmse']:.2f}")
    st.sidebar.metric("R² Score", f"{config['r2_score']:.2f}%")
    st.sidebar.info(f"Model dilatih pada: {config['training_date']}")
    
    # Layout utama
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Grafik Harga CPO")
        
        # Grafik data historis
        fig = go.Figure()
        
        # Data historis
        fig.add_trace(go.Scatter(
            x=df['Tanggal'],
            y=df['Penetapan_Harga'],
            mode='lines',
            name='Data Historis',
            line=dict(color='blue', width=2)
        ))
        
        # Jika tombol prediksi ditekan
        if predict_button:
            with st.spinner('Sedang membuat prediksi...'):
                # Buat prediksi
                predictions = predict_future_prices(model, scaler, last_data, days_ahead)
                
                # Buat tanggal untuk prediksi
                last_date = df['Tanggal'].max()
                future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
                
                # Tambahkan prediksi ke grafik
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions,
                    mode='lines+markers',
                    name='Prediksi',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                ))
                
                # Simpan hasil prediksi ke session state
                st.session_state['predictions'] = predictions
                st.session_state['future_dates'] = future_dates
        
        # Jika ada prediksi yang tersimpan, tampilkan
        if 'predictions' in st.session_state:
            fig.add_trace(go.Scatter(
                x=st.session_state['future_dates'],
                y=st.session_state['predictions'],
                mode='lines+markers',
                name='Prediksi',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ))
        
        # Update layout grafik
        fig.update_layout(
            title="Harga CPO Historis dan Prediksi",
            xaxis_title="Tanggal",
            yaxis_title="Harga (Rp)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("🎯 Hasil Prediksi")
        
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            future_dates = st.session_state['future_dates']
            
            # Statistik prediksi
            current_price = df['Penetapan_Harga'].iloc[-1]
            avg_prediction = np.mean(predictions)
            max_prediction = np.max(predictions)
            min_prediction = np.min(predictions)
            
            st.metric(
                "Harga Saat Ini", 
                f"Rp {current_price:,.0f}"
            )
            st.metric(
                "Rata-rata Prediksi", 
                f"Rp {avg_prediction:,.0f}",
                f"{avg_prediction - current_price:,.0f}"
            )
            st.metric(
                "Prediksi Tertinggi", 
                f"Rp {max_prediction:,.0f}"
            )
            st.metric(
                "Prediksi Terendah", 
                f"Rp {min_prediction:,.0f}"
            )
            
            # Tabel prediksi
            st.subheader("📋 Detail Prediksi")
            pred_df = pd.DataFrame({
                'Tanggal': future_dates,
                'Prediksi Harga': [f"Rp {p:,.0f}" for p in predictions]
            })
            
            st.dataframe(
                pred_df,
                use_container_width=True,
                height=300
            )
            
            # Tombol download
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Prediksi (CSV)",
                data=csv,
                file_name=f'prediksi_cpo_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        
        else:
            st.info("Klik tombol 'Buat Prediksi' untuk melihat hasil prediksi")
    
    # Statistik data historis
    st.markdown("---")
    st.header("📊 Analisis Data Historis")
    
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        st.metric("Total Data", f"{len(df):,} hari")
    
    with col4:
        st.metric("Harga Tertinggi", f"Rp {df['Penetapan_Harga'].max():,.0f}")
    
    with col5:
        st.metric("Harga Terendah", f"Rp {df['Penetapan_Harga'].min():,.0f}")
    
    with col6:
        st.metric("Rata-rata Harga", f"Rp {df['Penetapan_Harga'].mean():,.0f}")
    
    # Distribusi harga
    col7, col8 = st.columns(2)
    
    with col7:
        st.subheader("Distribusi Harga")
        fig_hist = px.histogram(
            df, 
            x='Penetapan_Harga',
            nbins=30,
            title="Distribusi Harga CPO"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col8:
        st.subheader("Trend Bulanan")
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly['Tanggal'].dt.to_period('M')
        monthly_avg = df_monthly.groupby('Month')['Penetapan_Harga'].mean().reset_index()
        monthly_avg['Month'] = monthly_avg['Month'].astype(str)
        
        fig_monthly = px.line(
            monthly_avg,
            x='Month',
            y='Penetapan_Harga',
            title="Rata-rata Harga CPO per Bulan"
        )
        fig_monthly.update_xaxis(tickangle=45)
        st.plotly_chart(fig_monthly, use_container_width=True)

# Footer
def show_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Aplikasi Prediksi Harga CPO menggunakan LSTM Neural Network</p>
        <p>⚠️ Prediksi ini hanya untuk referensi dan tidak menjamin akurasi 100%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    show_footer()