import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pickle
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow dengan error handling
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        from keras.models import load_model
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        st.error("⚠️ TensorFlow/Keras tidak tersedia. Aplikasi akan berjalan dalam mode demo.")

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
        model = None
        if TENSORFLOW_AVAILABLE:
            # Load model TensorFlow/Keras
            try:
                model = load_model('cpo_lstm_model.h5')
            except Exception as e:
                st.warning(f"Gagal memuat model TensorFlow: {str(e)}")
                # Fallback ke model sklearn jika ada
                try:
                    with open('cpo_sklearn_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    st.info("Menggunakan model scikit-learn sebagai fallback")
                except Exception as e2:
                    st.error(f"Gagal memuat semua model: {str(e2)}")
                    return None, None, None, None, None
        
        # Load scaler (coba beberapa metode)
        scaler = None
        for scaler_file in ['scaler.pkl', 'scaler.joblib']:
            try:
                if scaler_file.endswith('.pkl'):
                    with open(scaler_file, 'rb') as f:
                        scaler = pickle.load(f)
                else:
                    scaler = joblib.load(scaler_file)
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                st.warning(f"Error loading {scaler_file}: {str(e)}")
                continue
        
        if scaler is None:
            st.error("Gagal memuat scaler. Mencoba membuat scaler default...")
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        # Load konfigurasi
        config = {}
        try:
            with open('model_config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            # Default config jika file tidak ada
            config = {
                'rmse': 'N/A',
                'r2_score': 'N/A',
                'training_date': 'N/A',
                'sequence_length': 60
            }
        
        # Load data terakhir untuk prediksi
        last_data = None
        try:
            last_data = np.load('last_data_for_prediction.npy')
        except FileNotFoundError:
            st.warning("File last_data_for_prediction.npy tidak ditemukan")
        
        # Load dataset historis
        df = None
        try:
            df = pd.read_csv('processed_cpo_data.csv')
            df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        except FileNotFoundError:
            # Generate sample data jika file tidak ada
            st.warning("File data tidak ditemukan, menggunakan data sampel")
            dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
            np.random.seed(42)
            base_price = 12000
            prices = []
            current_price = base_price
            
            for i in range(len(dates)):
                # Simulasi fluktuasi harga dengan trend
                daily_change = np.random.normal(0, 200)
                seasonal_effect = 500 * np.sin(2 * np.pi * i / 365)
                current_price += daily_change + seasonal_effect * 0.1
                prices.append(max(8000, min(18000, current_price)))  # Batasi range harga
            
            df = pd.DataFrame({
                'Tanggal': dates,
                'Penetapan_Harga': prices
            })
        
        return model, scaler, config, last_data, df
    except Exception as e:
        st.error(f"Error loading model and data: {str(e)}")
        return None, None, None, None, None

# Fungsi prediksi
def predict_future_prices(model, scaler, last_data, df, days_ahead=30):
    try:
        if model is None:
            st.error("Model tidak tersedia untuk prediksi")
            return None
        
        # Jika menggunakan LSTM model
        if hasattr(model, 'predict') and last_data is not None:
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
        
        else:
            # Fallback: Simple prediction based on trend
            st.info("Menggunakan prediksi sederhana berdasarkan trend historis")
            recent_prices = df['Penetapan_Harga'].tail(30).values
            
            # Simple linear trend
            x = np.arange(len(recent_prices))
            coeffs = np.polyfit(x, recent_prices, 1)
            
            # Predict future values
            future_x = np.arange(len(recent_prices), len(recent_prices) + days_ahead)
            trend_predictions = np.polyval(coeffs, future_x)
            
            # Add some random variation
            np.random.seed(42)
            noise = np.random.normal(0, recent_prices.std() * 0.1, days_ahead)
            predictions = trend_predictions + noise
            
            return predictions
    
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None

# Main App
def main():
    st.title("🌴 Prediksi Harga CPO (Crude Palm Oil)")
    st.markdown("---")
    
    # Load model dan data
    model, scaler, config, last_data, df = load_model_and_data()
    
    if df is None:
        st.error("Gagal memuat data. Aplikasi tidak dapat berjalan.")
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
    
    if isinstance(config.get('rmse'), (int, float)):
        st.sidebar.metric("RMSE", f"{config['rmse']:.2f}")
    else:
        st.sidebar.metric("RMSE", str(config.get('rmse', 'N/A')))
    
    if isinstance(config.get('r2_score'), (int, float)):
        st.sidebar.metric("R² Score", f"{config['r2_score']:.2f}%")
    else:
        st.sidebar.metric("R² Score", str(config.get('r2_score', 'N/A')))
    
    st.sidebar.info(f"Model info: {config.get('training_date', 'N/A')}")
    
    # Status model
    if TENSORFLOW_AVAILABLE and model is not None:
        st.sidebar.success("✅ Model AI tersedia")
    else:
        st.sidebar.warning("⚠️ Menggunakan prediksi sederhana")
    
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
                predictions = predict_future_prices(model, scaler, last_data, df, days_ahead)
                
                if predictions is not None:
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
                    
                    st.success(f"✅ Berhasil membuat prediksi untuk {days_ahead} hari ke depan!")
        
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
        fig_monthly.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_monthly, use_container_width=True)

# Footer
def show_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Aplikasi Prediksi Harga CPO menggunakan Machine Learning</p>
        <p>⚠️ Prediksi ini hanya untuk referensi dan tidak menjamin akurasi 100%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    show_footer()