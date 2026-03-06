"""
Disaster Prediction System - Streamlit Dashboard
Interactive web app for natural disaster risk prediction and visualization.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Disaster Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header{font-size:2.5rem;font-weight:bold;color:#1f77b4;text-align:center;padding:1rem 0;border-bottom:3px solid #1f77b4;margin-bottom:2rem}
.risk-card{padding:1.5rem;border-radius:10px;text-align:center;font-size:1.2rem;font-weight:bold;margin:0.5rem 0}
.risk-LOW{background:#d4edda;color:#155724;border:2px solid #28a745}
.risk-MODERATE{background:#fff3cd;color:#856404;border:2px solid #ffc107}
.risk-HIGH{background:#f8d7da;color:#721c24;border:2px solid #dc3545}
.risk-CRITICAL{background:#f5c6cb;color:#491217;border:3px solid #a71d2a}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Generating training data...")
def load_flood_data(n_samples=4000):
    from src.data.data_generator import FloodDataGenerator
    return FloodDataGenerator(n_samples=n_samples, random_state=42).generate()

@st.cache_data(show_spinner="Generating earthquake data...")
def load_earthquake_data(n_samples=5000):
    from src.data.data_generator import EarthquakeDataGenerator
    return EarthquakeDataGenerator(n_samples=n_samples, random_state=42).generate()

@st.cache_resource(show_spinner="Training flood model...")
def get_flood_model(model_type='xgboost'):
    from src.models.flood_model import FloodGradientBoostingModel
    m = FloodGradientBoostingModel(model_type=model_type, random_state=42)
    m.train(load_flood_data())
    return m

@st.cache_resource(show_spinner="Training earthquake model...")
def get_earthquake_model(model_type='xgboost'):
    from src.models.earthquake_model import EarthquakeGradientBoostingModel
    m = EarthquakeGradientBoostingModel(model_type=model_type, random_state=42)
    m.train(load_earthquake_data())
    return m

@st.cache_resource(show_spinner="Training earthquake regressor...")
def get_earthquake_regressor():
    from src.models.earthquake_model import EarthquakeRiskRegressor
    m = EarthquakeRiskRegressor(model_type='xgboost', random_state=42)
    m.train(load_earthquake_data())
    return m


def risk_emoji(level):
    return {'LOW':'🟢','MODERATE':'🟡','HIGH':'🟠','CRITICAL':'🔴','MEDIUM':'🟡'}.get(level.upper(),'⚪')

def gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value*100,
        title={'text':title,'font':{'size':16}},
        number={'suffix':'%','font':{'size':24}},
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':'darkblue'},
            'steps':[
                {'range':[0,30],'color':'#d4edda'},
                {'range':[30,60],'color':'#fff3cd'},
                {'range':[60,80],'color':'#f8d7da'},
                {'range':[80,100],'color':'#dc3545'}
            ],
            'threshold':{'line':{'color':'red','width':4},'thickness':0.75,'value':70}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=20))
    return fig


def page_home():
    st.markdown('<div class="main-header">🌍 Disaster Prediction System</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.info("**🌊 Flood Prediction**\n\nXGBoost + LightGBM + LSTM time series models for flood risk based on rainfall, river levels and soil moisture")
    c2.warning("**🏔️ Earthquake Risk**\n\nGradient boosting ensembles for seismic risk classification using fault proximity, geology and historical data")
    c3.error("**📊 Analytics**\n\nFeature importance, global risk maps, distributions and time series visualizations")
    st.markdown("---")
    st.subheader("🤖 Model Architecture Overview")
    arch = pd.DataFrame({
        'Disaster':['Flood','Flood','Flood','Flood','Earthquake','Earthquake','Earthquake'],
        'Model':['XGBoost','LightGBM','LSTM','ARIMA','XGBoost','LightGBM','Ensemble'],
        'Type':['Gradient Boosting','Gradient Boosting','Time Series','Time Series',
                'Gradient Boosting','Gradient Boosting','Ensemble'],
        'Target Accuracy':[0.92,0.91,0.88,0.85,0.89,0.88,0.93]
    })
    fig = px.bar(arch, x='Model', y='Target Accuracy', color='Type', barmode='group',
                 facet_col='Disaster', title='Model Performance Targets',
                 color_discrete_map={'Gradient Boosting':'#1f77b4','Time Series':'#ff7f0e','Ensemble':'#2ca02c'},
                 height=400)
    st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌊 Flood Features")
        for f in ['Rainfall 24h / 72h / 7-day (mm)','River Level (m)','Soil Moisture (%)','Groundwater Level (m)','Elevation & Slope','Temperature & Humidity']:
            st.markdown(f"- {f}")
    with col2:
        st.subheader("🏔️ Earthquake Features")
        for f in ['Fault Distance & Type','Plate Velocity (mm/yr)','Historical Activity (5yr)','Rock Type & Vs30','Building Age & Code Compliance','Population Density']:
            st.markdown(f"- {f}")


def page_flood():
    st.header("🌊 Flood Prediction")
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Model Metrics", "⏱️ Time Series"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Parameters")
            with st.expander("🌧️ Rainfall & Hydrology", expanded=True):
                r24 = st.slider("Rainfall 24h (mm)", 0.0, 200.0, 25.0, 0.5)
                r72 = st.slider("Rainfall 72h (mm)", 0.0, 500.0, round(r24*2.8,1), 1.0)
                r7d = st.slider("Rainfall 7-day (mm)", 0.0, 1000.0, round(r72*2.5,1), 5.0)
                river = st.slider("River Level (m)", 0.0, 15.0, 3.0, 0.1)
                soil = st.slider("Soil Moisture (%)", 0.0, 100.0, 55.0, 1.0)
                gw = st.slider("Groundwater Level (m)", 0.0, 10.0, 3.0, 0.1)
            with st.expander("🗺️ Topography"):
                elev = st.slider("Elevation (m)", 0.0, 500.0, 50.0, 5.0)
                slope = st.slider("Slope (°)", 0.0, 45.0, 5.0, 0.5)
                drain = st.slider("Drainage Area (km²)", 1.0, 500.0, 100.0, 5.0)
            with st.expander("🌡️ Weather"):
                temp = st.slider("Temperature (°C)", -10.0, 45.0, 22.0, 0.5)
                humid = st.slider("Humidity (%)", 0.0, 100.0, 75.0, 1.0)
                wind = st.slider("Wind Speed (km/h)", 0.0, 100.0, 20.0, 1.0)
            mtype = st.selectbox("Model", ["xgboost","lightgbm","sklearn"])
        with col2:
            st.subheader("Prediction Results")
            inp = pd.DataFrame([{'rainfall_24h_mm':r24,'rainfall_72h_mm':r72,'rainfall_7day_mm':r7d,
                'river_level_m':river,'soil_moisture_pct':soil,'groundwater_level_m':gw,
                'elevation_m':elev,'slope_degrees':slope,'drainage_area_km2':drain,
                'temperature_c':temp,'humidity_pct':humid,'wind_speed_kmh':wind}])
            try:
                model = get_flood_model(mtype)
                res = model.predict(inp)
                prob = res['flood_probability'][0]
                risk = res['risk_level'][0]
                st.plotly_chart(gauge_chart(prob, "Flood Probability"), use_container_width=True)
                st.markdown(f'<div class="risk-card risk-{risk}">{risk_emoji(risk)} {risk} | {prob:.1%}</div>', unsafe_allow_html=True)
                st.markdown("### Recommendations")
                msgs = {'CRITICAL':('error','🚨 EVACUATE NOW – Contact emergency services immediately'),
                        'HIGH':('warning','⚠️ HIGH ALERT – Prepare for evacuation, monitor hourly'),
                        'MODERATE':('info','ℹ️ MONITOR – Check conditions every 6 hours'),
                        'LOW':('success','✅ SAFE – Normal monitoring recommended')}
                level_key = risk if risk in msgs else 'LOW'
                getattr(st, msgs[level_key][0])(msgs[level_key][1])
                factors = pd.DataFrame({'Factor':['Rainfall 24h','River Level','Soil Moisture','Rainfall 72h'],
                                        'Risk':[r24/200, river/15, soil/100, r72/500]})
                fig = px.bar(factors, x='Factor', y='Risk', color='Risk',
                             color_continuous_scale='RdYlGn_r', range_y=[0,1], title='Key Risk Factors', height=250)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")
    with tab2:
        st.subheader("Flood Model Performance")
        try:
            model = get_flood_model('xgboost')
            m = model.metrics
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Accuracy", f"{m.get('accuracy',0):.1%}")
            c2.metric("F1 Score", f"{m.get('f1_score',0):.1%}")
            c3.metric("ROC-AUC", f"{m.get('roc_auc',0):.1%}")
            c4.metric("Recall", f"{m.get('recall',0):.1%}")
            fi = model.get_feature_importance()
            fig = px.bar(fi.head(12), x='importance', y='feature', orientation='h',
                         title='Feature Importance (XGBoost)', color='importance',
                         color_continuous_scale='Blues', height=400)
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    with tab3:
        st.subheader("River Level Time Series")
        try:
            from src.data.data_generator import FloodDataGenerator
            stations = FloodDataGenerator(n_samples=200).generate_time_series(n_stations=3, n_timesteps=168)
            station = st.selectbox("Station", list(stations.keys()))
            sdf = stations[station]
            fig = make_subplots(rows=2, cols=1, subplot_titles=('River Level (m)','Rainfall (mm)'))
            fig.add_trace(go.Scatter(x=sdf['timestamp'], y=sdf['river_level_m'], mode='lines',
                                     name='River Level', line=dict(color='blue', width=2)), row=1, col=1)
            fig.add_trace(go.Bar(x=sdf['timestamp'], y=sdf['rainfall_mm'], name='Rainfall',
                                 marker_color='cyan', opacity=0.7), row=2, col=1)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")


def page_earthquake():
    st.header("🏔️ Earthquake Risk Assessment")
    tab1, tab2, tab3 = st.tabs(["🔮 Assess Risk", "📊 Distribution", "🗺️ Global Map"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Location & Parameters")
            with st.expander("📍 Location", expanded=True):
                lat = st.slider("Latitude", -60.0, 60.0, 37.5, 0.1)
                lon = st.slider("Longitude", -180.0, 180.0, -122.0, 0.5)
            with st.expander("⚡ Seismic Activity", expanded=True):
                fd = st.slider("Fault Distance (km)", 0.0, 200.0, 25.0, 1.0)
                ft = st.selectbox("Fault Type", ['strike-slip','thrust','normal','oblique'])
                pv = st.slider("Plate Velocity (mm/yr)", 1.0, 15.0, 7.0, 0.5)
                heq = st.slider("Historical EQs (5yr)", 0, 50, 8)
                le = st.slider("Years Since Last Major EQ", 0.0, 100.0, 20.0, 1.0)
            with st.expander("🪨 Geology"):
                rt = st.selectbox("Rock Type", ['hard_rock','soft_soil','alluvium','fill'])
                vs30 = st.slider("Vs30 (m/s)", 150.0, 1500.0, 450.0, 10.0)
                dob = st.slider("Depth to Bedrock (m)", 0.0, 100.0, 25.0, 1.0)
            with st.expander("🏙️ Infrastructure"):
                pd_val = st.slider("Population Density (/km²)", 0.0, 5000.0, 500.0, 10.0)
                ba = st.slider("Avg Building Age (yr)", 0.0, 100.0, 35.0, 1.0)
                cc = st.slider("Code Compliance", 0.0, 1.0, 0.65, 0.01)
            mtype = st.selectbox("Model", ["xgboost","lightgbm","sklearn"])
        with col2:
            st.subheader("Risk Assessment")
            sa = {'hard_rock':1.0,'soft_soil':3.0,'alluvium':2.0,'fill':4.0}[rt]
            inp = pd.DataFrame([{'latitude':lat,'longitude':lon,'fault_distance_km':fd,
                'fault_type':ft,'plate_velocity_mm_yr':pv,'historical_earthquakes_5yr':heq,
                'last_major_event_years':le,'rock_type':rt,'soil_amplification_factor':sa,
                'depth_to_bedrock_m':dob,'vs30_m_s':vs30,'population_density_km2':pd_val,
                'building_age_avg_years':ba,'seismic_code_compliance':cc}])
            try:
                clf = get_earthquake_model(mtype)
                reg = get_earthquake_regressor()
                clf_res = clf.predict(inp)
                risk_cat = clf_res['risk_category'][0]
                conf = clf_res['confidence'][0]
                risk_score = float(reg.predict_risk_score(inp)[0])
                st.plotly_chart(gauge_chart(risk_score, "Seismic Risk Score"), use_container_width=True)
                disp = risk_cat if risk_cat != 'MEDIUM' else 'MODERATE'
                st.markdown(f'<div class="risk-card risk-{disp}">{risk_emoji(risk_cat)} {risk_cat} | Conf: {conf:.1%} | Score: {risk_score:.3f}</div>', unsafe_allow_html=True)
                if 'class_probabilities' in clf_res:
                    probs = clf_res['class_probabilities']
                    classes = clf.classes_
                    pd_data = {'Risk':list(classes),'Probability':[probs.get(f'prob_{c.lower()}',[0])[0] for c in classes]}
                    fig = px.pie(pd.DataFrame(pd_data), names='Risk', values='Probability',
                                 color='Risk', color_discrete_map={'LOW':'#28a745','MEDIUM':'#ffc107','HIGH':'#dc3545'},
                                 title='Risk Category Probabilities', height=280)
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("### Mitigation Recommendations")
                recs = {'HIGH':('error','🚨 HIGH RISK: Enforce strict seismic codes. Mandatory structural audits.'),
                        'MEDIUM':('warning','⚠️ MEDIUM RISK: Upgrade older buildings. Seismic hazard assessment required.'),
                        'LOW':('success','✅ LOW RISK: Standard building codes sufficient. Basic preparedness recommended.')}
                r = recs.get(risk_cat, recs['LOW'])
                getattr(st, r[0])(r[1])
            except Exception as e:
                st.error(f"Error: {e}")
    with tab2:
        st.subheader("Risk Distribution Analysis")
        try:
            df = load_earthquake_data(2000)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x='earthquake_risk_score', color='risk_category', nbins=40,
                                   title='Risk Score Distribution',
                                   color_discrete_map={'LOW':'#28a745','MEDIUM':'#ffc107','HIGH':'#dc3545'})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                rc = df['risk_category'].value_counts()
                fig = px.pie(values=rc.values, names=rc.index, title='Risk Category Share',
                             color=rc.index, color_discrete_map={'LOW':'#28a745','MEDIUM':'#ffc107','HIGH':'#dc3545'})
                st.plotly_chart(fig, use_container_width=True)
            fig = px.violin(df, x='fault_type', y='earthquake_risk_score', color='fault_type',
                            box=True, title='Risk Score by Fault Type', height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    with tab3:
        st.subheader("Global Seismic Risk Map")
        try:
            df = load_earthquake_data(3000)
            fig = px.scatter_geo(df.sample(min(1500,len(df))), lat='latitude', lon='longitude',
                color='earthquake_risk_score', color_continuous_scale='RdYlGn_r',
                size='earthquake_risk_score', size_max=12, opacity=0.7,
                title='Global Earthquake Risk Distribution', projection='natural earth',
                hover_data={'earthquake_risk_score':':.3f','risk_category':True,'fault_type':True},
                height=600)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")


def page_analytics():
    st.header("📊 Risk Analytics & Insights")
    tab1, tab2 = st.tabs(["🌊 Flood Analytics", "🏔️ Earthquake Analytics"])
    with tab1:
        try:
            df = load_flood_data(3000)
            model = get_flood_model('xgboost')
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Records", f"{len(df):,}")
            c2.metric("Flood Events", f"{df['flood_occurred'].sum():,}")
            c3.metric("Flood Rate", f"{df['flood_occurred'].mean():.1%}")
            c4.metric("Avg Rain 24h", f"{df['rainfall_24h_mm'].mean():.1f}mm")
            cols = ['rainfall_24h_mm','rainfall_72h_mm','river_level_m','soil_moisture_pct','elevation_m','humidity_pct','flood_occurred']
            corr = df[cols].corr()
            fig = px.imshow(corr, text_auto='.2f', title='Feature Correlations', color_continuous_scale='RdBu_r', aspect='auto', height=400)
            st.plotly_chart(fig, use_container_width=True)
            fi = model.get_feature_importance()
            fig = px.bar(fi, x='feature', y='importance', color='importance',
                         title='Feature Importance – Flood XGBoost', color_continuous_scale='viridis', height=350)
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    with tab2:
        try:
            df = load_earthquake_data(3000)
            model = get_earthquake_model('xgboost')
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Records", f"{len(df):,}")
            c2.metric("High Risk Zones", f"{(df['risk_category']=='HIGH').sum():,}")
            c3.metric("Avg Risk Score", f"{df['earthquake_risk_score'].mean():.3f}")
            c4.metric("Tectonic Regions", str(df['region'].nunique()))
            rr = df.groupby('region')['earthquake_risk_score'].agg(['mean','std']).reset_index()
            rr.columns = ['Region','Mean Risk','Std Dev']
            fig = px.bar(rr, x='Region', y='Mean Risk', error_y='Std Dev',
                         title='Avg Earthquake Risk by Tectonic Region', color='Mean Risk',
                         color_continuous_scale='Reds', height=400)
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
            fi = model.get_feature_importance()
            fig = px.bar(fi.head(10), x='importance', y='feature', orientation='h',
                         title='Top 10 Earthquake Risk Factors', color='importance',
                         color_continuous_scale='Reds', height=400)
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")


def main():
    with st.sidebar:
        st.markdown("## 🌍 Disaster Prediction")
        page = st.radio("Navigate to", ["🏠 Home","🌊 Flood Prediction","🏔️ Earthquake Risk","📊 Analytics"])
        st.markdown("---")
        st.markdown("**Models**")
        st.markdown("- XGBoost (gradient boosting)\n- LightGBM (gradient boosting)\n- LSTM (time series)\n- ARIMA (forecasting)\n- Ensemble")
        st.markdown("---")
        try:
            fm = get_flood_model('xgboost')
            em = get_earthquake_model('xgboost')
            st.metric("Flood Model F1", f"{fm.metrics.get('f1_score',0):.1%}")
            st.metric("EQ Model Accuracy", f"{em.metrics.get('accuracy',0):.1%}")
        except:
            st.info("Models train on first use")
    
    if "Home" in page: page_home()
    elif "Flood" in page: page_flood()
    elif "Earthquake" in page: page_earthquake()
    elif "Analytics" in page: page_analytics()


if __name__ == '__main__':
    main()
