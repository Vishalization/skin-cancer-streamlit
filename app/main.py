import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from model_loader import load_models
from utils import preprocess, interpret_prediction
from config import CLASS_NAMES

st.set_page_config(page_title='Skin Cancer Detection', page_icon='🩺', layout='wide')

st.markdown('''
<style>
.stApp {background: #0e1117; color: white;}
.hero {padding: 1.2rem; border-radius: 18px; background: linear-gradient(90deg,#1f77b4,#6a5acd); text-align:center; font-size:2rem; font-weight:700; color:white; margin-bottom:1rem;}
.sub {text-align:center; color:#d0d0d0; margin-bottom:1.2rem;}
.card {background: rgba(255,255,255,0.05); padding:1rem; border-radius:16px; border:1px solid rgba(255,255,255,0.08);}
.small {color:#cfcfcf; font-size:0.9rem;}
</style>
''', unsafe_allow_html=True)

st.markdown("<div class='hero'>Skin Cancer Detection using Deep Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Premium AI screening demo using ResNet50 and EfficientNetB0</div>", unsafe_allow_html=True)

st.sidebar.title('Project Info')
st.sidebar.info('Dataset: HAM10000\n\nModels:\n- ResNet50\n- EfficientNetB0')
st.sidebar.warning('Educational use only. Not medical advice.')

class_meanings = {
    'akiec':'Actinic Keratoses',
    'bcc':'Basal Cell Carcinoma',
    'bkl':'Benign Keratosis-like Lesions',
    'df':'Dermatofibroma',
    'mel':'Melanoma',
    'nv':'Melanocytic Nevus',
    'vasc':'Vascular Lesions'
}

def make_pdf(df, img):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph('Skin Cancer Detection Report', styles['Title']))
    story.append(Spacer(1,12))
    story.append(Paragraph(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), styles['Normal']))
    story.append(Spacer(1,12))
    img_buf = BytesIO()
    img.save(img_buf, format='PNG')
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=220, height=220))
    story.append(Spacer(1,12))
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('ALIGN',(0,0),(-1,-1),'CENTER')]))
    story.append(table)
    story.append(Spacer(1,12))
    story.append(Paragraph('Disclaimer: Educational tool only.', styles['Normal']))
    doc.build(story)
    buf.seek(0)
    return buf

uploaded = st.file_uploader('Upload Skin Lesion Image', type=['jpg','jpeg','png'])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    col1,col2 = st.columns([1,1.3])
    with col1:
        st.image(img, caption='Uploaded Image', use_container_width=True)
    with col2:
        st.subheader('Predictions')
        with st.spinner('Running AI models...'):
            models = load_models()
            rows = []
            for model_name, model in models.items():
                x = preprocess(img, model_name)
                preds = model.predict(x, verbose=0)[0]
                class_name, conf, label = interpret_prediction([preds])
                rows.append({'Model':model_name,'Prediction':class_name.upper(),'Confidence %':round(conf*100,2),'Status':label})
                st.markdown(f"### {model_name}")
                if label == 'Suspicious':
                    st.error(f'⚠️ Suspicious ({class_name.upper()})')
                else:
                    st.success(f'✅ Not Suspicious ({class_name.upper()})')
                st.progress(float(conf))
                st.caption(f'Confidence: {conf*100:.2f}%')
                chart = {CLASS_NAMES[i].upper(): float(preds[i]) for i in range(len(CLASS_NAMES))}
                st.bar_chart(chart)
                st.divider()
        df = pd.DataFrame(rows)
        st.subheader('Model Comparison')
        st.dataframe(df, use_container_width=True)
        pdf = make_pdf(df, img)
        st.download_button('📄 Download PDF Report', data=pdf, file_name='skin_cancer_report.pdf', mime='application/pdf')

st.markdown('## HAM10000 Dataset')
st.write('A benchmark dataset containing 10,000+ dermoscopic skin lesion images across multiple diagnostic classes.')

st.markdown('## Class Meanings')
st.dataframe(pd.DataFrame({'Class':list(class_meanings.keys()),'Meaning':list(class_meanings.values())}), use_container_width=True)

st.markdown('## About This Project')
st.write('Final year project demonstrating deep learning based skin lesion classification deployed as an interactive cloud application.')

st.warning('This application is for educational and research purposes only and does not replace professional medical diagnosis.')