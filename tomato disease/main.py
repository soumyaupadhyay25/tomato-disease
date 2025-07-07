import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
import os
from datetime import datetime

# ================== CONSTANTS & TRANSLATIONS ==================
CLASS_NAMES = [
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__healthy"
]

# Language mapping
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta"
}

# Translations dictionary
T = {
    "title": {
        "en": "TOMATO DISEASE RECOGNITION SYSTEM",
        "hi": "टमाटर रोग पहचान प्रणाली",
        "ta": "தக்காளி நோய் கண்டறிதல் அமைப்பு"
    },
    "report_title": {
        "en": "Tomato Disease Detection Report",
        "hi": "टमाटर रोग पहचान रिपोर्ट",
        "ta": "தக்காளி நோய் கண்டறிதல் அறிக்கை"
    },
    "detected_disease": {
        "en": "Detected Disease:",
        "hi": "पता चला रोग:",
        "ta": "கண்டறியப்பட்ட நோய்:"
    },
    "treatment_label": {
        "en": "Recommended Treatment:",
        "hi": "अनुशंसित उपचार:",
        "ta": "பரிந்துரைக்கப்பட்ட சிகிச்சை:"
    },
    "home_content": {
        "en": """## 🍅🌱 Welcome to the Tomato Disease Prediction System

**🔍 What Does It Do?**  
- Detects 9 common tomato plant diseases  
- Classifies healthy and diseased leaves  
- Recommends treatment strategies

**🧠 Powered by Deep Learning**  
- Trained on 10,000+ tomato leaf images  
- Uses Convolutional Neural Network (CNN)  
- 95%+ validation accuracy

**🚀 How to Use**  
1. Go to Disease Recognition page  
2. Upload leaf image (jpg/png)  
3. Get instant diagnosis""",
        "hi": """## 🍅🌱 टमाटर रोग पूर्वानुमान प्रणाली में आपका स्वागत है

**🔍 सिस्टम क्या करता है?**  
- 9 सामान्य टमाटर रोगों का पता लगाता है  
- स्वस्थ और रोगग्रस्त पत्तियों को वर्गीकृत करता है  
- उपचार सुझाव देता है

**🧠 एआई तकनीक**  
- 10,000+ टमाटर पत्ती छवियों पर प्रशिक्षित  
- कन्वेन्शनल न्यूरल नेटवर्क मॉडल  
- 95%+ सटीकता

**🚀 उपयोग विधि**  
1. 'रोग पहचान' पेज पर जाएं  
2. पत्ती की तस्वीर अपलोड करें (jpg/png)  
3. तुरंत परिणाम प्राप्त करें""",
        "ta": """## 🍅🌱 தக்காளி நோய் முன்னறிவிப்பு அமைப்புக்கு வரவேற்கிறோம்

**🔍 அமைப்பு என்ன செய்கிறது?**  
- 9 பொதுவான தக்காளி நோய்களை கண்டறியும்  
- ஆரோக்கியமான மற்றும் நோய் பாதிக்கப்பட்ட இலைகளை வகைப்படுத்துகிறது  
- சிகிச்சை பரிந்துரைகள்

**🧠 டீப் லர்னிங் தொழில்நுட்பம்**  
- 10,000+ தக்காளி இலை படங்களில் பயிற்சி பெற்றது  
- கான்வல்யூஷனல் நியூரல் நெட்வொர்க் மாதிரி  
- 95%+ துல்லியம்

**🚀 பயன்படுத்தும் முறை**  
1. 'நோய் கண்டறிதல்' பக்கத்திற்குச் செல்லவும்  
2. இலையின் படத்தை பதிவேற்றம் செய்யவும் (jpg/png)  
3. உடனடி முடிவுகளைப் பெறவும்"""
    },
    "about_content": {
        "en": """## 📚 Dataset & Model Information

**📂 Dataset**  
- 10 classes (9 diseases + healthy)  
- 10,000 training images  
- 1,006 validation images

**🖼️ Image Specifications**  
- RGB format  
- 128x128 resolution  
- Various lighting conditions

**🤖 Model Architecture**  
- TensorFlow/Keras CNN   
- Data augmentation layers

**👨💻 Developers**  
Ayush Pandey & Soumya Upadhyay  
AI Engineering Team""",
        "hi": """## 📚 डेटासेट और मॉडल जानकारी

**📂 डेटासेट**  
- 10 वर्ग (9 रोग + स्वस्थ)  
- 10,000 प्रशिक्षण छवियाँ  
- 1,006 सत्यापन छवियाँ

**🖼️ छवि विवरण**  
- RGB प्रारूप  
- 128x128 रिज़ॉल्यूशन  
- विभिन्न प्रकाश स्थितियाँ

**🤖 मॉडल आर्किटेक्चर**  
- TensorFlow/Keras CNN    
- डेटा संवर्द्धन

**👨💻 डेवलपर्स**  
आयुष पांडेय और सौम्या उपाध्याय  
एआई इंजीनियरिंग टीम""",
        "ta": """## 📚 தரவுத்தொகுப்பு & மாதிரி தகவல்

**📂 தரவுத்தொகுப்பு**  
- 10 வகைகள் (9 நோய்கள் + ஆரோக்கியமானது)  
- 10,000 பயிற்சி படங்கள்  
- 1,006 சரிபார்ப்பு படங்கள்

**🖼️ பட விவரக்குறிப்புகள்**  
- RGB வடிவம்  
- 128x128 தீர்மானம்  
- பல்வேறு ஒளி நிலைமைகள்

**🤖 மாதிரி கட்டமைப்பு**  
- TensorFlow/Keras CNN    
- தரவு மேம்படுத்தல்

**👨💻 உருவாக்குனர்கள்**  
ஆயுஷ் பாண்டே மற்றும் சௌம்யா உபாத்யாய்  
AI பொறியியல் குழு"""
    }
}

# ================== Add Missing Treatment ==================
T['treatment'] = {
    'Tomato__Bacterial_spot': {
        "en": "Copper-based bactericides. Avoid overhead irrigation.",
        "hi": "तांबा-आधारित जीवाणुरोधी। ऊपर से सिंचाई न करें।",
        "ta": "தாமிரம் அடிப்படையிலான பாக்டீரியா எதிர்ப்பிகள். மேலிருந்து நீர்ப்பாசனம் தவிர்க்கவும்."
    },
    'Tomato__Early_blight': {
        "en": "Use fungicides like chlorothalonil or mancozeb. Remove infected leaves.",
        "hi": "क्लोरोथालोनिल या मैंकोजेब जैसे कवकनाशी का प्रयोग करें। संक्रमित पत्तियों को हटा दें।",
        "ta": "குளோரோத்தாலோனில் அல்லது மாங்கோசெப் போன்ற பூஞ்சைநாசினிகளை பயன்படுத்தவும். பாதிக்கப்பட்ட இலைகளை அகற்றவும்."
    },
    'Tomato__Late_blight': {
        "en": "Use fungicides with active ingredients like copper or chlorothalonil. Avoid wet foliage.",
        "hi": "कॉपर या क्लोरोथालोनिल जैसे तत्वों वाले कवकनाशी का प्रयोग करें। गीली पत्तियों से बचें।",
        "ta": "தாமிரம் அல்லது குளோரோத்தாலோனில் உள்ள பூஞ்சைநாசினிகளை பயன்படுத்தவும். ஈரமான இலைகளை தவிர்க்கவும்."
    },
    'Tomato__Leaf_Mold': {
        "en": "Ensure good air circulation. Use fungicides if necessary.",
        "hi": "अच्छा वायु प्रवाह सुनिश्चित करें। आवश्यकता होने पर कवकनाशी का उपयोग करें।",
        "ta": "நல்ல காற்றோட்டத்தை உறுதி செய்யவும். தேவையெனில் பூஞ்சைநாசினிகளை பயன்படுத்தவும்."
    },
    'Tomato__Septoria_leaf_spot': {
        "en": "Apply fungicides and remove debris around the plant.",
        "hi": "कवकनाशी का छिड़काव करें और पौधे के चारों ओर का मलबा हटा दें।",
        "ta": "பூஞ்சைநாசினிகளைத் தெளிக்கவும் மற்றும் செடியை சுற்றியுள்ள குப்பைகளை அகற்றவும்."
    },
    'Tomato__Spider_mites_Two_spotted_spider_mite': {
        "en": "Use miticides or insecticidal soap. Maintain humidity.",
        "hi": "माइटिसाइड या कीटनाशक साबुन का उपयोग करें। आर्द्रता बनाए रखें।",
        "ta": "மைட்டிசைடு அல்லது பூச்சி அழிக்கும் சோப்பைப் பயன்படுத்தவும். ஈரப்பதத்தை பராமரிக்கவும்."
    },
    'Tomato__Target_Spot': {
        "en": "Use preventive fungicides and crop rotation.",
        "hi": "रोकथाम के लिए कवकनाशी और फसल चक्रण करें।",
        "ta": "முன்கூட்டிய பூஞ்சைநாசினிகள் மற்றும் பயிர் சுழற்சி பயன்படுத்தவும்."
    },
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus': {
        "en": "Use insecticides to control whiteflies. Remove infected plants.",
        "hi": "सफेद मक्खियों को नियंत्रित करने के लिए कीटनाशकों का उपयोग करें। संक्रमित पौधों को हटा दें।",
        "ta": "வெள்ளை ஈயை கட்டுப்படுத்த பூச்சி நாசினிகளைப் பயன்படுத்தவும். பாதிக்கப்பட்ட தாவரங்களை அகற்றவும்."
    },
    'Tomato__Tomato_mosaic_virus': {
        "en": "Remove infected plants. Disinfect tools. Avoid tobacco use around plants.",
        "hi": "संक्रमित पौधों को हटाएं। उपकरणों को कीटाणुरहित करें। पौधों के पास तंबाकू का उपयोग न करें।",
        "ta": "பாதிக்கப்பட்ட தாவரங்களை அகற்றவும். கருவிகளை जीवाणுநாசினியுடன் சுத்தம் செய்யவும். தாவரங்களுக்குக் குறுகிய தூரத்தில் புகைபிடிப்பதை தவிர்க்கவும்."
    },
    'Tomato__healthy': {
        "en": "No treatment needed. Maintain proper care.",
        "hi": "उपचार की आवश्यकता नहीं। उचित देखभाल बनाए रखें।",
        "ta": "சிகிச்சை தேவையில்லை. சரியான பராமரிப்பை நிலைநாட்டவும்."
    }
}

# ================== Disease Recognition ==================
from fpdf import FPDF
from datetime import datetime


# Specify the path to your Noto Sans Regular font
font_path = r"C:\Users\ayush\Downloads\tomato disease\fonts\NotoSans-Regular.ttf"  # Correct path to your font file

# Generate PDF with Unicode support
def generate_pdf(disease_name, confidence, treatment, language):
    pdf = FPDF()
    pdf.add_page()

    # Add custom font that supports Unicode characters (Noto Sans Regular)
    pdf.add_font('NotoSans', '', font_path, uni=True)  # Load font with Unicode support
    pdf.set_font('NotoSans', '', 12)  # Use the custom font

    # Add content to the PDF
    pdf.cell(200, 10, T['report_title'][language], ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, f"{T['detected_disease'][language]} {disease_name}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, f"{T['treatment_label'][language]} {treatment}", ln=True)

    # Save the PDF to a file
    file_name = f"tomato_disease_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf.output(file_name)
    return file_name


if __name__ == '__main__':
    # Language selection and sidebar content
    language_selection = st.sidebar.selectbox("Select Language", ["English", "Hindi", "Tamil"])
    language = LANGUAGES[language_selection]  # Convert selection to language code

    # Set the main page content based on the sidebar selection
    page_selection = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

    if page_selection == "Home":
        st.title(T['title'][language])
        st.markdown(T['home_content'][language])
    elif page_selection == "About":
        st.title("📚 Dataset & Model Information")
        st.markdown(T['about_content'][language])
    elif page_selection == "Disease Recognition":
        st.title("🌱 Disease Recognition")
        st.write("Upload a leaf image to detect the disease.")
        uploaded_file = st.file_uploader("Upload Tomato Leaf Image", type=["jpg", "png"])

        if uploaded_file is not None:
            try:
                # Open the image using PIL
                image = Image.open(uploaded_file)
                image = image.convert('RGB')  # Ensure the image is in RGB

                # Display the image
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Preprocess the image for prediction
                model = tf.keras.models.load_model('full_model.h5')
                image = np.array(image.resize((128, 128))) / 255.0
                image = np.expand_dims(image, axis=0)

                # Make predictions
                predictions = model.predict(image)
                predicted_class = CLASS_NAMES[np.argmax(predictions)]
                confidence = np.max(predictions) * 100

                st.write(f"Predicted Disease: {predicted_class}")
                st.write(f"Confidence: {confidence:.2f}%")

                # Fetch treatment
                treatment = T['treatment'].get(predicted_class, T['treatment']['Tomato__healthy'])[language]

                st.write(f"Recommended Treatment: {treatment}")

                # If user wants to download a report
                if st.button("Download Report"):
                    generate_pdf(predicted_class, confidence, treatment, language)

            except Exception as e:
                st.error(f"Error: {e}")
