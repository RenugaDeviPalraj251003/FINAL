import streamlit as st
import pandas as pd
import shap
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
from joblib import load
from lime import lime_tabular
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from googletrans import Translator

# Add this at the beginning of your script (after imports)
if 'prev_language' not in st.session_state:
    st.session_state.prev_language = "English" 



# --- Load Datasets ---
recommendation_data = pd.read_csv("Expanded_Crop_Recommendation.csv")
yield_data = pd.read_csv("Crop_Yield_Expanded_Dataset.csv")

# --- Encode categorical variables ---
state_encoder = LabelEncoder()
season_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

yield_data['State'] = state_encoder.fit_transform(yield_data['State'])
yield_data['Season'] = season_encoder.fit_transform(yield_data['Season'])
yield_data['Crop'] = crop_encoder.fit_transform(yield_data['Crop'])

# --- Train Models ---
X_recommend = recommendation_data.drop(columns=['label'])
y_recommend = recommendation_data['label']
crop_model = RandomForestClassifier(n_estimators=200, random_state=42)
crop_model.fit(X_recommend, y_recommend)

X_yield = yield_data[['State', 'Season', 'Crop', 'Area', 'Pesticide', 'Fertilizer', 'Annual_Rainfall']]
y_yield = yield_data['Yield']
yield_model = RandomForestRegressor(n_estimators=200, random_state=42)
yield_model.fit(X_yield, y_yield)

# --- Load Waste and Compost Models ---
try:
    waste_model = load("waste_model_scaled.joblib")
    compost_model = load("compost_model_scaled.joblib")
    waste_scaler = load("waste_scaler.joblib")  # Load the waste scaler
    compost_scaler = load("compost_scaler.joblib")  # Load the compost scaler
except:
    # Create dummy models if files not found
    from sklearn.linear_model import LinearRegression
    waste_model = LinearRegression()
    compost_model = LinearRegression()
    # Create dummy scalers
    from sklearn.preprocessing import MinMaxScaler
    waste_scaler = MinMaxScaler()
    compost_scaler = MinMaxScaler()

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="CROP RECOMMENDATION,YIELD AND WASTE PREDICTION-பயிர் பரிந்துரை, விளைச்சல் மற்றும் கழிவு கணிப்பு", 
    page_icon="🌾", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Translator ---
translator = Translator()

# --- Sidebar for Language Selection ---
with st.sidebar:
    st.title("🌐 Language Settings")
    language = st.radio(
        "Select Language",
        ["English", "Tamil", "Hindi"],
        index=0
    )
     # Clear selections when language changes
    if st.session_state.prev_language != language:
        st.session_state.prev_language = language
        for key in ['selected_state_translated', 'selected_season_translated', 'selected_crop_translated']:
            if key in st.session_state:
                del st.session_state[key]
    
    # Add some space and additional info
    st.markdown("---")
    st.markdown("### ℹ️ About This App")
    st.markdown("This app helps farmers with:")
    st.markdown("- Crop recommendations")
    st.markdown("- Yield predictions")
    st.markdown("- Waste management")
    
# Set language code mapping
lang_code = {"English": "en", "Tamil": "ta", "Hindi": "hi"}[language]

# --- Translation Dictionaries ---
translations = {
    "crop_recommendation": {
        "title": {
            "en": "🔍 Enter Soil & Climate Conditions for Crop Recommendation",
            "ta": "🔍பயிர் பரிந்துரைக்கு மண் மற்றும் காலநிலை நிலைமைகளை உள்ளிடவும்",
            "hi": "🔍 फसल सिफारिश के लिए मिट्टी और जलवायु की स्थिति दर्ज करें"
        },
        "nitrogen": {
            "en": "Nitrogen (N)",
            "ta": "நைட்ரஜன்",
            "hi": "नाइट्रोजन"
        },
        "phosphorus": {
            "en": "Phosphorus (P)",
            "ta": "பாஸ்பரஸ்",
            "hi": "फास्फोरस"
        },
        "potassium": {
            "en": "Potassium (K)",
            "ta": "பொட்டாசியம்",
            "hi": "पोटेशियम"
        },
        "temperature": {
            "en": "Temperature (°C)",
            "ta": "வெப்பநிலை",
            "hi": "तापमान"
        },
        "humidity": {
            "en": "Humidity (%)",
            "ta": "ஈரப்பதம்",
            "hi": "नमी"
        },
        "ph": {
            "en": "pH",
            "ta": "pH",
            "hi": "pH"
        },
        "rainfall": {
            "en": "Rainfall (mm)",
            "ta": "மழைப்பொழிவு",
            "hi": "वर्षा"
        },
        "button": {
            "en": "Recommend Crop",
            "ta": "பயிர் பரிந்துரை",
            "hi": "फसल की सिफारिश करें"
        }
    },
    "yield_prediction": {
        "title": {
            "en": "📈 Enter Details for Yield Prediction",
            "ta": "📈மகசூல் கணிப்புக்கான விவரங்களை உள்ளிடவும்.",
            "hi": "📈 पैदावार भविष्यवाणी के लिए विवरण दर्ज करें"
        },
        "state": {
            "en": "Select State",
            "ta": "மாநிலத்தைத் தேர்ந்தெடுக்கவும்",
            "hi": "राज्य चुनें"
        },
        "season": {
            "en": "Select Season",
            "ta": "பருவத்தைத் தேர்ந்தெடுக்கவும்",
            "hi": "मौसम चुनें"
        },
        "area": {
            "en": "Area (hectares)",
            "ta": "பகுதி",
            "hi": "क्षेत्र"
        },
        "pesticide": {
            "en": "Pesticide (kg/ha)",
            "ta": "பூச்சிக்கொல்லி (கிலோ/எக்டர்)",
            "hi": "कीटनाशक"
        },
        "fertilizer": {
            "en": "Fertilizer (kg/ha)",
            "ta": "உரம் (கிலோ/எக்டர்)",
            "hi": "उर्वरक"
        },
        "rainfall": {
            "en": "Annual Rainfall (mm)",
            "ta": "ஆண்டு மழைப்பொழிவு (மிமீ)",
            "hi": "वार्षिक वर्षा"
        },
        "button": {
            "en": "Predict Yield",
            "ta": "விளைச்சலைக் கணிக்கவும்",
            "hi": "पैदावार का अनुमान लगाएं"
        }
    },
    "other_sections": {
        "feature_prediction": {
            "en": "🌟 Feature Prediction Module",
            "ta": "🌟 அம்ச கணிப்பு தொகுதி",
            "hi": "🌟 सुविधा भविष्यवाणी मॉड्यूल"
        },
        "select_crop": {
            "en": "Select Crop for Feature Expectations",
            "ta": "அம்ச எதிர்பார்ப்புகளுக்கான பயிரைத் தேர்ந்தெடுக்கவும்",
            "hi": "फीचर अपेक्षाओं के लिए फसल चुनें"
        },
        "get_features": {
            "en": "Get Expected Features",
            "ta": "எதிர்பார்க்கப்படும் அம்சங்களைப் பெறவும்",
            "hi": "अपेक्षित सुविधाएँ प्राप्त करें"
        },
        "waste_prediction": {
            "en": "♻️ Waste Prediction",
            "ta": "♻️ கழிவு கணிப்பு",
            "hi": "♻️ अपशिष्ट भविष्यवाणी"
        },
        "predict_waste": {
            "en": "Predict Waste",
            "ta": "கழிவை கணிக்கவும்",
            "hi": "अपशिष्ट का अनुमान लगाएं"
        },
        "compost_prediction": {
            "en": "🌿 Composting Yield Prediction",
            "ta": "🌿 கம்போஸ்ட் மகசூல் கணிப்பு",
            "hi": "🌿 कम्पोस्ट उपज भविष्यवाणी"
        },
        "moisture": {
            "en": "Moisture (%)",
            "ta": "ஈரப்பதம் (%)",
            "hi": "नमी (%)"
        },
        "compost_temp": {
            "en": "Temperature (°C)",
            "ta": "வெப்பநிலை (°C)",
            "hi": "तापमान (°C)"
        },
        "aeration": {
            "en": "Aeration Frequency (days)",
            "ta": "காற்றோட்டம் அதிர்வெண் (நாட்கள்)",
            "hi": "वातन आवृत्ति (दिन)"
        },
        "duration": {
            "en": "Composting Duration (weeks)",
            "ta": "கம்போஸ்டிங் காலம் (வாரங்கள்)",
            "hi": "कम्पोस्टिंग अवधि (सप्ताह)"
        },
        "predict_compost": {
            "en": "Predict Compost Yield",
            "ta": "கம்போஸ்ட் மகசூலை கணிக்கவும்",
            "hi": "कम्पोस्ट उपज का अनुमान लगाएं"
        }
    },
        "options": {
        "states": {
            "Andhra Pradesh": {"ta": "ஆந்திரப் பிரதேசம்", "hi": "आंध्र प्रदेश"},
            "Arunachal Pradesh": {"ta": "அருணாச்சல பிரதேசம்", "hi": "अरुणाचल प्रदेश"},
            "Assam": {"ta": "அசாம்", "hi": "असम"},
            "Bihar": {"ta": "பீகார்", "hi": "बिहार"},
            "Chhattisgarh": {"ta": "சத்தீஸ்கர்", "hi": "छत्तीसगढ़"},
            "Delhi": {"ta": "டெல்லி", "hi": "दिल्ली"},
            "Goa": {"ta": "கோவா", "hi": "गोवा"},
            "Gujarat": {"ta": "குஜராத்", "hi": "गुजरात"},
            "Haryana": {"ta": "ஹரியானா", "hi": "हरियाणा"},
            "Himachal Pradesh": {"ta": "இமாச்சல பிரதேசம்", "hi": "हिमाचल प्रदेश"},
            "Jammu and Kashmir": {"ta": "ஜம்மு காஷ்மீர்", "hi": "जम्मू और कश्मीर"},
            "Jharkhand": {"ta": "ஜார்க்கண்ட்", "hi": "झारखंड"},
            "Karnataka": {"ta": "கர்நாடகா", "hi": "कर्नाटक"},
            "Kerala": {"ta": "கேரளா", "hi": "केरल"},
            "Madhya Pradesh": {"ta": "மத்திய பிரதேசம்", "hi": "मध्य प्रदेश"},
            "Maharashtra": {"ta": "மகாராஷ்டிரா", "hi": "महाराष्ट्र"},
            "Manipur": {"ta": "மணிப்பூர்", "hi": "मणिपुर"},
            "Meghalaya": {"ta": "மேகாலயா", "hi": "मेघालय"},
            "Mizoram": {"ta": "மிசோரம்", "hi": "मिजोरम"},
            "Nagaland": {"ta": "நாகாலாந்து", "hi": "नागालैंड"},
            "Odisha": {"ta": "ஒடிசா", "hi": "ओडिशा"},
            "Puducherry": {"ta": "புதுச்சேரி", "hi": "पुदुच्चेरी"},
            "Punjab": {"ta": "பஞ்சாப்", "hi": "पंजाब"},
            "Sikkim": {"ta": "சிக்கிம்", "hi": "सिक्किम"},
            "Tamil Nadu": {"ta": "தமிழ்நாடு", "hi": "तमिलनाडु"},
            "Telangana": {"ta": "தெலுங்கானா", "hi": "तेलंगाना"},
            "Tripura": {"ta": "திரிபுரா", "hi": "त्रिपुरा"},
            "Uttar Pradesh": {"ta": "உத்தரப் பிரதேசம்", "hi": "उत्तर प्रदेश"},
            "Uttarakhand": {"ta": "உத்தரகண்ட்", "hi": "उत्तराखंड"},
            "West Bengal": {"ta": "மேற்கு வங்காளம்", "hi": "पश्चिम बंगाल"}
        },
        "seasons": {
            "Autumn": {"ta": "குளிர்காலம்", "hi": "पतझड़"},
            "Kharif": {"ta": "கறிஃப்", "hi": "खरीफ"},
            "Rabi": {"ta": "ரபி", "hi": "रबी"},
            "Summer": {"ta": "கோடை", "hi": "गर्मी"},
            "Whole Year": {"ta": "முழு ஆண்டு", "hi": "पूरा साल"},
            "Winter": {"ta": "குளிர்காலம்", "hi": "सर्दी"}
        },
        "crops": {
           "Arecanut": {"ta": "அரிகுட்டு", "hi": "पान"},
    "Arhar/Tur": {"ta": "துவரை", "hi": "अरहर"},
    "Bajra": {"ta": "கம்பு", "hi": "बाजरा"},
    "Banana": {"ta": "வாழை", "hi": "केला"},
    "Barley": {"ta": "பார்லி", "hi": "जौ"},
    "Black pepper": {"ta": "மிளகு", "hi": "काली मिर्च"},
    "Cardamom": {"ta": "ஏலக்காய்", "hi": "इलायची"},
    "Cashewnut": {"ta": "முந்திரி", "hi": "काजू"},
    "Castor seed": {"ta": "ரேகை", "hi": "अरंडी"},
    "Coconut": {"ta": "தோசை", "hi": "नारियल"},  # Added space after Coconut
    "Coriander": {"ta": "கொத்தமல்லி", "hi": "धनिया"},
    "Cotton(lint)": {"ta": "பஞ்சு", "hi": "कपास"},
    "Cowpea(Lobia)": {"ta": "கொம்பு", "hi": "लोबिया"},
    "Dry chillies": {"ta": "உலர்ந்த மிளகாய்", "hi": "सूखी मिर्च"},
    "Garlic": {"ta": "பூண்டு", "hi": "लहसुन"},
    "Ginger": {"ta": "இஞ்சி", "hi": "अदरक"},
    "Groundnut": {"ta": "பருத்தி", "hi": "मूंगफली"},
    "Guar seed": {"ta": "கொல்லு", "hi": "ग्वार"},
    "Horse-gram": {"ta": "குதிரை பருப்பு", "hi": "घास"},
    "Jowar": {"ta": "சோளம்", "hi": "ज्वार"},
    "Jute": {"ta": "ஜூட்", "hi": "जूट"},
    "Khesari": {"ta": "கேசரி", "hi": "केसरी"},
    "Linseed": {"ta": "அள்ளு", "hi": "अलसी"},
    "Maize": {"ta": "மக்காச்சோளம்", "hi": "मक्का"},
    "Masoor": {"ta": "மசூரி", "hi": "मसूर"},
    "Mesta": {"ta": "மெஸ்டா", "hi": "मेसता"},
    "Moong(Green Gram)": {"ta": "பச்சை பருப்பு", "hi": "मूंग"},
    "Moth": {"ta": "மொத்", "hi": "मोत"},
    "Niger seed": {"ta": "நிகர்", "hi": "निगर"},
    "Onion": {"ta": "வெங்காயம்", "hi": "प्याज"},
    "Peas & beans (Pulses)": {"ta": "பீன்ஸ்", "hi": "मटर"},
    "Potato": {"ta": "உருளைக்கிழங்கு", "hi": "आलू"},
    "Ragi": {"ta": "ரகி", "hi": "रागी"},
    "Rapeseed &Mustard": {"ta": "கொத்தமல்லி", "hi": "सरसों"},  # Removed space after &
    "Rice": {"ta": "அரிசி", "hi": "चावल"},
    "Safflower": {"ta": "சேம்பரத்தூள்", "hi": "सफ्लॉवर"},
    "Sannhamp": {"ta": "சன்னம்", "hi": "सन्नम"},
    "Sesamum": {"ta": "எள்ளு", "hi": "तिल"},
    "Soyabean": {"ta": "சோயா", "hi": "सोयाबीन"},
    "Sugarcane": {"ta": "சர்க்கரை cane", "hi": "गन्ना"},
    "Sunflower": {"ta": "சூரியக்கதிர்", "hi": "सूरजमुखी"},
    "Sweet potato": {"ta": "இனிப்பு உருளைக்கிழங்கு", "hi": "शकरकंद"},
    "Tapioca": {"ta": "கசவா", "hi": "साबूदाना"},
    "Tobacco": {"ta": "தம்பாக்கு", "hi": "तंबाकू"},
    "Turmeric": {"ta": "மஞ்சள்", "hi": "हल्दी"},
    "Urad": {"ta": "உருது", "hi": "उरद"},
    "Wheat": {"ta": "கோதுமை", "hi": "गेहूं"}
        }
    }
}

def translate_text(text, dest_lang):
    try:
        if pd.isna(text) or text == "":
            return text
        translated_text = GoogleTranslator(source='auto', target=dest_lang).translate(text)
        return translated_text  # No .text property needed
       
    except Exception as e:
        st.warning(f"Translation error for '{text}': {str(e)}")
        return text

def get_translation(key, section="crop_recommendation"):
    try:
        if section == "other_sections":
            return translations["other_sections"][key][lang_code]
        return translations[section][key][lang_code]
    except KeyError:
        return key

def get_translated_options(options, option_type):
    """Returns options in the selected language"""
    if language == "English":
        return options
    
    translated = []
    for opt in options:
        try:
            # Clean the option
            cleaned_opt = opt.strip()
            
            # For debugging - print the available keys
            if len(translated) == 0:  # Just for the first item
                available_keys = list(translations["options"][option_type].keys())
                st.write(f"Available {option_type} keys: {available_keys[:5]}...")
                st.write(f"Looking for: '{cleaned_opt}'")
            
            translated.append(translations["options"][option_type][cleaned_opt][lang_code])
        except KeyError:
            # Try case-insensitive match as fallback
            found = False
            for key in translations["options"][option_type]:
                if key.lower() == cleaned_opt.lower():
                    translated.append(translations["options"][option_type][key][lang_code])
                    found = True
                    break
            
            if not found:
                st.warning(f"Missing translation for '{opt}' in {option_type}")
                translated.append(opt)
    return translated

# --- Page Styling ---
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #e8f5e9;
            background-image: linear-gradient(to bottom, #e8f5e9, #c8e6c9);
        }
        .stButton>button {
            background-color: #2e7d32;
            color: white;
            border-radius: 10px;
            padding: 10px 24px;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #1b5e20;
            color: white;
        }
        .css-1aumxhk {
            background-color: #a5d6a7;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        h1, h2, h3, h4 {
            color: #1b5e20;
        }
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #a5d6a7 !important;
        }
        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
            color: #1b5e20 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# --- Session State Initialization ---
if "top_crops" not in st.session_state:
    st.session_state["top_crops"] = None
if "predicted_yield" not in st.session_state:
    st.session_state["predicted_yield"] = None
if "input_features" not in st.session_state:
    st.session_state["input_features"] = None
if "lime_explanations" not in st.session_state:
    st.session_state["lime_explanations"] = {}

# --- Page Title ---
st.title("🌾 CROP RECOMMENDATION, YIELD AND WASTE PREDICTION")          
st.title("பயிர் பரிந்துரை, விளைச்சல் மற்றும் கழிவு கணிப்பு" if language == "Tamil" else 
         "फसल सिफारिश, उपज और अपशिष्ट भविष्यवाणी" if language == "Hindi" else "")
st.header("🚜 Multi-Crop SHAP & LIME Explanation, Feature Expectations & Yield Prediction")

# ---------------------------
# 🚜 Crop Recommendation Module
# ---------------------------
st.subheader(get_translation("title"))
col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input(get_translation("nitrogen"), min_value=0, max_value=300, value=90)
    P = st.number_input(get_translation("phosphorus"), min_value=0, max_value=300, value=42)
with col2:
    K = st.number_input(get_translation("potassium"), min_value=0, max_value=300, value=43)
    temperature = st.number_input(get_translation("temperature"), min_value=0.0, max_value=50.0, value=27.5)
with col3:
    humidity = st.number_input(get_translation("humidity"), min_value=0, max_value=100, value=80)
    ph = st.number_input(get_translation("ph"), min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input(get_translation("rainfall"), min_value=0, max_value=5000, value=120)

if st.button(get_translation("button"), key="recommend_btn"):
    input_features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                                columns=X_recommend.columns)
    st.session_state["input_features"] = input_features

    probabilities = crop_model.predict_proba(input_features)[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]

    top_crops = [crop_model.classes_[i] for i in top_3_indices]
    top_confidences = [probabilities[i] for i in top_3_indices]
    
    st.session_state["top_crops"] = top_crops
    st.session_state["top_confidences"] = top_confidences

    # SHAP Explanation
    explainer = shap.TreeExplainer(crop_model)
    shap_values = explainer.shap_values(input_features)
    
    # Create SHAP plot for each top crop
    for i, crop_idx in enumerate(top_3_indices):
        if len(shap_values) > crop_idx:  # Check if index exists
            crop_title = translate_text(f"SHAP Explanation for {top_crops[i]} (Confidence: {top_confidences[i]*100:.2f}%)", lang_code)
            st.subheader(crop_title)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values[crop_idx], input_features, plot_type="bar", show=False)
            st.pyplot(fig)
            plt.close()

    # LIME Explanation
    st.subheader(translate_text("LIME Explanations", lang_code))
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_recommend.values,
        feature_names=X_recommend.columns,
        class_names=crop_model.classes_,
        verbose=True,
        mode='classification'
    )
    
    lime_explanations = {}
    for i, crop in enumerate(top_crops):
        exp = lime_explainer.explain_instance(
            input_features.values[0],
            crop_model.predict_proba,
            num_features=len(X_recommend.columns),
            top_labels=3
        )
        
        # Save explanation to session state
        lime_img = exp.as_pyplot_figure(label=exp.top_labels[0])
        # Modify the title after creating the figure
        plt.title(f"LIME Explanation for {crop}", fontsize=14)
    
        buf = BytesIO()
        lime_img.savefig(buf, format="png", bbox_inches='tight')
        plt.close(lime_img)
        lime_explanations[crop] = buf.getvalue()
    
    st.session_state["lime_explanations"] = lime_explanations
    
    # Display LIME explanations
    for crop, img_bytes in lime_explanations.items():
        st.subheader(translate_text(f"LIME Explanation for {crop}", lang_code))
        st.image(img_bytes, use_column_width=True)

    st.markdown(translate_text("### 📊 Top 3 Recommended Crops with Confidence Scores:", lang_code))
    for i, crop in enumerate(top_crops):
        st.write(translate_text(f"✅ {crop}: {top_confidences[i]*100:.2f}% confidence", lang_code))
    
    st.markdown(translate_text("### 🌱 Farmer-Friendly Explanation:", lang_code))
    for i, crop_index in enumerate(top_3_indices):
        if crop_index < len(shap_values):
            crop_name = crop_model.classes_[crop_index]
            shap_values_flat = shap_values[crop_index][0]

            explanation_text = ""
            for feature, value in zip(X_recommend.columns, shap_values_flat):
                impact = translate_text("increases" if value > 0 else "decreases", lang_code)
                farmer_tip = {
                    "Nitrogen (N)": translate_text("Essential for plant growth.", lang_code),
                    "Phosphorus (P)": translate_text("Promotes strong root development.", lang_code),
                    "Potassium (K)": translate_text("Improves disease resistance.", lang_code),
                    "Temperature": translate_text("Optimal temperature for crop growth.", lang_code),
                    "Humidity": translate_text("Regulates crop moisture balance.", lang_code),
                    "pH": translate_text("Balances soil nutrients.", lang_code),
                    "Rainfall": translate_text("Ensures consistent hydration.", lang_code)
                }.get(feature, translate_text("Optimize this feature for better results.", lang_code))

                explanation_text += f"""
                - {translate_text(feature, lang_code)}:
                    - {translate_text("Current value", lang_code)}: {input_features[feature].values[0]}
                    - {translate_text("This feature", lang_code)} {impact} {translate_text("the recommendation by", lang_code)} {abs(value):.4f}
                    - {translate_text("Farmer's Tip", lang_code)}: {farmer_tip}
                """

            st.markdown(f"#### 🌾 {translate_text(crop_name, lang_code)}")
            st.markdown(explanation_text)

# ---------------------------
# 📈 Yield Prediction Module (Updated with multilingual options)
# ---------------------------
st.subheader(get_translation("title", "yield_prediction"))
col1, col2 = st.columns(2)

with col1:
    # State selection with translations
    state_names = state_encoder.classes_
    translated_states = get_translated_options(state_names, "states")
    state_map = dict(zip(translated_states, state_names))
    
   # For State Selection (in Yield Prediction Module)
    selected_state_translated = st.selectbox(
    get_translation("state", "yield_prediction"),
    translated_states,
    key="selected_state_translated"  )
    selected_state_name = state_map[selected_state_translated]
    
    # Season selection with translations
    season_names = season_encoder.classes_
    translated_seasons = get_translated_options(season_names, "seasons")
    season_map = dict(zip(translated_seasons, season_names))
    
    # For Season Selection
    selected_season_translated = st.selectbox(
    get_translation("season", "yield_prediction"),
    translated_seasons,
    key="selected_season_translated"  )
    selected_season_name = season_map[selected_season_translated]

with col2:
    area = st.number_input(get_translation("area", "yield_prediction"), min_value=1, max_value=8629000, value=500)
    pesticide = st.number_input(get_translation("pesticide", "yield_prediction"), min_value=0.0, max_value=2674990.0, value=5.0)

fertilizer = st.number_input(get_translation("fertilizer", "yield_prediction"), min_value=0.0, max_value=1301253200.0, value=50.0)
annual_rainfall = st.number_input(get_translation("rainfall", "yield_prediction"), min_value=0.0, max_value=6552.7, value=900.0)

if st.button(get_translation("button", "yield_prediction"), key="yield_pred_btn"):
    selected_state_encoded = state_encoder.transform([selected_state_name])[0]
    selected_season_encoded = season_encoder.transform([selected_season_name])[0]

    input_df = pd.DataFrame([[selected_state_encoded, selected_season_encoded, 1,
                            area, pesticide, fertilizer, annual_rainfall]],
                          columns=["State", "Season", "Crop", "Area", "Pesticide", "Fertilizer", "Annual_Rainfall"])

    predicted_yield = yield_model.predict(input_df)[0]
    st.session_state["predicted_yield"] = predicted_yield
    st.session_state["yield_input_features"] = input_df

    # SHAP Explanation for Yield
    explainer_yield = shap.TreeExplainer(yield_model)
    shap_values_yield = explainer_yield.shap_values(input_df)
    
    st.subheader(translate_text("SHAP Explanation for Yield Prediction", lang_code))
    fig_yield_shap, ax = plt.subplots()
    shap.summary_plot(shap_values_yield, input_df, plot_type="bar", show=False)
    st.pyplot(fig_yield_shap)
    plt.close()

    # LIME Explanation for Yield
    lime_explainer_yield = lime_tabular.LimeTabularExplainer(
        X_yield.values,
        feature_names=X_yield.columns,
        verbose=True,
        mode='regression'
    )
    
    exp_yield = lime_explainer_yield.explain_instance(
        input_df.values[0],
        yield_model.predict,
        num_features=len(X_yield.columns))
    
    lime_img_yield = exp_yield.as_pyplot_figure()
    buf_yield = BytesIO()
    lime_img_yield.savefig(buf_yield, format="png", bbox_inches='tight')
    plt.close(lime_img_yield)
    st.session_state["lime_yield_explanation"] = buf_yield.getvalue()
    
    st.subheader(translate_text("LIME Explanation for Yield Prediction", lang_code))
    st.image(buf_yield.getvalue(), use_column_width=True)

    # Create multilingual graph
    fig_yield = go.Figure(data=[
        go.Bar(
            x=[translate_text(col, lang_code) for col in input_df.columns],
            y=input_df.values[0], 
            marker=dict(color='lightgreen')
        )
    ])
    
    fig_yield.update_layout(
        title=translate_text("📊 Yield Prediction Graph", lang_code),
        xaxis_title=translate_text("Features", lang_code),
        yaxis_title=translate_text("Values", lang_code),
        template="plotly_white",
        font=dict(
            family="Arial Unicode MS, sans-serif"  # Supports multiple languages
        )
    )
    st.plotly_chart(fig_yield)
    st.success(translate_text(f"🌾 Predicted Yield: {predicted_yield:.2f} tons/ha", lang_code))

# ---------------------------
# 🌟 Feature Prediction Module (Updated with multilingual crop selection)
# ---------------------------
st.subheader(get_translation("feature_prediction", "other_sections"))

# Get original crop names
crop_names = crop_model.classes_
# Get translated crop names for display
translated_crops = get_translated_options(crop_names, "crops")
# Create mapping between original and translated names
crop_map = dict(zip(translated_crops, crop_names))

# For Crop Selection (in Feature Prediction Module)
selected_crop_translated = st.selectbox(
    get_translation("select_crop", "other_sections"),
    translated_crops,
    key="selected_crop_translated"  # Add this key
)
selected_crop = crop_map[selected_crop_translated]

if st.button(get_translation("get_features", "other_sections"), key="features_btn"):
    crop_data = recommendation_data[recommendation_data['label'] == selected_crop]
    avg_values = crop_data.mean(numeric_only=True)
    st.write(translate_text("### 🌿 Expected Feature Values:", lang_code))
    for feature, value in avg_values.items():
        st.write(f"- {translate_text(feature, lang_code)}: {value:.2f}")
    st.session_state["expected_features"] = avg_values

# ---------------------------
# ♻️ Waste Prediction Module
# ---------------------------
# ---------------------------
# ♻️ Waste Prediction Module
# ---------------------------
st.subheader(get_translation("waste_prediction", "other_sections"))
if st.session_state.get("top_crops") is not None and st.session_state.get("predicted_yield") is not None:
    if st.button(get_translation("predict_waste", "other_sections"), key="waste_pred_btn"):
        selected_crop = st.session_state["top_crops"][0]
        predicted_yield_kg = st.session_state["predicted_yield"] * 1000  # Convert tons to kg
        
        try:
            waste_input = pd.DataFrame([[selected_crop, predicted_yield_kg]],
                                     columns=['Crop_Type', 'Yield_Amount_kg'])
            
            waste_input = pd.get_dummies(waste_input, columns=['Crop_Type'], drop_first=True)
            missing_cols = set(waste_model.feature_names_in_) - set(waste_input.columns)
            for col in missing_cols:
                waste_input[col] = 0
            waste_input = waste_input[waste_model.feature_names_in_]

            # Get the scaled prediction
            waste_pred_scaled = waste_model.predict(waste_input)[0]
            
            # Inverse transform to get the actual prediction in original units
            waste_pred = waste_scaler.inverse_transform([[waste_pred_scaled]])[0][0]
            
            st.session_state["waste_pred"] = waste_pred
            st.success(translate_text(f"🔋 Predicted Waste Amount: {waste_pred:.2f} kg", lang_code))
        except Exception as e:
            st.error(translate_text(f"Error in waste prediction: {str(e)}", lang_code))
            st.session_state["waste_pred"] = predicted_yield_kg * 0.2  # Default 20% waste if model fails
else:
    st.warning(translate_text("⚠️ Please recommend a crop and predict yield first!", lang_code))

# ---------------------------
# 🌿 Composting Yield Prediction
# ---------------------------
# ---------------------------
# 🌿 Composting Yield Prediction
# ---------------------------
st.subheader(get_translation("compost_prediction", "other_sections"))
moisture = st.number_input(get_translation("moisture", "other_sections"), min_value=0, max_value=100, value=60)
compost_temp = st.number_input(get_translation("compost_temp", "other_sections"), min_value=0, max_value=50, value=35)
aeration = st.number_input(get_translation("aeration", "other_sections"), min_value=1, max_value=30, value=5)
duration = st.number_input(get_translation("duration", "other_sections"), min_value=1, max_value=20, value=8)

if st.button(get_translation("predict_compost", "other_sections"), key="compost_pred_btn"):
    if "waste_pred" in st.session_state:
        compost_input = pd.DataFrame([[st.session_state["waste_pred"], moisture, 
                                    compost_temp, aeration, duration]],
                    columns=['Waste_Amount_kg', 'Moisture_%', 'Temperature_°C', 
                            'Aeration_Frequency_days', 'Composting_Duration_weeks'])
        
        try:
            # Get the scaled prediction
            compost_yield_scaled = compost_model.predict(compost_input)[0]
            
            # Inverse transform to get the actual prediction in original units
            compost_yield = compost_scaler.inverse_transform([[compost_yield_scaled]])[0][0]
            
            st.session_state["compost_yield"] = compost_yield
            st.success(translate_text(f"🌿 Predicted Compost Yield: {compost_yield:.2f} kg", lang_code))
        except Exception as e:
            st.error(translate_text(f"Error in compost prediction: {str(e)}", lang_code))
            st.session_state["compost_yield"] = st.session_state["waste_pred"] * 0.6  # Default 60% conversion if model fails
