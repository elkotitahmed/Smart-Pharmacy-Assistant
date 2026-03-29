import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ------------------- إعدادات الصفحة -------------------
st.set_page_config(
    page_title="Smart Pharmacy Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- الشعار والعنوان -------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # إذا كان لديك صورة logo.png في نفس المجلد، أضفها
    # st.image("logo.png", width=150)
    pass

st.markdown("<h1 style='text-align: center;'>💊 Smart Pharmacy Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>نظام ذكي لاقتراح الدواء المناسب باستخدام الذكاء الاصطناعي وبيانات حقيقية من السوق المصري</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------- تحميل البيانات -------------------
@st.cache_data
def load_training_data():
    df = pd.read_csv("data/drug_training.csv")
    # التأكد من الأعمدة المطلوبة
    required_cols = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"العمود {col} غير موجود في ملف التدريب!")
            st.stop()
    return df

@st.cache_data
def load_products_data():
    df = pd.read_csv("data/egyptian_drugs.csv")
    return df

df_train = load_training_data()
df_products = load_products_data()

# ------------------- تدريب النموذج -------------------
# ترميز الأعمدة النصية
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_drug = LabelEncoder()

df_train['Sex'] = le_sex.fit_transform(df_train['Sex'])
df_train['BP'] = le_bp.fit_transform(df_train['BP'])
df_train['Cholesterol'] = le_chol.fit_transform(df_train['Cholesterol'])
df_train['Drug'] = le_drug.fit_transform(df_train['Drug'])

X = df_train.drop('Drug', axis=1)
y = df_train['Drug']

# تقسيم البيانات (اختياري، يمكن استخدام كل البيانات للتدريب)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# تقييم الدقة (اختياري)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# لا نعرض الدقة في الواجهة، لكن نحتفظ بها للفحص

# ------------------- واجهة التطبيق (تبويبات) -------------------
tab1, tab2, tab3 = st.tabs(["🔮 توقع الدواء", "🔍 البحث عن دواء", "ℹ️ عن المشروع"])

with tab1:
    st.header("🏥 أدخل بيانات المريض")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("العمر", 1, 100, 45)
        sex = st.selectbox("الجنس", ["M", "F"])
        bp = st.selectbox("ضغط الدم", ["LOW", "NORMAL", "HIGH"])
    with col2:
        cholesterol = st.selectbox("الكوليسترول", ["NORMAL", "HIGH"])
        na_to_k = st.slider("نسبة الصوديوم إلى البوتاسيوم (Na_to_K)", 5.0, 30.0, 15.0)

    if st.button("🔍 توقع الدواء", type="primary"):
        # تحويل المدخلات
        sex_enc = le_sex.transform([sex])[0]
        bp_enc = le_bp.transform([bp])[0]
        chol_enc = le_chol.transform([cholesterol])[0]

        input_data = pd.DataFrame([[age, sex_enc, bp_enc, chol_enc, na_to_k]],
                                  columns=X.columns)
        pred_enc = model.predict(input_data)[0]
        predicted_drug = le_drug.inverse_transform([pred_enc])[0]

        # البحث عن الدواء في ملف المنتجات الكبير (للحصول على السعر والعبوة)
        product_info = df_products[df_products['name'].str.contains(predicted_drug, case=False, na=False)]

        if not product_info.empty:
            info = product_info.iloc[0]
            st.success(f"### 💊 الدواء المقترح: **{predicted_drug}**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("السعر التقريبي (جنيه)", info.get('price', 'غير متوفر'))
            with col_b:
                st.info(f"**العبوة:** {info.get('packaging', 'غير متوفر')}")
        else:
            st.success(f"### 💊 الدواء المقترح: **{predicted_drug}**")
            st.info("لم يتم العثور على معلومات إضافية عن هذا الدواء في قاعدة البيانات.")

        # عرض البدائل والتحذيرات من ملف التدريب (إن وجدت)
        training_info = df_train[df_train['Drug'] == predicted_drug]
        if not training_info.empty:
            tinfo = training_info.iloc[0]
            if 'Alternatives' in df_train.columns and pd.notna(tinfo.get('Alternatives')):
                st.write(f"**بدائل:** {tinfo['Alternatives']}")
            if 'Warnings' in df_train.columns and pd.notna(tinfo.get('Warnings')):
                st.warning(f"**تحذيرات:** {tinfo['Warnings']}")

        st.balloons()

with tab2:
    st.header("🔍 ابحث عن دواء في السوق المصري")
    search_term = st.text_input("أدخل اسم الدواء (أو جزء منه)", placeholder="مثال: Amlodipine")
    if search_term:
        results = df_products[df_products['name'].str.contains(search_term, case=False, na=False)]
        if not results.empty:
            st.write(f"عدد النتائج: {len(results)}")
            for idx, row in results.head(10).iterrows():
                with st.expander(f"💊 {row['name']}"):
                    st.write(f"**السعر:** {row.get('price', 'غير متوفر')} جنيه")
                    st.write(f"**العبوة:** {row.get('packaging', 'غير متوفر')}")
                    if 'discount_percentage' in row and pd.notna(row['discount_percentage']):
                        st.write(f"**خصم:** {row['discount_percentage']}")
        else:
            st.warning("لم يتم العثور على منتج بهذا الاسم.")

with tab3:
    st.header("ℹ️ عن المشروع")
    st.markdown("""
    **Smart Pharmacy Assistant** هو نظام ذكي يهدف إلى مساعدة الصيادلة والأطباء في:
    - اقتراح الدواء المناسب بناءً على بيانات المريض (العمر، الجنس، ضغط الدم، الكوليسترول، نسبة Na/K).
    - عرض معلومات إضافية عن الدواء مثل السعر والعبوة باستخدام بيانات حقيقية من السوق المصري.
    
    **التقنيات المستخدمة:**
    - Python
    - Streamlit (لواجهة المستخدم)
    - Scikit-learn (لتدريب نموذج Decision Tree)
    - Pandas (لمعالجة البيانات)
    
    **المطور:** [اسمك]
    **تاريخ الإصدار:** مارس 2026
    """)

# ------------------- الشريط الجانبي -------------------
with st.sidebar:
    st.markdown("## 📊 إحصائيات سريعة")
    st.metric("عدد الأدوية في قاعدة البيانات", len(df_products))
    st.metric("عدد الأدوية المستخدمة في التدريب", len(df_train))
    st.markdown("---")
    st.markdown("### 🧪 عن النموذج")
    st.markdown("نوع النموذج: **Decision Tree Classifier**")
    st.markdown(f"عدد العينات في التدريب: {len(df_train)}")
    st.markdown(f"دقة النموذج على بيانات الاختبار: {accuracy:.2%}")