import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import requests
import matplotlib.pyplot as plt
import warnings
from utils import DataPreprocessor
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== API CONFIG ====================
# Đổi URL này nếu deploy API lên server khác (VD: http://192.168.1.10:8000)
API_BASE_URL = os.environ.get("CREDIT_API_URL", "http://localhost:8000")

def api_is_available() -> bool:
    """Kiểm tra API server có đang chạy không"""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Credit Risk Assessment System",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Credit Risk Scoring System v1.0"}
)

# ==================== PROFESSIONAL STYLING ====================
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary-color: #1e40af;
        --success-color: #16a34a;
        --warning-color: #ea580c;
        --danger-color: #dc2626;
        --gray-light: #f3f4f6;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Score card styling */
    .score-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .score-card.high-risk {
        border-left-color: #dc2626;
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
    }
    
    .score-card.low-risk {
        border-left-color: #16a34a;
        background: linear-gradient(135deg, #f0fdf4 0%, #bbf7d0 100%);
    }
    
    .score-card.medium-risk {
        border-left-color: #ea580c;
        background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%);
    }
    
    /* Risk score display */
    .risk-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Factor box */
    .factor-box {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        border-left: 3px solid #1e40af;
    }
    
    .factor-positive {
        border-left-color: #16a34a;
        background: #f0fdf4;
    }
    
    .factor-negative {
        border-left-color: #dc2626;
        background: #fef2f2;
    }
    
    .factor-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.3rem;
    }
    
    .factor-value {
        font-size: 0.95rem;
        color: #6b7280;
    }
    
    /* Recommendation box */
    .recommendation {
        background: linear-gradient(135deg, #e0e7ff 0%, #dbeafe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1e40af;
        margin-top: 1.5rem;
    }
    
    .recommendation h3 {
        margin-top: 0;
        color: #1e40af;
    }
    
    .recommendation-item {
        margin: 0.5rem 0;
        color: #1f2937;
    }
    
    /* Data table */
    .dataframe {
        font-size: 0.95rem;
    }
    
    /* Info box */
    .info-section {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD ASSETS ====================
import streamlit as st
import os
import joblib
from utils import DataPreprocessor  # Import trực tiếp class từ utils

@st.cache_resource
def load_assets():
    """Load preprocessor từ utils và chuẩn bị model fallback"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Dùng is_training=True để datapreprocessing() tự build schema cột
        # từ data người dùng upload, không cần train_features từ pkl.
        # Việc align cột với model (reindex theo FEATURES) được xử lý bởi api.py,
        # nên main.py không cần biết schema training.
        processor = DataPreprocessor(is_training=True)
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo preprocessor: {e}")
        st.stop()

    # Nếu API không khả dụng → fallback load model local
    if not api_is_available():  # Giả sử hàm này đã được định nghĩa ở đâu đó
        st.warning("⚠️ API server không phản hồi — đang dùng model local làm dự phòng")
        model_path = os.path.join(current_dir, '..', 'models', 'lgbm_best_model.pkl')
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"❌ Không thể load model local: {e}")
            st.stop()
        return model, processor

    return None, processor  # model=None → sẽ dùng API

@st.cache_resource
def load_shap_explainer(_model):
    """Khởi tạo SHAP explainer nếu có model local"""
    if _model is None:
        return None  # SHAP sẽ gọi qua API endpoint /shap
    try:
        return shap.TreeExplainer(_model)
    except Exception as e:
        st.warning(f"⚠️ Không thể tải SHAP Explainer: {e}")
        return None

# Load assets
model, processor = load_assets()
explainer = load_shap_explainer(model)
USE_API = model is None  # True → dùng API, False → dùng model local

# ==================== HELPER FUNCTIONS ====================
def load_if_exists(file, file_name=""):
    """Load CSV file if provided with better error handling"""
    if file is None:
        return None
    
    try:
        df = pd.read_csv(file, encoding='utf-8')
        if df.empty:
            st.warning(f"⚠️ File '{file_name}' trống hoặc không có dữ liệu")
            return None
        return df
    except UnicodeDecodeError:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin-1')
            if df.empty:
                st.warning(f"⚠️ File '{file_name}' trống hoặc không có dữ liệu")
                return None
            return df
        except Exception as e:
            st.error(f"❌ Lỗi đọc file '{file_name}': {str(e)}")
            return None
    except Exception as e:
        st.error(f"❌ Lỗi đọc file '{file_name}': {str(e)}")
        return None

# FIX 1: Removed @st.cache_data — DataFrames are not reliably hashable by Streamlit.
# Instead, predictions are computed once and stored in st.session_state.
def get_prediction_results(processed_df, features):
    """
    Dự đoán xác suất rủi ro.
    - Nếu USE_API=True  → gọi POST /predict trên API server
    - Nếu USE_API=False → dùng model local (fallback)
    """
    if USE_API:
        try:
            records = processed_df[features].fillna(0).to_dict(orient="records")
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"records": records, "features": features},
                timeout=60,
            )
            response.raise_for_status()
            return np.array(response.json()["probabilities"])
        except requests.exceptions.Timeout:
            st.error("❌ API timeout — server xử lý quá lâu. Thử lại hoặc giảm số lượng khách hàng.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Không kết nối được API tại {API_BASE_URL}. Kiểm tra server đang chạy chưa.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Lỗi gọi API: {str(e)}")
            st.stop()
    else:
        # Fallback local: reindex về đúng thứ tự cột model đã train
        # vì is_training=True nên processed_df có thể thiếu/thừa cột
        try:
            model_features = model.feature_name_
        except AttributeError:
            model_features = list(model.booster_.feature_name())
        X = processed_df.reindex(columns=model_features, fill_value=0).fillna(0).astype(float)
        return model.predict_proba(X)[:, 1]

def get_risk_category(prob, threshold_low=0.2, threshold_high=0.5):
    """Categorize risk level with 3 categories"""
    if prob < threshold_low:
        return "LOW", "🟢 Rủi ro Thấp"
    elif prob < threshold_high:
        return "MEDIUM", "🟡 Rủi ro Trung bình"
    else:
        return "HIGH", "🔴 Rủi ro Cao"

def generate_business_explanation(customer_data, probability, features):
    """Generate business-friendly explanation based on banking factors"""
    explanation = {
        "positive_factors": [],
        "negative_factors": [],
        "risk_assessment": "",
        "recommendations": []
    }

    # ── 1. Tỷ lệ Nợ/Thu nhập ──────────────────────────────────────────────────
    # Ưu tiên cột CREDIT_BY_INCOME (đã là ratio). Nếu không có thì bỏ qua —
    # tránh dùng AMT_CREDIT (số tiền tuyệt đối) như là ratio gây nhầm lẫn.
    ratio_col = next(
        (c for c in ['CREDIT_INCOME_PERCENT', 'CREDIT_BY_INCOME', 'AMT_CREDIT_BY_INCOME']
         if c in customer_data.columns and not pd.isna(customer_data[c].values[0])),
        None
    )
    if ratio_col:
        ratio_val = customer_data[ratio_col].values[0]
        if ratio_val < 5:
            explanation["positive_factors"].append(
                f"📊 Tỷ lệ Nợ/Thu nhập hợp lý ({ratio_val:.2f}x) — khả năng trả nợ tốt"
            )
        elif ratio_val <= 8:
            explanation["negative_factors"].append(
                f"⚠️ Tỷ lệ Nợ/Thu nhập ở mức cao ({ratio_val:.2f}x) — cần theo dõi thêm"
            )
        else:
            explanation["negative_factors"].append(
                f"🚨 Tỷ lệ Nợ/Thu nhập rất cao ({ratio_val:.2f}x) — gánh nặng nợ vượt mức an toàn"
            )

    # ── 2. Tỷ lệ Trả góp/Thu nhập ────────────────────────────────────────────
    annuity_col = next(
        (c for c in ['ANNUITY_INCOME_PERCENT', 'ANNUITY_BY_INCOME', 'AMT_ANNUITY_BY_INCOME']
         if c in customer_data.columns and not pd.isna(customer_data[c].values[0])),
        None
    )
    if annuity_col:
        ann_val = customer_data[annuity_col].values[0]
        if ann_val < 0.15:
            explanation["positive_factors"].append(
                f"💳 Tỷ lệ trả góp/thu nhập thấp ({ann_val:.1%}) — dòng tiền tháng ổn định"
            )
        elif ann_val > 0.35:
            explanation["negative_factors"].append(
                f"💳 Tỷ lệ trả góp/thu nhập cao ({ann_val:.1%}) — áp lực dòng tiền hàng tháng lớn"
            )

    # ── 3. Điểm tín dụng bên ngoài (EXT_SOURCE) — tổng hợp, không lặp ────────
    ext_score_cols = [c for c in features if 'EXT_SOURCE' in c and c in customer_data.columns]
    ext_vals = [customer_data[c].values[0] for c in ext_score_cols if not pd.isna(customer_data[c].values[0])]

    if ext_vals:
        avg_ext = np.mean(ext_vals)
        min_ext = np.min(ext_vals)
        scores_str = ", ".join(f"{v:.2f}" for v in ext_vals)

        if avg_ext >= 0.6:
            explanation["positive_factors"].append(
                f"✅ Điểm tín dụng bên ngoài tốt (TB: {avg_ext:.2f}/1.0) — lịch sử tín dụng lành mạnh"
            )
        elif avg_ext >= 0.4:
            explanation["negative_factors"].append(
                f"⚠️ Điểm tín dụng bên ngoài trung bình (TB: {avg_ext:.2f}/1.0; các nguồn: {scores_str}) — cần xem xét thêm"
            )
        else:
            explanation["negative_factors"].append(
                f"❌ Điểm tín dụng bên ngoài thấp (TB: {avg_ext:.2f}/1.0; các nguồn: {scores_str}) — lịch sử tín dụng yếu"
            )
    
    # ── 4. Nợ tại các tổ chức tín dụng khác (Bureau) ────────────────────────
    bureau_debt_cols = [
        c for c in features
        if c.startswith('BUREAU') and 'DEBT' in c and c in customer_data.columns
    ]
    bureau_debt_vals = [
        customer_data[c].values[0]
        for c in bureau_debt_cols
        if not pd.isna(customer_data[c].values[0])
    ]
    bureau_debt = sum(bureau_debt_vals) if bureau_debt_vals else 0

    # Số lượng khoản vay đang hoạt động tại bureau
    bureau_active_cols = [
        c for c in features
        if c.startswith('BUREAU') and 'ACTIVE' in c and c in customer_data.columns
    ]
    bureau_active = next(
        (customer_data[c].values[0] for c in bureau_active_cols
         if not pd.isna(customer_data[c].values[0])),
        None
    )

    if bureau_debt > 0:
        detail = f" ({bureau_active:.0f} khoản đang hoạt động)" if bureau_active else ""
        explanation["negative_factors"].append(
            f"🏦 Có dư nợ tại các tổ chức tín dụng khác{detail} — tổng dư nợ {bureau_debt:,.0f}"
        )
    else:
        explanation["positive_factors"].append(
            "🏦 Không có dư nợ tại các tổ chức tín dụng khác — tình trạng tín dụng sạch"
        )

    # ── 5. Tuổi khách hàng (chỉ lấy cột đầu tiên tìm được) ──────────────────
    age_col = next(
        (c for c in features if 'AGE' in c and c in customer_data.columns
         and not pd.isna(customer_data[c].values[0])),
        None
    )
    if age_col:
        age = customer_data[age_col].values[0]
        if 30 <= age <= 55:
            explanation["positive_factors"].append(
                f"👤 Tuổi {int(age)} — độ tuổi lao động ổn định, năng lực trả nợ cao"
            )
        elif age < 25:
            explanation["negative_factors"].append(
                f"👤 Tuổi {int(age)} — còn trẻ, lịch sử tín dụng còn hạn chế"
            )
        elif age > 60:
            explanation["negative_factors"].append(
                f"👤 Tuổi {int(age)} — gần độ tuổi nghỉ hưu, cần xem xét khả năng trả nợ dài hạn"
            )

    # ── 6. Lịch sử thanh toán (nếu có) ───────────────────────────────────────
    late_cols = [
        c for c in features
        if any(k in c for k in ['DAYS_PAST_DUE', 'DPD', 'LATE'])
        and c in customer_data.columns
    ]
    late_vals = [
        customer_data[c].values[0] for c in late_cols
        if not pd.isna(customer_data[c].values[0]) and customer_data[c].values[0] > 0
    ]
    if late_vals:
        explanation["negative_factors"].append(
            f"⏰ Có lịch sử thanh toán trễ hạn — {len(late_vals)} chỉ số quá hạn được ghi nhận"
        )

    # ── 7. Đánh giá tổng thể & khuyến nghị ──────────────────────────────────
    n_pos = len(explanation["positive_factors"])
    n_neg = len(explanation["negative_factors"])

    if probability < 0.2:
        explanation["risk_assessment"] = (
            f"Khách hàng có hồ sơ tài chính tốt với {n_pos} yếu tố tích cực"
            + (f" và {n_neg} điểm cần lưu ý" if n_neg else "")
            + ". Xác suất vỡ nợ thấp, đủ điều kiện cấp tín dụng."
        )
        explanation["recommendations"] = [
            "✅ Phê duyệt cấp tín dụng — có thể áp dụng lãi suất ưu đãi",
            "💳 Hạn mức đề xuất phù hợp với thu nhập và nhu cầu thực tế",
            "📈 Xem xét nâng hạn mức trong kỳ đánh giá tiếp theo nếu thanh toán đúng hạn",
        ]
    elif probability < 0.5:
        explanation["risk_assessment"] = (
            f"Khách hàng có mức rủi ro trung bình ({n_pos} yếu tố tích cực, {n_neg} yếu tố tiêu cực). "
            "Cần thẩm định bổ sung trước khi quyết định."
        )
        explanation["recommendations"] = [
            "⚠️ Phê duyệt có điều kiện — yêu cầu bổ sung tài liệu chứng minh thu nhập",
            "📋 Xác minh lịch sử tín dụng tại các ngân hàng và tổ chức tín dụng khác",
            "🔒 Cân nhắc yêu cầu tài sản đảm bảo hoặc người bảo lãnh",
            "📉 Đề xuất hạn mức thấp hơn mức yêu cầu để kiểm soát rủi ro",
        ]
    else:
        explanation["risk_assessment"] = (
            f"Khách hàng có rủi ro tín dụng cao ({n_neg} yếu tố tiêu cực nổi bật). "
            "Hồ sơ hiện tại chưa đủ điều kiện cấp tín dụng."
        )
        explanation["recommendations"] = [
            "❌ Từ chối cấp tín dụng ở thời điểm hiện tại",
            "📞 Liên hệ khách hàng để tư vấn cải thiện hồ sơ tín dụng",
            "📄 Yêu cầu cung cấp thêm tài liệu về thu nhập và tài sản nếu muốn tái xét",
            "⏰ Đề nghị tái đánh giá sau 6–12 tháng khi các chỉ số được cải thiện",
        ]

    return explanation

def plot_shap_for_customer(shap_values, X_data, feature_names, max_features=10):
    """Create SHAP bar plot for a single customer"""
    try:
        if isinstance(shap_values, list):
            actual_shap = shap_values[1]
        else:
            actual_shap = shap_values
        
        if isinstance(actual_shap, np.ndarray):
            if actual_shap.ndim == 2:
                actual_shap = actual_shap[0]
        else:
            actual_shap = np.array(actual_shap)
        
        shap_abs = np.abs(actual_shap)
        top_indices = np.argsort(shap_abs)[-max_features:][::-1]
        
        values = actual_shap[top_indices]
        names = [feature_names[i] for i in top_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#dc2626' if v > 0 else '#16a34a' for v in values]
        ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("SHAP Value (Tác động đến xác suất rủi ro)", fontsize=10)
        ax.set_title("Top 10 Yếu tố Ảnh hưởng Nhất", fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Lỗi vẽ SHAP: {str(e)}")
        return None

# ==================== MAIN APPLICATION ====================

st.markdown("""
<div class="main-header">
    <h1>💰 Hệ Thống Đánh Giá Rủi Ro Tín Dụng</h1>
    <p>Phân tích và dự đoán rủi ro nợ xấu bằng Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR: FILE UPLOAD ====================
st.sidebar.title("📁 Tải lên Dữ liệu")
st.sidebar.markdown("---")

# Hiển thị trạng thái API
with st.sidebar:
    if api_is_available():
        st.success(f"🟢 API Online  \n`{API_BASE_URL}`")
    else:
        st.warning(f"🟡 API Offline — dùng model local  \n`{API_BASE_URL}`")
st.sidebar.markdown("---")
st.sidebar.markdown("**Hướng dẫn:** Tải lên các file CSV của bạn. Chỉ cần file 'Application' là bắt buộc.")

app_file = st.sidebar.file_uploader("📄 Application (Bắt buộc)*", type="csv", key="app")
st.sidebar.markdown("**Các file bổ trợ (tùy chọn):**")

bureau_file = st.sidebar.file_uploader("Bureau", type="csv", key="bureau")
bureau_bal_file = st.sidebar.file_uploader("Bureau Balance", type="csv", key="bureau_bal")
prev_app_file = st.sidebar.file_uploader("Previous Application", type="csv", key="prev_app")
inst_file = st.sidebar.file_uploader("Installments Payments", type="csv", key="inst")
pos_file = st.sidebar.file_uploader("POS CASH Balance", type="csv", key="pos")
cc_file = st.sidebar.file_uploader("Credit Card Balance", type="csv", key="cc")

if app_file is not None:
    try:
        st.info(f"📄 File được tải: {app_file.name} ({app_file.size} bytes)")
        
        df_app = pd.read_csv(app_file, encoding='utf-8')
        
        if df_app.empty:
            st.error("❌ File Application trống hoặc không có dữ liệu")
            st.stop()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Số khách hàng", f"{df_app.shape[0]:,}")
        with col2:
            st.metric("📈 Số trường dữ liệu", f"{df_app.shape[1]}")
        with col3:
            files_loaded = sum([f is not None for f in [bureau_file, bureau_bal_file, prev_app_file, inst_file, pos_file, cc_file]])
            st.metric("📁 File bổ trợ", f"{files_loaded}/6")
        
        st.success("✅ Dữ liệu đã sẵn sàng để xử lý")
        
    except pd.errors.EmptyDataError:
        st.error("❌ Lỗi đọc file Application: File trống hoặc không có dữ liệu")
        st.stop()
    except UnicodeDecodeError:
        st.error("❌ Lỗi đọc file Application: Mã hóa file không hợp lệ. Vui lòng kiểm tra file là UTF-8 hoặc ANSI")
        st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi đọc file Application: {str(e)}")
        st.info("💡 Gợi ý: Kiểm tra file CSV có đúng định dạng, không trống, và không bị lỗi")
        st.stop()

    df_bureau = load_if_exists(bureau_file, "Bureau")
    df_bureau_bal = load_if_exists(bureau_bal_file, "Bureau Balance")
    df_prev = load_if_exists(prev_app_file, "Previous Application")
    df_inst = load_if_exists(inst_file, "Installments Payments")
    df_pos = load_if_exists(pos_file, "POS CASH Balance")
    df_cc = load_if_exists(cc_file, "Credit Card Balance")

    if st.button("🚀 Bắt Đầu Phân Tích Rủi Ro", use_container_width=True, key="predict_btn"):
        with st.spinner("⏳ Đang xử lý và phân tích dữ liệu... Vui lòng chờ"):
            try:
                processed_df = processor.datapreprocessing(
                    app_data=df_app,
                    bureau=df_bureau,
                    bureau_balance=df_bureau_bal,
                    prev_app=df_prev,
                    inst=df_inst,
                    pos=df_pos,
                    cc=df_cc
                )
                
                # FIX 2: Reset index after preprocessing to ensure iloc and index-based
                # lookups are consistent throughout the application.
                processed_df = processed_df.reset_index(drop=True)
                
                features = [c for c in processed_df.columns if c not in ['TARGET', 'SK_ID_CURR']]
                probs = get_prediction_results(processed_df, features)
                
                result_df = pd.DataFrame({
                    'SK_ID_CURR': processed_df['SK_ID_CURR'],
                    'Xác suất rủi ro': probs,
                    'Phân loại': [get_risk_category(p)[1] for p in probs]
                # FIX 3: Reset result_df index too so it aligns with processed_df.
                }).reset_index(drop=True)
                
                st.session_state.result_df = result_df
                st.session_state.processed_df = processed_df
                st.session_state.features = features
                st.session_state.probs = probs
                
                st.success("✅ Phân tích hoàn thành!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Lỗi xử lý dữ liệu: {str(e)}")
                st.stop()

    # FIX 4: Use `'result_df' in st.session_state` instead of `hasattr(...)`.
    # hasattr on st.session_state is unreliable — the recommended Streamlit pattern
    # is the `in` operator.
    if 'result_df' in st.session_state:
        st.markdown("---")
        
        st.subheader("📊 Tóm Tắt Kết Quả Phân Tích")
        col1, col2, col3, col4 = st.columns(4)
        
        high_risk = (st.session_state.probs > 0.5).sum()
        medium_risk = ((st.session_state.probs >= 0.2) & (st.session_state.probs <= 0.5)).sum()
        low_risk = (st.session_state.probs < 0.2).sum()
        
        with col1:
            st.metric("🔴 Rủi ro Cao", f"{high_risk:,}", f"{(high_risk/len(st.session_state.probs)*100):.1f}%")
        with col2:
            st.metric("🟡 Rủi ro Trung bình", f"{medium_risk:,}", f"{(medium_risk/len(st.session_state.probs)*100):.1f}%")
        with col3:
            st.metric("🟢 Rủi ro Thấp", f"{low_risk:,}", f"{(low_risk/len(st.session_state.probs)*100):.1f}%")
        with col4:
            st.metric("📈 Xác suất TB", f"{st.session_state.probs.mean():.2%}")
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Danh sách kết quả", "🔍 Phân tích Chi tiết", "📊 SHAP Analysis", "📥 Tải xuống"])
        
        with tab1:
            st.subheader("Kết Quả Dự Đoán")
            
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sắp xếp theo:", ["Xác suất rủi ro (Cao → Thấp)", "Xác suất rủi ro (Thấp → Cao)", "ID Khách hàng"])
            with col2:
                filter_risk = st.selectbox("Lọc theo:", ["Tất cả", "Rủi ro Cao", "Rủi ro Trung bình", "Rủi ro Thấp"])
            
            display_df = st.session_state.result_df.copy()
            
            if filter_risk == "Rủi ro Cao":
                display_df = display_df[display_df['Xác suất rủi ro'] > 0.5]
            elif filter_risk == "Rủi ro Trung bình":
                display_df = display_df[(display_df['Xác suất rủi ro'] >= 0.2) & (display_df['Xác suất rủi ro'] <= 0.5)]
            elif filter_risk == "Rủi ro Thấp":
                display_df = display_df[display_df['Xác suất rủi ro'] < 0.2]
            
            if "Cao → Thấp" in sort_by:
                display_df = display_df.sort_values('Xác suất rủi ro', ascending=False)
            elif "Thấp → Cao" in sort_by:
                display_df = display_df.sort_values('Xác suất rủi ro', ascending=True)
            else:
                display_df = display_df.sort_values('SK_ID_CURR')
            
            display_df['Xác suất rủi ro'] = display_df['Xác suất rủi ro'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(
                display_df.head(50),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'SK_ID_CURR': st.column_config.TextColumn('ID Khách hàng', width=150),
                    'Xác suất rủi ro': st.column_config.TextColumn('Xác suất Rủi ro', width=150),
                    'Phân loại': st.column_config.TextColumn('Phân loại', width=150)
                }
            )
            
            st.info(f"Hiển thị {len(display_df)} / {len(st.session_state.result_df)} khách hàng")
        
        with tab2:
            st.subheader("Phân Tích Chi Tiết Khách Hàng")
            
            customer_id = st.selectbox(
                "Chọn khách hàng để xem chi tiết:",
                st.session_state.result_df['SK_ID_CURR'],
                format_func=lambda x: f"ID: {x}"
            )
            
            if customer_id:
                # FIX 5: Use .loc to find position in result_df, then map to processed_df
                # via positional index — avoids index mismatch after sorting/filtering.
                mask = st.session_state.result_df['SK_ID_CURR'] == customer_id
                row_pos = st.session_state.result_df.index[mask][0]
                score = st.session_state.probs[row_pos]
                risk_cat, risk_label = get_risk_category(score)
                
                risk_class = "high-risk" if risk_cat == "HIGH" else ("medium-risk" if risk_cat == "MEDIUM" else "low-risk")
                st.markdown(f"""
                <div class="score-card {risk_class}">
                    <div style="text-align: center;">
                        <div style="font-size: 0.9rem; color: #666;">Đánh Giá Rủi Ro</div>
                        <div class="risk-score">{score:.1%}</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">{risk_label}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("📋 Phân Tích Kinh Doanh")
                explanation = generate_business_explanation(
                    st.session_state.processed_df.iloc[[row_pos]],
                    score,
                    st.session_state.features
                )
                
                st.markdown(f"""
                <div class="info-section">
                    <h4>🎯 Đánh Giá Rủi Ro</h4>
                    <p>{explanation['risk_assessment']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if explanation["positive_factors"]:
                    st.markdown("### ✅ Các Yếu Tố Tích Cực")
                    for factor in explanation["positive_factors"]:
                        st.markdown(f'<div class="factor-box factor-positive"><div class="factor-title">{factor}</div></div>', unsafe_allow_html=True)
                
                if explanation["negative_factors"]:
                    st.markdown("### ❌ Các Yếu Tố Tiêu Cực")
                    for factor in explanation["negative_factors"]:
                        st.markdown(f'<div class="factor-box factor-negative"><div class="factor-title">{factor}</div></div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="recommendation">
                    <h3>💼 Khuyến Nghị Hành Động</h3>
                    {"".join([f'<div class="recommendation-item">{rec}</div>' for rec in explanation["recommendations"]])}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("📊 Chỉ Số Chi Tiết")
                
                metric_cols = st.columns(2)
                with metric_cols[0]:
                    st.metric("Xác suất rủi ro", f"{score:.2%}", delta=f"{(score - 0.45)*100:.1f}pp từ ngưỡng")
                with metric_cols[1]:
                    st.metric("Độ tin cậy", f"{max(score, 1-score):.2%}")
        
        with tab3:
            st.subheader("🔬 Phân Tích SHAP - Yếu Tố Kỹ Thuật")
            st.info("SHAP (SHapley Additive exPlanations) giải thích tác động của từng biến đến dự đoán")

            # Hiển thị nguồn đang dùng
            if USE_API:
                st.caption(f"🌐 Đang dùng API: `{API_BASE_URL}/shap`")
            else:
                st.caption("💻 Đang dùng model local")

            customer_id = st.selectbox(
                "Chọn khách hàng:",
                st.session_state.result_df['SK_ID_CURR'],
                format_func=lambda x: f"ID: {x}",
                key="shap_selectbox"
            )

            shap_available = USE_API or (explainer is not None)

            if customer_id and shap_available:
                mask = st.session_state.result_df['SK_ID_CURR'] == customer_id
                row_pos = st.session_state.result_df.index[mask][0]

                with st.spinner("Đang tính toán SHAP values..."):
                    try:
                        X_customer = st.session_state.processed_df.iloc[[row_pos]][st.session_state.features]
                        X_customer_numeric = X_customer.fillna(0).astype(float)

                        if USE_API:
                            # Gọi endpoint /shap trên API server
                            record = X_customer_numeric.iloc[0].to_dict()
                            response = requests.post(
                                f"{API_BASE_URL}/shap",
                                json={
                                    "record": record,
                                    "features": st.session_state.features,
                                    "top_n": 10,
                                },
                                timeout=30,
                            )
                            response.raise_for_status()
                            shap_data = response.json()

                            # Vẽ từ dữ liệu API trả về
                            fig, ax = plt.subplots(figsize=(10, 6))
                            values = shap_data["shap_values"]
                            names = shap_data["feature_names"]
                            colors = ['#dc2626' if v > 0 else '#16a34a' for v in values]
                            ax.barh(range(len(names)), values, color=colors)
                            ax.set_yticks(range(len(names)))
                            ax.set_yticklabels(names, fontsize=9)
                            ax.set_xlabel("SHAP Value (Tác động đến xác suất rủi ro)", fontsize=10)
                            ax.set_title("Top 10 Yếu tố Ảnh hưởng Nhất", fontsize=11, fontweight='bold')
                            ax.invert_yaxis()
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                        else:
                            # Dùng explainer local
                            shap_values = explainer.shap_values(X_customer_numeric)
                            fig = plot_shap_for_customer(
                                shap_values, X_customer_numeric,
                                st.session_state.features, max_features=10
                            )
                            if fig:
                                st.pyplot(fig, use_container_width=True)
                            else:
                                st.info("Không thể tạo biểu đồ SHAP cho khách hàng này.")

                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Lỗi gọi API SHAP: {str(e)}")
                    except Exception as e:
                        st.error(f"Lỗi SHAP: {str(e)}")

            elif not shap_available:
                st.warning("⚠️ SHAP Explainer không khả dụng (API offline và không có model local)")
        
        with tab4:
            st.subheader("📥 Tải Xuống Kết Quả")
            
            csv = st.session_state.result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Tải toàn bộ kết quả (CSV)",
                csv,
                f"credit_risk_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
            
            summary_data = {
                'Chỉ Số': [
                    'Tổng khách hàng',
                    'Khách hàng rủi ro cao (>50%)',
                    'Khách hàng rủi ro trung bình (20%-50%)',
                    'Khách hàng rủi ro thấp (<20%)',
                    'Xác suất rủi ro trung bình',
                    'Xác suất rủi ro cao nhất',
                    'Xác suất rủi ro thấp nhất'
                ],
                'Giá Trị': [
                    len(st.session_state.result_df),
                    (st.session_state.probs > 0.5).sum(),
                    ((st.session_state.probs >= 0.2) & (st.session_state.probs <= 0.5)).sum(),
                    (st.session_state.probs < 0.2).sum(),
                    f"{st.session_state.probs.mean():.2%}",
                    f"{st.session_state.probs.max():.2%}",
                    f"{st.session_state.probs.min():.2%}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            csv_summary = summary_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                "📊 Tải tóm tắt (CSV)",
                csv_summary,
                f"credit_risk_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )

else:
    st.markdown("""
    <div class="info-section">
        <h3>🚀 Bắt Đầu Sử Dụng</h3>
        <p><strong>Bước 1:</strong> Tải file 'Application' từ thanh bên trái (bắt buộc)</p>
        <p><strong>Bước 2:</strong> (Tùy chọn) Tải các file bổ trợ khác để có kết quả chính xác hơn</p>
        <p><strong>Bước 3:</strong> Nhấn nút 'Bắt Đầu Phân Tích Rủi Ro' để xem kết quả</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-section">
        <h3>📚 Hướng Dẫn Sử Dụng</h3>
        <ul>
            <li><strong>Danh sách kết quả:</strong> Xem toàn bộ khách hàng với phân loại rủi ro</li>
            <li><strong>Phân tích chi tiết:</strong> Nhận lời giải thích kinh doanh chi tiết cho từng khách hàng</li>
            <li><strong>SHAP Analysis:</strong> Xem tác động kỹ thuật của từng yếu tố</li>
            <li><strong>Tải xuống:</strong> Xuất kết quả dưới dạng CSV để sử dụng tiếp</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)