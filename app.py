import re
import ssl
import socket
import requests
import ipaddress
import whois
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing Website Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem; font-weight: 800;
        color: #1F4E79; text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem; color: #555;
        text-align: center; margin-bottom: 2rem;
    }
    .result-phishing {
        background: linear-gradient(135deg, #FDEDEC, #F5B7B1);
        border-left: 6px solid #E74C3C; border-radius: 10px;
        padding: 24px; text-align: center;
        font-size: 1.8rem; font-weight: bold; color: #922B21;
    }
    .result-legit {
        background: linear-gradient(135deg, #EAFAF1, #A9DFBF);
        border-left: 6px solid #2ECC71; border-radius: 10px;
        padding: 24px; text-align: center;
        font-size: 1.8rem; font-weight: bold; color: #1D8348;
    }
    .model-card {
        background: #F8F9FA; border-radius: 10px;
        padding: 18px; text-align: center;
        border: 1px solid #DEE2E6;
    }
    .feature-safe   { color: #1D8348; font-weight: bold; }
    .feature-danger { color: #922B21; font-weight: bold; }
    .feature-warn   { color: #D35400; font-weight: bold; }
    .url-box {
        background: #F0F4FA; border-radius: 10px;
        padding: 16px; border-left: 5px solid #1F4E79;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1F4E79, #2E86C1);
        color: white; font-weight: bold; font-size: 1.1rem;
        padding: 0.65rem 2rem; border-radius: 8px;
        border: none; width: 100%;
    }
    .section-header {
        font-size: 1.35rem; font-weight: 700; color: #1F4E79;
        border-bottom: 3px solid #AED6F1;
        padding-bottom: 6px; margin-top: 1.4rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR — extracts the same 30 features as UCI dataset from a URL
# ══════════════════════════════════════════════════════════════════════════════
SHORTENERS = [
    'bit.ly','goo.gl','shorte.st','go2l.ink','x.co','ow.ly','t.co',
    'tinyurl.com','tr.im','is.gd','cli.gs','yfrog.com','migre.me',
    'ff.im','tiny.cc','url4.eu','twit.ac','su.pr','twurl.nl','snipurl.com',
    'short.to','BudURL.com','ping.fm','post.ly','Just.as','bkite.com',
    'snipr.com','fic.kr','loopt.us','doiop.com','short.ie','kl.am',
    'wp.me','rubyurl.com','om.ly','to.ly','bit.do','t.co','lnkd.in',
    'db.tt','qr.ae','adf.ly','bitly.com','cur.lv','tinyurl.com','ity.im',
    'q.gs','po.st','bc.vc','twitthis.com','u.to','j.mp','buzurl.com',
    'cutt.us','u.bb','yourls.org','prettylinkpro.com','scrnch.me',
    'filoops.info','vzturl.com','qr.net','1url.com','tweez.me','v.gd',
    'tr.im','link.zip.net'
]

def extract_features(url):
    """Extract all 30 UCI phishing features from a URL."""
    features = {}
    
    # Ensure URL has scheme
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'https://' + url

    parsed   = urlparse(url)
    domain   = parsed.netloc or parsed.path.split('/')[0]
    hostname = domain.replace('www.', '')

    # ── Try to fetch the page ─────────────────────────────────────────────────
    html_content = ""
    soup         = None
    response     = None
    try:
        headers  = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=8, headers=headers, verify=False, allow_redirects=True)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception:
        pass

    # 1. having_IP_Address — IP address used as domain?
    try:
        ipaddress.ip_address(hostname)
        features['having_IP_Address'] = -1
    except ValueError:
        features['having_IP_Address'] = 1

    # 2. URL_Length
    url_len = len(url)
    if url_len < 54:
        features['URL_Length'] = 1
    elif url_len <= 75:
        features['URL_Length'] = 0
    else:
        features['URL_Length'] = -1

    # 3. Shortining_Service
    features['Shortining_Service'] = -1 if any(s in domain for s in SHORTENERS) else 1

    # 4. having_At_Symbol
    features['having_At_Symbol'] = -1 if '@' in url else 1

    # 5. double_slash_redirecting — // after position 7
    features['double_slash_redirecting'] = -1 if '//' in url[7:] else 1

    # 6. Prefix_Suffix — dash in domain
    features['Prefix_Suffix'] = -1 if '-' in hostname else 1

    # 7. having_Sub_Domain — count dots
    dots = hostname.count('.')
    if dots == 1:
        features['having_Sub_Domain'] = 1
    elif dots == 2:
        features['having_Sub_Domain'] = 0
    else:
        features['having_Sub_Domain'] = -1

    # 8. SSLfinal_State — HTTPS + valid cert
    if parsed.scheme == 'https':
        try:
            ctx  = ssl.create_default_context()
            conn = ctx.wrap_socket(socket.socket(), server_hostname=hostname)
            conn.settimeout(5)
            conn.connect((hostname, 443))
            cert     = conn.getpeercert()
            exp_date = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
            age_days = (exp_date - datetime.now()).days
            features['SSLfinal_State'] = 1 if age_days > 0 else -1
            conn.close()
        except Exception:
            features['SSLfinal_State'] = 0
    else:
        features['SSLfinal_State'] = -1

    # 9. Domain_registeration_length — from WHOIS
    try:
        w = whois.whois(hostname)
        exp = w.expiration_date
        if isinstance(exp, list):
            exp = exp[0]
        if exp:
            remaining = (exp - datetime.now()).days
            features['Domain_registeration_length'] = 1 if remaining > 365 else -1
        else:
            features['Domain_registeration_length'] = -1
    except Exception:
        features['Domain_registeration_length'] = -1

    # 10. Favicon — from external domain?
    if soup:
        favicon_tag = soup.find('link', rel=lambda x: x and 'icon' in x)
        if favicon_tag and favicon_tag.get('href'):
            fav_url  = favicon_tag['href']
            fav_dom  = urlparse(fav_url).netloc
            features['Favicon'] = -1 if (fav_dom and fav_dom != domain) else 1
        else:
            features['Favicon'] = 1
    else:
        features['Favicon'] = 0

    # 11. port — non-standard port
    port = parsed.port
    standard_ports = [80, 443, 21, 22, 23, 25, 53, 110, 143]
    if port is None:
        features['port'] = 1
    elif port in standard_ports:
        features['port'] = 0
    else:
        features['port'] = -1

    # 12. HTTPS_token — "https" word in domain part
    features['HTTPS_token'] = -1 if 'https' in hostname.lower() else 1

    # 13. Request_URL — external objects percentage
    if soup:
        total = 0
        external = 0
        for tag in soup.find_all(['img', 'video', 'audio']):
            src = tag.get('src', '')
            if src:
                total += 1
                if domain not in src and src.startswith('http'):
                    external += 1
        if total == 0:
            features['Request_URL'] = 1
        else:
            pct = external / total * 100
            features['Request_URL'] = 1 if pct < 22 else (0 if pct <= 61 else -1)
    else:
        features['Request_URL'] = 0

    # 14. URL_of_Anchor — anchor tags with external/empty links
    if soup:
        anchors  = soup.find_all('a', href=True)
        total_a  = len(anchors)
        unsafe_a = 0
        for a in anchors:
            href = a['href']
            if href in ['#', '#content', '#skip', 'javascript::void(0)'] or \
               (href.startswith('http') and domain not in href):
                unsafe_a += 1
        if total_a == 0:
            features['URL_of_Anchor'] = 1
        else:
            pct = unsafe_a / total_a * 100
            features['URL_of_Anchor'] = 1 if pct < 31 else (0 if pct <= 67 else -1)
    else:
        features['URL_of_Anchor'] = 0

    # 15. Links_in_tags — meta/script/link tags with external URLs
    if soup:
        tag_links = soup.find_all(['meta', 'script', 'link'])
        total_t   = len(tag_links)
        ext_t     = 0
        for t in tag_links:
            src = t.get('src') or t.get('href') or ''
            if src.startswith('http') and domain not in src:
                ext_t += 1
        if total_t == 0:
            features['Links_in_tags'] = 1
        else:
            pct = ext_t / total_t * 100
            features['Links_in_tags'] = 1 if pct < 17 else (0 if pct <= 81 else -1)
    else:
        features['Links_in_tags'] = 0

    # 16. SFH — Server Form Handler
    if soup:
        forms = soup.find_all('form', action=True)
        sfh_vals = []
        for form in forms:
            action = form['action']
            if action in ['', 'about:blank']:
                sfh_vals.append(-1)
            elif action.startswith('http') and domain not in action:
                sfh_vals.append(0)
            else:
                sfh_vals.append(1)
        features['SFH'] = min(sfh_vals) if sfh_vals else 1
    else:
        features['SFH'] = 0

    # 17. Submitting_to_email — mailto in form action
    if soup:
        forms  = soup.find_all('form', action=True)
        mailto = any('mailto:' in f['action'] for f in forms)
        features['Submitting_to_email'] = -1 if mailto else 1
    else:
        features['Submitting_to_email'] = 1

    # 18. Abnormal_URL — hostname in URL
    features['Abnormal_URL'] = 1 if hostname in url else -1

    # 19. Redirect — number of redirects
    if response:
        num_redirects = len(response.history)
        features['Redirect'] = 1 if num_redirects <= 1 else (0 if num_redirects <= 4 else -1)
    else:
        features['Redirect'] = 0

    # 20. on_mouseover — status bar changes
    features['on_mouseover'] = -1 if html_content and 'onmouseover' in html_content.lower() else 1

    # 21. RightClick — right click disabled
    features['RightClick'] = -1 if html_content and 'contextmenu' in html_content.lower() else 1

    # 22. popUpWidnow — popup with text fields
    features['popUpWidnow'] = -1 if html_content and 'window.open' in html_content else 1

    # 23. Iframe — iframe tag present
    features['Iframe'] = -1 if soup and soup.find('iframe') else 1

    # 24. age_of_domain — domain age in months
    try:
        w = whois.whois(hostname)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            age_months = (datetime.now() - created).days // 30
            features['age_of_domain'] = 1 if age_months >= 6 else -1
        else:
            features['age_of_domain'] = -1
    except Exception:
        features['age_of_domain'] = -1

    # 25. DNSRecord
    try:
        socket.gethostbyname(hostname)
        features['DNSRecord'] = 1
    except Exception:
        features['DNSRecord'] = -1

    # 26. web_traffic — rough heuristic from URL structure
    popular_domains = [
        'google','facebook','youtube','amazon','wikipedia','twitter',
        'instagram','linkedin','microsoft','apple','netflix','github',
        'stackoverflow','reddit','yahoo','bing','dropbox','paypal',
        'ebay','shopify','wordpress','wix','squarespace'
    ]
    is_popular = any(pd in hostname.lower() for pd in popular_domains)
    features['web_traffic'] = 1 if is_popular else 0

    # 27. Page_Rank — heuristic: well-known TLDs
    trusted_tlds = ['.gov', '.edu', '.org', '.com', '.net']
    has_trusted  = any(hostname.endswith(t) for t in trusted_tlds)
    features['Page_Rank'] = 1 if has_trusted else -1

    # 28. Google_Index — check if indexed (simplified)
    features['Google_Index'] = 1 if features['DNSRecord'] == 1 else -1

    # 29. Links_pointing_to_page — heuristic
    features['Links_pointing_to_page'] = 1 if features['web_traffic'] == 1 else 0

    # 30. Statistical_report — known phishing patterns
    suspicious_keywords = [
        'secure','account','update','login','signin','banking',
        'verify','confirm','password','paypal','apple','microsoft'
    ]
    has_suspicious = any(kw in url.lower() for kw in suspicious_keywords)
    features['Statistical_report'] = -1 if has_suspicious else 1

    return features


FEATURE_DESCRIPTIONS = {
    'having_IP_Address'           : 'URL Contains IP Address',
    'URL_Length'                  : 'URL Length',
    'Shortining_Service'          : 'URL Shortening Service Used',
    'having_At_Symbol'            : '@ Symbol in URL',
    'double_slash_redirecting'    : 'Double Slash Redirect',
    'Prefix_Suffix'               : 'Dash (-) in Domain Name',
    'having_Sub_Domain'           : 'Number of Sub-Domains',
    'SSLfinal_State'              : 'SSL / HTTPS Certificate',
    'Domain_registeration_length' : 'Domain Registration Length',
    'Favicon'                     : 'Favicon from External Domain',
    'port'                        : 'Non-Standard Port Used',
    'HTTPS_token'                 : '"HTTPS" in Domain Name',
    'Request_URL'                 : 'External Objects in Page',
    'URL_of_Anchor'               : 'Unsafe Anchor Tag Links',
    'Links_in_tags'               : 'External Links in Tags',
    'SFH'                         : 'Server Form Handler',
    'Submitting_to_email'         : 'Form Submits to Email',
    'Abnormal_URL'                : 'Abnormal URL Pattern',
    'Redirect'                    : 'Number of Redirects',
    'on_mouseover'                : 'onMouseOver Status Change',
    'RightClick'                  : 'Right Click Disabled',
    'popUpWidnow'                 : 'Pop-up Window Present',
    'Iframe'                      : 'iFrame Tag Present',
    'age_of_domain'               : 'Domain Age',
    'DNSRecord'                   : 'DNS Record Found',
    'web_traffic'                 : 'Website Traffic',
    'Page_Rank'                   : 'Page Rank / TLD Trust',
    'Google_Index'                : 'Indexed by Search Engine',
    'Links_pointing_to_page'      : 'Inbound Links',
    'Statistical_report'          : 'Suspicious Keywords in URL',
}


# ── Train Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    data, meta = arff.loadarff('Training Dataset.arff')
    df = pd.DataFrame(data)
    df = df.apply(lambda col: col.map(lambda x: int(x) if isinstance(x, bytes) else x))

    X = df.drop('Result', axis=1)
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)

    rf.fit(X_train_scaled, y_train)
    svm.fit(X_train_scaled, y_train)
    knn.fit(X_train_scaled, y_train)

    models  = {'Random Forest': rf, 'SVM': svm, 'KNN': knn}
    metrics = {}
    for name, model in models.items():
        pred = model.predict(X_test_scaled)
        metrics[name] = {
            'accuracy' : accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred),
            'recall'   : recall_score(y_test, pred),
            'f1'       : f1_score(y_test, pred),
            'cv'       : cross_val_score(model, X_train_scaled, y_train, cv=5).mean(),
            'cm'       : confusion_matrix(y_test, pred),
        }

    return rf, svm, knn, scaler, X, y, X_test, y_test, metrics, df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Phishing Detector")
    st.markdown("*Machine Learning (Supervised)*")
    st.divider()
    st.markdown("Models")
    st.markdown("Random Forest")
    st.markdown("Support Vector Machine")
    st.markdown("K-Nearest Neighbors")
    st.divider()
    page = st.radio("Navigate to:", [
        "URL Checker",
        "Model Performance",
        "Feature Analysis"
    ], index=0)

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Training models... Please wait (may take ~1 min for SVM)"):
    try:
        rf, svm, knn, scaler, X, y, X_test, y_test, metrics, df = load_and_train()
        models_loaded = True
    except FileNotFoundError:
        models_loaded = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — URL CHECKER
# ══════════════════════════════════════════════════════════════════════════════
if page == "URL Checker":

    st.markdown('<div class="main-title">Phishing Website Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter any website URL to check if it is legitimate or a phishing site</div>', unsafe_allow_html=True)

    if not models_loaded:
        st.error("**Dataset not found!** Place 'Training Dataset.arff' in the same folder as app.py")
        st.stop()

    st.success("All 3 models trained and ready!")
    st.markdown("---")

    # ── URL Input ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Enter Website URL</div>', unsafe_allow_html=True)

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        user_url = st.text_input(
            label="Website URL",
            placeholder="e.g. https://www.google.com  or  paypal-secure-login.xyz",
            label_visibility="collapsed"
        )
    with col_btn:
        analyse_clicked = st.button("Analyse URL")

    st.caption("Supports any URL. Add https:// if not included. Feature extraction may take 5–15 seconds.")

    # ── Examples ──────────────────────────────────────────────────────────────
    with st.expander("Example URLs to try"):
        st.markdown("**Likely Legitimate:**")
        st.code("https://www.google.com")
        st.code("https://www.github.com")
        st.code("https://www.wikipedia.org")
        st.markdown("**Likely Phishing (suspicious patterns):**")
        st.code("http://paypal-secure-login.update-account.xyz")
        st.code("http://192.168.1.1/secure/banking/login.php")
        st.code("http://bit.ly/free-prize-claim")

    st.markdown("---")

    # ── Analysis ─────────────────────────────────────────────────────────────
    if analyse_clicked and user_url.strip():
        url = user_url.strip()

        with st.spinner(f"Analysing {url} ..."):
            try:
                features_dict = extract_features(url)

                # Build input array in correct feature order
                feature_order = list(X.columns)
                feature_array = np.array([features_dict.get(f, 0) for f in feature_order]).reshape(1, -1)
                feature_scaled = scaler.transform(feature_array)

                # Predict
                rf_pred  = rf.predict(feature_scaled)[0]
                svm_pred = svm.predict(feature_scaled)[0]
                knn_pred = knn.predict(feature_scaled)[0]
                rf_prob  = rf.predict_proba(feature_scaled)[0]
                svm_prob = svm.predict_proba(feature_scaled)[0]
                knn_prob = knn.predict_proba(feature_scaled)[0]

                votes      = [rf_pred, svm_pred, knn_pred]
                final      = -1 if votes.count(-1) >= 2 else 1
                phish_vote = votes.count(-1)
                legit_vote = votes.count(1)

                extraction_success = True
            except Exception as e:
                st.error(f"Could not analyse URL: {str(e)}")
                extraction_success = False

        if extraction_success:

            # ── Final Verdict ─────────────────────────────────────────────────
            st.markdown('<div class="section-header">Final Verdict</div>', unsafe_allow_html=True)

            parsed_display = urlparse(url if url.startswith('http') else 'https://' + url)
            st.markdown(f"""
            <div class="url-box">
                <strong>URL Analysed:</strong><br>
                <code style="font-size:0.95rem;">{url}</code>
            </div>
            """, unsafe_allow_html=True)

            if final == -1:
                st.markdown(f"""
                <div class="result-phishing">
                    PHISHING WEBSITE DETECTED<br>
                    <span style="font-size:1rem;font-weight:normal;">
                        {phish_vote} out of 3 models flagged this as phishing
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                    LEGITIMATE WEBSITE<br>
                    <span style="font-size:1rem;font-weight:normal;">
                        {legit_vote} out of 3 models classified this as legitimate
                    </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Individual Model Results ───────────────────────────────────────
            st.markdown('<div class="section-header">Individual Model Predictions</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            model_results = [
                ('Random Forest', rf_pred, rf_prob),
                ('SVM',           svm_pred, svm_prob),
                ('KNN',           knn_pred, knn_prob),
            ]

            for col, (name, pred, prob) in zip([c1, c2, c3], model_results):
                with col:
                    label = "🚨 PHISHING" if pred == -1 else "✅ LEGITIMATE"
                    color = "#E74C3C"     if pred == -1 else "#2ECC71"
                    st.markdown(f"""
                    <div class="model-card">
                        <div style="font-size:1.05rem;font-weight:bold;color:#1F4E79;">{name}</div>
                        <div style="font-size:1.4rem;font-weight:bold;color:{color};margin:12px 0;">{label}</div>
                        <hr style="margin:8px 0;">
                        <div style="font-size:0.85rem;color:#555;">
                            🔴 Phishing: <strong>{prob[0]*100:.1f}%</strong><br>
                            🟢 Legitimate: <strong>{prob[1]*100:.1f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Probability Bar Chart ─────────────────────────────────────────
            st.markdown('<div class="section-header">Phishing Probability by Model</div>', unsafe_allow_html=True)

            model_names_chart = ['Random Forest', 'SVM', 'KNN']
            phish_probs = [rf_prob[0]*100, svm_prob[0]*100, knn_prob[0]*100]
            bar_colors  = ['#E74C3C' if p >= 50 else '#2ECC71' for p in phish_probs]

            fig, ax = plt.subplots(figsize=(8, 3.5))
            bars = ax.barh(model_names_chart, phish_probs, color=bar_colors,
                           edgecolor='white', linewidth=1.5, height=0.5)
            ax.axvline(x=50, color='grey', linestyle='--', linewidth=1.5,
                       label='Decision Threshold (50%)')
            for bar, val in zip(bars, phish_probs):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontweight='bold', fontsize=11)
            ax.set_xlim(0, 110)
            ax.set_xlabel('Phishing Probability (%)', fontsize=11)
            ax.set_title('Phishing Probability per Model', fontsize=13,
                         fontweight='bold', color='#1F4E79')
            ax.legend(fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("---")

            # ── Feature Breakdown ─────────────────────────────────────────────
            st.markdown('<div class="section-header">Extracted Feature Breakdown</div>', unsafe_allow_html=True)
            st.caption("These are the 30 features extracted from the URL and used by the models to make predictions.")

            feat_df_data = []
            for feat, val in features_dict.items():
                desc   = FEATURE_DESCRIPTIONS.get(feat, feat)
                if val == 1:
                    status = "✅ Safe"
                    risk   = "Low"
                elif val == 0:
                    status = "🟡 Suspicious"
                    risk   = "Medium"
                else:
                    status = "🔴 Risky"
                    risk   = "High"
                feat_df_data.append({"Feature": desc, "Value": val, "Status": status, "Risk Level": risk})

            feat_df = pd.DataFrame(feat_df_data)

            # Show risky ones first
            feat_df_sorted = pd.concat([
                feat_df[feat_df['Risk Level'] == 'High'],
                feat_df[feat_df['Risk Level'] == 'Medium'],
                feat_df[feat_df['Risk Level'] == 'Low'],
            ])

            risky_count = (feat_df['Risk Level'] == 'High').sum()
            safe_count  = (feat_df['Risk Level'] == 'Low').sum()
            warn_count  = (feat_df['Risk Level'] == 'Medium').sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("🔴 Risky Features",     risky_count, f"out of 30")
            m2.metric("🟡 Suspicious Features", warn_count,  f"out of 30")
            m3.metric("✅ Safe Features",        safe_count,  f"out of 30")

            st.dataframe(
                feat_df_sorted[['Feature', 'Status', 'Risk Level']],
                use_container_width=True,
                hide_index=True
            )

    elif analyse_clicked and not user_url.strip():
        st.warning("Please enter a URL before clicking Analyse.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":

    st.markdown('<div class="main-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluation metrics trained on the UCI Phishing Websites Dataset</div>', unsafe_allow_html=True)

    if not models_loaded:
        st.error("Dataset not found!")
        st.stop()

    model_names  = ['Random Forest', 'SVM', 'KNN']
    metric_keys  = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Metrics cards
    st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
    for mname in model_names:
        m = metrics[mname]
        st.markdown(f"#### {mname}")
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, mk, ml in zip([c1, c2, c3, c4], metric_keys, metric_labels):
            col.metric(ml, f"{m[mk]*100:.2f}%")
        c5.metric("CV Score (5-Fold)", f"{m['cv']*100:.2f}%")
        st.markdown("---")

    # Comparison bar chart
    st.markdown('<div class="section-header">Side-by-Side Comparison</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    x      = np.arange(len(metric_labels))
    width  = 0.25
    colors = ['#2E86C1', '#E74C3C', '#2ECC71']
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        vals = [metrics[mname][mk] for mk in metric_keys]
        bars = ax.bar(x + i*width, vals, width, label=mname,
                      color=color, edgecolor='white', linewidth=1.5, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0.80, 1.05)
    ax.set_title('Model Performance Comparison', fontsize=14,
                 fontweight='bold', color='#1F4E79')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Confusion matrices
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax2, mname, cmap in zip(axes, model_names, ['Blues', 'Reds', 'Greens']):
        cm = metrics[mname]['cm']
        sns.heatmap(cm, annot=True, fmt='d', ax=ax2, cmap=cmap,
                    xticklabels=['Phishing', 'Legitimate'],
                    yticklabels=['Phishing', 'Legitimate'],
                    linewidths=0.5, annot_kws={'size': 13, 'weight': 'bold'})
        ax2.set_title(mname, fontsize=13, fontweight='bold', color='#1F4E79')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Results table
    st.markdown('<div class="section-header">Summary Table</div>', unsafe_allow_html=True)
    result_data = {
        'Model'    : model_names,
        'Accuracy' : [f"{metrics[m]['accuracy']*100:.2f}%"  for m in model_names],
        'Precision': [f"{metrics[m]['precision']*100:.2f}%" for m in model_names],
        'Recall'   : [f"{metrics[m]['recall']*100:.2f}%"    for m in model_names],
        'F1 Score' : [f"{metrics[m]['f1']*100:.2f}%"        for m in model_names],
        'CV Score' : [f"{metrics[m]['cv']*100:.2f}%"        for m in model_names],
    }
    st.dataframe(pd.DataFrame(result_data).set_index('Model'), use_container_width=True)
    best = max(model_names, key=lambda m: metrics[m]['accuracy'])
    st.success(f"Best Model: {best} with {metrics[best]['accuracy']*100:.2f}% accuracy")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Analysis":

    st.markdown('<div class="main-title">Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Dataset overview and Random Forest feature importance</div>', unsafe_allow_html=True)

    if not models_loaded:
        st.error("Dataset not found!")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples",    f"{len(df):,}")
    c2.metric("Total Features",   "30")
    c3.metric("Phishing Sites",   f"{(df['Result']==-1).sum():,}")
    c4.metric("Legitimate Sites", f"{(df['Result']==1).sum():,}")

    # Class distribution
    st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
    counts = df['Result'].value_counts()
    labels = ['Phishing', 'Legitimate']
    colors = ['#E74C3C', '#2ECC71']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=2, width=0.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 60, str(v), ha='center', fontweight='bold', fontsize=12)
    axes[0].set_title('Count', fontsize=13, fontweight='bold', color='#1F4E79')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140, explode=(0.05, 0.05))
    axes[1].set_title('Percentage', fontsize=13, fontweight='bold', color='#1F4E79')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Feature importance
    st.markdown('<div class="section-header">Top 15 Feature Importances (Random Forest)</div>', unsafe_allow_html=True)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances.index = [FEATURE_DESCRIPTIONS.get(f, f) for f in importances.index]
    top15 = importances.sort_values(ascending=True).tail(15)
    fig2, ax = plt.subplots(figsize=(12, 7))
    colors_imp = plt.cm.Blues(np.linspace(0.4, 0.9, len(top15)))
    bars = ax.barh(top15.index, top15.values, color=colors_imp, edgecolor='white')
    for bar, val in zip(bars, top15.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Top 15 Most Important Features', fontsize=13,
                 fontweight='bold', color='#1F4E79')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
