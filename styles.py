"""CSS style blocks for VerifyAI – premium dark-mode Streamlit app."""

FONTS = '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">'

LANDING_CSS = """
<div><style>
@keyframes fadeUp{from{opacity:0;transform:translateY(32px)}to{opacity:1;transform:translateY(0)}}
@keyframes shimmer{0%{background-position:-200% center}100%{background-position:200% center}}
@keyframes pulse{0%,100%{opacity:.6}50%{opacity:1}}
@keyframes gridFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes rotateBg{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}

.stApp{
  background:#0B0F19 !important;
  color:#e2e8f0;
  font-family:'Inter',sans-serif;
}
.block-container{max-width:100% !important;padding:0.4rem 3.5rem 3rem !important;}
[data-testid="stAppViewContainer"]{min-height:100vh;}
header,header[data-testid="stHeader"]{background:transparent !important;}
[data-testid="stSidebar"]{display:none;}

/* animated bg */
.stApp::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 80% 50% at 20% 10%,rgba(99,102,241,.12) 0%,transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%,rgba(6,182,212,.08) 0%,transparent 60%),
    radial-gradient(ellipse 50% 60% at 50% 50%,rgba(139,92,246,.05) 0%,transparent 70%);
}

/* navbar */
.navbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:.9rem 0 1.4rem;border-bottom:1px solid rgba(255,255,255,.06);
  margin-bottom:0;animation:fadeUp .5s ease-out both;
}
.nav-logo{
  font-size:1.35rem;font-weight:800;letter-spacing:-.02em;
  background:linear-gradient(135deg,#6366f1,#06b6d4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.nav-links{display:flex;gap:2rem;align-items:center;}
.nav-link{color:#94a3b8;font-size:.88rem;font-weight:500;text-decoration:none;transition:color .2s;}
.nav-link:hover{color:#e2e8f0;}
.nav-cta{
  padding:.46rem 1.1rem;border-radius:8px;font-size:.88rem;font-weight:600;
  background:linear-gradient(135deg,#6366f1,#4f46e5);color:#fff;border:none;cursor:pointer;
  transition:transform .2s,box-shadow .2s;
}
.nav-cta:hover{transform:translateY(-1px);box-shadow:0 8px 24px rgba(99,102,241,.35);}

/* hero */
.hero{padding:5rem 0 3.5rem;position:relative;animation:fadeUp .7s ease-out .1s both;}
.hero-kicker{
  display:inline-flex;align-items:center;gap:.5rem;
  padding:.32rem .9rem;border-radius:999px;
  border:1px solid rgba(99,102,241,.3);background:rgba(99,102,241,.08);
  color:#a5b4fc;font-size:.78rem;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;margin-bottom:1.6rem;
}
.hero-kicker-dot{width:6px;height:6px;border-radius:50%;background:#6366f1;animation:pulse 2s infinite;}
.hero-title{
  font-size:clamp(3.2rem,7vw,5.5rem);font-weight:900;line-height:1.06;
  letter-spacing:-.04em;margin:0 0 1.5rem;color:#f1f5f9;
}
.hero-title .gradient{
  background:linear-gradient(135deg,#6366f1 0%,#06b6d4 50%,#6366f1 100%);
  background-size:200% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:shimmer 5s linear infinite;
}
.hero-sub{
  font-size:1.15rem;color:#94a3b8;line-height:1.75;max-width:560px;
  margin-bottom:2.2rem;font-weight:400;
}
.hero-tags{display:flex;flex-wrap:wrap;gap:.6rem;margin-top:1.8rem;margin-bottom:2rem;}
.hero-tag{
  padding:.38rem .88rem;border-radius:999px;
  border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.04);
  color:#94a3b8;font-size:.75rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
}

/* hero image */
.hero-img-wrap{position:relative;animation:fadeUp .7s ease-out .2s both;}
.hero-img-card{
  border-radius:24px;overflow:hidden;
  border:1px solid rgba(255,255,255,.08);
  background:rgba(255,255,255,.03);
  box-shadow:0 0 60px rgba(99,102,241,.15),0 32px 64px rgba(0,0,0,.5);
  backdrop-filter:blur(12px);
}
.hero-img-card img{width:100%;display:block;border-radius:24px;}

/* stats row */
.stats-row{
  display:flex;gap:2rem;flex-wrap:wrap;padding:2rem 0;
  border-top:1px solid rgba(255,255,255,.06);border-bottom:1px solid rgba(255,255,255,.06);
  margin:2rem 0;animation:fadeUp .7s ease-out .3s both;
}
.stat-item{display:flex;flex-direction:column;gap:.2rem;}
.stat-value{font-size:1.8rem;font-weight:800;color:#f1f5f9;letter-spacing:-.03em;}
.stat-label{font-size:.8rem;color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.06em;}

/* buttons */
.stButton>button{
  border-radius:10px !important;border:none !important;
  background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
  color:#fff !important;font-weight:700 !important;font-size:.94rem !important;
  padding:.75rem 1.4rem !important;width:100%;letter-spacing:.01em;
  transition:transform .2s,box-shadow .25s !important;
}
.stButton>button:hover{
  transform:translateY(-2px) scale(1.01) !important;
  box-shadow:0 12px 32px rgba(99,102,241,.4) !important;
}

h1,h2,h3,h4,h5,h6{color:#f1f5f9 !important;}
p,.stMarkdown p{color:#94a3b8 !important;}
[data-testid="stCaption"]{color:#64748b !important;}
code{background:rgba(99,102,241,.15) !important;color:#a5b4fc !important;border-radius:4px !important;}

@media(max-width:900px){
  .block-container{padding:.7rem 1.2rem 2.5rem !important;}
  .navbar{padding:.7rem 0 1rem;}
  .hero{padding:2.5rem 0 2rem;}
  .stats-row{gap:1.2rem;}
}
</style></div>
"""

DASHBOARD_CSS = """
<div><style>
@keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:.5}50%{opacity:1}}
@keyframes glow-green{0%,100%{box-shadow:0 0 30px rgba(16,185,129,.2)}50%{box-shadow:0 0 55px rgba(16,185,129,.35)}}
@keyframes glow-red{0%,100%{box-shadow:0 0 30px rgba(239,68,68,.2)}50%{box-shadow:0 0 55px rgba(239,68,68,.35)}}
@keyframes bar-fill{from{width:0%}to{width:var(--conf-width)}}

.stApp{background:#0B0F19 !important;color:#e2e8f0;font-family:'Inter',sans-serif;}
.block-container{max-width:100% !important;padding:1.2rem 3.5rem 3rem !important;}
[data-testid="stAppViewContainer"]{min-height:100vh;}
header,header[data-testid="stHeader"]{background:transparent !important;}
[data-testid="stSidebar"]{display:none;}
.stApp::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 70% 50% at 15% 10%,rgba(99,102,241,.1) 0%,transparent 60%),
    radial-gradient(ellipse 50% 40% at 85% 85%,rgba(6,182,212,.07) 0%,transparent 60%);
}

/* page header */
.page-header{
  display:flex;align-items:center;justify-content:space-between;
  padding:1rem 0 1.6rem;border-bottom:1px solid rgba(255,255,255,.06);
  margin-bottom:1.6rem;animation:fadeUp .5s ease-out both;
}
.page-logo{
  font-size:1.25rem;font-weight:800;letter-spacing:-.02em;
  background:linear-gradient(135deg,#6366f1,#06b6d4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.page-title-block{animation:fadeUp .5s ease-out .05s both;}
.page-kicker{color:#6366f1;font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem;}
.page-title{font-size:clamp(1.8rem,3.5vw,2.8rem);font-weight:800;color:#f1f5f9;margin:0;line-height:1.1;}

/* glass card base */
.glass-card{
  border-radius:16px;padding:1.4rem;
  background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.08);
  backdrop-filter:blur(12px);
  box-shadow:0 8px 32px rgba(0,0,0,.3);
  animation:fadeUp .5s ease-out both;
  transition:border-color .25s,box-shadow .25s;
}
.glass-card:hover{border-color:rgba(99,102,241,.3);box-shadow:0 12px 40px rgba(0,0,0,.4);}

/* mode cards */
.mode-card{
  border-radius:14px;padding:1.1rem 1.2rem;
  background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.07);
  transition:border-color .2s,background .2s;margin-bottom:.6rem;
}
.mode-card.active{border-color:rgba(99,102,241,.45);background:rgba(99,102,241,.08);}
.mode-card-title{color:#f1f5f9;font-size:1rem;font-weight:700;}
.mode-card-desc{color:#64748b;font-size:.86rem;margin-top:.3rem;line-height:1.5;}

/* sample card */
.sample-card{
  border-radius:14px;padding:1.1rem;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);
  margin-bottom:.6rem;transition:border-color .2s,transform .2s;cursor:pointer;
}
.sample-card:hover{border-color:rgba(99,102,241,.3);transform:translateY(-2px);}
.sample-card.selected{border-color:rgba(99,102,241,.5);background:rgba(99,102,241,.07);}
.sample-card-num{color:#6366f1;font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
.sample-card-title{color:#f1f5f9;font-size:1.1rem;font-weight:700;margin:.3rem 0 .4rem;}
.sample-card-desc{color:#64748b;font-size:.84rem;line-height:1.55;}
.sample-card-status{margin-top:.7rem;font-size:.78rem;font-weight:700;}
.sample-card-status.selected{color:#6366f1;}
.sample-card-status.ready{color:#10b981;}

/* result cards */
.result-card{
  border-radius:16px;padding:1.5rem;
  background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.08);
  backdrop-filter:blur(12px);
  animation:fadeUp .5s ease-out both;
}
.result-card.real{border-color:rgba(16,185,129,.3);animation:fadeUp .5s ease-out both,glow-green 3s ease-in-out 1s infinite;}
.result-card.fake{border-color:rgba(239,68,68,.3);animation:fadeUp .5s ease-out both,glow-red 3s ease-in-out 1s infinite;}
.result-badge{
  display:inline-block;padding:.28rem .75rem;border-radius:999px;
  font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  background:rgba(99,102,241,.12);color:#a5b4fc;margin-bottom:1rem;
}
.result-label{font-size:2.8rem;font-weight:900;letter-spacing:-.04em;margin:0 0 .3rem;line-height:1;}
.result-label.real{color:#10b981;}
.result-label.fake{color:#ef4444;}
.result-confidence{color:#94a3b8;font-size:.9rem;margin-bottom:1rem;}
.conf-bar-wrap{background:rgba(255,255,255,.06);border-radius:999px;height:7px;overflow:hidden;margin-bottom:1rem;}
.conf-bar{height:100%;border-radius:999px;transition:width 1s ease-out;}
.conf-bar.real{background:linear-gradient(90deg,#059669,#10b981);}
.conf-bar.fake{background:linear-gradient(90deg,#dc2626,#ef4444);}
.risk-badge{
  display:inline-block;padding:.28rem .75rem;border-radius:999px;
  font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:1rem;
}
.risk-low{background:rgba(16,185,129,.15);color:#10b981;}
.risk-medium{background:rgba(245,158,11,.15);color:#f59e0b;}
.risk-high{background:rgba(239,68,68,.15);color:#ef4444;}

/* explanation card */
.explain-card{
  border-radius:14px;padding:1.2rem;
  background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
  margin-top:1rem;
}
.explain-title{font-size:.95rem;font-weight:700;color:#f1f5f9;margin-bottom:.8rem;}
.explain-item{
  display:flex;align-items:flex-start;gap:.7rem;
  padding:.55rem 0;border-bottom:1px solid rgba(255,255,255,.05);
  font-size:.85rem;color:#94a3b8;line-height:1.5;
}
.explain-item:last-child{border-bottom:none;}
.explain-icon{font-size:1rem;flex-shrink:0;margin-top:.05rem;}

/* preview section */
.preview-label{color:#64748b;font-size:.75rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;margin-bottom:.5rem;}
.preview-filename{color:#f1f5f9;font-size:.9rem;font-weight:600;margin-bottom:.8rem;}

/* file uploader override */
[data-testid="stFileUploader"]{
  background:rgba(255,255,255,.03) !important;border-radius:14px !important;
  border:1.5px dashed rgba(99,102,241,.3) !important;padding:.5rem !important;
}
[data-testid="stFileUploader"]:hover{border-color:rgba(99,102,241,.55) !important;}

/* spinner/status */
[data-testid="stSpinner"]{color:#6366f1 !important;}
.stAlert{border-radius:12px !important;}

/* buttons */
.stButton>button{
  border-radius:10px !important;border:1px solid rgba(99,102,241,.3) !important;
  background:rgba(99,102,241,.12) !important;color:#a5b4fc !important;
  font-weight:700 !important;font-size:.88rem !important;
  padding:.68rem 1rem !important;width:100%;
  transition:all .2s !important;
}
.stButton>button:hover{
  background:rgba(99,102,241,.22) !important;border-color:rgba(99,102,241,.55) !important;
  transform:translateY(-1px) !important;box-shadow:0 8px 24px rgba(99,102,241,.2) !important;
  color:#c7d2fe !important;
}
.stButton>button:disabled{opacity:.35 !important;transform:none !important;}

/* run button – primary */
.stButton:last-of-type>button{
  background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
  color:#fff !important;border:none !important;
}
.stButton:last-of-type>button:hover{box-shadow:0 12px 32px rgba(99,102,241,.4) !important;}

h1,h2,h3,h4,h5,h6{color:#f1f5f9 !important;}
p,.stMarkdown p{color:#94a3b8 !important;}
[data-testid="stCaption"]{color:#64748b !important;}
.stSelectbox label,.stFileUploader label{color:#94a3b8 !important;}
[data-testid="stInfo"]{background:rgba(99,102,241,.08) !important;border:1px solid rgba(99,102,241,.2) !important;border-radius:12px !important;}
[data-testid="stError"]{border-radius:12px !important;}
code{background:rgba(99,102,241,.15) !important;color:#a5b4fc !important;border-radius:4px !important;}
[data-testid="stVerticalBlock"]{gap:.5rem !important;}

@media(max-width:900px){
  .block-container{padding:1rem 1.2rem 2.5rem !important;}
  .page-header{padding:.6rem 0 1rem;}
}
</style></div>
"""
