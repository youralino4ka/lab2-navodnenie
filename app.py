# -*- coding: utf-8 -*-
"""FloodLab — интерактивное приложение для моделирования последствий наводнения."""
import io, csv, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from model import VARS, f, scaled_rhs

st.set_page_config(page_title="FloodLab · Модель наводнения", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

# --- visual system: inspired by the supplied reference dashboard, adapted to floods ---
st.markdown(r"""
<style>
:root { --ink:#f6f7fb; --muted:#9ca3af; --panel:#171922; --panel2:#20232e; --line:#343846; --accent:#f05a9d; --accent2:#ff8fbd; }
.stApp { background: radial-gradient(ellipse at 80% 0%, #22212e 0%, #101116 48%, #0d0e13 100%); color:var(--ink); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#20212b,#171820); border-right:1px solid #343542; }
[data-testid="stSidebar"] * { color:#f4f4f8; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1600px; }
h1,h2,h3 { letter-spacing:-.025em; }
h1 { font-size:2.25rem !important; font-weight:800 !important; }
h2 { font-size:1.35rem !important; font-weight:750 !important; }
h3 { font-size:1.05rem !important; font-weight:700 !important; }
.small-muted { color:#a3a7b4; font-size:.92rem; }
.hero { padding:1.35rem 1.55rem; border:1px solid #393343; border-radius:18px; background:linear-gradient(120deg,rgba(240,90,157,.14),rgba(35,35,48,.82) 55%,rgba(110,90,150,.12)); margin-bottom:1.1rem; }
.hero-kicker { color:#ff9dc7; text-transform:uppercase; font-size:.76rem; letter-spacing:.16em; font-weight:800; }
.hero h1 { margin:.25rem 0 .3rem 0; }
.hero p { color:#b5b7c3; margin:0; }
div[data-testid="stMetric"] { background:linear-gradient(150deg,#20232e,#181a23); border:1px solid #343746; border-radius:14px; padding:14px 16px; }
div[data-testid="stMetricLabel"] { color:#b4b7c3; }
div[data-testid="stMetricValue"] { color:#fff; font-weight:750; }
.stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid #f05a9d; background:linear-gradient(135deg,#ef5b9e,#d84488); color:white; font-weight:700; padding:.58rem 1rem; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:#ffb4d2; color:white; background:#d84488; }
.stTabs [data-baseweb="tab-list"] { gap:1.2rem; border-bottom:1px solid #343746; }
.stTabs [data-baseweb="tab"] { color:#d0d2dc; padding:12px 4px; }
.stTabs [aria-selected="true"] { color:#ff83b6 !important; border-bottom-color:#f05a9d !important; }
[data-testid="stExpander"] { background:#181a23; border:1px solid #333644; border-radius:12px; }
[data-testid="stDataFrame"] { border:1px solid #343746; border-radius:12px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:12px; }
hr { border-color:#333644; }
label, .stMarkdown, p, li { color:#e5e7ef; }
input, textarea { color:#f5f6fb !important; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

DEFAULT_Z = np.ones(14, dtype=float)
DRIVERS = {
    "Rw": ("Rw(t)", 1.0, "Внешняя функция времени; численная зависимость в PDF не задана."),
    "Hw": ("Hw(t)", 1.0, "Внешняя функция времени; численная зависимость в PDF не задана."),
    "Sem": ("Sem(t)", 1.0, "Внешняя функция времени; численная зависимость в PDF не задана."),
    "Vhr": ("Vhr(t)", 1.0, "Внешняя функция времени; численная зависимость в PDF не задана."),
    "Chr": ("Chr(t)", 1.0, "Внешняя функция времени; численная зависимость в PDF не задана."),
    "Sev": ("Sev(t)", 1.0, "Внешняя функция времени; численная зависимость в PDF не задана."),
    "Ch": ("Ch(t)", 1.0, "Внешняя функция времени; численная зависимость в PDF не задана."),
    "Hr": ("Hr(t)", 1.0, "Внешняя функция времени; численная зависимость в PDF не задана."),
    "Vh": ("Vh(t)", 1.0, "Внешняя функция времени; численная зависимость в PDF не задана."),
    "Lr": ("Lr(t)", 1.0, "Внешняя функция времени; численная зависимость в PDF не задана."),
}

FORMULAS = [
(1,"f₁(Z₂)","Число погибших зависит от длительности поражающего воздействия", "0.6111·Z₂⁴ + 0.7217·Z₂³ + 0.7446·Z₂² + 0.9823·Z₂ + 0.0148"),
(2,"f₂(Z₉)","Число погибших зависит от длительности аварийного периода", "0.0000057·Z₉⁴ − 0.00068·Z₉³ − 0.011·Z₉² + 1.37·Z₉ + 224.77"),
(3,"f₃(Z₁₄)","Число погибших убывает с ростом ущерба территории", "0.8985·Z₁₄⁴ + 0.176·Z₁₄³ + 0.7223·Z₁₄² + 0.9847·Z₁₄ + 0.927"),
(4,"f₄(Z₇)","Длительность поражающего воздействия растёт с площадью земель вне оборота", "0.4677·Z₇⁴ + 0.7069·Z₇³ + 0.0868·Z₇² + 0.1717·Z₇ + 0.9457"),
(5,"f₅(Z₄)","Длительность воздействия связана с потерей имущества первой необходимости", "exp(0.8471 + 0.5422/Z₄ + 0.3951·log₁₀(Z₄))"),
(6,"f₆(Z₁₁)","Длительность воздействия растёт с числом пострадавших животных", "0.7952·Z₁₁ + 0.3401"),
(7,"f₇(Z₁)","Площадь зоны ЧС растёт с числом погибших", "0.0032·Z₁⁴ − 1.8·Z₁³ + 336.03·Z₁² − 24105.24·Z₁ + 695052.48"),
(8,"f₈(Z₇)","Площадь зоны ЧС растёт с площадью земель вне оборота", "0.0077·Z₇² + 0.9946·Z₇ + 0.1374"),
(9,"f₉(Z₉)","Площадь зоны ЧС зависит от длительности аварийного периода", "−0.064·Z₉⁴ + 15.23·Z₉³ − 1140.77·Z₉² + 24335.32·Z₉ + 403947.48"),
(10,"f₁₀(Z₂)","Площадь зоны ЧС убывает с длительностью поражающего воздействия", "0.4986·Z₂² + 0.0854·Z₂ + 0.492"),
(11,"f₁₁(Z₁₀)","Площадь зоны ЧС убывает с длительностью восстановления", "0.9716·Z₁₀⁴ + 0.7811·Z₁₀³ + 0.5336·Z₁₀² + 0.6221·Z₁₀ + 0.3907"),
(12,"f₁₂(Z₁)","Число людей с утратой имущества связано с числом погибших", "−0.14·Z₁³ + 48.83·Z₁² − 4402.026·Z₁ + 283563.063"),
(13,"f₁₃(Z₃)","Число людей с утратой имущества зависит от площади зоны ЧС", "11707.78·⁸√Z₃ − 1950.012·⁴√Z₃ − 63.8·√Z₃ − 0.11·Z₃ + 554.5"),
(14,"f₁₄(Z₅)","Потеря имущества растёт с ущербом коммерческим организациям", "exp(0.762 + 0.114/Z₅ + 0.3895·log₁₀(Z₅))"),
(15,"f₁₅(Z₇)","Потеря имущества зависит от площади земель вне оборота", "0.0217·Z₇² + 0.7927·Z₇ + 0.4721"),
(16,"f₁₆(Z₁₄)","Потеря имущества растёт с ущербом территории", "0.8302·Z₁₄² + 0.4483·Z₁₄ + 0.3633"),
(17,"f₁₇(Z₆)","Ущерб коммерческим организациям растёт с объёмом загрязнённого грунта", "0.5845·Z₆⁴ + 0.8868·Z₆³ + 0.8836·Z₆² + 0.6114·Z₆ + 0.865"),
(18,"f₁₈(Z₈)","Ущерб коммерческим организациям растёт при снижении плодородия", "exp(0.6776 + 0.9149/Z₈ + 0.5615·log₁₀(Z₈))"),
(19,"f₁₉(Z₁₂)","Ущерб коммерческим организациям зависит от потерь урожая", "exp(0.4224 + 0.9753/Z₁₂ + 0.1117·log₁₀(Z₁₂))"),
(20,"f₂₀(Z₂)","Ущерб коммерческим организациям убывает с длительностью воздействия", "exp(0.9321 + 0.17/Z₂ + 0.347·log₁₀(Z₂))"),
(21,"f₂₁(Z₁₂)","Ущерб коммерческим организациям убывает с величиной потерь урожая", "exp(0.3229 + 0.7388/Z₁₂ + 0.2312·log₁₀(Z₁₂))"),
(22,"f₂₂(Z₄)","Загрязнение грунта растёт с числом людей, потерявших имущество", "0.9833·Z₄⁴ + 0.0954·Z₄³ + 0.841·Z₄² + 0.0554·Z₄ + 0.1397"),
(23,"f₂₃(Z₇)","Загрязнение грунта растёт с площадью земель вне оборота", "exp(0.2076 + 0.1655/Z₇ + 0.6113·log₁₀(Z₇))"),
(24,"f₂₄(Z₉)","Загрязнение грунта растёт с длительностью аварийного периода", "0.4161·Z₉⁴ + 0.8724·Z₉³ + 0.9778·Z₉² + 0.2006·Z₉ + 0.3202"),
(25,"f₂₅(Z₁₃)","Загрязнение грунта растёт с площадью повреждённых лесов", "0.5896·Z₁₃² + 0.8183·Z₁₃ + 0.1977"),
(26,"f₂₆(Z₁₀)","Загрязнение грунта убывает с длительностью восстановления", "0.283·Z₁₀² + 0.9196·Z₁₀ + 0.1487"),
(27,"f₂₇(Z₁₁)","Загрязнение грунта убывает с числом пострадавших животных", "0.8713·Z₁₁⁴ + 0.5895·Z₁₁³ + 0.0583·Z₁₁² + 0.6532·Z₁₁ + 0.4761"),
(28,"f₂₈(Z₁₂)","Загрязнение грунта убывает с потерями урожая", "0.3343·Z₁₂² + 0.0034·Z₁₂ + 0.5951"),
(29,"f₂₉(Z₉)","Площадь земель вне оборота растёт с длительностью аварийного периода", "exp(0.3608 + 0.0292/Z₉ + 0.4383·log₁₀(Z₉))"),
(30,"f₃₀(Z₆)","Снижение плодородия растёт с загрязнением грунта", "0.5193·Z₆⁴ + 0.1307·Z₆³ + 0.9083·Z₆² + 0.4975·Z₆ + 0.8935"),
(31,"f₃₁(Z₁₁)","Снижение плодородия растёт с числом пострадавших животных", "exp(0.5722 + 0.2981/Z₁₁ + 0.714·log₁₀(Z₁₁))"),
(32,"f₃₂(Z₄)","Снижение плодородия убывает с числом людей, потерявших имущество", "0.0741·Z₄⁴ + 0.2611·Z₄³ + 0.4477·Z₄² + 0.4958·Z₄ + 0.8284"),
(33,"f₃₃(Z₁₀)","Снижение плодородия убывает с длительностью восстановления", "0.5528·Z₁₀² + 0.5976·Z₁₀ + 0.7432"),
(34,"f₃₄(Z₁₃)","Снижение плодородия убывает с повреждением лесов", "0.3872·Z₁₃² + 0.6341·Z₁₃ + 0.489"),
(35,"f₃₅(Z₄)","Длительность аварийного периода растёт с потерей имущества", "0.9601·Z₄² + 0.27·Z₄ + 0.1453"),
(36,"f₃₆(Z₂)","Длительность аварийного периода зависит от длительности воздействия", "exp(0.5809 + 0.2149/Z₂ + 0.8216·log₁₀(Z₂))"),
(37,"f₃₇(Z₅)","Длительность аварийного периода убывает с ущербом коммерческим организациям", "0.5853·Z₅² + 0.7931·Z₅ + 0.0132"),
(38,"f₃₈(Z₁₀)","Длительность аварийного периода убывает с длительностью восстановления", "0.6748·Z₁₀⁴ + 0.1345·Z₁₀³ + 0.8504·Z₁₀² + 0.9407·Z₁₀ + 0.8215"),
(39,"f₃₉(Z₂)","Длительность восстановления зависит от длительности воздействия", "0.0503·Z₂ + 0.8566"),
(40,"f₄₀(Z₄)","Длительность восстановления растёт с потерей имущества", "0.4528·Z₄ + 0.2391"),
(41,"f₄₁(Z₁₁)","Длительность восстановления растёт с числом пострадавших животных", "exp(0.5853 + 0.8867/Z₁₁ + 0.0981·log₁₀(Z₁₁))"),
(42,"f₄₂(Z₁₂)","Длительность восстановления растёт с потерями урожая", "0.8644·Z₁₂² + 0.4174·Z₁₂ + 0.1021"),
(43,"f₄₃(Z₁₄)","Длительность восстановления растёт с ущербом территории", "0.6815·Z₁₄ + 0.3011"),
(44,"f₄₄(Z₃)","Пострадавшие животные зависят от площади зоны ЧС", "0.521·Z₃² + 0.0578·Z₃ + 0.9765"),
(45,"f₄₅(Z₇)","Пострадавшие животные растут с площадью земель вне оборота", "exp(0.7846 + 0.0503/Z₇ + 0.4584·log₁₀(Z₇))"),
(46,"f₄₆(Z₉)","Пострадавшие животные растут с длительностью аварийного периода", "exp(0.2664 + 0.6038/Z₉ + 0.5652·log₁₀(Z₉))"),
(47,"f₄₇(Z₄)","Пострадавшие животные убывают с потерей имущества", "0.4522·Z₄² + 0.8492·Z₄ + 0.2404"),
(48,"f₄₈(Z₁₃)","Пострадавшие животные убывают с повреждением лесов", "0.2115·Z₁₃² + 0.3728·Z₁₃ + 0.7422"),
(49,"f₄₉(Z₆)","Потери урожая зависят от загрязнения грунта", "exp(0.0096 + 0.6482/Z₆ + 0.0958·log₁₀(Z₆))"),
(50,"f₅₀(Z₈)","Потери урожая растут со снижением плодородия", "0.8223·Z₈² + 0.584·Z₈ + 0.2215"),
(51,"f₅₁(Z₁₃)","Потери урожая растут с повреждением лесов", "0.6571·Z₁₃² + 0.1468·Z₁₃ + 0.7845"),
(52,"f₅₂(Z₁)","Потери урожая убывают с числом погибших", "0.466·Z₁² + 0.5152·Z₁ + 0.1969"),
(53,"f₅₃(Z₇)","Потери урожая убывают с площадью земель вне оборота", "0.9482·Z₇⁴ + 0.5353·Z₇³ + 0.52·Z₇² + 0.2103·Z₇ + 0.5825"),
(54,"f₅₄(Z₁₁)","Потери урожая убывают с числом пострадавших животных", "0.5398·Z₁₁² + 0.3594·Z₁₁ + 0.7191"),
(55,"f₅₅(Z₇)","Повреждение лесов растёт с площадью земель вне оборота", "0.2407·Z₇⁴ + 0.0158·Z₇³ + 0.6595·Z₇² + 0.6349·Z₇ + 0.6759"),
(56,"f₅₆(Z₃)","Площадь повреждённых лесов убывает с площадью зоны ЧС", "exp(0.0963 + 0.4879/Z₃ + 0.8474·log₁₀(Z₃))"),
(57,"f₅₇(Z₉)","Площадь повреждённых лесов убывает с длительностью аварийного периода", "exp(0.0107 + 0.9191/Z₉ + 0.9822·log₁₀(Z₉))"),
(58,"f₅₈(Z₂)","Ущерб территории растёт с длительностью воздействия", "0.692·Z₂² + 0.2629·Z₂ + 0.1879"),
(59,"f₅₉(Z₃)","Ущерб территории растёт с площадью зоны ЧС", "0.4349·Z₃ + 0.6437"),
(60,"f₆₀(Z₅)","Ущерб территории растёт с ущербом коммерческим организациям", "0.0623·Z₅⁴ + 0.526·Z₅³ + 0.7183·Z₅² + 0.9025·Z₅ + 0.447"),
(61,"f₆₁(Z₈)","Ущерб территории растёт со снижением плодородия", "0.8509·Z₈² + 0.7456·Z₈ + 0.0468"),
(62,"f₆₂(Z₁₁)","Ущерб территории растёт с числом пострадавших животных", "0.8446·Z₁₁⁴ + 0.9258·Z₁₁³ + 0.6296·Z₁₁² + 0.5082·Z₁₁ + 0.3431"),
(63,"f₆₃(Z₁₂)","Ущерб территории растёт с потерями урожая", "exp(0.4367 + 0.82/Z₁₂ + 0.803·log₁₀(Z₁₂))"),
(64,"f₆₄(Z₁₃)","Ущерб территории растёт с площадью повреждённых лесов", "0.9985·Z₁₃² + 0.8721·Z₁₃ + 0.8347"),
(65,"f₆₅(Z₇)","Ущерб территории убывает с площадью земель вне оборота", "0.3843·Z₇ + 0.3849"),
(66,"f₆₆(Z₉)","Ущерб территории убывает с длительностью аварийного периода", "0.7881·Z₉ + 0.8325"),
]

SYSTEM_EQS = [
"dZ₁/dt = [f₁(Z₂)·f₂(Z₉)·Rw − f₃(Z₁₄)·Vhr·Chr·Sev] / Z₁",
"dZ₂/dt = [f₄(Z₇)·Rw·Hw − f₅(Z₄)·f₆(Z₁₁)] / Z₂",
"dZ₃/dt = [f₇(Z₁)·f₈(Z₇)·f₉(Z₉)·Sem − f₁₀(Z₂)·f₁₁(Z₁₀)·Rw·Lr] / Z₃",
"dZ₄/dt = [f₁₂(Z₁)·f₁₃(Z₃)·f₁₄(Z₅)·f₁₅(Z₇)·f₁₆(Z₁₄) − Rw] / Z₄",
"dZ₅/dt = [f₁₇(Z₆)·f₁₈(Z₈)·f₁₉(Z₁₂) − f₂₀(Z₂)·f₂₁(Z₁₂)] / Z₅",
"dZ₆/dt = [f₂₂(Z₄)·f₂₃(Z₇)·f₂₄(Z₉)·f₂₅(Z₁₃)·Sem − f₂₆(Z₁₀)·f₂₇(Z₁₁)·f₂₈(Z₁₂)·Chr·Hr·Vhr] / Z₆",
"dZ₇/dt = [f₂₉(Z₉)·Chr·Hr·Vhr − Rw·Sev·Lr] / Z₇",
"dZ₈/dt = [f₃₀(Z₆)·f₃₁(Z₁₁)·Sev − f₃₂(Z₄)·f₃₃(Z₁₀)·f₃₄(Z₁₃)] / Z₈",
"dZ₉/dt = [f₃₅(Z₄) − f₃₆(Z₂)·f₃₇(Z₅)·f₃₈(Z₁₀)] / Z₉",
"dZ₁₀/dt = [f₃₉(Z₂)·f₄₀(Z₄)·f₄₁(Z₁₁)·f₄₂(Z₁₂)·f₄₃(Z₁₄)·Rw − Chr·Hr·Vhr] / Z₁₀",
"dZ₁₁/dt = [f₄₄(Z₃)·f₄₅(Z₇)·f₄₆(Z₉) − f₄₇(Z₄)·f₄₈(Z₁₃)] / Z₁₁",
"dZ₁₂/dt = [f₄₉(Z₆)·f₅₀(Z₈)·f₅₁(Z₁₃)·Sem − f₅₂(Z₁)·f₅₃(Z₇)·f₅₄(Z₁₁)] / Z₁₂",
"dZ₁₃/dt = [f₅₅(Z₇)·Rw·Sem − f₅₆(Z₃)·f₅₇(Z₉)] / Z₁₃",
"dZ₁₄/dt = [f₅₈(Z₂)·f₅₉(Z₃)·f₆₀(Z₅)·f₆₁(Z₈)·f₆₂(Z₁₁)·f₆₃(Z₁₂)·f₆₄(Z₁₃) − f₆₅(Z₇)·f₆₆(Z₉)] / Z₁₄",
]

# Session state
if "z0" not in st.session_state: st.session_state.z0 = DEFAULT_Z.copy()
if "drivers" not in st.session_state: st.session_state.drivers = {k:v[1] for k,v in DRIVERS.items()}
if "horizon" not in st.session_state: st.session_state.horizon = 1.0
if "dt" not in st.session_state: st.session_state.dt = 0.01
if "alpha" not in st.session_state: st.session_state.alpha = 1e-12
if "result" not in st.session_state: st.session_state.result = None

# Sidebar controls
with st.sidebar:
    st.markdown("<div style='padding:.3rem 0 1rem'><div style='font-size:1.7rem;font-weight:850;letter-spacing:-.04em'>🌊 Flood<span style='color:#ff86b9'>Lab</span></div><div style='color:#aeb1bf;font-size:.9rem'>Модель последствий наводнения</div></div>", unsafe_allow_html=True)
    st.markdown("### Генерация данных")
    if st.button("🎲 Рандомизировать параметры", use_container_width=True):
        rng=np.random.default_rng()
        st.session_state.z0 = np.exp(rng.uniform(-0.3,0.5,14))
        st.session_state.drivers = {k:float(rng.uniform(0.5,1.5)) for k in DRIVERS}
        for i,(key,_) in enumerate(VARS):
            st.session_state[f"input_{key}"] = float(st.session_state.z0[i])
        for k in DRIVERS:
            st.session_state[f"driver_{k}"] = float(st.session_state.drivers[k])
        st.session_state.result=None
        st.rerun()
    if st.button("↺ Сбросить значения по умолчанию", use_container_width=True):
        st.session_state.z0=DEFAULT_Z.copy(); st.session_state.drivers={k:v[1] for k,v in DRIVERS.items()}
        for key,_ in VARS:
            st.session_state[f"input_{key}"] = 1.0
        for k,(_,default,_) in DRIVERS.items():
            st.session_state[f"driver_{k}"] = float(default)
        st.session_state.horizon=1.0; st.session_state.dt=0.01; st.session_state.alpha=1e-12
        st.session_state["horizon_input"] = 1.0
        st.session_state["dt_input"] = 0.01
        st.session_state["alpha_input"] = 1e-12
        st.session_state.result=None
        st.rerun()
    st.divider()
    st.markdown("### Параметры расчёта")
    st.session_state.horizon=st.number_input("Горизонт T", min_value=0.01, max_value=10.0, value=float(st.session_state.horizon), step=0.1, help="Интервал интегрирования от 0 до T.", key="horizon_input")
    st.session_state.dt=st.number_input("Шаг вывода Δt", min_value=0.0001, max_value=0.2, value=float(st.session_state.dt), step=0.005, format="%.4f", key="dt_input")
    st.session_state.alpha=st.number_input("Масштаб производных α (доп. параметр)", min_value=1e-14, max_value=1e-4, value=float(st.session_state.alpha), step=1e-12, format="%.2e", help="Умножает правые части системы на α для исследования динамики. В исходном PDF такого коэффициента нет; при α ≠ 1 рассчитывается масштабированный сценарий.", key="alpha_input")
    st.caption("Для расчёта используются f₁–f₆₆ из PDF. Начальные Zᵢ и внешние воздействия — сценарные параметры: источник не задаёт их численные значения.")

st.markdown("""
<div class="hero">
 <div class="hero-kicker">Системная динамика · 14 переменных · 66 вспомогательных функций</div>
 <h1>Моделирование последствий наводнения</h1>
 <p>Настройте исходные значения, задайте внешние воздействия и исследуйте динамику показателей во времени.</p>
</div>
""", unsafe_allow_html=True)

main_tabs = st.tabs(["Ввод данных", "Результаты расчёта", "Графики и диаграммы", "Формулы модели"])

with main_tabs[0]:
    input_tabs=st.tabs(["Общие параметры", "Показатели Zᵢ", "Возмущения", "Функции fₖ"])
    with input_tabs[0]:
        st.subheader("Общие параметры моделирования")
        c1,c2,c3=st.columns(3)
        c1.metric("Количество показателей", "14")
        c2.metric("Вспомогательные функции", "66")
        c3.metric("Численный метод", "Runge–Kutta 4/5")
        st.info("Уравнения и вспомогательные функции перенесены из предоставленного PDF. В исходнике нет достоверных начальных условий и временных рядов возмущений, поэтому они задаются пользователем как сценарий.")
        st.markdown("**Метод расчёта**")
        st.write("Адаптивный интегратор `scipy.solve_ivp` (RK45), с выводом результатов на равномерной сетке Δt. Параметр α умножает правые части для исследования динамики; он явно отмечен как дополнительный параметр приложения, отсутствующий в PDF.")
        st.markdown("**Проверки входных данных**")
        st.write("Все Zᵢ должны быть строго положительными, поскольку исходные функции содержат деление на переменные и логарифмы. Внешние воздействия также должны быть неотрицательными.")
    with input_tabs[1]:
        st.subheader("Начальные значения показателей")
        st.caption("Начальные значения 1.0 — условные значения по умолчанию, не наблюдения из PDF. Укажите значения из своего задания, если они заданы отдельно.")
        cols=st.columns(2)
        z_new=[]
        for i,(key,name) in enumerate(VARS):
            with cols[i%2]:
                val=st.number_input(f"{key} · {name}", min_value=0.000001, max_value=1e9, value=float(st.session_state.z0[i]), format="%.6g", key=f"input_{key}")
                z_new.append(val)
        st.session_state.z0=np.asarray(z_new,dtype=float)
    with input_tabs[2]:
        st.subheader("Внешние возмущения и параметры сценария")
        st.caption("PDF задаёт обозначения функций времени, но не приводит численные временные ряды. В приложении каждый драйвер задаётся постоянным на интервале значением.")
        dcols=st.columns(2)
        dnew={}
        for i,(k,(label,default,desc)) in enumerate(DRIVERS.items()):
            with dcols[i%2]:
                dnew[k]=st.number_input(label, min_value=0.0, max_value=1e6, value=float(st.session_state.drivers[k]), step=0.1, format="%.5g", help=desc, key=f"driver_{k}")
                st.caption(desc)
        st.session_state.drivers=dnew
    with input_tabs[3]:
        st.subheader("Справочник вспомогательных функций f₁–f₆₆")
        query=st.text_input("Поиск по номеру, переменной или описанию", placeholder="Например: f₁₃ или урожая")
        df_form=pd.DataFrame(FORMULAS,columns=["№","Функция","Смысл зависимости","Формула"])
        if query:
            mask=df_form.astype(str).apply(lambda c:c.str.contains(query,case=False,regex=False)).any(axis=1)
            df_form=df_form[mask]
        st.dataframe(df_form, use_container_width=True, hide_index=True, height=500)

with main_tabs[1]:
    st.subheader("Результаты численного моделирования")
    left, right=st.columns([1,4])
    with left:
        run=st.button("▶ Рассчитать модель", use_container_width=True, type="primary", key="run_result")
        st.caption("Расчёт выполняется по текущим параметрам слева и во вкладке «Ввод данных».")
    if run:
        try:
            if np.any(st.session_state.z0<=0): raise ValueError("Все начальные значения Zᵢ должны быть больше нуля.")
            n=max(2,int(math.ceil(st.session_state.horizon/st.session_state.dt))+1)
            t_eval=np.linspace(0.0,st.session_state.horizon,n)
            pars=dict(st.session_state.drivers)
            sol=solve_ivp(lambda t,y:scaled_rhs(t,y,pars,st.session_state.alpha),(0.0,st.session_state.horizon),st.session_state.z0,
                          method="RK45",t_eval=t_eval,rtol=1e-6,atol=1e-8,max_step=max(st.session_state.dt/2,1e-6))
            if not sol.success: raise RuntimeError(sol.message)
            if not np.isfinite(sol.y).all(): raise ValueError("Получены нечисловые значения. Уменьшите T или α, либо проверьте исходные параметры.")
            if np.max(np.abs(sol.y))>1e12: raise ValueError("Траектория вышла за пределы устойчивого численного диапазона. Уменьшите T или α; формулы PDF могут быть очень чувствительны к начальным значениям.")
            st.session_state.result={"t":sol.t,"y":sol.y,"z0":st.session_state.z0.copy(),"drivers":pars,"horizon":st.session_state.horizon,"dt":st.session_state.dt,"alpha":st.session_state.alpha}
        except Exception as e:
            st.error(f"Не удалось завершить расчёт: {e}")
    res=st.session_state.result
    if res:
        t=res["t"]; y=res["y"]; y0=res["z0"]
        end=y[:,-1]; mean=np.mean(y,axis=1); maxs=np.max(y,axis=1); mins=np.min(y,axis=1)
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Конечный ущерб территории Z₁₄", f"{end[13]:,.5g}", f"{(end[13]-y0[13]):+,.4g} к началу")
        m2.metric("Конечное число погибших Z₁", f"{end[0]:,.5g}", f"{(end[0]-y0[0]):+,.4g} к началу")
        m3.metric("Среднее Z₃ · площадь зоны ЧС", f"{mean[2]:,.5g}")
        m4.metric("Рассчитано точек", f"{len(t):,}")
        st.markdown("#### Сводная таблица по показателям")
        rows=[]
        for i,(key,name) in enumerate(VARS):
            change=(end[i]-y0[i])/abs(y0[i])*100 if y0[i] else np.nan
            rows.append({"Показатель":key,"Описание":name,"Z(0)":y0[i],"Z(T)":end[i],"Изменение":end[i]-y0[i],"Изменение, %":change,"Среднее":mean[i],"Максимум":maxs[i],"Минимум":mins[i]})
        result_df=pd.DataFrame(rows)
        st.dataframe(result_df.style.format({c:"{:,.5g}" for c in result_df.columns if c not in ["Показатель","Описание"]}),use_container_width=True,hide_index=True,height=520)
        st.download_button("⬇ Скачать таблицу результатов CSV", data=result_df.to_csv(index=False,sep=";",decimal=",",encoding="utf-8-sig"), file_name="floodlab_summary.csv", mime="text/csv")
        st.download_button("⬇ Скачать временной ряд CSV", data=pd.DataFrame({"t":t,**{VARS[i][0]:y[i] for i in range(14)}}).to_csv(index=False,sep=";",decimal=",",encoding="utf-8-sig"), file_name="floodlab_timeseries.csv", mime="text/csv")
        st.warning("Результаты отражают выбранный сценарий и постоянные внешние воздействия. Это не оперативный прогноз реального наводнения; исходный PDF не задаёт начальные данные и ряды возмущений.")
    else:
        st.info("Пока нет результатов. Нажмите «Рассчитать модель», чтобы построить таблицу и метрики.")

with main_tabs[2]:
    st.subheader("Графики и диаграммы")
    res=st.session_state.result
    if not res:
        st.info("Сначала выполните расчёт во вкладке «Результаты расчёта».")
    else:
        t=res["t"]; y=res["y"]
        graph_tabs=st.tabs(["Динамика Zᵢ(t)","Все показатели", "Возмущения", "Лепестковая диаграмма", "Сечения по времени", "Матрица взаимосвязи"])
        with graph_tabs[0]:
            chosen=st.multiselect("Показатели для отображения",options=[v[0] for v in VARS],default=["Z1","Z3","Z5","Z14"])
            fig=go.Figure()
            for i,(key,name) in enumerate(VARS):
                if key in chosen: fig.add_trace(go.Scatter(x=t,y=y[i],name=f"{key} · {name}",mode="lines",line=dict(width=2.5)))
            fig.update_layout(template="plotly_dark",height=520,margin=dict(l=15,r=15,t=35,b=15),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(18,20,28,.7)",xaxis_title="Время t",yaxis_title="Значение показателя",legend=dict(orientation="h",y=-.22))
            st.plotly_chart(fig,use_container_width=True)
        with graph_tabs[1]:
            long=pd.DataFrame(y.T,columns=[v[0] for v in VARS]); long.insert(0,"t",t)
            melted=long.melt(id_vars="t",var_name="Показатель",value_name="Значение")
            fig=px.line(melted,x="t",y="Значение",color="Показатель",template="plotly_dark",title="Динамика всех 14 показателей")
            fig.update_layout(height=620,margin=dict(t=55,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(18,20,28,.7)")
            st.plotly_chart(fig,use_container_width=True)
        with graph_tabs[2]:
            driver_df=pd.DataFrame({"Возмущение":list(res["drivers"].keys()),"Значение":list(res["drivers"].values()),"Описание":[DRIVERS[k][2] for k in res["drivers"]]})
            fig=px.bar(driver_df,x="Возмущение",y="Значение",color="Значение",color_continuous_scale=["#8f3567","#f05a9d","#ffb3d2"],template="plotly_dark",title="Заданные постоянные внешние воздействия")
            fig.update_layout(height=430,showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(18,20,28,.7)")
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Временные зависимости внешних функций Rw(t), Hw(t) и остальных в PDF не заданы численно; здесь используются постоянные сценарные значения.")
        with graph_tabs[3]:
            end=y[:,-1]; start=y[:,0]
            norm=np.maximum(np.abs(start),1e-9)
            vals=np.clip(end/norm,0,10)
            fig=go.Figure(go.Scatterpolar(r=list(vals)+[vals[0]],theta=[v[0] for v in VARS]+[VARS[0][0]],fill="toself",line=dict(color="#f05a9d",width=3),fillcolor="rgba(240,90,157,.2)"))
            fig.update_layout(template="plotly_dark",title="Отношение конечного значения к начальному (обрезка шкалы 0–10)",height=560,polar=dict(radialaxis=dict(visible=True,range=[0,10])),paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Для сопоставимости показатели показаны как Z(T)/Z(0); значения выше 10 на графике ограничены 10 и остаются точными в таблице.")
        with graph_tabs[4]:
            fractions=st.multiselect("Моменты времени",options=[0.0,0.25,0.5,0.75,1.0],default=[0.0,0.5,1.0],format_func=lambda x:f"{x:.0%} интервала")
            rows=[]
            for frac in fractions:
                j=int(round(frac*(len(t)-1)))
                for i,(key,name) in enumerate(VARS): rows.append({"Момент":f"{frac:.0%}","Показатель":key,"Значение":y[i,j]})
            if rows:
                df=pd.DataFrame(rows)
                fig=px.line(df,x="Момент",y="Значение",color="Показатель",markers=True,template="plotly_dark",title="Сравнение выбранных временных сечений")
                fig.update_layout(height=520,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(18,20,28,.7)")
                st.plotly_chart(fig,use_container_width=True)
        with graph_tabs[5]:
            corr=np.corrcoef(y)
            corr=np.nan_to_num(corr,nan=0.0)
            fig=px.imshow(corr,x=[v[0] for v in VARS],y=[v[0] for v in VARS],zmin=-1,zmax=1,color_continuous_scale="RdBu",origin="lower",text_auto=".2f",aspect="auto",template="plotly_dark",title="Корреляция временных траекторий показателей")
            fig.update_layout(height=680,paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Это корреляция рассчитанных временных рядов, а не отдельная причинно-следственная матрица из исходного PDF.")

with main_tabs[3]:
    st.subheader("Математическая модель")
    st.markdown("**Обозначения:** Z₁–Z₁₄ — переменные состояния; Rw, Hw, Sem, Vhr, Chr, Sev, Ch, Hr, Vh, Lr — внешние функции времени. Поскольку их временные зависимости в PDF не заданы численно, расчёт использует заданные постоянные значения.")
    st.markdown("### Система дифференциальных уравнений")
    st.caption("Структура правых частей переписана по странице «Система уравнений» в файле «Наводнения.pdf». Коэффициент α умножает правые части только в дополнительном режиме масштабированного сценария и не входит в опубликованные формулы.")
    for i,eq in enumerate(SYSTEM_EQS,1):
        with st.container(border=True):
            st.markdown(f"**Уравнение {i}**")
            st.code(eq,language=None)
    st.markdown("### Вспомогательные функции f₁–f₆₆")
    st.dataframe(pd.DataFrame(FORMULAS,columns=["№","Функция","Описание из пояснений PDF","Формула"]),use_container_width=True,hide_index=True,height=700)
    with st.expander("Численная реализация и ограничения"):
        st.markdown("- Интегратор: `scipy.integrate.solve_ivp`, метод RK45.\n- Для логарифмов и деления исходные Zᵢ должны быть положительными.\n- Внешние функции времени представлены постоянными пользовательскими значениями.\n- Исходный PDF не содержит набора начальных условий или измеренных временных рядов возмущений; значения по умолчанию — демонстрационный сценарий.\n- При резком росте показателей уменьшите горизонт T или масштаб производных α и проверьте значения входных параметров.")

st.markdown("<div style='margin-top:2rem;padding:1rem 0;border-top:1px solid #343746;color:#8e92a1;font-size:.82rem'>FloodLab · учебная система моделирования · математические зависимости: предоставленный файл «Наводнения.pdf» · интерфейс адаптирован по примеру приложения моделирования характеристик хранилища данных.</div>",unsafe_allow_html=True)
