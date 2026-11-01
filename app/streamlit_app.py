# -*- coding: utf-8 -*-
"""
CallSense — لوحة المشرف (Streamlit)
المكان: app/streamlit_app.py
التشغيل من مجلد المشروع الرئيسي:  streamlit run app/streamlit_app.py
"""

import html
import io
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# نخلي مجلد المشروع الرئيسي يظهر لبايثون عشان يشتغل: from src.xxx import ...
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

# مهم: نستورد هذي المكتبات قبل src.unified_inference
# لأن unified_inference يفرض الوضع offline، وهذا الترتيب يخلي e5 وQwen يتحملون عادي
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- الإعدادات
EMBED_MODEL = "intfloat/multilingual-e5-base"
LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CSV = ROOT / "data" / "raw" / "team_calls" / "team_transcriptions_final.csv"

INTENT_AR = {
    "billing": "الفواتير",
    "technical": "مشكلة تقنية",
    "cancellation": "إلغاء",
    "complaint": "شكوى",
    "information": "استفسار",
}

st.set_page_config(page_title="CallSense", page_icon="🎧", layout="wide")

# ---------------------------------------------------------------- التصميم
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

:root{
  --ink:#0E1A24; --panel:#15242F; --line:#243645;
  --signal:#2EC4B6; --amber:#F4B942; --alert:#E5484D;
  --text:#E8EEF2; --muted:#8FA3B3;
}
html, body, [class*="css"], .stApp {
  font-family:'IBM Plex Sans Arabic', sans-serif !important;
  direction:rtl;
}
.stApp { background:var(--ink); color:var(--text); }
header[data-testid="stHeader"] { background:transparent; }

/* نثبّت لون النص بغض النظر عن ثيم ستريملت */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li,
.stApp label, .stApp span, .stApp small, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stWidgetLabel"] {
  color:var(--text);
}
.stApp [data-testid="stCaptionContainer"] { color:var(--muted); }

/* الحقول والأزرار */
.stApp input, .stApp textarea {
  background:var(--panel) !important; color:var(--text) !important;
  border:1px solid var(--line) !important; border-radius:10px;
}
.stApp [data-testid="stFileUploaderDropzone"] {
  background:var(--panel); border:1px dashed var(--line);
}
.stApp [data-testid="stFileUploaderDropzone"] * { color:var(--text); }
.stApp .stButton>button[kind="primary"] {
  background:var(--signal); color:#06211e; border:none;
}
.stApp .stButton>button[kind="secondary"] {
  background:var(--panel); color:var(--text); border:1px solid var(--line);
}
.stApp [data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:12px; }
.stApp [data-baseweb="tab-list"] { gap:6px; }
.block-container { padding-top:1.5rem; max-width:1200px; }
section[data-testid="stSidebar"] { background:var(--panel); border-left:1px solid var(--line); }

/* الهيدر: الموجة الصوتية هي العنصر المميز */
.hero{
  background:linear-gradient(135deg,#15242F 0%,#10202B 100%);
  border:1px solid var(--line); border-radius:18px;
  padding:26px 30px; margin-bottom:20px;
  display:flex; align-items:center; justify-content:space-between; gap:20px;
}
.hero h1{ margin:0; font-size:2.1rem; font-weight:700; color:var(--text); }
.hero p{ margin:6px 0 0; color:var(--muted); font-size:1rem; }
.wave{ display:flex; align-items:center; gap:4px; height:56px; direction:ltr; }
.wave span{
  width:5px; border-radius:3px; background:var(--signal); opacity:.9;
  animation:pulse 1.6s ease-in-out infinite;
}
@keyframes pulse{ 0%,100%{transform:scaleY(.35)} 50%{transform:scaleY(1)} }
@media (prefers-reduced-motion: reduce){ .wave span{animation:none} }

/* فقاعات المحادثة */
.turn{ padding:12px 16px; border-radius:14px; margin:8px 0; max-width:82%;
       line-height:1.9; border:1px solid var(--line); }
.turn .who{ font-size:.78rem; color:var(--muted); margin-bottom:2px; }
.turn.customer{ background:#12343a; border-color:#1d5560; margin-left:auto; margin-right:0; }
.turn.agent{ background:var(--panel); margin-right:auto; margin-left:0; }
.turn.unknown{ background:transparent; border-style:dashed; color:var(--muted); }

/* شرائط النوايا */
.bar-row{ margin:10px 0; }
.bar-head{ display:flex; justify-content:space-between; font-size:.95rem; margin-bottom:4px; }
.bar-bg{ background:var(--line); border-radius:8px; height:10px; overflow:hidden; direction:ltr; }
.bar-fill{ height:100%; border-radius:8px; background:var(--signal); }
.bar-fill.off{ background:#4b6071; }
.tag{ display:inline-block; padding:2px 10px; border-radius:20px; font-size:.78rem;
      background:rgba(46,196,182,.15); color:var(--signal); margin-right:8px; }

/* بطاقة التصعيد */
.esc{ border-radius:16px; padding:20px; border:1px solid var(--line); background:var(--panel); }
.esc .big{ font-size:2.6rem; font-weight:700; line-height:1.1; }
.esc.high{ border-color:var(--alert); } .esc.high .big{ color:var(--alert); }
.esc.low{ border-color:var(--signal); } .esc.low .big{ color:var(--signal); }
.esc .note{ color:var(--muted); font-size:.85rem; margin-top:8px; }

/* نتائج البحث */
.hit{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
      padding:14px 18px; margin:10px 0; }
.hit .meta{ color:var(--muted); font-size:.82rem; margin-bottom:6px; }
.hit .score{ color:var(--amber); font-weight:600; }

.digest{ background:var(--panel); border:1px solid var(--line); border-right:4px solid var(--signal);
         border-radius:14px; padding:18px 22px; line-height:2; white-space:pre-wrap; }

.stTabs [data-baseweb="tab"]{ font-weight:600; font-size:1rem; }
.stButton>button{ border-radius:12px; font-weight:600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def build_logo():
    """شعار CallSense: موجة صوتية داخل أيقونة + الاسم. العمود العنبري = لحظة الإحساس."""
    heights = [14, 26, 40, 58, 34, 22, 12]
    bars_svg = ""
    for i, h in enumerate(heights):
        color = "#F4B942" if i == 3 else "#2EC4B6"
        bars_svg += (
            f'<rect class="lb" x="{22 + i * 8.5}" y="{44 - h / 2}" width="5" height="{h}" '
            f'rx="2.5" fill="{color}" style="animation-delay:{i * 0.12:.2f}s"/>'
        )
    # مهم: السطر الواحد بدون فراغات في البداية، عشان markdown ما يحوله لكود
    return (
        '<svg viewBox="0 0 330 88" width="330" height="88" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="CallSense">'
        '<style>.lb{transform-box:fill-box;transform-origin:center;animation:lg 1.6s ease-in-out infinite}'
        '@keyframes lg{0%,100%{transform:scaleY(.45)}50%{transform:scaleY(1)}}'
        '@media (prefers-reduced-motion:reduce){.lb{animation:none}}</style>'
        '<rect x="2" y="2" width="84" height="84" rx="26" fill="#15242F" stroke="#2EC4B6" stroke-width="2"/>'
        f'{bars_svg}'
        '<text x="104" y="57" font-family="IBM Plex Sans Arabic, sans-serif" font-size="46" letter-spacing="-0.5">'
        '<tspan font-weight="400" fill="#E8EEF2">Call</tspan>'
        '<tspan font-weight="700" fill="#2EC4B6">Sense</tspan></text>'
        '</svg>'
    )


st.markdown(
    f'<div class="hero"><div style="direction:ltr">{build_logo()}</div>'
    '<p style="max-width:380px;text-align:right">من صوت المكالمة إلى النية، خطر التصعيد، والمكالمات المشابهة</p></div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- تحميل الموديلات (مرة وحدة)
@st.cache_resource(show_spinner="جاري تحميل Whisper وpyannote وCAMeLBERT...")
def load_audio_pipeline(hf_token):
    from src.audio_intent_pipeline import AudioIntentPipeline
    return AudioIntentPipeline(hf_token=hf_token)


@st.cache_resource(show_spinner="جاري تحميل موديل التصعيد...")
def load_escalation_model():
    from src.unified_inference import UnifiedInferencePipeline
    return UnifiedInferencePipeline()


@st.cache_resource(show_spinner="جاري تحميل موديل البحث (e5)...")
def load_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device)


@st.cache_resource(show_spinner="جاري تحميل Qwen لتوليد الملخص (يحتاج وقت)...")
def load_llm():
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    # Apple Silicon: FP16 على MPS بدل FP32 + offload إلى القرص.
    if torch.backends.mps.is_available():
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_NAME, dtype=torch.float16, low_cpu_mem_usage=True
        ).to("mps")
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
    llm.eval()
    return tok, llm


# ---------------------------------------------------------------- البيانات والبحث
def read_calls(source):
    df = pd.read_csv(source)
    required = {"call_id", "whisper_transcript"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"أعمدة ناقصة: {missing}. الموجود: {list(df.columns)}")
    df = df.dropna(subset=["whisper_transcript"]).copy()
    df["transcript"] = df["whisper_transcript"].astype(str).str.strip()
    df = df[df["transcript"].ne("")].reset_index(drop=True)
    df["audio_id"] = df["call_id"].astype(str)
    return df


@st.cache_data(show_spinner="جاري بناء فهرس البحث...")
def build_embeddings(texts: tuple):
    embedder = load_embedder()
    return embedder.encode(
        ["passage: " + t for t in texts],
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def retrieve(query, df, embeddings, k=5):
    embedder = load_embedder()
    q = embedder.encode(
        ["query: " + query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    scores = embeddings @ q
    idx = np.argsort(scores)[::-1][:k]
    return [
        {
            "audio_id": df.iloc[i]["audio_id"],
            "text": df.iloc[i]["transcript"],
            "score": float(scores[i]),
            "dialect": str(df.iloc[i].get("dialect", "")),
        }
        for i in idx
    ]


# ---------------------------------------------------------------- الملخص المبني على المصادر
def clean_digest_note(note):
    """تنظيف مخرجات النموذج الصغيرة دون إضافة معلومات غير موجودة."""
    note = re.sub(r"\[S\d+\]", "", note)
    note = note.replace("\\n", " ").replace("\n", " ")
    note = re.sub(r"(?<!\w)-n(?!\w)", " ", note)
    note = re.sub(r"(?i)^(?:الملخص|الإجابة|الشكوى|ملخص المكالمة)\s*[:：]\s*", "", note.strip())
    note = re.sub(r"\s+", " ", note).strip(" -–•:،؛")
    # إزالة التكرار الحرفي الطويل، دون تغيير معنى النص
    parts = re.split(r"(?<=[.!؟])\s+", note)
    unique = list(dict.fromkeys(x.strip() for x in parts if x.strip()))
    return " ".join(unique)[:320].strip()


def extractive_note(transcript, query):
    """مقتطف حرفي من المصدر عند تعثر التوليد؛ لا نضيف حقائق."""
    text = re.sub(r"\s+", " ", str(transcript)).strip()
    # مقاطع صغيرة من المكالمة، مع الاحتفاظ بالنص الأصلي.
    parts = re.split(r"(?<=[.!؟؛])\s+|(?=\b(?:العميل|الموظف|خدمة العملاء)\s*[:：])", text)
    parts = [p.strip() for p in parts if len(p.strip()) >= 18]
    if not parts:
        parts = [text] if text else []
    if not parts:
        return ""
    qwords = set(re.findall(r"[\u0621-\u064a]{3,}", query))
    topic_words = {"فاتور", "رسوم", "مبلغ", "دفع", "اشتراك", "انترنت", "شبك", "خدم", "الغاء", "شكوى", "مشكلة", "رصيد"}
    def score(part):
        words = set(re.findall(r"[\u0621-\u064a]{3,}", part))
        topical = sum(any(w.startswith(root) for w in words) for root in topic_words)
        overlap = len(words & qwords)
        noise = sum(x in part for x in ("أهلا", "مرحبا", "كيف أساعدك", "السلام عليكم", "شكراً"))
        return topical * 2 + overlap * 3 - noise * 3
    best = max(parts, key=score)
    # اقتطاع حرفي فقط، وليس تلخيصًا توليديًا.
    if len(best) > 260:
        best = best[:260].rsplit(" ", 1)[0] + "…"
    return best


def is_supported_note(note, transcript):
    """فحص محافظ؛ التشابه اللفظي ليس إثباتًا دلاليًا."""
    if not note or len(note) < 18 or len(note) > 220:
        return False
    bad = ("كمساعد", "يجب أن", "عذراً على", "المعلومات المتاحة", "لا أستطيع", "بصفتي", "كنموذج")
    if any(x in note for x in bad) or "�" in note or "�" in note:
        return False
    tokens = set(re.findall(r"[\u0621-\u064a]{4,}", note))
    source_tokens = set(re.findall(r"[\u0621-\u064a]{4,}", transcript))
    if len(tokens) < 4:
        return False
    return len(tokens & source_tokens) / len(tokens) >= 0.72


def make_grounded_digest(query, sources, max_new_tokens=75):
    """Qwen أولاً، ثم مقتطف حرفي واضح إذا كانت النتيجة غير موثوقة."""
    notes = []
    # تحميل النموذج مرة واحدة؛ فشل تحميله لا يمنع عرض مقتطفات المصادر.
    try:
        tok, llm = load_llm()
    except Exception:
        tok, llm = None, None
    for i, hit in enumerate(sources, 1):
        transcript = str(hit["text"]).strip()
        if not transcript:
            continue
        note = ""
        if tok is not None:
            try:
                messages = [
                    {"role": "system", "content": "استخرج شكوى العميل في جملة عربية واحدة من نص المكالمة فقط. لا تكتب رد خدمة العملاء أو تعليمات أو اعتذارات."},
                    {"role": "user", "content": f"نص المكالمة: {transcript[:1100]}\nاكتب شكوى العميل فقط في جملة قصيرة."},
                ]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1300).to(llm.device)
                with torch.inference_mode():
                    output = llm.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False, repetition_penalty=1.12,
                                          pad_token_id=tok.eos_token_id)
                note = clean_digest_note(tok.decode(
                    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
            except Exception:
                note = ""
        if is_supported_note(note, transcript):
            notes.append(f"- {note} [S{i}]")
        else:
            excerpt = extractive_note(transcript, query)
            if excerpt:
                notes.append(f"- {excerpt} [S{i}]")
    return "\n".join(notes) if notes else "لا توجد نصوص كافية لعرض مقتطفات موثقة."


def check_citations(digest, sources):
    valid = {f"S{i}" for i in range(1, len(sources) + 1)}
    found = re.findall(r"\[(S\d+)\]", digest)
    return found, sorted(set(found) - valid)


@st.cache_resource(show_spinner="جاري تحميل نموذج الصوت العربي MMS-TTS...")
def load_tts():
    from transformers import VitsModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ara")
    model = VitsModel.from_pretrained("facebook/mms-tts-ara")
    model.eval()
    return tokenizer, model


def speak(text):
    """نطق عربي باستخدام MMS-TTS مع تنظيف الاستشهادات والرموز."""
    import soundfile as sf
    import numpy as np
    clean = re.sub(r"\[S\d+\]", "", text)
    clean = clean.replace("\\n", " ").replace("\n", " ")
    clean = re.sub(r"(?<!\w)-n(?!\w)", " ", clean)
    clean = re.sub(r"(?m)^\s*[-•]\s*", "", clean)
    clean = re.sub(r"[^\w\s،؛.!؟]", " ", clean, flags=re.UNICODE)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        raise ValueError("الملخص فارغ.")
    tokenizer, model = load_tts()
    # مقاطع أقصر لتحسين وضوح النطق وتخفيف الذاكرة
    chunks = re.split(r"(?<=[.!؟])\s+", clean)
    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        chunks = [clean]
    audio = []
    sr = model.config.sampling_rate
    for chunk in chunks:
        for start in range(0, len(chunk), 180):
            inputs = tokenizer(chunk[start:start + 180], return_tensors="pt")
            with torch.inference_mode():
                wave = model(**inputs).waveform.squeeze().cpu().numpy()
            audio.append(wave)
            audio.append(np.zeros(int(sr * 0.25), dtype=np.float32))
    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(audio), sr, format="WAV")
    return buffer.getvalue()


# ---------------------------------------------------------------- دوال العرض
def build_turns(words):
    """دمج الكلمات المتتالية لنفس المتحدث في دور واحد."""
    turns = []
    for w in words:
        if turns and turns[-1]["speaker"] == w["speaker"]:
            turns[-1]["text"] += " " + w["word"]
            turns[-1]["end"] = w["end_sec"]
        else:
            turns.append(
                {"speaker": w["speaker"], "text": w["word"],
                 "start": w["start_sec"], "end": w["end_sec"]}
            )
    return turns


def render_turns(turns, customer_speaker):
    for t in turns:
        if t["speaker"] == customer_speaker:
            cls, who = "customer", "العميل"
        elif t["speaker"] == "UNKNOWN":
            cls, who = "unknown", "غير محدد"
        else:
            cls, who = "agent", "الموظف"
        st.markdown(
            f'<div class="turn {cls}"><div class="who">{who} · {t["start"]:.1f}s – {t["end"]:.1f}s</div>'
            f'{html.escape(t["text"])}</div>',
            unsafe_allow_html=True,
        )


def render_intents(probs, preds):
    for label, p in sorted(probs.items(), key=lambda x: -x[1]):
        on = preds[label]
        tag = '<span class="tag">مُفعّل</span>' if on else ""
        st.markdown(
            f"""<div class="bar-row">
  <div class="bar-head"><span>{INTENT_AR.get(label, label)}{tag}</span><span>{p*100:.0f}%</span></div>
  <div class="bar-bg"><div class="bar-fill {'' if on else 'off'}" style="width:{p*100:.1f}%"></div></div>
</div>""",
            unsafe_allow_html=True,
        )


def render_escalation(prob, flag):
    cls = "high" if flag else "low"
    title = "خطر تصعيد محتمل" if flag else "لا يوجد تصعيد محتمل"
    st.markdown(
        f"""<div class="esc {cls}">
  <div>{title}</div><div class="big">{prob*100:.0f}%</div>
  <div class="note">هذا مؤشر مساعد وليس بديلًا عن الاستماع للمكالمة. غياب التصعيد لا يعني غياب المشكلة.</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_hits(hits):
    for i, h in enumerate(hits, 1):
        dialect = f" · {html.escape(h['dialect'])}" if h["dialect"] not in ("", "nan") else ""
        st.markdown(
            f"""<div class="hit">
  <div class="meta">[S{i}] call_id = {html.escape(str(h['audio_id']))}{dialect} · تشابه <span class="score">{h['score']:.3f}</span></div>
  {html.escape(h['text'])}
</div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- الشريط الجانبي
with st.sidebar:
    st.header("الإعدادات")

    hf_token = st.text_input(
        "HF Token (لفصل المتحدثين)",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
    )
    customer_speaker = st.radio(
        "من هو العميل؟", ["SPEAKER_00", "SPEAKER_01"], horizontal=True,
        help="pyannote يعطي المتحدثين أسماء عشوائية. جرّبي الاثنين إذا طلعت النتيجة معكوسة.",
    )
    strategy = st.radio(
        "طريقة دمج النوافذ", ["max_pooling", "mean_pooling"], horizontal=True
    )

    st.divider()
    st.subheader("قاعدة المكالمات")
    csv_up = st.file_uploader("ملف CSV (call_id + whisper_transcript)", type="csv")
    st.caption(f"إذا ما رفعتِ ملف، يُستخدم: {DEFAULT_CSV.name}")

# تحميل بيانات البحث
calls_df, calls_emb = None, None
try:
    if csv_up is not None:
        calls_df = read_calls(csv_up)
    elif DEFAULT_CSV.exists():
        calls_df = read_calls(DEFAULT_CSV)
except Exception as e:
    st.sidebar.error(str(e))

if calls_df is not None:
    st.sidebar.success(f"تم تحميل {len(calls_df)} مكالمة")

tab_call, tab_search, tab_digest = st.tabs(["تحليل مكالمة", "بحث في المكالمات", "ملخص المشرف"])

# ================================================================ تبويب 1: تحليل مكالمة
with tab_call:
    audio = st.file_uploader("ارفعي تسجيل المكالمة", type=["wav", "mp3", "m4a", "ogg", "flac"])

    if audio is not None:
        st.audio(audio)

    run = st.button("حلّلي المكالمة", type="primary", disabled=audio is None)

    if run and audio is not None:
        if not hf_token:
            st.error("أدخلي HF Token في الشريط الجانبي (مطلوب لفصل المتحدثين).")
        else:
            suffix = Path(audio.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio.getvalue())
                tmp_path = tmp.name

            try:
                pipe = load_audio_pipeline(hf_token)
                esc_model = load_escalation_model()

                t0 = time.time()
                with st.spinner("Whisper + فصل المتحدثين + تصنيف النية..."):
                    result = pipe.predict(
                        tmp_path,
                        customer_speaker=customer_speaker,
                        call_id=Path(audio.name).stem,
                        strategy=strategy,
                    )
                with st.spinner("تقدير خطر التصعيد..."):
                    esc = esc_model.predict(result["call_id"], result["customer_transcript"])
                result["escalation_probability"] = esc["escalation_probability"]
                result["escalation_flag"] = esc["escalation_flag"]
                result["latency"] = time.time() - t0
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"صار خطأ أثناء التحليل: {e}")
            finally:
                os.unlink(tmp_path)

    res = st.session_state.get("result")
    if res:
        c1, c2, c3 = st.columns(3)
        c1.metric("زمن المعالجة", f"{res['latency']:.1f} ث")
        c2.metric("عدد الكلمات", len(res["aligned_words"]))
        c3.metric("المتحدث المختار كعميل", res["customer_speaker"])

        left, right = st.columns([3, 2], gap="large")

        with left:
            st.subheader("المحادثة")
            render_turns(build_turns(res["aligned_words"]), res["customer_speaker"])

        with right:
            st.subheader("خطر التصعيد")
            render_escalation(res["escalation_probability"], res["escalation_flag"])
            st.subheader("نية العميل")
            render_intents(res["intent_probabilities"], res["intent_predictions"])

        st.subheader("مكالمات سابقة مشابهة")
        if calls_df is None:
            st.info("ارفعي ملف CSV للمكالمات من الشريط الجانبي عشان تظهر المكالمات المشابهة.")
        else:
            calls_emb = build_embeddings(tuple(calls_df["transcript"]))
            render_hits(retrieve(res["customer_transcript"], calls_df, calls_emb, k=2))

        with st.expander("النص الكامل (Whisper)"):
            st.write(res["full_transcript"])
        with st.expander("نص العميل فقط"):
            st.write(res["customer_transcript"])

# ================================================================ تبويب 2: البحث
with tab_search:
    if calls_df is None:
        st.info("ارفعي ملف CSV للمكالمات من الشريط الجانبي.")
    else:
        calls_emb = build_embeddings(tuple(calls_df["transcript"]))
        q = st.text_input("اكتبي سؤالك بالعربي أو الإنجليزي", placeholder="عملاء منزعجون من الفواتير")
        k = st.slider("عدد النتائج", 1, 10, 5)
        if q.strip():
            render_hits(retrieve(q, calls_df, calls_emb, k=k))

# ================================================================ تبويب 3: الملخص
with tab_digest:
    if calls_df is None:
        st.info("ارفعي ملف CSV للمكالمات من الشريط الجانبي.")
    else:
        calls_emb = build_embeddings(tuple(calls_df["transcript"]))
        topic = st.text_input("موضوع الملخص", value="أبرز شكاوى العملاء المتعلقة بالفواتير")
        k2 = st.slider("عدد المصادر", 3, 10, 8, key="k_digest")
        want_voice = st.checkbox("اقرئي الملخص صوتيًا (MMS-TTS العربي)", value=False)

        if st.button("ولّدي الملخص", type="primary"):
            sources = retrieve(topic, calls_df, calls_emb, k=k2)
            with st.spinner("جاري إعداد ملخص مدعوم بالمصادر..."):
                digest = make_grounded_digest(topic, sources)
            st.session_state["digest"] = (digest, sources, want_voice)

        saved = st.session_state.get("digest")
        if saved:
            digest, sources, voice = saved
            st.markdown(f'<div class="digest">{html.escape(digest)}</div>', unsafe_allow_html=True)

            found, invalid = check_citations(digest, sources)
            if not found:
                st.warning("الملخص ما فيه أي استشهاد [S#].")
            elif invalid:
                st.error(f"استشهادات غير موجودة: {invalid}")
            else:
                st.caption("قد تظهر مقتطفات حرفية بدل الملخص عند ضعف مخرجات Qwen. الاستشهادات تشير للمصادر، لكن راجعي الدقة يدويًا.")

            if voice:
                if found and not invalid:
                    try:
                        with st.spinner("جاري توليد الصوت باستخدام MMS-TTS..."):
                            audio_bytes = speak(digest)
                        st.audio(audio_bytes, format="audio/wav")
                    except Exception as exc:
                        st.error(f"تعذّر توليد الصوت باستخدام MMS-TTS: {exc}")
                else:
                    st.info("لن يتم توليد الصوت قبل وجود ملخص مع استشهادات صالحة.")

            st.subheader("المصادر المستخدمة")
            render_hits(sources)