from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import time
import re

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


app = Flask(__name__)
app.secret_key = "senior-project-key"
CORS(app)


LAWS_INDEX_FOLDER = "faiss_laws_index"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

FEES_FILE_2025 = (
    "amman_building_reg_amendments_2019_2022_2025_consolidated_ar_with_fees_tables.md" 
)


# Domain-aware keyword routing for retrieval prioritization
FIRECODE_KEYWORDS = [
    "كود الحريق", "كودة الحريق", "الوقاية من الحريق", "الدفاع المدني", "حريق", "إنذار", "رشاش", "مرشات", "sprinkler",
    "طفاية", "إطفاء", "مخارج", "مخرج", "وسائل الخروج", "مسار خروج", "سعة", "نهاية مسدودة", "سلالم", "سلم", "ممر", "عرض الممر",
    "حمولة الإشغال", "occupant load", "اشغال", "إشغال", "تصنيف الإشغال", "خطورة المحتويات",
]

DEFINITION_KEYWORDS = [
    "تعريف", "التعريف", "ما معنى", "المقصود", "يعني", "يُقصد", "تعريفات", "glossary"
]

def is_firecode_query(q: str) -> bool:
    q = q or ""
    return any(k in q for k in FIRECODE_KEYWORDS)

def is_definition_query(q: str) -> bool:
    q = q or ""
    return any(k in q for k in DEFINITION_KEYWORDS)

def dedup_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        key = (d.page_content.strip()[:200], d.metadata.get("filename"), d.metadata.get("id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out



ARABIC_NUMBERS = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

def normalize_arabic_numbers(text: str) -> str:
    """Convert Arabic numerals to Western for consistent parsing."""
    if not text or not isinstance(text, str):
        return ""

    for ar, en in ARABIC_NUMBERS.items():
        text = text.replace(ar, en)
    return text



# Load embeddings model once at startup

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={'normalize_embeddings': True},
)


def make_legal_splitter() -> RecursiveCharacterTextSplitter:
    """Custom splitter respecting Arabic legal document structure."""
    return RecursiveCharacterTextSplitter(
        separators=[
            "\nالمادة ",  # Article boundary
            "\nالبند ",   # Clause boundary
            "\nالفصل ",  # Chapter boundary
            "\n\n",      # Paragraph break
            ". ",        # Sentence boundary
            "\n"         # Line break
        ],
        chunk_size=1000,
        chunk_overlap=200,
    )

def _load_markdown_and_jsonl_documents(folder: str) -> List[Document]:
    """Load and parse structured legal documents (Markdown + JSONL)."""
    docs: List[Document] = []
    
    # JSONL: pre-chunked fire code with metadata (source, page numbers)
    for name in os.listdir(folder):
        if name.lower().endswith(".jsonl"):
            path = os.path.join(folder, name)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        import json
                        obj = json.loads(line)
                    except Exception:
                        continue
                    content = obj.get("content")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    meta = {k: v for k, v in obj.items() if k != "content"}
                    if "filename" not in meta:
                        meta["filename"] = meta.get("source_file", name)
                    if "source" not in meta:
                        meta["source"] = meta.get("doc_id") or "unknown"
                    meta["format"] = "jsonl_chunk"
                    docs.append(Document(page_content=content, metadata=meta))
        
        # Markdown: building regulations with hierarchical structure
        elif name.lower().endswith(".md"):
            path = os.path.join(folder, name)
            try:
                text = open(path, "r", encoding="utf-8").read()
            except Exception:
                continue
            splitter = make_legal_splitter()
            for d in splitter.create_documents([text]):
                d.metadata.update({"filename": name, "format": "markdown", "source": "laws"})
                docs.append(d)
    return docs

def load_base_vectorstore():
    """Load or rebuild FAISS index from cleaned legal documents."""
    t0 = time.time()
    try:
        # Try safe load first (no deserialization vulnerabilities)
        vs = FAISS.load_local(LAWS_INDEX_FOLDER, embeddings)
        print(f"[startup] Loaded FAISS from '{LAWS_INDEX_FOLDER}' in {time.time()-t0:.2f}s")
        return vs
    except Exception as e:
        # Fallback: allow pickle-based deserialization if explicitly enabled
        allow_dangerous = os.getenv("ALLOW_DANGEROUS_FAISS_LOAD", "1").lower() in {"1","true","yes"}
        if allow_dangerous:
            print("[startup] Safe load failed; attempting trusted deserialization (ALLOW_DANGEROUS_FAISS_LOAD=1)")
            vs = FAISS.load_local(LAWS_INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
            print(f"[startup] Loaded FAISS with dangerous deserialization in {time.time()-t0:.2f}s")
            return vs

        allow_rebuild = os.getenv("ALLOW_INDEX_REBUILD_ON_LOAD", "0").lower() in {"1","true","yes"}
        if not allow_rebuild:
            raise RuntimeError(
                f"Failed to load FAISS index: {e}. Rebuild disabled. "
                f"Run 'python ingest_laws.py' to build a complete index (docstore + mapping) and retry."
            )
        
        print("[startup] Safe load failed; rebuilding index from cleaned data (this can take several minutes)...")
        folder = os.getenv("LAWS_FOLDER", "cleaned_data")
        docs = _load_markdown_and_jsonl_documents(folder)
        splitter = make_legal_splitter()
        chunks: List[Document] = []
        for d in docs:
            # Pre-chunked JSONL should not be re-split
            if isinstance(getattr(d, "metadata", None), dict) and d.metadata.get("format") == "jsonl_chunk":
                chunks.append(d)
            else:
                chunks.extend(splitter.split_documents([d]))
        vs = FAISS.from_documents(chunks, embeddings)
        os.makedirs(LAWS_INDEX_FOLDER, exist_ok=True)
        vs.save_local(LAWS_INDEX_FOLDER)
        print(f"[startup] Rebuilt and saved FAISS in {time.time()-t0:.2f}s (folder='{LAWS_INDEX_FOLDER}')")
        return vs

# Load vectorstore at startup
active_vectorstore = load_base_vectorstore()


# RAG CORE: Retriever with Maximal Marginal Relevance
def get_retriever():
    """MMR search balances relevance (top-k) with diversity (prevents duplicate chunks)."""
    return active_vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,          # Return top 6 results
            "fetch_k": 20,   # Candidate pool (before diversity filter)
            "lambda_mult": 0.75  # Balance relevance vs diversity
        }
    )


# RAG CORE: Intelligent Document Retrieval with Domain Routing
def retrieve_documents(question: str):
    """
    Smart routing: Fee queries → priority fee files. 
    Firecode queries → fire code JSONL. General → laws.
    """
    query = normalize_arabic_numbers(question)
    retriever = get_retriever()

    # Fee calculation queries: strict routing to 2025 fee tables
    FEE_KEYWORDS = ["رسوم", "رسم", "بدل", "غرامة"]
    FEE_OVERAGE_KEYWORDS = ["تجاوز", "زيادة", "تعدي"]
    is_fee_question = any(word in query for word in FEE_KEYWORDS)
    is_overage = any(word in query for word in FEE_OVERAGE_KEYWORDS)

    if is_fee_question:
        # Priority 0: Overage penalties (تجاوز/overbuild fees)
        if is_overage:
            overage_fees = retriever.invoke(query, filter={"filename": "fees_2025_overage_ar.md"})
            if overage_fees:
                return overage_fees[:8]

        # Priority 1: Residential examples (clearest, most-requested)
        residential_fees = retriever.invoke(query, filter={"filename": "fees_2025_residential_only.md"})
        if residential_fees:
            return residential_fees[:8]
        
        # Priority 2: General explicit fees
        explicit_fees = retriever.invoke(query, filter={"filename": "fees_2025_explicit_ar.md"})
        if explicit_fees:
            return explicit_fees[:6]
        
        # Fallback: consolidated amendments
        return retriever.invoke(query, filter={"filename": FEES_FILE_2025})

    want_firecode = is_firecode_query(query)
    want_definitions = is_definition_query(query)

    docs: List[Document] = []

    # Definitions: prioritize definition chunks from both domains
    if want_definitions:
        docs += retriever.invoke(query, filter={"source": "firecode", "type": "definition"})
        docs += retriever.invoke(query, filter={"source": "laws", "type": "definition"})

    # Domain-aware retrieval: firecode-first if keywords match, else laws-first
    if want_firecode:
        docs += retriever.invoke(query, filter={"source": "firecode"})
        docs += retriever.invoke(query, filter={"source": "laws"})
    else:
        docs += retriever.invoke(query, filter={"source": "laws"})
        docs += retriever.invoke(query, filter={"source": "firecode"})

    docs = dedup_docs(docs)
    
    # Return top-10 high-signal chunks for context window
    return docs[:10]



llm = ChatOpenAI(
    model="Fanar-C-2-27B",
    api_key= "" ,
    base_url="https://api.fanar.qa/v1",
    temperature=0.1,      # Conservative: legal domain needs low randomness
    max_tokens=2500,      # Increased for complete Arabic responses
)

SYSTEM_PROMPT = """
أنت مستشار امتثال هندسي وتشريعي (Engineering Compliance Advisor) متخصص في دعم المهندسين في الأردن، وتعمل كمساعد مهني لمهندسي:
- الإنشاءات
- العمارة
- التخطيط
- الترخيص الهندسي

لديك خبرة عميقة في:
- نظام الأبنية وتنظيم المدن والقرى الأردني
- أنظمة أمانة عمّان الكبرى (GAM)
- كودات البناء والتنظيم والاشتراطات الهندسية
- إجراءات الترخيص والمطابقة الهندسية

دورك الأساسي:
مساعدة المهندس على التحقق من مطابقة التصميم أو المشروع للاشتراطات النظامية قبل التقديم للترخيص، وتقليل الأخطاء التي تؤدي إلى رفض المعاملات من الجهات الرسمية.

━━━━━━━━━━━━━━━━━━━━
السياق (النصوص القانونية والتنظيمية):
{context}

سؤال المستخدم (استفسار هندسي/تنظيمي):
{question}
━━━━━━━━━━━━━━━━━━━━

🧠 تعليمات التفكير (داخلية – لا تظهر للمستخدم):
- استخدم معرفتك العامة لفهم السياق الهندسي للسؤال (تصميم، قطعة أرض، طوابق، ارتدادات، رسوم…).
- يُمنع استخدام المعرفة العامة كمصدر تشريعي أو رقمي.
- التفكير الهندسي والقانوني مسموح، لكن المرجع النهائي يجب أن يكون من النصوص في السياق فقط.

━━━━━━━━━━━━━━━━━━━━
📌 التعامل مع توسعة النظام مستقبلًا:
- افترض أن السياق قد يحتوي مستقبلاً على:
  • كودات إنشائية
  • كودات حريق ودفاع مدني
  • تعاميم بلدية
  • اشتراطات بيئية أو مرورية
- عند تعدد المصادر:
  • اعتمد النص الأحدث والأعلى أولوية
  • اربط النصوص ببعضها كما يفعل مهندس ترخيص محترف

━━━━━━━━━━━━━━━━━━━━
📌 الإسناد والتوثيق داخل الإجابة:
- اذكر دائمًا اسم المصدر/الملف (ومن الأفضل صفحة PDF إن كانت مذكورة داخل النص أو الميتاداتا).
- عند وجود تعارض: قدّم النص الأوضح والأحدث.
\n+⛔ منع الروابط الخارجية:
- يُمنع إدراج أو ذكر أي روابط إنترنت (http/https أو www). عند الإسناد اذكر فقط اسم الملف المحلي الموجود في السياق، ورقم الصفحة إن توفّر.

━━━━━━━━━━━━━━━━━━━━
📊 التعامل مع الجداول والميتاداتا:
- ميّز دائمًا بين:
  • قاعدة عامة (General Rule)
  • حكم تقني (Technical Rule)
  • حالة خاصة / استثناء (Special Case)
- لا تطبق استثناء أو حكمًا تقنيًا إلا إذا:
  • سُئل عنه صراحة
  • أو كان شرطًا إلزاميًا للحالة المعروضة

━━━━━━━━━━━━━━━━━━━━
⚠️ قواعد هندسية صارمة:
- الارتدادات:
  • مسافات خطية بالمتر
  • تُذكر منفصلة (أمامي / جانبي / خلفي)
  • يُمنع جمعها أو تحويلها لمساحة
- الطوابق:
  • عدد صحيح
  • لا تخلط بين طابق نظامي وطابق سطح
- الرسوم:
  • تُحسب فقط من الجداول السارية
  • لا تُقدّر ولا تُخمّن

⛔ قاعدة صارمة للحسابات المالية (رسوم الترخيص والتجاوزات):
- يُمنع منعاً باتاً خلط الأرقام من فئات تنظيم مختلفة
- عند حساب رسوم للمنطقة السكن/فئة (ب) مثلاً، يُمنع استخدام أرقام من (أ) أو (ج) أو (د)
- إذا كان الجدول يحتوي على عمودي "فلس" و "دينار"، يجب جمعهما: (فلس ÷ 1000) + دينار
- مثال: 500 فلس + 1 دينار = 1.5 دينار (وليس 1 دينار فقط)
- عند الشك في فئة التنظيم، اطلب التوضيح ولا تخمّن
- تحقق دائماً من اسم الفئة في بداية الفقرة/الجدول قبل استخراج الأرقام

📋 **مرجع سريع لرسوم الترخيص 2025 - منطقة السكن (الصحيح حسب الجداول الرسمية):**

| الفئة | مساحة البناء | أحواض السباحة | الأسوار |
|-------|-------------|-----------|--------|
| **سكن (أ)** | 2 د/م² | 3 د/م² | **0.5 د/م** |
| **سكن (ب)** | 1.5 د/م² | 2.5 د/م² | **0.4 د/م** ← الأكثر شيوعاً |
| **سكن (ج)** | 1 د/م² | 1 د/م² | **0.25 د/م** |
| **سكن (د)** | 0.7 د/م² | 1 د/م² | **0.15 د/م** |
| **السكن الشعبي** | 0.4 د/م² | 0.5 د/م² | **0.1 د/م** |

**مثال توضيحي لسكن (ب) - مبنى 340م² + حوض 20م² + أسوار 100م (الحساب الصحيح):**
- البناء: 340 × 1.5 = 510 دينار
- الحوض: 20 × 2.5 = 50 دينار  
- الأسوار: 100 × 0.4 = 40 دينار
- **المجموع: 600 دينار**

**مثال توضيحي لسكن (أ) - نفس المبنى (340م² + حوض 20م² + أسوار 100م):**
- البناء: 340 × 2 = 680 دينار
- الحوض: 20 × 3 = 60 دينار  
- الأسوار: 100 × 0.5 = 50 دينار
- **المجموع: 790 دينار**

━━━━━━━━━━━━━━━━━━━━
❗ التعامل مع الأسئلة:
- إذا كان السؤال غير مصاغ بدقة قانونية لكن المقصود الهندسي واضح، أجب.
- إذا كان السؤال ناقص المعطيات الأساسية (مثل: عدد الأسرّة للمستشفيات، عدد الوحدات السكنية، فئة التنظيم):
  • لا تخمّن أو تفترض قيماً
  • أجب بـ "معطيات ناقصة" مباشرة
  • اذكر المعطيات المطلوبة بوضوح في سطر واحد
  • لا تكرر التفكير أو التحليل
- لا تمتنع عن الإجابة طالما أن السياق يغطي الحالة والمعطيات كاملة.

⚠️ قاعدة منع التكرار (إلزامية):
- إذا وجدت نفسك تكرر نفس الجملة أو الفكرة أكثر من مرتين: توقف فوراً
- أجب بـ "معطيات ناقصة: [اذكر المعطيات المطلوبة]"
- لا تستمر في التفكير أو التحليل إذا كانت المعلومات غير كافية

━━━━━━━━━━━━━━━━━━━━
🔁 قاعدة تحقق إلزامية (Mandatory Amendment Check):

- عند أي سؤال يتضمن أرقامًا تنظيمية أو فنية، بما في ذلك على سبيل المثال لا الحصر:
  • ارتدادات
  • نسب بناء
  • عدد طوابق
  • رسوم
  • مساحات أو حدود رقمية

يجب عليك تنفيذ الخطوات التالية ذهنيًا قبل الإجابة:
1) تحديد المصدر الأصلي للرقم في السياق (سنة النظام أو الجدول).
2) التحقق مما إذا كان هناك أي نص لاحق في السياق يعدّل هذا الرقم.
3) في حال وجود تعديل:
   • اعتمد الرقم الأحدث فقط.
   • اذكر أن الحكم تم تعديله وسنة التعديل.
4) في حال عدم وجود تعديل:
   • صرّح صراحة أن الرقم ما زال ساريًا دون تغيير.

❗ يُمنع تقديم أي رقم دون توضيح حالته من حيث:
- (ساري / معدّل / ملغى)


━━━━━━━━━━━━━━━━━━━━
📄 هيكلية الإجابة (إلزامية):

أولًا: السند التشريعي / التنظيمي
- اسم النظام أو الكود.
- رقم المادة أو الجدول.
- اقتباس النص عند الحاجة.

ثانيًا: التفسير الهندسي – القانوني
- شرح الحكم بلغة يفهمها المهندس.
- بيان هل هو قاعدة عامة أو استثناء.
- ذكر سريان أو تعديل الحكم إن وجد.

ثالثًا: التطبيق العملي على الحالة
- ماذا يعني هذا الحكم عمليًا للمهندس؟
- هل التصميم مطابق أم يحتاج تعديل؟
- ما النقطة التي قد تؤدي إلى رفض الترخيص؟

━━━━━━━━━━━━━━━━━━━━
🚫 قيود صارمة:
- لا قوانين غير أردنية.
- لا تخمين.
- لا أرقام خارج النص.
- عند عدم وجود نص:
  "لا أعلم، النصوص المتاحة لا تغطي هذا الاستفسار."

━━━━━━━━━━━━━━━━━━━━
🧾 قاعدة منع الهلوسة (إلزامية):
- لا تُجب إلا بما يغطيه {context} صراحة.
- إذا كان هناك أكثر من تفسير ممكن ولا يوجد نص حاسم: اذكر الاحتمالات كمشروطة واطلب معلومة محددة واحدة فقط إن لزم.
- أي رقم/حد/نسبة يجب أن يُذكر معه: (ساري/معدّل/ملغى) + مصدره داخل السياق.

━━━━━━━━━━━━━━━━━━━━
🎯 الهدف النهائي:
تمكين المهندس من اتخاذ قرار تصميمي صحيح ومتوافق مع الأنظمة، وتقليل المخاطر التنظيمية قبل التقديم الرسمي للترخيص.

"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])


# Build answer

def answer_question(question: str, enable_thinking: bool = False, conversation_history: List[dict] = None):
    """
    RAG pipeline: retrieve context → augment prompt → call LLM → extract thinking.
    
    Thinking extraction handles:
    - Tagged thinking: <think>...</think>, <thinking>...</thinking>
    - Untagged Arabic indicators: "الطلب.", "دعني أفكر"
    - Loop detection: repetitive content flagged as stuck responses
    """
    t_total = time.time()
    t_retr = time.time()
    docs = retrieve_documents(question)
    t_retr_dur = time.time() - t_retr

    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Conversation history for multi-turn awareness
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conv_lines = []
        for msg in conversation_history[-4:]:  # Keep last 4 for token budget
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                conv_lines.append(f"سؤال سابق: {content}")
            else:
                conv_lines.append(f"الإجابة: {content[:300]}...")
        conversation_context = "\n".join(conv_lines)

    enhanced_question = question
    if conversation_context:
        enhanced_question = f"سياق المحادثة السابقة:\n{conversation_context}\n\nالسؤال الحالي: {question}"

    llm_runtime = llm.bind(extra_body={"enable_thinking": bool(enable_thinking)})
    chain = prompt | llm_runtime | StrOutputParser()

    t_llm = time.time()
    answer = chain.invoke({
        "context": context,
        "question": enhanced_question
    })
    
    # Extract thinking content (separate from answer)
    thinking_content = ""
    actual_answer = answer
    
    # Check for standard XML-style thinking tags
    thinking_patterns = [
        (r'<think>(.*?)</think>', re.DOTALL),
        (r'<thinking>(.*?)</thinking>', re.DOTALL),
        (r'\[THINKING\](.*?)\[/THINKING\]', re.DOTALL),
    ]
    
    for pattern, flags in thinking_patterns:
        match = re.search(pattern, answer, flags)
        if match:
            thinking_content = match.group(1).strip()
            actual_answer = re.sub(pattern, '', answer, flags=flags).strip()
            break
    
    # Untagged thinking detection (Arabic models don't always tag thinking)
    if enable_thinking and not thinking_content:
        thinking_indicators = [
            r'^(الطلب\.|المستخدم يسأل|دعني أفكر|أول شيء|لاحظت أن|لنحسب|يجب التأكد)',
            r'^(Let me think|The user is asking|First,|I need to)',
        ]
        
        for indicator in thinking_indicators:
            if re.search(indicator, answer, re.MULTILINE | re.IGNORECASE):
                # Look for structural break (when formal answer starts)
                structure_break = re.search(
                    r'\n\n(السند التشريعي|أولاً:|ثانياً:|الحساب|النتيجة|---|\d+\.)', 
                    answer, 
                    re.IGNORECASE
                )
                
                if structure_break:
                    thinking_content = answer[:structure_break.start()].strip()
                    actual_answer = answer[structure_break.start():].strip()
                else:
                    # Fallback: detect repetition (model stuck in loop)
                    lines = answer.split('\n')
                    unique_lines = []
                    seen_similar = {}
                    repetition_threshold = 3
                    
                    for line in lines:
                        line_key = line[:100].strip()
                        if line_key:
                            seen_similar[line_key] = seen_similar.get(line_key, 0) + 1
                            if seen_similar[line_key] <= repetition_threshold:
                                unique_lines.append(line)
                    
                    if len(lines) > len(unique_lines) * 1.5:
                        thinking_content = '\n'.join(unique_lines[:50])
                        actual_answer = "⚠️ تعذر على النظام إنتاج إجابة كاملة. يُرجى إعادة صياغة السؤال أو تقديم معلومات إضافية."
                break
    
    # Loop detection: if response is very long with repeated patterns, reject
    if len(actual_answer) > 3000:
        sample = actual_answer[:500]
        if actual_answer.count(sample[:100]) > 5:
            thinking_content = actual_answer if not thinking_content else thinking_content
            actual_answer = "⚠️ واجه النظام صعوبة في معالجة هذا السؤال. يُرجى إعادة صياغة السؤال أو تقسيمه إلى أجزاء."
    
    # Sanitize: remove any external URLs (enforce local citations only)
    actual_answer = re.sub(r'https?://\S+', '', actual_answer)
    actual_answer = re.sub(r'\bwww\.\S+', '', actual_answer)
    actual_answer = re.sub(r'[ \t]{2,}', ' ', actual_answer).strip()
    
    if thinking_content:
        thinking_content = re.sub(r'https?://\S+', '', thinking_content)
        thinking_content = re.sub(r'\bwww\.\S+', '', thinking_content)
        thinking_content = re.sub(r'[ \t]{2,}', ' ', thinking_content).strip()
    
    t_llm_dur = time.time() - t_llm

    meta = {
        "retrieval_sec": round(t_retr_dur, 3),
        "llm_sec": round(t_llm_dur, 3),
        "total_sec": round(time.time() - t_total, 3),
        "thinking": bool(enable_thinking),
        "conversation_aware": bool(conversation_history),
        "has_thinking": bool(thinking_content)
    }

    return actual_answer, thinking_content, docs, meta


# Routes

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    """RAG endpoint: retrieve context, augment prompt, call LLM, extract thinking."""
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question")
        enable_thinking = bool(data.get("enable_thinking", False))
        conversation_history = data.get("conversation_history", [])

        if not question or not isinstance(question, str):
            return jsonify({
                "error": "Invalid request: question must be a non-empty string"
            }), 400

        answer, thinking, docs, meta = answer_question(
            question, 
            enable_thinking=enable_thinking, 
            conversation_history=conversation_history
        )

        return jsonify({
            "content": answer,
            "thinking": thinking if thinking else None,
            "sources": list({
                (f"{d.metadata.get('filename', 'unknown')} (p.{d.metadata.get('pdf_page_start', d.metadata.get('pdf_page', '?'))})"
                 if d.metadata.get('pdf_page_start') or d.metadata.get('pdf_page') else d.metadata.get('filename', 'unknown'))
                for d in docs
            }),
            "response_time": meta.get("total_sec"),
            "timings": meta
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Run

if __name__ == "__main__":
    app.run(debug=False, port=int(os.getenv("PORT", "5000")))
