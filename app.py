import os
import streamlit as st
from dotenv import load_dotenv
from rag import generate_chat_response_stream, get_chunk_count, get_api_key

load_dotenv()

st.set_page_config(
    page_title="국제거래정보통합보고서 제출안내 AI",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 1. 시범 운영용 비밀번호 인증 시스템
# -------------------------------------------------------------
def check_auth():
    auth_required = os.getenv("AUTH_REQUIRED", "True").lower() in ("true", "1", "yes")
    correct_password = os.getenv("APP_ACCESS_PASSWORD", "1234")
    
    if hasattr(st, "secrets"):
        if "AUTH_REQUIRED" in st.secrets:
            auth_required = str(st.secrets["AUTH_REQUIRED"]).lower() in ("true", "1", "yes")
        if "APP_ACCESS_PASSWORD" in st.secrets:
            correct_password = str(st.secrets["APP_ACCESS_PASSWORD"])

    if not auth_required:
        return True

    if st.session_state.get("authenticated", False):
        return True

    st.markdown("## 🔒 국제거래정보통합보고서 AI 챗봇 (시범운영)")
    st.info("본 서비스는 현재 시범운영(Closed Beta) 중입니다. 안내받으신 접속 비밀번호를 입력해주세요.")
    
    col1, col2, _ = st.columns([2, 1, 3])
    with col1:
        input_pw = st.text_input("접속 비밀번호", type="password", key="input_pw_key")
    with col2:
        st.write("")
        st.write("")
        submit_btn = st.button("접속하기", type="primary")

    if submit_btn:
        if input_pw == correct_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다. 관리자에게 문의하세요.")
    return False

if not check_auth():
    st.stop()

# -------------------------------------------------------------
# 2. UI 세션 상태 및 기본 설정
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요! **국제거래정보통합보고서 제출안내 AI 도우미**입니다.\n\n"
                "개별기업보고서, 통합기업보고서, 국가별보고서 제출의무자 요건, 작성 방법, 제출기한, "
                "과태료 및 오류사항 등에 대해 궁금한 점을 질문해 주세요!"
            ),
            "sources": []
        }
    ]

# -------------------------------------------------------------
# 3. 사이드바 (정보 및 빠른 질문)
# -------------------------------------------------------------
with st.sidebar:
    st.title("📑 안내 데스크")
    
    # DB 상태 표시
    chunk_count = get_chunk_count()
    if chunk_count > 0:
        st.success(f"✅ RAG 지침서 준비 완료 ({chunk_count}개 청크)")
    else:
        st.warning("⚠️ RAG 데이터베이스가 아직 구축되지 않았습니다. `ingest.py`를 먼저 실행해주세요.")

    st.markdown("---")
    st.subheader("💡 자주 묻는 질문 (FAQ)")
    
    sample_questions = [
        "국가별보고서 제출의무자 요건은 어떻게 되나요?",
        "개별기업보고서와 통합기업보고서의 차이는 무엇인가요?",
        "보고서 미제출 또는 거짓 제출 시 과태료는 얼마인가요?",
        "국가별보고서 작성 시 자주 발생하는 일반적인 오류사항은?",
        "해외현지법인 명세서와 국제거래명세서의 제출기한은?"
    ]

    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ 대화 내역 초기화", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

    st.markdown("---")
    st.caption(
        "⚠️ **면책 조항**: 본 AI 챗봇이 제공하는 답변은 국세청 공식 가이드라인 및 관련 법령에 기반하여 생성된 참고 자료입니다. "
        "실제 세무 신고 및 불복 청구 등 중요한 의사결정 시에는 반드시 관할 세무서 또는 공인 세무전문가와 상담하시기 바랍니다."
    )

# -------------------------------------------------------------
# 4. 메인 화면 헤더
# -------------------------------------------------------------
st.title("🌐 국제거래정보통합보고서 AI 가이드")
st.markdown("국세조세조정에 관한 법률 및 국세청 공식 가이드라인에 기반한 맞춤형 안내를 제공합니다.")

# API 키 점검
api_key = get_api_key()
if not api_key or api_key == "your_gemini_api_key_here":
    st.error("🚨 Google Gemini API 키가 설정되지 않았습니다. `.env` 파일에 발급받으신 키를 등록해주세요.")
    st.stop()

# -------------------------------------------------------------
# 5. 기존 대화 내역 렌더링
# -------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📌 참고한 지침서 및 법령 근거"):
                seen = set()
                for s in msg["sources"]:
                    src_file = s["metadata"].get("source", "출처 미상")
                    src_page = s["metadata"].get("page", "?")
                    key = (src_file, src_page)
                    if key not in seen:
                        seen.add(key)
                        st.markdown(f"- **{src_file}** (p.{src_page})")

# -------------------------------------------------------------
# 6. 사용자 입력 처리
# -------------------------------------------------------------
prompt = st.chat_input("국제거래정보통합보고서에 대해 무엇이든 물어보세요...")

if "pending_query" in st.session_state and st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("지침서 및 법령 문서를 검색하고 있습니다..."):
                response_stream, sources = generate_chat_response_stream(prompt, st.session_state.messages)

            for text_chunk in response_stream:
                full_response += text_chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)

            if sources:
                with st.expander("📌 참고한 지침서 및 법령 근거"):
                    seen = set()
                    for s in sources:
                        src_file = s["metadata"].get("source", "출처 미상")
                        src_page = s["metadata"].get("page", "?")
                        key = (src_file, src_page)
                        if key not in seen:
                            seen.add(key)
                            st.markdown(f"- **{src_file}** (p.{src_page})")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })

        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
