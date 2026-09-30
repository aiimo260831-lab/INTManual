import os
import time
import pickle
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

VECTOR_STORE_PATH = Path(__file__).parent / "vector_store.pkl"

_CLIENT = None
_CACHED_STORE = None
_CACHED_MATRIX = None

def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return api_key

def get_gemini_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    api_key = get_api_key()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일이나 배포 Secrets에 키를 설정해주세요.")
    _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT

def load_vector_store():
    """벡터 스토어 및 정규화 행렬을 로드하고 메모리에 캐싱합니다."""
    global _CACHED_STORE, _CACHED_MATRIX
    if _CACHED_STORE is not None:
        return _CACHED_STORE, _CACHED_MATRIX

    if not VECTOR_STORE_PATH.exists():
        return [], None

    try:
        with open(VECTOR_STORE_PATH, "rb") as f:
            _CACHED_STORE = pickle.load(f)
        
        if _CACHED_STORE:
            matrix = np.array([item["embedding"] for item in _CACHED_STORE], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            _CACHED_MATRIX = matrix / norms
    except Exception as e:
        print(f"벡터 스토어 로드 실패: {e}")
        return [], None

    return _CACHED_STORE, _CACHED_MATRIX

def get_chunk_count():
    store, _ = load_vector_store()
    return len(store)

def search_relevant_chunks(query: str, top_k: int = 5):
    """사용자 질의와 가장 관련된 문서 청크를 코사인 유사도로 검색합니다."""
    store, matrix = load_vector_store()
    if not store or matrix is None or len(store) == 0:
        return []

    client = get_gemini_client()
    query_resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query],
    )
    query_vec = np.array(query_resp.embeddings[0].values, dtype=np.float32)
    q_norm = np.linalg.norm(query_vec)
    if q_norm > 0:
        query_vec = query_vec / q_norm

    similarities = np.dot(matrix, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        item = store[idx]
        results.append({
            "text": item["text"],
            "metadata": item["metadata"],
            "score": float(similarities[idx])
        })
    return results

SYSTEM_PROMPT = """당신은 대한민국 국세청의 '국제거래정보통합보고서(개별기업보고서, 통합기업보고서, 국가별보고서)' 제출 안내 전문 AI 세무 어시스턴트입니다.

사용자의 질문에 대해 아래 제공되는 [참고 문서 자료]를 기반으로 정확하고 신뢰성 있게 답변해야 합니다.

### [답변 지침]
1. **정확성과 근거 중심**:
   - 제공된 [참고 문서 자료]의 내용에 엄격하게 기반하여 사실만을 설명하세요.
   - 답변 시 인용한 근거가 되는 문서명과 페이지(예: `[국제거래정보 통합보고서 제출 안내, p.12]`)를 문장 또는 단락 끝에 명확히 표기하세요.
2. **보고서 유형 구분**:
   - 질문에 따라 개별기업보고서, 통합기업보고서, 국가별보고서 중 어떤 서식/보고서에 해당하는지 명확하게 구분하여 설명하세요.
   - 제출의무자 요건, 제출기한, 제출서류, 서식 작성 시 주의사항 등을 일목요연하게 글머리 기호(Bulleted list)나 표를 활용해 가독성 있게 정리하세요.
3. **불명확하거나 없는 정보에 대한 처리**:
   - 참고 자료에 명시되어 있지 않은 내용은 억지로 지어내지 말고, "제공된 지침 문서에는 해당 내용이 명시되어 있지 않으므로, 국세청 관할 세무서 또는 국세상담센터(126)를 통해 확인하시기 바랍니다."라고 정직하게 안내하세요.
4. **어조**:
   - 정중하고 전문적인 공공 세무 상담원 톤앤매너를 유지하세요.
"""

def generate_chat_response_stream(query: str, chat_history: list = None, max_retries_per_model: int = 2):
    """
    RAG 검색을 수행하고 최적의 Gemini Flash 모델 스트리밍 생성 제너레이터를 반환합니다.
    특정 모델에서 503(과부하/일시적 지연) 또는 429 에러 발생 시 다른 안정적인 모델로 자동 대체(Fallback)하여 답변을 생성합니다.
    """
    chunks = search_relevant_chunks(query, top_k=5)

    if not chunks:
        context_str = "참고할 수 있는 사전 지침 문서가 데이터베이스에 등록되어 있지 않습니다."
    else:
        context_parts = []
        for i, c in enumerate(chunks, 1):
            src = c["metadata"].get("source", "알 수 없음")
            page = c["metadata"].get("page", "?")
            context_parts.append(f"[문서 {i}] 출처: {src} (p.{page})\n{c['text']}")
        context_str = "\n\n".join(context_parts)

    user_content = f"""[참고 문서 자료]
{context_str}

[사용자 질문]
{query}

위 참고 문서 자료를 바탕으로 사용자의 질문에 친절하고 정확하게 답변해 주세요."""

    client = get_gemini_client()

    # 가용 모델 후보군 (우선순위 순서대로 시도)
    custom_model = os.getenv("GEMINI_MODEL")
    candidate_models = ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
    if custom_model:
        if custom_model in candidate_models:
            candidate_models.remove(custom_model)
        candidate_models.insert(0, custom_model)

    def stream_generator():
        last_error = None
        has_yielded = False

        for model_name in candidate_models:
            wait_time = 1.5
            for attempt in range(max_retries_per_model):
                try:
                    response = client.models.generate_content_stream(
                        model=model_name,
                        contents=user_content,
                        config={
                            "system_instruction": SYSTEM_PROMPT,
                            "temperature": 0.2,
                        }
                    )
                    for chunk in response:
                        if chunk.text:
                            has_yielded = True
                            yield chunk.text
                    # 성공적으로 스트리밍을 마쳤으므로 종료
                    return
                except Exception as e:
                    last_error = e
                    err_msg = str(e)
                    
                    # 이미 텍스트가 일부 출력된 상태에서 중단된 경우 중복 출력을 피하기 위해 안내 메시지 출력 후 종료
                    if has_yielded:
                        yield f"\n\n⚠️ 답변 스트리밍 도중 일시적인 네트워크 지연이 발생했습니다: {e}"
                        return
                    
                    is_transient = any(code in err_msg for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "502"])
                    if is_transient and attempt < max_retries_per_model - 1:
                        time.sleep(wait_time)
                        wait_time *= 2
                    else:
                        # 다음 후보 모델로 폴백
                        break

        # 모든 모델 시도 실패 시
        yield f"\n\n⚠️ AI 서버 일시적 지연으로 인해 답변 생성이 지연되었습니다. 잠시 후 다시 시도해 주세요. (오류 내용: {last_error})"

    return stream_generator(), chunks

