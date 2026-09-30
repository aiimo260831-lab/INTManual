# 📑 국제거래정보통합보고서 제출안내 RAG AI 챗봇

Google Gemini API와 Streamlit을 기반으로 구축된 **국제거래정보통합보고서(개별기업보고서, 통합기업보고서, 국가별보고서)** 제출 안내 전문 AI 챗봇입니다.

---

## 🌟 주요 기능
1. **정밀 RAG (검색 증강 생성)**: 국세청 지침서 및 관련 법령 15종 문서를 분석하여 정확한 답변과 인용 출처(문서명 및 페이지)를 제공합니다.
2. **최신 Gemini 모델 탑재**: Google의 `gemini-2.5-flash` 및 `text-embedding-004`를 적용하여 빠르고 신뢰성 높은 세무 답변을 생성합니다.
3. **시범운영 보안 잠금**: 웹에 배포 시 지정된 비밀번호(PIN)를 입력해야만 접속할 수 있어, 개발 및 시범운영 기간 동안 비인가자의 접근을 차단합니다.
4. **추천 질문 칩**: 이용자가 자주 찾는 핵심 질문을 한 번의 클릭으로 질의할 수 있습니다.

---

## 📁 프로젝트 구조
```
INTManual/
├── Source/                   # 국세청 공식 지침서 및 법령 PDF (15종)
├── chroma_db/                # 생성된 벡터 데이터베이스 (로컬 캐시)
├── .streamlit/
│   ├── config.toml           # UI 테마 설정
│   └── secrets.toml.example  # 클라우드 배포용 환경변수 템플릿
├── app.py                    # Streamlit 프론트엔드 메인 앱
├── rag.py                    # RAG 검색 및 Gemini 스트리밍 로직
├── ingest.py                 # PDF 파싱 및 ChromaDB 임베딩 인덱싱 스크립트
├── requirements.txt          # 파이썬 의존성 패키지
├── .env.example              # 로컬 환경변수 템플릿
└── .gitignore
```

---

## 🚀 빠른 시작 가이드 (로컬 실행)

### 1. API 키 및 환경변수 설정
프로젝트 폴더에 `.env` 파일을 생성하고 발급받으신 Gemini API 키를 입력합니다.
```env
GEMINI_API_KEY=발급받으신_Gemini_API_키
APP_ACCESS_PASSWORD=1234
AUTH_REQUIRED=True
```

### 2. 가상환경 활성화 및 패키지 설치
```bash
# 가상환경 활성화 (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 패키지 설치 (이미 설치된 경우 생략)
pip install -r requirements.txt
```

### 3. 지침서 문서 벡터 DB 인덱싱 (최초 1회 실행)
`Source/` 폴더 내의 15개 PDF 문서를 읽어 벡터 데이터베이스를 구축합니다.
```bash
python ingest.py
```
*(성공 시 `chroma_db/` 폴더에 임베딩 벡터가 영구 저장됩니다.)*

### 4. 챗봇 웹 애플리케이션 실행
```bash
streamlit run app.py
```
브라우저에서 `http://localhost:8501`이 열리면 설정한 비밀번호(기본: `1234`)를 입력하여 사용을 시작합니다.

---

## 🌐 인터넷 배포 및 시범운영 링크 공유 방법 (Streamlit Cloud)

1. **GitHub 저장소 업로드**:
   - 본 프로젝트 폴더를 GitHub 개인 저장소(Public 또는 Private)로 푸시합니다.
   - *주의: `.env` 파일과 `chroma_db/`는 `.gitignore`에 의해 자동으로 제외됩니다.*
2. **Streamlit Community Cloud 접속**:
   - [share.streamlit.io](https://share.streamlit.io)에 로그인 후 **"New app"** 클릭
   - 방금 올린 GitHub 저장소 및 브랜치(`main`), 메인 파일(`app.py`) 선택
3. **Secrets(환경변수) 설정**:
   - 배포 설정 중 **Advanced settings > Secrets** 창에 `.streamlit/secrets.toml.example` 내용을 복사하여 입력합니다:
     ```toml
     GEMINI_API_KEY = "발급받으신_Gemini_API_키"
     APP_ACCESS_PASSWORD = "전달할_비밀번호"
     AUTH_REQUIRED = true
     ```
4. **Deploy 클릭**:
   - 수 분 후 생성된 고유 웹 링크(예: `https://your-app-name.streamlit.app`)가 생성됩니다.
   - 시범운영 기간 동안 테스트 대상자들에게 **해당 URL과 접속 비밀번호**만 공유하시면 됩니다.
5. **정식 서비스 전환 (누구나 접속 허용)**:
   - Streamlit Cloud의 Settings > Secrets에서 `AUTH_REQUIRED = false`로 변경하면 비밀번호 입력 없이 누구나 즉시 이용할 수 있습니다.
