"""챗봇 검색 임계값(RAG_MIN_SCORE) 보정용 측정 스크립트.

보안 질문과 무관한 질문 각각의 '최상위 기사 유사도'를 재서, 임계값별로 몇 개가 통과/차단되는지 보여준다.
기사 수나 임베딩 입력이 크게 바뀌면(요약 길이, 태그 포함 등) 다시 돌려 chatbot/rag.py의 MIN_SCORE를 점검한다.
OpenAI 임베딩 호출이 질문 수만큼(1회 배치) 발생한다.

  python scripts/calibrate_threshold.py
"""
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot import rag  # noqa: E402
from core import llm  # noqa: E402
from store import vector  # noqa: E402

IN_DOMAIN = [
    "랜섬웨어 침해 사고 사례 알려줘", "최근 개인정보 유출 사고", "이번 주 주요 취약점 정리해줘", "OpenSSH 취약점",
    "보안 컨퍼런스 소식", "해킹 그룹 APT 활동", "피싱 공격 수법", "제로데이 취약점 패치", "국내 기업 해킹 피해",
    "데이터 유출 사건", "악성코드 유포 사례", "금융권 보안 사고", "CVE 최신 정보", "AI를 악용한 사이버 공격",
    "중국 해커 그룹", "클라우드 보안 서비스 출시", "ransomware attack on hospital", "zero-day exploited in the wild",
    "data breach customers notified", "phishing campaign targeting banks", "랜섬웨어", "취약점", "해킹", "보안", "개인정보",
]
OFF_TOPIC = [
    "오늘 서울 날씨 어때?", "피자 맛집 추천해줘", "비트코인 가격 전망", "손흥민 경기 결과", "파이썬 리스트 정렬 방법",
    "주말에 볼만한 영화 추천", "아침 식단 레시피", "강아지 훈련 방법", "삼성전자 주가", "부동산 대출 금리",
    "제주도 여행 코스", "파스타 만드는 법", "올림픽 메달 순위", "건강한 다이어트 방법", "자동차 엔진오일 교체 주기",
    "best pizza near me", "how to learn guitar", "weather forecast tomorrow", "football world cup winner",
    "chocolate cake recipe", "안녕하세요", "너 누구야", "고마워", "내일 회의 일정", "점심 뭐 먹지",
]


def top1_scores(store, queries):
    vecs = llm.embed(queries)
    return [(store.search(v, 1)[0][1], q) for q, v in zip(queries, vecs)]


def main():
    store = vector.VectorStore()
    if store.total == 0:
        print("인덱스가 비어 있습니다. 먼저 python scripts/run_pipeline.py 를 실행하세요.")
        return 1
    inside, outside = top1_scores(store, IN_DOMAIN), top1_scores(store, OFF_TOPIC)
    for name, rows in (("보안 질문", inside), ("무관 질문", outside)):
        s = sorted(x[0] for x in rows)
        print(f"{name}: 최저 {s[0]:.2f} / 중앙 {st.median(s):.2f} / 최고 {s[-1]:.2f}")
    print("가장 낮은 보안 질문:", [(round(x[0], 2), x[1]) for x in sorted(inside)[:3]])
    print("가장 높은 무관 질문:", [(round(x[0], 2), x[1]) for x in sorted(outside)[-3:]])
    print(f"\n현재 MIN_SCORE = {rag.MIN_SCORE}")
    for t in (0.20, 0.25, 0.27, 0.30, 0.35, 0.40):
        mark = "  ← 현재" if abs(t - rag.MIN_SCORE) < 1e-9 else ""
        print(f"T={t:.2f}  보안 질문 통과 {sum(x[0] >= t for x in inside)}/{len(inside)}  "
              f"무관 질문 차단 {sum(x[0] < t for x in outside)}/{len(outside)}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
