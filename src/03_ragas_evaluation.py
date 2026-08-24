"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# ── Compatibility Shim cho RAGAS ──────────────────────────────────────────
import types
try:
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    try:
        import langchain_google_vertexai.chat_models as vertexai_mod
        sys.modules['langchain_community.chat_models.vertexai'] = vertexai_mod
    except ModuleNotFoundError:
        dummy = types.ModuleType('langchain_community.chat_models.vertexai')
        dummy.ChatVertexAI = None
        sys.modules['langchain_community.chat_models.vertexai'] = dummy

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (tái sử dụng từ Bước 2) ───────────────────────────
SYSTEM_V1 = """Bạn là trợ lý AI hữu ích. Chỉ sử dụng thông tin từ context được cung cấp dưới đây để trả lời câu hỏi.
Giữ câu trả lời ngắn gọn, trực tiếp và chính xác trong từ 2 đến 4 câu. Nếu không tìm thấy thông tin trong context, hãy trả lời rõ ràng là không biết.

Context:
{context}"""

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = """Bạn là một chuyên gia phân tích dữ liệu AI cao cấp. Đọc kỹ context được cung cấp dưới đây, xác định các sự thật (facts) liên quan và trình bày câu trả lời một cách rõ ràng, có cấu trúc logic (3 đến 5 câu).
Yêu cầu:
- Sử dụng giọng văn chuyên nghiệp và chính xác.
- Đưa ra thông tin trọng tâm dựa trên duy nhất context được cung cấp.
- Nếu thông tin không đủ trong context, hãy ghi rõ không thể trả lời từ tài liệu.

Context:
{context}"""

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.
    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ vài phút)")

    dataset = build_ragas_dataset(rag_results)

    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }

    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    report_path = data_dir / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")

    # Copy report to evidence directory
    evidence_dir = Path(__file__).parent.parent / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    evidence_report_path = evidence_dir / "03_ragas_report.json"
    evidence_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã sao chép báo cáo vào {evidence_report_path}")


if __name__ == "__main__":
    main()
