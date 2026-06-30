from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from mlx_embeddings import load
from openai import OpenAI


MODEL_NAME = "majentik/Qwen3-Embedding-0.6B-MLX-4bit"
MAX_LENGTH = 32768
ovseaMrktNews_top_k = 10
compSucsCase_top_k = 3
globalSupplyInsights_top_k = 3
SUMMARY_ARTICLES = 10
GPT_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "gpt-5-mini",
    "gpt-5-nano",
]

DATASETS = {
    "해외시장뉴스": {
        "index_path": Path("data/faiss/ovseaMrktNews_qwen3_embedding/index.faiss"),
        "metadata_path": Path("data/faiss/ovseaMrktNews_qwen3_embedding/metadata.parquet"),
        "body_col": "본문",
        "top_k": ovseaMrktNews_top_k,
        "table_cols": ["score", "뉴스제목", "게시물 URL", "키워드", "뉴스게시일자", "국가"],
    },
    "기업성공사례": {
        "index_path": Path("data/faiss/compSucsCase_qwen3_embedding/index.faiss"),
        "metadata_path": Path("data/faiss/compSucsCase_qwen3_embedding/metadata.parquet"),
        "body_col": "본문텍스트",
        "top_k": compSucsCase_top_k,
        "table_cols": ["score", "기업명", "제목", "국가", "지역", "산업분류", "게시일자"],
    },
    "글로벌 공급망 인사이트": {
        "index_path": Path("data/faiss/globalSupplyInsights_pdf_qwen3_embedding/index.faiss"),
        "metadata_path": Path("data/faiss/globalSupplyInsights_pdf_qwen3_embedding/metadata.parquet"),
        "body_col": "chunk_text",
        "top_k": globalSupplyInsights_top_k,
        "table_cols": ["score", "게시글제목", "PDF파일명", "chunk_id", "게시물 URL", "공개일시"],
    },
}


@st.cache_resource
def load_embedding_model():
    return load(MODEL_NAME)


@st.cache_resource
def load_faiss_index(index_path):
    return faiss.read_index(index_path)


@st.cache_data
def load_metadata(metadata_path, metadata_mtime):
    return pd.read_parquet(metadata_path)


def existing_columns(df, columns):
    return [col for col in columns if col in df.columns]


def embed_query(query, model, tokenizer):
    task = "Given a Korean natural language search query, retrieve relevant KOTRA documents."
    query_text = f"Instruct: {task}\nQuery: {query}"
    inputs = tokenizer(
        [query_text],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="mlx",
    )
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    embedding = np.array(outputs.text_embeds).astype("float32")
    faiss.normalize_L2(embedding)
    return embedding


def search_dataset(query_embedding, dataset_name, config):
    index = load_faiss_index(str(config["index_path"]))
    metadata_path = config["metadata_path"]
    metadata = load_metadata(str(metadata_path), metadata_path.stat().st_mtime)

    scores, indices = index.search(query_embedding, config["top_k"])
    valid = indices[0] >= 0
    results = metadata.iloc[indices[0][valid]].copy()
    results.insert(0, "score", scores[0][valid])
    results.insert(1, "데이터셋", dataset_name)
    return results


def format_news(row, rank):
    return f"""[해외시장뉴스 {rank}]
제목: {row.get("뉴스제목", "")}
URL: {row.get("게시물 URL", "")}
키워드: {row.get("키워드", "")}
게시일자: {row.get("뉴스게시일자", "")}
국가: {row.get("국가", "")}

본문:
{row.get("본문", "")}
"""


def format_case(row, rank):
    return f"""[기업성공사례 {rank}]
기업명: {row.get("기업명", "")}
제목: {row.get("제목", "")}
국가: {row.get("국가", "")}
지역: {row.get("지역", "")}
산업분류: {row.get("산업분류", "")}
게시일자: {row.get("게시일자", "")}

본문:
{row.get("본문텍스트", "")}
"""


def format_supply_insight(row, rank):
    return f"""[글로벌 공급망 인사이트 {rank}]
제목: {row.get("게시글제목", "")}
PDF파일명: {row.get("PDF파일명", "")}
chunk_id: {row.get("chunk_id", "")}
URL: {row.get("게시물 URL", "")}
공개일시: {row.get("공개일시", "")}

본문:
{row.get("chunk_text", "")}
"""


def build_summary_prompt(query, news_results, case_results, supply_results):
    news_text = "\n\n".join(
        format_news(row, rank)
        for rank, (_, row) in enumerate(news_results.head(SUMMARY_ARTICLES).iterrows(), start=1)
    )
    case_text = "\n\n".join(
        format_case(row, rank)
        for rank, (_, row) in enumerate(case_results.head(SUMMARY_ARTICLES).iterrows(), start=1)
    )
    supply_text = "\n\n".join(
        format_supply_insight(row, rank)
        for rank, (_, row) in enumerate(supply_results.head(SUMMARY_ARTICLES).iterrows(), start=1)
    )

    return f"""아래는 검색어와 벡터 검색으로 찾은 KOTRA 해외시장뉴스, 기업성공사례, 글로벌 공급망 인사이트입니다.

검색어:
{query}

해외시장뉴스:
{news_text}

기업성공사례:
{case_text}

글로벌 공급망 인사이트:
{supply_text}

요청:
1. 검색어와 관련된 시장 동향을 한국어로 요약해줘.
2. 기업성공사례에서 확인되는 실무적 성공 요인을 정리해줘.
3. 글로벌 공급망 인사이트에서 확인되는 공급망 리스크와 구조적 변화를 정리해줘.
4. 시장 동향, 성공사례, 공급망 인사이트를 연결해서 기업이 참고할 시사점을 bullet로 정리해줘.
5. 마지막에 참고한 해외시장뉴스 제목/URL, 기업성공사례 제목/기업명, 공급망 인사이트 PDF명을 간단히 목록으로 붙여줘.
"""


st.title("KOTRA 뉴스 검색 요약")

query = st.text_input("검색어", placeholder="예: 러시아 화장품 시장 전망")
selected_gpt_model = st.selectbox("요약 모델", GPT_MODELS)
run = st.button("검색 및 요약", type="primary", disabled=not query.strip())

if run:
    load_dotenv()

    progress = st.progress(0)
    status = st.empty()

    try:
        status.write("1/4 리소스 로드 중")
        model, tokenizer = load_embedding_model()
        progress.progress(25)

        status.write("2/4 검색어 임베딩 중")
        query_embedding = embed_query(query.strip(), model, tokenizer)
        progress.progress(50)

        status.write("3/4 FAISS 검색 중")
        news_results = search_dataset(query_embedding, "해외시장뉴스", DATASETS["해외시장뉴스"])
        case_results = search_dataset(query_embedding, "기업성공사례", DATASETS["기업성공사례"])
        supply_results = search_dataset(
            query_embedding,
            "글로벌 공급망 인사이트",
            DATASETS["글로벌 공급망 인사이트"],
        )
        progress.progress(75)

        status.write(f"4/4 {selected_gpt_model} 통합 요약 중")
        client = OpenAI()
        response = client.responses.create(
            model=selected_gpt_model,
            input=build_summary_prompt(query.strip(), news_results, case_results, supply_results),
            reasoning={"effort": "minimal"},
            max_output_tokens=8192,
        )
        progress.progress(100)
        status.write("완료")

        st.subheader("통합 요약")
        st.write(response.output_text)

        st.subheader("해외시장뉴스 검색 결과")
        st.dataframe(
            news_results[existing_columns(news_results, DATASETS["해외시장뉴스"]["table_cols"])],
            use_container_width=True,
        )

        st.subheader("기업성공사례 검색 결과")
        st.dataframe(
            case_results[existing_columns(case_results, DATASETS["기업성공사례"]["table_cols"])],
            use_container_width=True,
        )

        st.subheader("글로벌 공급망 인사이트 검색 결과")
        st.dataframe(
            supply_results[existing_columns(supply_results, DATASETS["글로벌 공급망 인사이트"]["table_cols"])],
            use_container_width=True,
        )

        st.subheader("글로벌 공급망 인사이트 보고서 관련 부분")
        for rank, (_, row) in enumerate(supply_results.iterrows(), start=1):
            title = row.get("게시글제목", "")
            pdf_name = row.get("PDF파일명", "")
            chunk_id = row.get("chunk_id", "")
            score = row.get("score", 0)
            with st.expander(f"{rank}. {title} / chunk {chunk_id} / score {score:.4f}"):
                st.caption(pdf_name)
                st.write(row.get("chunk_text", ""))

    except Exception as e:
        status.write("오류 발생")
        st.error(f"{type(e).__name__}: {e}")
