# -*- coding: utf-8 -*-
"""重排序模块实现

支持 BGE-reranker、cross-encoder 和 API 调用重排序。
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.retrievers.base_retriever import SearchResult

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_local_rerank_model(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return model_name
    if Path(str(model_name)).exists():
        return str(model_name)
    if "/" not in str(model_name):
        return model_name

    org, name = str(model_name).split("/", 1)
    local_path = PROJECT_ROOT / "models" / org / name
    if local_path.exists():
        return str(local_path)
    return model_name


@dataclass
class RerankSearchResult:
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

# 尝试导入相关库
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "未安装 sentence-transformers。请使用以下命令安装：pip install sentence-transformers"
    )

try:
    from FlagEmbedding import FlagReranker
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False
    logger.warning(
        "未安装 FlagEmbedding。请使用以下命令安装：pip install FlagEmbedding"
    )


FLAG_EMBEDDING_IMPORT_AVAILABLE = FLAG_EMBEDDING_AVAILABLE


class Reranker:
    """重排序器
    
    使用 cross-encoder 模型对候选文档进行重排序。
    支持 BGE-reranker 和其他 cross-encoder 模型。
    
    Attributes:
        model_name: 模型名称
        model: 模型实例
        batch_size: 批处理大小
        max_length: 最大序列长度
    """
    
    def __init__(
        self,
        use_fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        """初始化重排序器
        
        Args:
            use_fp16: 是否使用半精度浮点
            batch_size: 批处理大小
            max_length: 最大序列长度
            device: 设备类型 (cuda/cpu)，None 则自动选择
            **kwargs: 额外参数
        """
        config = get_config()
        
        self.model_name = resolve_local_rerank_model(config.rerank_model)
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.device = device
        
        self.model = None
        self._init_model(**kwargs)
        
        logger.debug(f"初始化重排序器: model={self.model_name}")
    
    def _init_model(self, **kwargs: Any) -> None:
        global FLAG_EMBEDDING_AVAILABLE
        requested_backend = str(kwargs.pop("backend", "") or os.getenv("RERANK_BACKEND", "sentence_transformers")).lower()
        flag_requested = requested_backend in {"flag", "flag_embedding", "flagembedding"}
        FLAG_EMBEDDING_AVAILABLE = FLAG_EMBEDDING_IMPORT_AVAILABLE and flag_requested
        """初始化模型
        
        Args:
            **kwargs: 额外参数
        """
        # 优先使用 FlagEmbedding（针对 BGE 模型优化）
        if FLAG_EMBEDDING_AVAILABLE and "bge" in self.model_name.lower():
            self.model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            self._backend = "flag_embedding"
        # 使用 sentence-transformers
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
            self._backend = "sentence_transformers"
        else:
            raise ImportError(
                "No reranker backend available. "
                "Install sentence-transformers or FlagEmbedding."
            )
    
    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """对候选文档进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回结果数量（可选）
            
        Returns:
            重排序后的结果列表
        """
        if not candidates:
            return []
        
        if self.model is None:
            logger.warning("模型未初始化，返回原始候选列表")
            return candidates[:top_k] if top_k else candidates
        
        # 准备查询-文档对
        pairs = [
            (query, candidate.content)
            for candidate in candidates
        ]
        
        # 计算相关性分数
        scores = self._compute_scores(pairs)
        
        # 更新分数并排序
        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)
            candidate.metadata["rerank_score"] = float(score)
        
        # 按分数降序排序
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # 截断
        if top_k is not None:
            candidates = candidates[:top_k]
        
        logger.debug(f"重排序完成: candidates={len(candidates)}")
        
        return candidates
    
    def _compute_scores(
        self,
        pairs: List[Tuple[str, str]],
    ) -> np.ndarray:
        """计算查询-文档对的相关性分数
        
        Args:
            pairs: (query, document) 元组列表
            
        Returns:
            分数数组
        """
        if self._backend == "flag_embedding":
            # FlagEmbedding 方式
            scores = self.model.compute_score(
                pairs,
                batch_size=self.batch_size,
                max_length=self.max_length,
            )
            # 确保返回数组
            if isinstance(scores, (int, float)):
                scores = [scores]
            return np.array(scores)
        
        elif self._backend == "sentence_transformers":
            # sentence-transformers 方式
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
            )
            return np.array(scores)
        
        else:
            raise ValueError(f"未知的后端: {self._backend}")
    
    def rerank_batch(
        self,
        queries: List[str],
        candidates_list: List[List[SearchResult]],
        top_k: Optional[int] = None,
    ) -> List[List[SearchResult]]:
        """批量重排序
        
        Args:
            queries: 查询文本列表
            candidates_list: 候选文档列表的列表
            top_k: 每个查询返回的结果数量（可选）
            
        Returns:
            重排序后的结果列表的列表
        """
        results = []
        
        for query, candidates in zip(queries, candidates_list):
            reranked = self.rerank(query, candidates, top_k)
            results.append(reranked)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model_name": self.model_name,
            "backend": self._backend,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "use_fp16": self.use_fp16,
            "device": self.device,
        }


class BGEReranker(Reranker):
    """Backward-compatible BGE reranker name used by experiment scripts."""

    pass


class LLMBasedReranker:
    """基于 LLM 的重排序器
    
    使用大语言模型对候选文档进行重排序。
    适用于需要复杂推理的场景。
    
    Attributes:
        llm_client: LLM 客户端
        prompt_template: 提示词模板
    """
    
    DEFAULT_PROMPT = """你是一个文档相关性评估专家。请评估以下文档与查询的相关性，并给出一个 0 到 1 之间的相关性分数。

查询: {query}

文档:
{document}

请只返回一个 0 到 1 之间的数字，表示相关性分数。不要返回其他内容。

相关性分数:"""
    
    def __init__(
        self,
        llm_client: Any,
        prompt_template: Optional[str] = None,
        batch_size: int = 10,
        **kwargs: Any,
    ):
        """初始化 LLM 重排序器
        
        Args:
            llm_client: LLM 客户端实例
            prompt_template: 提示词模板（可选）
            batch_size: 批处理大小
            **kwargs: 额外参数
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.batch_size = batch_size
        
        logger.debug("初始化 LLM 重排序器")
    
    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """使用 LLM 对候选文档进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回结果数量（可选）
            
        Returns:
            重排序后的结果列表
        """
        if not candidates:
            return []
        
        # 批量处理
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            
            for candidate in batch:
                # 构建提示词
                prompt = self.prompt_template.format(
                    query=query,
                    document=candidate.content[:1000],  # 限制文档长度
                )
                
                # 调用 LLM
                try:
                    response = self.llm_client.chat(prompt)
                    score_text = response.content.strip()
                    
                    # 解析分数
                    score = self._parse_score(score_text)
                    candidate.score = score
                    candidate.metadata["llm_rerank_score"] = score
                    
                except Exception as e:
                    logger.error(f"LLM 重排序失败: {e}")
                    candidate.metadata["llm_rerank_error"] = str(e)
        
        # 按分数降序排序
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # 截断
        if top_k is not None:
            candidates = candidates[:top_k]
        
        return candidates
    
    def _parse_score(self, score_text: str) -> float:
        """解析 LLM 返回的分数
        
        Args:
            score_text: 分数文本
            
        Returns:
            分数值
        """
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except ValueError:
            numbers = re.findall(r"[\d.]+", score_text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            else:
                logger.warning(f"无法解析分数: {score_text}")
                return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "type": "llm_based",
            "llm_model": getattr(self.llm_client, "model", "unknown"),
            "batch_size": self.batch_size,
        }


class APIBasedReranker:
    """基于 API 的重排序器
    
    使用 OpenAI 兼容 API 进行重排序。
    支持 SiliconFlow、Jina AI 等提供 rerank API 的服务。
    
    Attributes:
        model: 模型名称
        api_key: API 密钥
        base_url: API 基础 URL
        client: OpenAI 客户端实例
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """初始化 API 重排序器
        
        Args:
            model: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL
            **kwargs: 额外参数
        """
        config = get_config()
        
        self.model = model or config.rerank_model
        self.api_key = api_key or config.rerank_api_key
        self.base_url = base_url or config.rerank_base_url
        
        self._init_client()
        
        logger.debug(f"初始化 API 重排序器: model={self.model}")
    
    def _init_client(self) -> None:
        """初始化 OpenAI 客户端"""
        try:
            import httpx
            self._http_client = httpx.Client(timeout=60.0)
        except ImportError:
            raise ImportError("请安装 httpx 库: pip install httpx")
    
    def _call_rerank_api(
        self,
        query: str,
        documents: List[str],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """调用 rerank API
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回数量
            
        Returns:
            重排序结果列表
        """
        url = f"{self.base_url}/rerank"
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        response = self._http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        return result.get("results", [])
    
    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """使用 API 对候选文档进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回结果数量（可选）
            
        Returns:
            重排序后的结果列表
        """
        if not candidates:
            return []
        
        try:
            documents = [candidate.content for candidate in candidates]
            
            logger.debug(
                f"调用 Rerank API: model={self.model}, "
                f"query_len={len(query)}, docs_count={len(documents)}"
            )
            
            results = self._call_rerank_api(
                query=query,
                documents=documents,
                top_n=top_k or len(candidates),
            )
            
            reranked = []
            for item in results:
                idx = item.get("index", 0)
                score = item.get("relevance_score", 0.0)
                
                candidate = candidates[idx]
                candidate.score = float(score)
                candidate.metadata["api_rerank_score"] = float(score)
                reranked.append(candidate)
            
            reranked.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug(f"API 重排序完成: results={len(reranked)}")
            
            return reranked
            
        except Exception as e:
            logger.error(f"API 重排序失败: {e}")
            for candidate in candidates:
                candidate.metadata["api_rerank_error"] = str(e)
            return candidates[:top_k] if top_k else candidates
    
    def rerank_batch(
        self,
        queries: List[str],
        candidates_list: List[List[SearchResult]],
        top_k: Optional[int] = None,
    ) -> List[List[SearchResult]]:
        """批量重排序
        
        Args:
            queries: 查询文本列表
            candidates_list: 候选文档列表的列表
            top_k: 每个查询返回的结果数量（可选）
            
        Returns:
            重排序后的结果列表的列表
        """
        results = []
        
        for query, candidates in zip(queries, candidates_list):
            reranked = self.rerank(query, candidates, top_k)
            results.append(reranked)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "type": "api_based",
            "model": self.model,
            "base_url": self.base_url,
        }


def create_reranker(
    mode: str = "auto",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """创建重排序器的便捷函数
    
    Args:
        mode: 模式 ("local", "api", "llm", "auto")
            - local: 使用本地模型 (FlagEmbedding/sentence-transformers)
            - api: 使用 API 服务
            - llm: 使用 LLM 进行重排序
            - auto: 根据配置自动选择
        model: 模型名称
        api_key: API 密钥 (仅 api/llm 模式)
        base_url: API 基础 URL (仅 api 模式)
        **kwargs: 其他参数
            
    Returns:
        重排序器实例
        
    Examples:
        # API 方式
        reranker = create_reranker(
            mode="api",
            model="Pro/BAAI/bge-reranker-v2-m3",
            api_key="sk-xxx",
            base_url="https://api.siliconflow.cn/v1"
        )
        
        # 本地模型
        reranker = create_reranker(
            mode="local",
            model="BAAI/bge-reranker-large"
        )
        
        # 自动选择
        reranker = create_reranker(mode="auto")
    """
    config = get_config()
    
    if mode == "auto":
        mode = config.rerank_mode or "local"
    
    if mode == "api":
        return APIBasedReranker(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    elif mode == "llm":
        from src.llms.deepseek_client import create_client
        llm_client = create_client(
            model=api_key,
            api_key=api_key or config.openai_api_key,
            base_url=base_url or config.openai_api_base,
        )
        return LLMBasedReranker(
            llm_client=llm_client,
            **kwargs,
        )
    else:  # local
        return Reranker(**kwargs)
