# -*- coding: utf-8 -*-
"""全局配置管理模块

使用 pydantic-settings 实现配置管理，支持从环境变量加载配置。
"""

import os
from pathlib import Path
from typing import Optional, Literal
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Config(BaseSettings):
    """全局配置类
    
    使用 pydantic-settings 管理所有配置项，支持：
    - 从环境变量加载
    - 从 .env 文件加载
    - 类型验证和转换
    - 默认值设置
    """
    
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ==================== API Keys ====================
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"),
        description="OpenAI API Key"
    )
    openai_api_base: Optional[str] = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL"),
        description="OpenAI API Base URL"
    )
    deepseek_api_key: Optional[str] = Field(
        default=None,
        description="DeepSeek API Key"
    )
    deepseek_base_url: Optional[str] = Field(
        default=None,
        description="DeepSeek API Base URL"
    )
    deepseek_model: Optional[str] = Field(
        default=None,
        description="DeepSeek model name"
    )

    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API Key"
    )

    embed_api_key: Optional[str] = Field(
        default=None,
        description="embeddubg API Key"
    )
    embed_base_url: Optional[str] = Field(
        default=None,
        description="embeddubg API Base URL"
    )
    
    # ==================== Ollama 配置 ====================
    ollama_base_url: Optional[str] = Field(
        default="http://localhost:11434",
        description="Ollama 服务地址"
    )
    ollama_model: Optional[str] = Field(
        default=None,
        description="Ollama 模型名称"
    )
    
    # ==================== 数据库配置 ====================
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j 数据库连接 URI"
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j 用户名"
    )
    neo4j_password: str = Field(
        default="password",
        description="Neo4j 密码"
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j 数据库名称"
    )
    
    # 向量数据库
    vector_db_type: str = Field(
        default="faiss",
        description="向量数据库类型 (milvus, pinecone, chroma, faiss)"
    )
    vector_db_host: str = Field(
        default="localhost",
        description="向量数据库主机地址"
    )
    vector_db_port: int = Field(
        default=19530,
        description="向量数据库端口"
    )
    vector_db_collection: str = Field(
        default="rag_collection",
        description="向量数据库集合名称"
    )
    
    # FAISS 索引路径
    faiss_index_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "faiss_index"),
        description="FAISS 索引存储路径"
    )
    
    # Local Graph 存储路径
    local_graph_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "local_graph"),
        description="本地图存储路径"
    )
    
    # Redis 缓存
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis 连接 URL"
    )
    
    # PostgreSQL (可选，用于存储文档元数据)
    postgres_url: Optional[str] = Field(
        default=None,
        description="PostgreSQL 连接 URL"
    )
    
    # ==================== 检索参数 ====================
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="检索返回的文档数量"
    )
    rerank_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="重排序后返回的文档数量"
    )
    vector_dim: int = Field(
        default=1024,
        description="向量维度 (OpenAI: 1536, DeepSeek: 1024, BGE: 1024)"
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="相似度阈值"
    )
    
    # ==================== 分块参数 ====================
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=4096,
        description="文档分块大小"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="分块重叠大小"
    )
    
    # ==================== LLM 参数 ====================
    llm_model: str = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        description="LLM 模型名称"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM 温度参数"
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="LLM 最大 token 数"
    )
    
    # ==================== 嵌入模型参数 ====================
    embedding_mode: str = Field(
        default="local",
        description="嵌入模式: api, local, bge, auto"
    )
    embedding_model: str = Field(
        default_factory=lambda: str(PROJECT_ROOT / "models" / "BAAI" / "bge-m3")
        if (PROJECT_ROOT / "models" / "BAAI" / "bge-m3").exists()
        else "BAAI/bge-m3",
        description="嵌入模型名称"
    )
    embedding_device: Optional[str] = Field(
        default="auto",
        description="嵌入模型设备: cuda, cpu, auto"
    )
    embedding_use_fp16: bool = Field(
        default=True,
        description="是否使用半精度 (仅 BGE 模型)"
    )
    
    # ==================== 重排序配置 ====================
    rerank_model: Optional[str] = Field(
        default="BAAI/bge-reranker-base",
        description="重排序模型名称"
    )
    rerank_api_key: Optional[str] = Field(
        default=None,
        description="重排序 API Key"
    )
    rerank_base_url: Optional[str] = Field(
        default=None,
        description="重排序 API Base URL"
    )
    rerank_mode: str = Field(
        default="local",
        description="重排序模式: local (本地模型), api (API调用)"
    )

    @model_validator(mode="after")
    def prefer_deepseek_env(self) -> "Config":
        """Use DeepSeek-specific variables when present.

        Some environments keep OPENAI_* variables around. This project uses a
        DeepSeek OpenAI-compatible endpoint for generation/judging, so the
        DEEPSEEK_* variables are authoritative when they are set.
        """
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or self.deepseek_api_key
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL") or self.deepseek_base_url
        deepseek_model = os.getenv("DEEPSEEK_MODEL") or self.deepseek_model
        if deepseek_api_key:
            self.openai_api_key = deepseek_api_key
        if deepseek_base_url:
            self.openai_api_base = deepseek_base_url
        if deepseek_model:
            self.llm_model = deepseek_model
        return self
    
    # ==================== Agent 参数 ====================
    agent_max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Agent 最大迭代次数"
    )
    agent_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Agent 超时时间（秒）"
    )
    
    # ==================== 应用配置 ====================
    environment: str = Field(
        default="development",
        description="运行环境"
    )
    debug: bool = Field(
        default=False,
        description="调试模式"
    )
    log_level: str = Field(
        default="INFO",
        description="日志级别"
    )
    log_file: str = Field(
        default="",
        description="日志文件路径"
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper
    
    @field_validator("vector_db_type")
    @classmethod
    def validate_vector_db_type(cls, v: str) -> str:
        """验证向量数据库类型"""
        valid_types = ["milvus", "pinecone", "chroma", "faiss", "weaviate", "qdrant"]
        v_lower = v.lower()
        if v_lower not in valid_types:
            raise ValueError(f"Invalid vector db type: {v}. Must be one of {valid_types}")
        return v_lower
    
    @field_validator("embedding_mode")
    @classmethod
    def validate_embedding_mode(cls, v: str) -> str:
        """验证嵌入模式"""
        valid_modes = ["api", "local", "bge", "auto"]
        v_lower = v.lower()
        if v_lower not in valid_modes:
            raise ValueError(f"Invalid embedding mode: {v}. Must be one of {valid_modes}")
        return v_lower
    
    def get_neo4j_config(self) -> dict:
        """获取 Neo4j 配置字典"""
        return {
            "uri": self.neo4j_uri,
            "user": self.neo4j_user,
            "password": self.neo4j_password,
            "database": self.neo4j_database,
        }
    
    def get_vector_db_config(self) -> dict:
        """获取向量数据库配置字典"""
        return {
            "type": self.vector_db_type,
            "host": self.vector_db_host,
            "port": self.vector_db_port,
            "collection": self.vector_db_collection,
        }
    
    def get_llm_config(self) -> dict:
        """获取 LLM 配置字典"""
        return {
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
        }
    
    def get_embedding_config(self) -> dict:
        """获取嵌入模型配置字典"""
        return {
            "mode": self.embedding_mode,
            "model": self.embedding_model,
            "device": self.embedding_device,
            "use_fp16": self.embedding_use_fp16,
            "dimension": self.vector_dim,
        }
    
    def get_retrieval_config(self) -> dict:
        """获取检索配置字典"""
        return {
            "top_k": self.top_k,
            "rerank_top_k": self.rerank_top_k,
            "vector_dim": self.vector_dim,
            "similarity_threshold": self.similarity_threshold,
        }


@lru_cache()
def get_config() -> Config:
    """获取配置单例
    
    使用 lru_cache 实现单例模式，确保配置只加载一次。
    
    Returns:
        Config: 配置实例
    """
    return Config()


config = get_config()


if __name__ == '__main__':
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f".env 文件路径: {ENV_FILE}")
    print(f".env 文件存在: {ENV_FILE.exists()}")
    print(f"OPENAI_API_KEY: {config.openai_api_key[:10]}..." if config.openai_api_key else "未设置")
    print(f"OPENAI_API_BASE: {config.openai_api_base}")
    print(f"LLM_MODEL: {config.llm_model}")
    print(f"EMBEDDING_MODE: {config.embedding_mode}")
    print(f"EMBEDDING_MODEL: {config.embedding_model}")
