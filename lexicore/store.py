from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from .core import Record, classify, norm

DEFAULT_MODEL = os.getenv("LEXICORE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_COLLECTION = os.getenv("LEXICORE_COLLECTION", "lexicore_evidence_v3")


@dataclass
class Hit:
    id: str
    text: str
    source: str
    source_family: str
    category: str
    citation: str
    distance: float
    rank: int
    metadata: dict[str, Any]

    @property
    def relevance(self) -> float:
        # Ranking indicator only. It is NOT probability or factual confidence.
        return 1.0 / (1.0 + max(self.distance, 0.0))

    def to_record(self) -> Record:
        return Record(self.id, self.text, self.source, self.source_family, self.category, self.citation,
                      self.metadata.get("segment_type", ""), self.metadata.get("language", ""), self.metadata.get("dataset", ""), self.metadata)


class EvidenceStore:
    def __init__(self, path: str, collection: str = DEFAULT_COLLECTION, model_name: str = DEFAULT_MODEL):
        self.path=path
        self.collection_name=collection
        self.model_name=model_name
        self.client=chromadb.PersistentClient(path=path)
        self.collection=self.client.get_collection(collection)
        self.model=SentenceTransformer(model_name)

    @classmethod
    def open_or_create(cls, path: str, collection: str = DEFAULT_COLLECTION, model_name: str = DEFAULT_MODEL):
        client=chromadb.PersistentClient(path=path)
        coll=client.get_or_create_collection(name=collection, metadata={"hnsw:space":"cosine","schema_version":"3"})
        obj=cls.__new__(cls)
        obj.path=path; obj.collection_name=collection; obj.model_name=model_name; obj.client=client; obj.collection=coll
        obj.model=SentenceTransformer(model_name)
        return obj

    def count(self): return self.collection.count()

    def add_records(self, records: list[Record], batch_size: int=256):
        if not records: return
        for start in range(0,len(records),batch_size):
            batch=records[start:start+batch_size]
            emb=self.model.encode([r.text for r in batch], normalize_embeddings=True, show_progress_bar=False).tolist()
            self.collection.upsert(ids=[r.id for r in batch], documents=[r.text for r in batch], metadatas=[r.chroma_metadata() for r in batch], embeddings=emb)

    def inspect(self, sample_size:int=1000):
        data=self.collection.get(limit=sample_size, include=["metadatas"])
        fields={}
        for m in data.get("metadatas") or []:
            for k,v in (m or {}).items():
                f=fields.setdefault(k,{"types":set(),"examples":[]}); f["types"].add(type(v).__name__)
                if len(f["examples"])<5 and v not in f["examples"]: f["examples"].append(v)
        for v in fields.values(): v["types"]=sorted(v["types"])
        return {"path":self.path,"collection":self.collection_name,"count":self.count(),"metadata":fields}

    def search(self, query:str, n:int=20, categories:list[str]|None=None, families:list[str]|None=None):
        q=norm(query)
        if not q: return []
        where=[]
        if categories: where.append({"category":{"$in":categories}})
        if families: where.append({"source_family":{"$in":families}})
        kwargs={"query_embeddings":[self.model.encode(q,normalize_embeddings=True).tolist()],"n_results":min(max(n,1),100),"include":["documents","metadatas","distances"]}
        if len(where)==1: kwargs["where"]=where[0]
        elif len(where)>1: kwargs["where"]={"$and":where}
        res=self.collection.query(**kwargs)
        docs=res.get("documents",[[]])[0]; metas=res.get("metadatas",[[]])[0]; ids=res.get("ids",[[]])[0]; ds=res.get("distances",[[]])[0]
        out=[]
        for i,doc in enumerate(docs):
            m=metas[i] or {}; source=norm(m.get("source") or m.get("scripture_source") or "Unknown")
            category,family=classify(source,m.get("dataset", ""),m.get("segment_type", ""))
            category=norm(m.get("category")) or category; family=norm(m.get("source_family")) or family
            out.append(Hit(str(ids[i]),str(doc),source,family,category,norm(m.get("citation") or m.get("citation_ref")),float(ds[i] or 0),i+1,m))
        return out

    def delete_collection(self): self.client.delete_collection(self.collection_name)
