from __future__ import annotations
import argparse,json,os
import chromadb

p=argparse.ArgumentParser(description="Inspect LexiCore Chroma databases without loading an embedding model.")
p.add_argument("--db",default=os.getenv("LEXICORE_DB_PATH","./chroma_db"))
p.add_argument("--collection",default=None)
p.add_argument("--sample",type=int,default=1000)
a=p.parse_args()
client=chromadb.PersistentClient(path=a.db)
names=client.list_collections()
names=[getattr(x,"name",str(x)) for x in names]
out={"db":a.db,"collections":{}}
for name in names:
    if a.collection and name!=a.collection: continue
    c=client.get_collection(name)
    data=c.get(limit=a.sample,include=["metadatas"])
    fields={}
    for m in data.get("metadatas") or []:
        for k,v in (m or {}).items():
            f=fields.setdefault(k,{"types":set(),"examples":[]}); f["types"].add(type(v).__name__)
            if len(f["examples"])<5 and v not in f["examples"]: f["examples"].append(v)
    for v in fields.values(): v["types"]=sorted(v["types"])
    out["collections"][name]={"count":c.count(),"metadata":fields}
print(json.dumps(out,indent=2,ensure_ascii=False))
