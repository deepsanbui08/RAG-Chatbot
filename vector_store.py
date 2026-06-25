from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class VectorStore:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def create_index(self, chunks):

        embeddings = self.model.encode(chunks)

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(
            dimension
        )

        index.add(embeddings)

        return index

    def search(
        self,
        question,
        index,
        chunks,
        k=5
    ):

        query_embedding = self.model.encode(
            [question]
        )

        query_embedding = np.array(
            query_embedding,
            dtype="float32"
        )

        distances, indices = index.search(
            query_embedding,
            min(k, len(chunks))
        )

        retrieved_chunks = []

        for idx in indices[0]:

            if idx < len(chunks):
                retrieved_chunks.append(
                    chunks[idx]
                )

        return retrieved_chunks