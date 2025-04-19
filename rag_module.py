# rag_module.py
import numpy as np
import faiss
import traceback
from sentence_transformers import SentenceTransformer

# Import config variables
from config import SENTENCE_TRANSFORMER_MODEL

class FlightRAG:
    """Handles encoding flight data and querying using Sentence Transformers and FAISS."""
    def __init__(self, model_name=SENTENCE_TRANSFORMER_MODEL):
        self.encoder = None
        self.index = None
        self.flight_data = []
        self.dimension = 0
        self.is_initialized = False
        try:
            # Consider device='cuda' if GPU is available
            self.encoder = SentenceTransformer(model_name)
            self.dimension = self.encoder.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(self.dimension) # Initialize empty index
            self.is_initialized = True
            print(f"RAG Encoder ({model_name}) initialized.")
        except Exception as e:
            print(f"FATAL: Failed to initialize SentenceTransformer model '{model_name}': {e}")
            traceback.print_exc()

    def _create_flight_description(self, flight, index):
        """Helper to create a textual description for a flight."""
        try:
            parts = []
            origin = flight.get('origin', 'N/A')
            dest = flight.get('destination', 'N/A')
            airline = flight.get('airline', 'Unknown')
            price = flight.get('price')
            currency = flight.get('currency', 'USD')
            dep_time = flight.get('departure_time', 'N/A')
            arr_time = flight.get('arrival_time', 'N/A')
            duration = flight.get('duration', 'N/A')
            stops = flight.get('stops', 'N/A')

            parts.append(f"Option {index+1}: {airline} from {origin} to {dest}.")
            if price is not None: parts.append(f"Price approx ${price:.0f} {currency}.")
            parts.append(f"Departs {dep_time}, Arrives {arr_time}.")
            parts.append(f"Duration {duration}, {stops} stops.")
            if flight.get('is_round_trip'):
                parts.append(f"Return duration {flight.get('return_duration', 'N/A')}, {flight.get('return_stops', 'N/A')} stops.")
            return " ".join(parts)
        except Exception as e:
            print(f"Warn: Error creating desc for flight {index}: {e}")
            return None

    def encode_flight_data(self, flights):
        """Encodes a list of flight dictionaries into the FAISS index."""
        if not self.is_initialized:
            print("RAG Error: Cannot encode, RAG system not initialized.")
            return False

        if not flights:
            self.flight_data = []
            self.index = faiss.IndexFlatL2(self.dimension) # Reset index
            print("RAG Info: No flight data provided, resetting index.")
            return True # Success encoding nothing

        self.flight_data = flights # Store original data
        flight_texts = [self._create_flight_description(f, i) for i, f in enumerate(flights)]
        valid_texts = [t for t in flight_texts if t is not None] # Filter out failed descriptions

        if not valid_texts:
            print("RAG Warn: No valid flight descriptions generated.")
            self.index = faiss.IndexFlatL2(self.dimension); return False

        try:
            embeddings = self.encoder.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
            if embeddings.dtype != np.float32: embeddings = embeddings.astype(np.float32)

            self.index = faiss.IndexFlatL2(self.dimension) # Create new index
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            print(f"RAG Info: Encoded {self.index.ntotal} flights into index.")
            return True
        except Exception as e:
            print(f"RAG Error: FAISS indexing failed: {e}")
            traceback.print_exc()
            self.flight_data = []
            self.index = faiss.IndexFlatL2(self.dimension) # Reset on failure
            return False

    def query(self, question, top_k=3):
        """Queries the FAISS index with a natural language question."""
        if not self.is_initialized or self.index is None or self.index.ntotal == 0:
            print("RAG Warn: Query attempted on uninitialized/empty index.")
            return []
        if not question: return []

        try:
            q_embedding = self.encoder.encode([question], convert_to_numpy=True, show_progress_bar=False)
            if q_embedding.dtype != np.float32: q_embedding = q_embedding.astype(np.float32)
            faiss.normalize_L2(q_embedding)

            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(q_embedding, k)

            # Map FAISS indices back to original flight_data list indices
            # This assumes flight_data and the indexed embeddings correspond 1:1
            relevant_flights = [self.flight_data[idx] for idx in indices[0] if idx < len(self.flight_data)]
            return relevant_flights
        except Exception as e:
            print(f"RAG Error: Query failed: {e}")
            traceback.print_exc()
            return []