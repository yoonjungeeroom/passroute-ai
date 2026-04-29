import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODELS_DIR = "/app/models"
ONNX_MODEL_PATH = f"{MODELS_DIR}/kr-sbert-uint8.onnx"
TOKENIZER_PATH = f"{MODELS_DIR}/tokenizer"
MAX_TOKEN_LENGTH = 512


class OnnxEmbedder:
    """ONNX 양자화 모델 기반 텍스트 임베딩 서비스.

    snunlp/KR-SBERT-V40K-klueNLI-augSTS 모델의 ONNX QUInt8 양자화 버전을 사용한다.
    Consumer와 동일한 후처리(Mean Pooling + L2 정규화)를 적용하여
    ChromaDB 유사도 검색이 정상 동작하도록 보장한다.
    """

    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self._session = ort.InferenceSession(
            ONNX_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    def embed(self, text: str) -> list[float]:
        """텍스트를 768차원 임베딩 벡터로 변환한다."""
        encoded = self._tokenizer(
            [text],
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
        )

        ort_inputs = {
            name: encoded[name]
            for name in self._input_names
            if name in encoded
        }
        outputs = self._session.run(None, ort_inputs)
        token_embeddings: np.ndarray = outputs[0]

        # Mean Pooling
        attention_mask = encoded["attention_mask"]
        mask_expanded = np.expand_dims(attention_mask, axis=-1)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        mean_embedding = (sum_embeddings / sum_mask)[0]

        # L2 정규화
        norm = max(float(np.linalg.norm(mean_embedding)), 1e-9)
        return (mean_embedding / norm).tolist()
