"""SentencePiece tokenizer wrapper for ASR."""

from pathlib import Path
from typing import Optional, Union

import sentencepiece as spm


class Tokenizer:
    """SentencePiece tokenizer wrapper.

    Handles text tokenization for ASR training.
    Token ID 0 is reserved for CTC blank.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        vocab_size: int = 5000,
    ):
        """Initialize Tokenizer.

        Args:
            model_path: Path to trained SentencePiece model.
            vocab_size: Vocabulary size (used when training new model).
        """
        self.model_path = Path(model_path) if model_path else None
        self.vocab_size = vocab_size
        self._sp = None

        if self.model_path and self.model_path.exists():
            self.load(self.model_path)

    @property
    def sp(self) -> spm.SentencePieceProcessor:
        """Get SentencePiece processor."""
        if self._sp is None:
            raise RuntimeError("Tokenizer not loaded. Call load() or train() first.")
        return self._sp

    def load(self, model_path: Union[str, Path]) -> None:
        """Load a trained SentencePiece model.

        Args:
            model_path: Path to .model file.
        """
        self.model_path = Path(model_path)
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(self.model_path))

    def train(
        self,
        texts: list[str],
        model_prefix: Union[str, Path],
        vocab_size: Optional[int] = None,
    ) -> None:
        """Train a new SentencePiece model.

        Args:
            texts: List of training texts.
            model_prefix: Output model prefix (without extension).
            vocab_size: Vocabulary size (uses self.vocab_size if None).
        """
        import tempfile

        vocab_size = vocab_size or self.vocab_size
        model_prefix = Path(model_prefix)
        model_prefix.parent.mkdir(parents=True, exist_ok=True)

        # Write texts to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for text in texts:
                f.write(text.strip() + "\n")
            temp_path = f.name

        # Train SentencePiece
        spm.SentencePieceTrainer.Train(
            input=temp_path,
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,  # Reserve 0 for CTC blank
            unk_id=1,
            bos_id=-1,  # No BOS
            eos_id=-1,  # No EOS
            user_defined_symbols=["<blank>"],
        )

        # Clean up temp file
        Path(temp_path).unlink()

        # Load the trained model
        self.load(f"{model_prefix}.model")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Note: Token IDs are shifted by 1 to reserve 0 for CTC blank.

        Args:
            text: Input text.

        Returns:
            List of token IDs.
        """
        # SentencePiece IDs + 1 to reserve 0 for blank
        ids = self.sp.EncodeAsIds(text)
        return [id + 1 for id in ids]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text.
        """
        # Remove blank tokens (0) and shift back
        filtered_ids = [id - 1 for id in ids if id > 0]
        return self.sp.DecodeIds(filtered_ids)

    def __len__(self) -> int:
        """Return vocabulary size including blank token."""
        return self.sp.GetPieceSize() + 1  # +1 for blank

    @property
    def blank_id(self) -> int:
        """CTC blank token ID."""
        return 0

    @property
    def unk_id(self) -> int:
        """Unknown token ID."""
        return 2  # SentencePiece unk (1) + 1 for blank offset
