import sentencepiece as spm


#### Load SentencePiece model
def load_sentencepiece(model_path: str):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


#### Subword encoding with SentencePiece BPE and post-padding
def subword_encode(
    text: str,
    sp,
    *,
    max_len: int = 300,
    pad_id: int = 0,
    unk_id: int | None = None
) -> list[int]:
    """
    Encode text using SentencePiece BPE and apply post-padding.
    """
    if not text:
        return [pad_id] * max_len

    ids = sp.encode(text, out_type=int)

    if unk_id is not None:
        ids = [i if i < sp.get_piece_size() else unk_id for i in ids]

    # truncate
    ids = ids[:max_len]

    # post-padding
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))

    return ids


#### Character-level encoding with post-padding
def char_encode(
    text: str,
    char_vocab: dict[str, int],
    *,
    max_len: int = 800,
    pad_id: int = 0,
    unk_id: int = 1
) -> list[int]:
    """
    Character-level encoding with post-padding.
    """
    if not text:
        return [pad_id] * max_len

    ids = [char_vocab.get(c, unk_id) for c in text]

    # truncate
    ids = ids[:max_len]

    # post-padding
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))

    return ids


#### Encode inputs for XLSTM model
def encode_xlstm_inputs(
    clean_text: str,
    *,
    sp,
    char_vocab: dict[str, int],
    Tmax: int = 300,
    Tchar: int = 800,
    pad_id: int = 0,
    char_unk_id: int = 1
) -> dict[str, list[int]]:
    """
    Encode inputs for XLSTM model: subword IDs and character IDs with post-padding.
    """
    word_ids = subword_encode(
        clean_text,
        sp,
        max_len=Tmax,
        pad_id=pad_id,
        unk_id=sp.unk_id()
    )

    char_ids = char_encode(
        clean_text,
        char_vocab,
        max_len=Tchar,
        pad_id=pad_id,
        unk_id=char_unk_id
    )

    return {
        "word_ids": word_ids,
        "char_ids": char_ids
    }
