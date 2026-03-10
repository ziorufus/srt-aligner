import re
import math
import os
from pathlib import Path
import numpy as np
import pysrt
import torch

from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")


def align_text_to_srt_advanced(
    srt_path: str,
    target_text: str,
    output_path: Optional[str] = None,
    model: Optional[SentenceTransformer] = None,
    max_src_group: int = 3,
    max_tgt_group: int = 3,
    max_chars_per_line: int = 42,
):
    """
    Riallinea un testo libero a un file SRT sorgente e restituisce
    un nuovo SRT con gli stessi timestamp originali.

    Parametri
    ---------
    srt_path : str
        Percorso del file SRT sorgente.
    target_text : str
        Testo di destinazione libero, in qualsiasi lingua.
    output_path : str | None
        Se fornito, salva l'SRT risultante su disco.
    model : SentenceTransformer
        Modello SentenceTransformer multilingue.
    max_src_group : int
        Numero massimo di cue sorgente aggregabili in DP.
    max_tgt_group : int
        Numero massimo di segmenti target aggregabili in DP.
    max_chars_per_line : int
        Lunghezza massima per riga nel wrapping finale.

    Ritorna
    -------
    pysrt.SubRipFile
        Oggetto SRT tradotto.
    """
    if model is None:
        model = _load_default_model()

    subs = pysrt.open(srt_path, encoding="utf-8")

    if len(subs) == 0:
        raise ValueError("Il file SRT è vuoto.")

    src_cues_raw = [sub.text for sub in subs]
    src_cues_norm = [_normalize_subtitle_text(x) for x in src_cues_raw]

    tgt_segments = _segment_target_text(target_text)
    if not tgt_segments:
        raise ValueError("Non sono riuscito a segmentare il testo target.")

    # Cache per embedding di gruppi concatenati
    group_emb_cache: Dict[Tuple[str, int, int], np.ndarray] = {}

    def get_group_embedding(kind: str, start: int, end: int) -> np.ndarray:
        key = (kind, start, end)
        if key in group_emb_cache:
            return group_emb_cache[key]

        if kind == "src":
            text = " ".join(src_cues_norm[start:end]).strip()
        elif kind == "tgt":
            text = " ".join(tgt_segments[start:end]).strip()
        else:
            raise ValueError("kind deve essere 'src' o 'tgt'")

        emb = model.encode([text], convert_to_numpy=True)[0]
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        group_emb_cache[key] = emb
        return emb

    alignment = _monotonic_group_alignment(
        src_cues_norm=src_cues_norm,
        tgt_segments=tgt_segments,
        get_group_embedding=get_group_embedding,
        max_src_group=max_src_group,
        max_tgt_group=max_tgt_group,
    )

    aligned_texts = [""] * len(src_cues_norm)

    for i0, i1, j0, j1 in alignment:
        src_count = i1 - i0
        tgt_text = " ".join(tgt_segments[j0:j1]).strip()

        if not tgt_text:
            continue

        distributed = _distribute_target_text_over_cues(
            source_texts=src_cues_norm[i0:i1],
            target_text=tgt_text,
        )

        if len(distributed) != src_count:
            distributed = [tgt_text] * src_count

        for offset, piece in enumerate(distributed):
            aligned_texts[i0 + offset] = _wrap_subtitle_text(
                piece,
                max_chars_per_line=max_chars_per_line
            )

    # Fallback: se qualche cue resta vuoto, propaghiamo in modo conservativo
    aligned_texts = _fill_empty_slots(aligned_texts)

    for sub, txt in zip(subs, aligned_texts):
        sub.text = txt

    if output_path:
        subs.save(output_path, encoding="utf-8")

    return subs


def translate_srt_with_alignment_advanced(
    srt_path: str,
    english_text: str,
    output_path: Optional[str] = None,
    model: Optional[SentenceTransformer] = None,
    max_src_group: int = 3,
    max_tgt_group: int = 3,
    max_chars_per_line: int = 42,
):
    """
    Alias retrocompatibile della funzione generica.
    """
    return align_text_to_srt_advanced(
        srt_path=srt_path,
        target_text=english_text,
        output_path=output_path,
        model=model,
        max_src_group=max_src_group,
        max_tgt_group=max_tgt_group,
        max_chars_per_line=max_chars_per_line,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_default_model() -> SentenceTransformer:
    device = _select_device()
    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )
    print(f"Model loaded on device: {model.device}")
    return model


def _normalize_subtitle_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"<[^>]+>", " ", text)  # rimuove tag html semplici
    text = re.sub(r"\{\\.*?\}", " ", text)  # rimuove tag stile ASS/SSA basilari
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _segment_target_text(text: str) -> List[str]:
    """
    Segmenta il testo target in:
    1) frasi
    2) sotto-frasi/clausole se la frase è molto lunga
    """
    text = _normalize_subtitle_text(text)
    if not text:
        return []

    sents = _split_into_sentences(text)

    segments = []
    for sent in sents:
        pieces = _split_long_sentence(sent)
        for p in pieces:
            p = p.strip()
            if p:
                segments.append(p)

    # Merge di segmenti troppo piccoli
    merged = []
    buffer = ""
    for seg in segments:
        if len(seg.split()) <= 2 and merged:
            merged[-1] = (merged[-1] + " " + seg).strip()
        elif len(seg.split()) <= 2:
            buffer = (buffer + " " + seg).strip()
        else:
            if buffer:
                seg = (buffer + " " + seg).strip()
                buffer = ""
            merged.append(seg)

    if buffer:
        if merged:
            merged[-1] = (merged[-1] + " " + buffer).strip()
        else:
            merged.append(buffer)

    return merged


def _split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentence_pattern = r"(?<=[.!?…。！？؟۔])\s+|(?<=\.\.\.)\s+"
    parts = [part.strip() for part in re.split(sentence_pattern, text) if part.strip()]
    if len(parts) <= 1:
        return [text]
    return parts


def _split_long_sentence(sentence: str) -> List[str]:
    """
    Spezza frasi lunghe con euristiche generiche basate su punteggiatura e lunghezza.
    """
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if not sentence:
        return []

    # Sotto una certa soglia, lasciamo stare
    if len(sentence) <= 80 and len(sentence.split()) <= 14:
        return [sentence]

    # Split su punteggiatura morbida e separatori comuni, senza dipendere da una lingua specifica.
    parts = re.split(r"(?<=[,;:，；：])\s+|\s+[/-]\s+", sentence)
    if len(parts) == 1:
        parts = re.split(
            r"(?<=[,;:，；：])|(?<=[)\]】])\s+|(?<=\s[-–—])\s+",
            sentence,
        )

    # Se ancora troppo lungo, split ulteriore
    final_parts = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > 100 or len(part.split()) > 18:
            subparts = re.split(
                r"(?<=[,;:，；：])\s+|\s+[/-]\s+",
                part,
            )
            final_parts.extend([x.strip() for x in subparts if x.strip()])
        else:
            final_parts.append(part)

    # Evita segmenti ridicoli
    cleaned = []
    buffer = ""
    for part in final_parts:
        if len(part.split()) <= 3 and cleaned:
            cleaned[-1] = cleaned[-1] + " " + part
        elif len(part.split()) <= 3:
            buffer = (buffer + " " + part).strip()
        else:
            if buffer:
                part = (buffer + " " + part).strip()
                buffer = ""
            cleaned.append(part)

    if buffer:
        if cleaned:
            cleaned[-1] += " " + buffer
        else:
            cleaned.append(buffer)

    return cleaned if cleaned else [sentence]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _monotonic_group_alignment(
    src_cues_norm: List[str],
    tgt_segments: List[str],
    get_group_embedding,
    max_src_group: int = 3,
    max_tgt_group: int = 3,
) -> List[Tuple[int, int, int, int]]:
    """
    Dynamic programming monotono many-to-many.
    Ogni stato allinea un gruppo di cue sorgente con un gruppo di segmenti target.
    """
    n = len(src_cues_norm)
    m = len(tgt_segments)

    NEG_INF = -1e18
    dp = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    back: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}

    dp[0, 0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i, j] <= NEG_INF / 2:
                continue

            for di in range(1, max_src_group + 1):
                if i + di > n:
                    break

                for dj in range(1, max_tgt_group + 1):
                    if j + dj > m:
                        break

                    score = _group_alignment_score(
                        src_cues_norm=src_cues_norm,
                        tgt_segments=tgt_segments,
                        i0=i,
                        i1=i + di,
                        j0=j,
                        j1=j + dj,
                        get_group_embedding=get_group_embedding,
                    )

                    cand = dp[i, j] + score
                    if cand > dp[i + di, j + dj]:
                        dp[i + di, j + dj] = cand
                        back[(i + di, j + dj)] = (i, j, di, dj)

    if (n, m) not in back and not (n == 0 and m == 0):
        raise RuntimeError("Allineamento fallito: impossibile trovare un path completo.")

    alignment = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, di, dj = back[(i, j)]
        alignment.append((pi, pi + di, pj, pj + dj))
        i, j = pi, pj

    alignment.reverse()
    return alignment


def _group_alignment_score(
    src_cues_norm: List[str],
    tgt_segments: List[str],
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    get_group_embedding,
) -> float:
    src_text = " ".join(src_cues_norm[i0:i1]).strip()
    tgt_text = " ".join(tgt_segments[j0:j1]).strip()

    src_emb = get_group_embedding("src", i0, i1)
    tgt_emb = get_group_embedding("tgt", j0, j1)
    sim = _cosine(src_emb, tgt_emb)

    src_len = max(len(src_text), 1)
    tgt_len = max(len(tgt_text), 1)

    # Penalità differenza lunghezza relativa
    len_ratio = tgt_len / src_len
    len_penalty = abs(math.log(len_ratio + 1e-9)) * 0.15

    # Penalità lieve per gruppi grandi: favorisce allineamenti più locali
    size_penalty = ((i1 - i0 - 1) + (j1 - j0 - 1)) * 0.04

    # Bonus lieve se entrambe le estremità sembrano "chiudere" una frase
    punct_bonus = 0.0
    if re.search(r"[.!?…]$", src_text) and re.search(r"[.!?…]$", tgt_text):
        punct_bonus += 0.03

    # Bonus per numeri/token capitalizzati condivisi come ancore deboli
    anchor_bonus = _anchor_bonus(src_text, tgt_text)

    return sim - len_penalty - size_penalty + punct_bonus + anchor_bonus


def _anchor_bonus(it_text: str, en_text: str) -> float:
    bonus = 0.0

    nums_it = set(re.findall(r"\d+", it_text))
    nums_en = set(re.findall(r"\d+", en_text))
    if nums_it and nums_en and nums_it & nums_en:
        bonus += 0.05

    caps_it = set(re.findall(r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]+\b", it_text))
    caps_en = set(re.findall(r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]+\b", en_text))
    if caps_it and caps_en:
        overlap = len(caps_it & caps_en)
        bonus += min(overlap * 0.02, 0.06)

    return bonus


def _distribute_target_text_over_cues(
    source_texts: List[str],
    target_text: str,
) -> List[str]:
    """
    Distribuisce il testo target su N cue sorgente.
    Strategia:
    1. split in micro-unità testuali
    2. allocazione proporzionale alla lunghezza dei cue sorgente
    3. merge finale per ottenere esattamente N pezzi
    """
    n = len(source_texts)
    if n == 1:
        return [target_text.strip()]

    units = _split_text_for_distribution(target_text)
    if len(units) == 1:
        return _split_text_proportionally(target_text, source_texts)

    weights = np.array([max(len(t.strip()), 1) for t in source_texts], dtype=float)
    weights = weights / weights.sum()

    unit_lens = np.array([max(len(u), 1) for u in units], dtype=float)
    total_len = unit_lens.sum()
    targets = weights * total_len

    buckets = [[] for _ in range(n)]
    bucket_lens = np.zeros(n, dtype=float)

    current_idx = 0
    for unit, ulen in zip(units, unit_lens):
        # scegli il bucket corrente o il successivo in modo monotono
        while current_idx < n - 1 and bucket_lens[current_idx] >= targets[current_idx]:
            current_idx += 1
        buckets[current_idx].append(unit)
        bucket_lens[current_idx] += ulen

    pieces = [" ".join(bucket).strip() for bucket in buckets]

    # Se ci sono buchi, prova a riequilibrare
    pieces = _rebalance_empty_pieces(pieces)

    if any(not p.strip() for p in pieces):
        pieces = _split_text_proportionally(target_text, source_texts)

    return pieces


def _split_text_for_distribution(text: str) -> List[str]:
    """
    Spezza un gruppo testuale in micro-unità più facili da distribuire sui cue.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[,;:.!?，；：。！？؟۔])\s+", text)
    refined = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(part) > 70 or len(part.split()) > 12:
            sub = re.split(
                r"(?<=[,;:，；：])\s+|\s+[/-]\s+",
                part,
            )
            refined.extend([x.strip() for x in sub if x.strip()])
        else:
            refined.append(part)

    # Merge segmenti troppo piccoli
    merged = []
    for part in refined:
        if merged and len(part.split()) <= 2:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    return merged if merged else [text]


def _split_text_proportionally(text: str, source_texts: List[str]) -> List[str]:
    """
    Fallback: spezza il testo in N pezzi cercando confini naturali vicini a target proporzionali.
    """
    n = len(source_texts)
    if n == 1:
        return [text.strip()]

    words = text.split()
    if len(words) <= n:
        out = []
        for i in range(n):
            out.append(words[i] if i < len(words) else "")
        return _rebalance_empty_pieces(out)

    weights = np.array([max(len(t.strip()), 1) for t in source_texts], dtype=float)
    weights = weights / weights.sum()

    cum_targets = np.cumsum(weights)[:-1]
    total_words = len(words)
    split_positions = [int(round(x * total_words)) for x in cum_targets]

    # Aggiusta gli split verso punteggiatura vicina
    adjusted = []
    for pos in split_positions:
        best = pos
        best_score = 10**9
        lo = max(1, pos - 4)
        hi = min(total_words - 1, pos + 4)

        for candidate in range(lo, hi + 1):
            prev_word = words[candidate - 1]
            score = abs(candidate - pos)

            if re.search(r"[,;:.!?，；：。！？؟۔]$", prev_word):
                score -= 1.5

            if score < best_score:
                best_score = score
                best = candidate

        adjusted.append(best)

    # Costruzione pezzi
    pieces = []
    start = 0
    for pos in adjusted:
        pieces.append(" ".join(words[start:pos]).strip())
        start = pos
    pieces.append(" ".join(words[start:]).strip())

    pieces = _fix_piece_count(pieces, n)
    pieces = _rebalance_empty_pieces(pieces)
    return pieces


def _fix_piece_count(pieces: List[str], n: int) -> List[str]:
    if len(pieces) == n:
        return pieces

    pieces = pieces[:]
    while len(pieces) < n:
        # split del pezzo più lungo
        idx = max(range(len(pieces)), key=lambda i: len(pieces[i]))
        words = pieces[idx].split()
        if len(words) < 2:
            pieces.insert(idx + 1, "")
            continue
        mid = len(words) // 2
        left = " ".join(words[:mid]).strip()
        right = " ".join(words[mid:]).strip()
        pieces[idx] = left
        pieces.insert(idx + 1, right)

    while len(pieces) > n:
        # merge dei due più corti adiacenti
        best_i = 0
        best_len = 10**9
        for i in range(len(pieces) - 1):
            cand_len = len(pieces[i]) + len(pieces[i + 1])
            if cand_len < best_len:
                best_len = cand_len
                best_i = i
        pieces[best_i] = (pieces[best_i] + " " + pieces[best_i + 1]).strip()
        del pieces[best_i + 1]

    return pieces


def _rebalance_empty_pieces(pieces: List[str]) -> List[str]:
    pieces = pieces[:]
    for i in range(len(pieces)):
        if pieces[i].strip():
            continue

        # prova a "rubare" dalla sinistra
        if i > 0 and len(pieces[i - 1].split()) >= 4:
            words = pieces[i - 1].split()
            take = max(1, len(words) // 3)
            pieces[i - 1] = " ".join(words[:-take]).strip()
            pieces[i] = " ".join(words[-take:]).strip()
            continue

        # prova a "rubare" dalla destra
        if i < len(pieces) - 1 and len(pieces[i + 1].split()) >= 4:
            words = pieces[i + 1].split()
            take = max(1, len(words) // 3)
            pieces[i] = " ".join(words[:take]).strip()
            pieces[i + 1] = " ".join(words[take:]).strip()

    return pieces


def _wrap_subtitle_text(text: str, max_chars_per_line: int = 42) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars_per_line:
        return text

    words = text.split()
    lines = []
    current = []

    for word in words:
        candidate = " ".join(current + [word]).strip()
        if len(candidate) <= max_chars_per_line:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    # Preferisci al massimo 2 righe, se possibile
    if len(lines) <= 2:
        return "\n".join(lines)

    # Ricompattazione a 2 righe bilanciate
    all_words = text.split()
    best_split = 1
    best_diff = 10**9

    for i in range(1, len(all_words)):
        left = " ".join(all_words[:i])
        right = " ".join(all_words[i:])
        diff = abs(len(left) - len(right))
        overflow = max(0, len(left) - max_chars_per_line) + max(0, len(right) - max_chars_per_line)
        score = diff + overflow * 10
        if score < best_diff:
            best_diff = score
            best_split = i

    left = " ".join(all_words[:best_split]).strip()
    right = " ".join(all_words[best_split:]).strip()
    return left + "\n" + right


def _fill_empty_slots(texts: List[str]) -> List[str]:
    """
    Evita cue completamente vuoti dopo la distribuzione.
    """
    texts = texts[:]

    for i in range(len(texts)):
        if texts[i].strip():
            continue

        prev_text = texts[i - 1].strip() if i > 0 else ""
        next_text = texts[i + 1].strip() if i < len(texts) - 1 else ""

        if prev_text and next_text:
            texts[i] = next_text
        elif prev_text:
            texts[i] = prev_text
        elif next_text:
            texts[i] = next_text
        else:
            texts[i] = "..."

    return texts


if __name__ == "__main__":
    model = _load_default_model()
    target_text = open("translation.txt", "r", encoding="utf-8").read()

    aligned_subs = align_text_to_srt_advanced(
        srt_path="input.srt",
        target_text=target_text,
        output_path="output.srt",
        model=model,
    )
