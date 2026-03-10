# Aligner

Servizio Python per riallineare un file SRT a un testo tradotto libero, mantenendo i timestamp originali.

Il progetto espone:

- una libreria locale in [aligner.py](/Users/alessio/aligner/aligner.py) per l'allineamento
- una API FastAPI in [server.py](/Users/alessio/aligner/server.py) protetta con Bearer token

## Come funziona

L'algoritmo:

- carica il file SRT sorgente
- segmenta il testo target con euristiche generiche, non dipendenti dalla lingua
- usa un modello `sentence-transformers` multilingue per confrontare i segmenti
- esegue un allineamento monotono many-to-many
- distribuisce il testo target sui cue originali e produce un nuovo SRT

## Requisiti

- Python 3.12 consigliato
- ambiente virtuale attivo oppure uso esplicito di `env/bin/python`

## Installazione

```bash
python3 -m venv env
env/bin/pip install -r requirements.txt
```

## Configurazione

Copia `.env.example` in `.env` e imposta i valori:

```bash
cp .env.example .env
```

Variabili disponibili:

- `API_BEARER_TOKEN`: token richiesto per autenticarsi verso l'API
- `MODEL_NAME`: modello `sentence-transformers` da usare per l'allineamento

Le stesse variabili possono anche essere passate come normali variabili d'ambiente del processo.

## Avvio del server

```bash
env/bin/uvicorn server:app --reload
```

All'avvio il servizio stampa il device usato per il modello:

```text
Model loaded on device: cpu
```

La selezione del device avviene in questo ordine:

- `mps` se disponibile
- altrimenti `cuda`
- altrimenti `cpu`

## Endpoint

### `GET /health`

Richiede header:

```text
Authorization: Bearer <token>
```

Esempio:

```bash
curl http://127.0.0.1:8000/health \
  -H "Authorization: Bearer il-tuo-token"
```

### `POST /align`

Richiede header:

```text
Authorization: Bearer <token>
```

Parametri `multipart/form-data`:

- `srt_file`: file `.srt` obbligatorio
- `translation_text`: testo tradotto in chiaro, opzionale
- `translation_file`: file di testo UTF-8 con la traduzione, opzionale

Devi fornire uno solo tra `translation_text` e `translation_file`.

Esempio con file di traduzione:

```bash
curl -X POST http://127.0.0.1:8000/align \
  -H "Authorization: Bearer il-tuo-token" \
  -F "srt_file=@input.srt" \
  -F "translation_file=@translation.txt" \
  -o output.srt
```

Esempio con testo inline:

```bash
curl -X POST http://127.0.0.1:8000/align \
  -H "Authorization: Bearer il-tuo-token" \
  -F "srt_file=@input.srt" \
  -F "translation_text=$(cat translation.txt)" \
  -o output.srt
```

## Uso come libreria

Puoi usare direttamente la funzione `align_text_to_srt_advanced`:

```python
from aligner import align_text_to_srt_advanced

target_text = open("translation.txt", "r", encoding="utf-8").read()

align_text_to_srt_advanced(
    srt_path="input.srt",
    target_text=target_text,
    output_path="output.srt",
)
```

## Note

- L'API si aspetta file SRT e testo di traduzione in UTF-8
- Il modello viene caricato all'avvio del server, non a ogni richiesta
- Se il modello non è disponibile localmente, `sentence-transformers` proverà a scaricarlo
