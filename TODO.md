# TODO

## Preprocessing

- Fix clipping
    - If wet audio has values over 1, normalize wet and dry accordingly

- Understand and apply spectral loss
    - plot spectrums of x and y
    - plot distance spectrums

- Fix batches name in AudioDataset use couples or similar
- Think about mono and stereo aglomeration in preprocessing
    - Do not throw away the mono samples

- Understand and apply pqmf
    - do decomposition
    - listen to audio after decomposition
    - Compare the shapes
    - do reconstruction
