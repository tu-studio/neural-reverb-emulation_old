# TODO

## Preprocessing

- Fix clipping
    - If wet audio has values over 1, normalize wet and dry accordingly --> done 

- Understand and apply spectral loss --> done
    - plot spectrums of x and y --> to check
    - plot distance spectrums --> to check

- Fix batches name in AudioDataset use couples or similar --> done
- Think about mono and stereo aglomeration in preprocessing
    - Do not throw away the mono samples --> done

- Understand and apply pqmf
    - do decomposition
    - listen to audio after decomposition
    - Compare the shapes
    - do reconstruction

- Do the block results list directly in the encoder achitecture

- Normalize clipping by pair not individually !

- Do spectral loss plots locally rather than on tensorboard.

- Also plot signals of the PQMF. 