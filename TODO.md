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
    - do decomposition  --> done
    - listen to audio after decomposition --> DOne 
    - Compare the shapes --> Done
    - do reconstruction --> Done

- Do the block results list directly in the encoder achitecture --> Done

- Normalize clipping by pair not individually !  --> Done

- Do spectral loss plots locally rather than on tensorboard. --> Done

- Also plot signals of the PQMF. time domain/ spectral/ --> Done

- plot both in time and then in fft. --> Done