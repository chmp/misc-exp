# kwdetect - simple keyword detection using tensorflow

See [the notebook](kwdetect.ipynb) for an overview.

Libraries used to handle audio data:

- sounddevice: recording and playing audio data
- pysoundfile: saving and reading audio data
- python-speech-features: compute Mel-frequency cepstrum coefficients

## Usage

First install the `chmp-app-kwdetect` package:

    pip install chmp-app-kwdetect

To listen for keywords and classify them with a fitted model, run:

    python -m chmp.app.kwdetect detect --model ./run/model/model.pickle ./data

To collect data without a model, just leave out the model argument:

    python -m chmp.app.kwdetect detect ./data

To label collected data use:

    python -m chmp.app.kwdetect label ./data
