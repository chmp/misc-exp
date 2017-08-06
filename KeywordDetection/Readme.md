# kwdetect - simple keyword detection

See [the notebook](kwdetect.ipynb) for an overview.

## Detecting keywords with a trained model

Run the detect command. Note, it will write out snippets to the given path.

    python -m kwdetect detect ./run/model/model.pickle ./data

## Acquiring data

First run the record mode:

    python -m kwdetect record ./data

Then label the recorded data:

    python -m kwdetect label ./data
