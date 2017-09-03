# Labelling helpers

## `chmp.label`

Helpers to label files.

For any data file, i.e., `image.png`, the corresponding label files has the
same base, but the extension `.label`, i.e., `image.label`.

IPython widgtes for annotating data sets:

- [TextAnnotator](#textannotator)
- [ImageAnnotator](#imageannotator)
- [AudioAnnotator](#audioannotator)
- [Annotator](#annotator)


### `chmp.label.TextAnnotator`
`chmp.label.TextAnnotator(classes, history_length, history_length=10, context_size=1)`

IPython widget to annotate a text document line by line.

Usage:

```
classes = [
    {'label': 'Skip', 'style': 'primary'},
    {'label': 'Pose', 'style': 'success'},
    {'label': 'Background', 'style': 'danger'},
]
annotator = TextAnnotator(classes)
annotator.annotate('my-key', lines)

display(annotator)
```

To limit the number or change the order of the lines to display, pass the
order argument to `annotate`:

```
annotator.annotate('my-key', lines, order=[10, 5, 6])
```



### `chmp.label.ImageAnnotator`
`chmp.label.ImageAnnotator(classes, history_length, history_length=10, context_size=1)`

IPython widget to annotate image files.



### `chmp.label.AudioAnnotator`
`chmp.label.AudioAnnotator(classes, history_length, history_length=10, context_size=1)`

IPython widget to annotate audio files.



### `chmp.label.Annotator`
`chmp.label.Annotator(classes, history_length, history_length=10, context_size=1)`

IPython widget to quickly annotate data sets.

