
## `chmp.label`

Helpers to label files.

For any data file, i.e., `image.png`, the corresponding label files has the
same base, but the extension `.label`, i.e., `image.label`.

IPython widgtes for annotating data sets:

- [ImageAnnotator](#imageannotator)
- [AudioAnnotator](#audioannotator)
- [Annotator](#annotator)
Distributed as part of `https://github.com/chmp/misc-exp` under the MIT
license, (c) 2017 Christopher Prohm.



### `chmp.label.annotate`
`chmp.label.annotate()`

Annotate data inside the ipython notebook.

This function constructs an IPython widget and displays it to the user. The
returned list will be filled with the labels as the user interacts with the
widget.

#### Parameters

* **items** (*Sequence[Any]*):
  the collection of items to label. The interpretation of the items
  depends on the [Annotator](#annotator) chosen by the `cls` argument.  For
  image and audio, each item should be a filename. For text, the items
  should be the lines of the text. For custom `display_value` each item
  is passed as is.
* **classes** (*Sequence[str]*):
  the classes to choose from.
* **history_length** (*int*):
  the number of previously labeled items to show for changes in case of
  errors.
* **display_value** (*Optional[Callable[str,Any]]*):
  if given, a callable that accepts an item and returns a HTML
  representation to show to the user.
* **cls** (*Optional[Union[str,class]]*):
  the type of annotator to use. Can be either a class directly or one
  of `'image'`, `'audio'`. If not given, but `display_value` is
  given, it will be used to display the result to the user. If neither
  one is given, the `repr` will be shown to the user.
* **kwargs** (*Optional[Mapping[Str,Any]]*):
  if given, additional keyword arguments passed on constucting the
  annotator object. Note, `history_length` and `display_value`
  are set by the parameters of this function.

#### Returns

{body}



### `chmp.label.ImageAnnotator`
`chmp.label.ImageAnnotator(classes, history_length=10)`

IPython widget to annotate image files.

The widget expects a list of filenames.



### `chmp.label.AudioAnnotator`
`chmp.label.AudioAnnotator(classes, history_length=10)`

IPython widget to annotate audio files.

The widget expects a list of filenames.



### `chmp.label.Annotator`
`chmp.label.Annotator(classes, history_length=10)`

IPython widget to quickly annotate data sets.

