## Notes on Data Processing

*TL;DR:*
Each training example is processed as a dictionary of multiple text fields.
Then the `TextProcessor` will process text fields according to its configurations, and return the final tokens.

### TextProcessor Configurations
Here are the configurable options for TextProcessor:
* `fields`: A comma separated list of text fields to process.
* `fields_from_example`: Whether to use the keys of the input example as the
  text fields to process. If this option is set, the `fields` argument will
  be ignored.
* `subfield_separator`: The text separator to use when concatenating subfields
  of a texts.
* `add_eos_token`: Whether to add an EOS token to the end of the text.
* `prepend_text`: The text to prepended to the beginning.

The most important configuration for TextProcessor is the `fields` argument. It
is a comma separated list of text fields to process. Each field consists of one
or more subfields, which are separated by a `+`. Each subfield represent a key
used to extract the text from the input example dictionary. The TextProcessor
joins the extracted subfields of texts with the `subfield_separator` in the text
level and then tokenize the joined text. Finally, the TextProcessor will concatenate
the tokenized text fields at the token level, and add the EOS token if specified.

Other than the keys in the input example, you can also use the following special
keys to indicate a special token for a text field:
* `<|bos|>`: Beginning of sentence token.
* `<|eos|>`: End of sentence token.

For each text field, you can encapulate the subfields with `[]` to specify that
the loss should not be computed for this field. Doing so will make the loss
masks to be 0 for this field. This is useful when you want to use the text field
as a prompt for the model.


### TextProcessor Examples

To give a concrete example, if the input example looks like this:
```python
{
    'question': 'Would ice float on water?',
    'prompt': 'Think step by step.',
    'answer': 'The density of ice is 0.92 g/cm3, and the density of water is 1.0 g/cm3. So ice will float on water.',
}
```
To use the `question` and `prompt` as the input text, and `answer` as the target,
we can specify the following configuration for the `fields` argument:
```
[question+prompt],answer
```

The `question+prompt` indicates that the `question` and `prompt` should be joined
togather with the `subfield_separator`, which is a space by default. The `[]`
indicates that the loss should not be computed for this field. The `answer` field
is then concatenated at the token level, where the loss will be computed.
