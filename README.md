# text2tashkeel

text2tashkeel restores the missing vowel marks (tashkeel) in Arabic text. It is
not one model but a model picker: a single small API over interchangeable
diacritization models, all running on `onnxruntime`. It needs no PyTorch and
no API keys, and it works offline by default. Pick the model that fits your
accuracy, speed, or size budget. The only runtime dependencies are `numpy`
and `onnxruntime`.

```python
from text2tashkeel import Diacritizer
Diacritizer().diacritize("بسم الله الرحمن الرحيم")              # default model - 2.03% DER
Diacritizer("rawi-v2-int8").diacritize("بسم الله الرحمن الرحيم")  # lean single model
```

Most diacritizers add only the short-vowel marks, and only to text that is
already spelled correctly. The default rawi models also restore the hamza
(ء) and the silent dagger-alef, so they fix real, inconsistently spelled
input, such as a bare `ا` typed for `أ`, not just clean text. See
[what makes rawi different](docs/11-what-makes-rawi-different.md#111-a-wider-task-rawi-restores-hamza-and-the-dagger-alef-not-just-vowels)
for the reasons and the method.

## Install

```bash
pip install --pre text2tashkeel
```

The default model and `rawi-v2-int8` ship in the wheel and run offline. See
[install, models and CLI](docs/12-install-models-cli.md) for the optional
`[hf]` extra, the model list, custom models and the command line.

## Documentation

[`docs/index.md`](docs/index.md) links every page. The benchmark command and
the scores of every model are in [`benchmarks/README.md`](benchmarks/README.md#reproduce).

## Related projects

- [`phoonnx`](https://github.com/TigreGotico/phoonnx) — offline text-to-speech, which reads diacritized text for pronunciation.

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and
[`docs/07-credits-and-license.md`](docs/07-credits-and-license.md) for the
license of each bundled model.
