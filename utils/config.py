import argparse, json, pathlib, sys
from typing import Any, Dict, Iterable

class Config(dict):
    def __init__(self, json_path, argv = None):
        super().__init__()
        self._path = pathlib.Path(json_path)
        self.update(self._load_json())
        overrides = self._parse_overrides(argv)
        self.update(overrides)

    def _load_json(self):
        with self._path.open("r") as f:
            return json.load(f)

    def _parse_overrides(self, argv):
        parser = argparse.ArgumentParser(
            prog=sys.argv[0],
            description="Flags here override values in config.json",
        )

        for key, default_val in self.items():
            if not key.isidentifier():
                continue  # skip odd keys like "drop-out"
            t = self._bool_conv if isinstance(default_val, bool) else type(default_val)
            parser.add_argument(f"--{key}", type=t, default=None)

        args = parser.parse_args(list(argv) if argv is not None else None)
        return {k: v for k, v in vars(args).items() if v is not None}

    @staticmethod
    def _bool_conv(s):
        if isinstance(s, bool):
            return s
        return str(s).lower() in {"1", "true", "yes", "on"}

    def save(self, path = None):
        target = pathlib.Path(path or self._path)
        target.write_text(json.dumps(self, indent=2))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(*e.args) from None

    __setattr__ = dict.__setitem__