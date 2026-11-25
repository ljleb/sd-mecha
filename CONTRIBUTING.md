# Contributing

Thanks for your interest in improving sd-mecha.
Before contributing, please read the following.

## Propose Changes

Target the `dev` branch to submit PRs:

 ```sh
 git clone -b dev https://github.com/ljleb/sd-mecha.git
 cd sd-mecha
 git checkout -b my-branch
 ```

## Build the docs

From the repository root, install the documentation theme and build the site.

```sh
python -m pip install mkdocs-material
python -m mkdocs build --strict
```

Run `python -m mkdocs serve` to preview changes locally.
