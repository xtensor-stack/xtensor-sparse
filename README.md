# xtensor-sparse

Multi-dimensional sparse arrays based on [xtensor](https://github.com/xtensor-stack/xtensor)

## Introduction

** xtensor-sparse is in development **

## Installation

For now, only installation from sources is available. Be sure to 
install the dependencies before trying to install `xtensor-sparse`.

`xtensor-sparse` is a header-only library. you can directly install it
from the sources:

```bash
cmake -DCMAKE_INSTALL_PREFIX=your_install_prefix
make install
```

To build and run the test suite:

```bash
cmake -DDOWNLOAD_GTEST=ON -DCMAKE_INSTALL_PREFIX=your_install_prefix
make xtest
```

## Dependencies

No dependencies yet.

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the
[LICENSE](LICENSE) file for details.
