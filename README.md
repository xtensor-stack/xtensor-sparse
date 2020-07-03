# xtensor-sparse

[![Travis](https://travis-ci.org/xtensor-stack/xtensor-sparse.svg?branch=master)](https://travis-ci.org/xtensor-stack/xtensor)
[![Appveyor](https://ci.appveyor.com/api/projects/status/aubhh8lvrihw2odx/branch/master?svg=true)](https://ci.appveyor.com/project/xtensor-stack/xtensor-sparse/branch/master)
[![Azure](https://dev.azure.com/xtensor-stack/xtensor-stack/_apis/build/status/xtensor-stack.xtensor-sparse?branchName=master)](https://dev.azure.com/xtensor-stack/xtensor-stack/_build/latest?definitionId=4&branchName=master)
[![Join the Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/QuantStack/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Multi-dimensional sparse arrays based on [xtensor](https://github.com/xtensor-stack/xtensor)

## Introduction

**xtensor-sparse is in development**

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

`xtensor-sparse` depends on the [xtensor](https://github.com/xtensor-stack/xtensor) library:

| `xtensor-sparse` |    `xtensor`    |
|------------------|-----------------|
|    master        | >=0.21.4, <0.22 |


## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the
[LICENSE](LICENSE) file for details.
