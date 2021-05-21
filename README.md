# Combinators

Code for *Learning Proposals for Probabilistic Programs with Inference Combinators* in `combinators/inference.py`

To cite, use:

``` bibtex
@misc{stites-zimmerman2021LearningProposals,
      title={Learning Proposals for Probabilistic Programs with Inference Combinators}, 
      author={Sam Stites and Heiko Zimmerman and Hao Wu and Eli Sennesh and Jan-Willem van de Meent},
      shortauthor={Stites and Zimmermann},
      year={2021},
      eprint={2103.00668},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

### Development

#### Nix and python
To get access to the binary cache, use: `cachix use combinators`.

To access the development shell, a flake is provided which can be accessed via
`nix develop`. Experimentally [https://github.com/numtide/devshell](`devshell`)
is used, but this is used from within the flake and the devshell pip package is
untested.
