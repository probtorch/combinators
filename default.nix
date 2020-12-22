{ use_jupyter ? true, ... }:
let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs {
    overlays = [
      (_: _: { inherit sources; })
      (_: super: {
        mach-nix = import (builtins.fetchGit {
          url = "https://github.com/DavHau/mach-nix/";
          ref = "refs/tags/3.1.1";
        }) {
          pkgs = super;
          python = "python38";
          # optionally update pypi data revision from https://github.com/DavHau/pypi-deps-db
          pypiDataRev = "4a14f99";  # 12-22-2020
          pypiDataSha256 = "1a3a8r2wrd2rh6yznkjjvcx7cxbx070kwxnpycpjbrvn2i96d4i3";
        };
      })
    ];
  };
in
with pkgs;

mach-nix.mkPython {
  requirements = (builtins.readFile ./requirements.txt) + (lib.optionalString use_jupyter ''
    jupyterlab
  '');

  packagesExtra = [
    # nvi-dev branch on probtorch
    "https://github.com/probtorch/probtorch/tarball/1a9af26"
  ];

  providers = {
    # disallow wheels by default
    _default = "nixpkgs,sdist";
    # allow wheels only for torch
    jupyterlab-server = "wheel";
    torch = "wheel";
    hydra-core = "wheel";
    torchvision = "wheel";
    Sphinx = "wheel";
    json5 = "wheel";
  };
}
