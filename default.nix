{ use_jupyter ? true, ... }:
let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs {
    overlays = [
      (_: _: { inherit sources; })
      (_: super: {
        mach-nix = import sources.mach-nix {
          pkgs = super;
          python = "python37";
          # optionally update pypi data revision from https://github.com/DavHau/pypi-deps-db
          pypiDataRev = "f3122fc";
          pypiDataSha256 = "1rzj1v9868gk6p1q73bxybbxvx0k53cprw0hj2z16jlpqjcqy9gz";
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
    json5 = "wheel";
  };
}
