let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs {
    overlays = [
      (_: _: { inherit sources; })
      (_: super: {
        mach-nix = import sources.mach-nix {
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

pkgs.mach-nix.mkPython {
  requirements = (builtins.readFile ./requirements.txt)  + "\njupyterlab";

  packagesExtra = [
    ./probtorch # local branch starting from nvi-dev on probtorch
    # "https://github.com/probtorch/probtorch/tarball/1a9af26"
  ];

  providers = {
    # disallow wheels by default
    _default = "nixpkgs,sdist";
    # allow wheels only for torch
    torch = "wheel";
    hydra-core = "wheel";
    jupyterlab = "wheel";
    torchvision = "wheel";
    Sphinx = "wheel";
    json5 = "wheel";
  };
}
