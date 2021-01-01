let
  pynixifyOverlay = self: super: {
    python3 = super.python3.override { inherit packageOverrides; };
    python37 = super.python37.override { inherit packageOverrides; };
    python38 = super.python38.override { inherit packageOverrides; };
  };

  packageOverrides = self: super: {
    antlr4-python3-runtime =
      self.callPackage ./nix/antlr4.nix { };

    hydra-core = self.callPackage ./nix/hydra.nix { };

    # omegaconf = self.callPackage ./packages/omegaconf { };

  };

  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs {
    overlays = [
      (_: _: { inherit sources; })
      pynixifyOverlay
      (self: super: {
        mach-nix = import sources.mach-nix {
          pkgs = self;
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
  requirements = (builtins.readFile ./requirements.txt)
   # + "\njupyterlab\njson5==0.9.5"
   # + "\nhydra-core==1.0.4"
  ;

  packagesExtra = [
    ./probtorch # local branch starting from nvi-dev on probtorch
    # "https://github.com/probtorch/probtorch/tarball/1a9af26"
  ];
  providers = {
    # disallow wheels by default
    _default = "nixpkgs,sdist";
    # allow wheels only for torch
    torch = "wheel";
    # jupyterlab = "wheel";
    torchvision = "wheel";
    Sphinx = "wheel";
    json5 = "wheel";
  };
  # _.hydra-core.buildInputs.add = [ pkgs.jre ];
  # _.hydra-core.patches.add = [
  #   ''substituteInPlace build_helpers/build_helpers.py --replace "java" "${pkgs.jre}/bin/java"''
  # ];
}
