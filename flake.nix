{
  description = "Flake for portmanteau - portfolio manager to end all uncertainties";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    mach-nix = {
      url = github:DavHau/mach-nix/1ec92303acd142aa1a3b60bb97745544cf049312; # v3.1.1 release tag
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
      flake = false;
    };
  };
  outputs = { self, nixpkgs, mach-nix, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = (import nixpkgs { inherit system; }).pkgs;
        mach-nix-utils = import mach-nix {
          inherit pkgs;
          python = "python3";
        };
        requirements = (builtins.readFile ./requirements.txt);
        packagesExtra = [
          # ./probtorch # local branch starting from nvi-dev on probtorch
        #  "https://github.com/probtorch/probtorch/tarball/1a9af26"
        ];
        providers = {
          # disallow wheels by default
          _default = "nixpkgs,sdist";
          # allow wheels only for torch
          torch = "wheel";
          hydra-core = "wheel";
          torchvision = "wheel";
          Sphinx = "wheel";
          json5 = "wheel";
        };
      in
        rec {
          devShell = pkgs.mkShell {
            buildInputs = [
              (mach-nix-utils.mkPythonShell {
                inherit packagesExtra providers;
                requirements = requirements + "\n" + pkgs.lib.strings.concatStringsSep "\n" [
                  "jupyterlab"
                  "mypy"
                ];
              })
            ];
          };

        packages.combinators = mach-nix-utils.mkPython {
          inherit requirements packagesExtra providers;
        };

        defaultPackage = packages.combinators;
      }
  );
}
