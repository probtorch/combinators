let
  defaults = import ./nix/defaults.nix;
in
{ compiler ? defaults.compiler

, pkgs ? defaults.pkgs

}:

pkgs.haskell-nix.project {
  # 'cleanGit' cleans a source directory based on the files known by git
  src = pkgs.haskell-nix.haskellLib.cleanGit {
    name = "proto-combinators";
    src = ./.;
  };
  # For `cabal.project` based projects specify the GHC version to use.
  compiler-nix-name = compiler; # Not used for `stack.yaml` based projects.
  projectFileName = "cabal.project"; # Not used for `stack.yaml` based projects.
}
