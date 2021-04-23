{ ... }:
let
  sources = import ./nix/sources.nix;
  mach-nix = import sources.mach-nix {
    python = "python38";
    # optionally update pypi data revision from https://github.com/DavHau/pypi-deps-db
    pypiDataRev = "1d6a42de350651280e66c7a20b8166301515f2fd";  # 01-08-2020
    pypiDataSha256 = "01yccf8z15m4mgf971bv670xx7l1p3i8khjz4c8662nrjirslm43";
  };
  inherit (mach-nix.nixpkgs) lib;
in
mach-nix.buildPythonPackage {

  requirements = lib.strings.concatStringsSep "\n" [
    (builtins.readFile ./requirements.txt)
    # for development we also need:
    "jupyterlab"
  ];

  packagesExtra = [
    # branch starting from nvi-dev on probtorch
    #"https://github.com/probtorch/probtorch/tarball/1a9af26"
    ../probtorch
    mach-nix.nixpkgs.qt512.full # for nixos matplotlib support
  ];

  # enumerate this manually -- originally done to build a problematic hydra-core
  # with a java dependency, but now for docutmentat.
  providers = {
    _default = "nixpkgs,wheel";
  } // {
    # wheels will have the most up to date versions but are also black-box.
    # Ideally we keep the derivations in nixpkgs, but the following need to be
    # "bleeding edge."
    torch = "wheel";
    torchvision = "wheel";
    Sphinx = "wheel";
    protobuf = "wheel"; # ...otherwise infinite recursion.
    jupyterlab = "wheel";
    anyio = "wheel";
    mypy = "wheel";
    jedi = "wheel";
  };
}
