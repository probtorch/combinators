{ use-nix-science ? true }:
let
  sources = import ./nix/sources.nix;
  mach-nix = import sources.mach-nix {
    python = "python38";
    # optionally update pypi data revision from https://github.com/DavHau/pypi-deps-db
    pypiDataRev = "1d6a42de350651280e66c7a20b8166301515f2fd";  # 01-08-2020
    pypiDataSha256 = "01yccf8z15m4mgf971bv670xx7l1p3i8khjz4c8662nrjirslm43";
  };
  hydra-core = (mach-nix.buildPythonPackage {
    src = sources.hydra;
    requirements = builtins.readFile "${sources.hydra}/requirements/requirements.txt";
    providers.antlr4-python3-runtime = "nixpkgs";
  }).overrideAttrs (old:{
    postPatch = ''
      substituteInPlace build_helpers/build_helpers.py --replace "java" "${mach-nix.nixpkgs.jre}/bin/java"
    '';
  });
  inherit (mach-nix.nixpkgs) lib;
in
mach-nix.mkPython {
  requirements = lib.strings.concatStringsSep "\n" [
    (builtins.readFile ./requirements.txt)
  ];

  packagesExtra = [
    ./probtorch # local branch starting from nvi-dev on probtorch
    # "https://github.com/probtorch/probtorch/tarball/1a9af26"
  ];

          providers = {
            _default = "wheel";
            torch = "wheel";
            tqdm = "wheel";
            numpy = "wheel";
            scipy = "wheel";
            torchvision = "wheel";
            protobuf = "wheel"; # ...otherwise infinite recursion.
          };
}
