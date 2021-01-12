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
    "hydra-core"
    "hydra_colorlog"
    "jupyterlab"
  ];

  packagesExtra = [
    ./probtorch # local branch starting from nvi-dev on probtorch
    # "https://github.com/probtorch/probtorch/tarball/1a9af26"
    hydra-core
    mach-nix.nixpkgs.qt512.full
  ];
  providers = let fast = false; in {
    # disallow wheels by default
    _default = "wheel,nixpkgs,sdist";
  } // {
    torch = "wheel"; # allow wheels only for torch
    torchvision = "wheel";
    Sphinx = "wheel";
    protobuf = "wheel"; # otherwise infinite recursion..
    jupyterlab = "wheel";
    anyio = "wheel";
    mypy = "wheel"; # must be up-to-date
  } // {
    antlr4-python3-runtime = "nixpkgs";
    tornado = "nixpkgs";
    tensorboard = "nixpkgs";
    jinja2 = "nixpkgs";
    six = "nixpkgs";
    requests = "nixpkgs";
    urllib3 = "nixpkgs";
    pyzmq = "nixpkgs";
    prometheus-client = "nixpkgs";
    packaging = "nixpkgs";
    pygments = "nixpkgs";
    traitlets = "nixpkgs";
    python-dateutil = "nixpkgs";
    typing-extensions ="nixpkgs";
    mypy-extensions ="nixpkgs";
    typed-ast ="nixpkgs";
    pillow = "nixpkgs";
    tqdm = "nixpkgs";
    typeguard = "nixpkgs";

    joblib = "nixpkgs";
    threadpoolctl = "nixpkgs";

    matplotlib = "wheel";
      kiwisolver = "nixpkgs";
      certifi = "nixpkgs";
      cycler = "nixpkgs";
      pyparsing = "nixpkgs";

    pytz = "nixpkgs";
    pytest-mock = "nixpkgs";
    pytest = "nixpkgs";
      iniconfig = "nixpkgs";
      pluggy = "nixpkgs";
      py = "nixpkgs";
      attrs = "nixpkgs";
      toml = "nixpkgs";

  } // (lib.optionalAttrs (use-nix-science) {
    numpy = "nixpkgs";
    scipy = "nixpkgs";
    scikit-learn = "nixpkgs";
    scikit-image = "nixpkgs";
    seaborn = "nixpkgs";
    pandas = "nixpkgs";
    matplotlib = "nixpkgs";

#     tensorboard = "nixpkgs";
#       pywavelets = "nixpkgs";
#       tifffile = "nixpkgs";
#       imageio = "nixpkgs";
#       networkx = "nixpkgs";
#       decorator = "nixpkgs";
#       werkzeug = "nixpkgs";
       six = "nixpkgs";
#       requests = "nixpkgs";
#       abs-py = "nixpkgs";
#       markdown = "nixpkgs";
#       google-auth = "nixpkgs";
#       cachetools = "nixpkgs";
#       rsa = "nixpkgs";
#       pyasn1 = "nixpkgs";
#       pyasn1-modules = "nixpkgs";
#       setuptools = "nixpkgs";
#       google-auth-oauthlib = "nixpkgs";
#       requests-oauthlib = "nixpkgs";
#       oauthlib = "nixpkgs";
#       grpcio = "nixpkgs";
  });
}
