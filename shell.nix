{ use_jupyter ? false }:
let
  mypython = import ./.;
  sources = import ./nix/sources.nix;
  host_pkgs = (import sources.nixpkgs {});
  jupyter = import sources.jupyterWith {};

  iPython = jupyter.kernels.iPythonWith {
    name = "mach-nix-jupyter";
    python3 = mypython.python;
    packages = p: (mypython.python.pkgs.selectPkgs p) ++ [
      p.mypy
      p.pytest
      host_pkgs.python-language-server
    ];
  };

  jupyterEnvironment = jupyter.jupyterlabWith {
    kernels = [ iPython ];
    ## The generated directory for extensions
    directory = ./.jupyterlab;
    extraPackages = p: [p.python38Packages.pytest p.mypy p.python-language-server];
    extraJupyterPath = pkgs:
      "${mypython}/lib/python3.8/site-packages";
  };
in
if use_jupyter
then jupyterEnvironment.env
else with host_pkgs;
mkShell {
  buildInputs = [
    mypython
    python-language-server
    black
  ];
}
