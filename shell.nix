{ use_jupyter ? true, start_jupyter ? false }:
let
  mypython = import ./. { inherit use_jupyter; };
  sources = import ./nix/sources.nix;
  host_pkgs = (import sources.nixpkgs {});
in
host_pkgs.mkShell {
  buildInputs = [
    mypython
    host_pkgs.mypy
  ];
  shellHook = host_pkgs.lib.optionalString (use_jupyter && start_jupyter) ''
    jupyter lab --notebook-dir=$PWD
  '';
}
