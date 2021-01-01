{ use_jupyter ? true, start_jupyter ? false }:
let
  mypython = import ./.;
  sources = import ./nix/sources.nix;
  host_pkgs = (import sources.nixpkgs {});
in
# ) + (pkgs.lib.strings.concatStringsSep "\n" [
#    (lib.optionalString use_jupyter "jupyterlab")
#  ]);
host_pkgs.mkShell {
  buildInputs = [
    mypython
    # host_pkgs.mypy
    # host_pkgs.python-language-server
  ];
  shellHook = host_pkgs.lib.optionalString (use_jupyter && start_jupyter) ''
    jupyter lab --notebook-dir=$PWD
  '';
}
