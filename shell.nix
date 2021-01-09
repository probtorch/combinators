{ use_jupyter ? false }:
let
  mypython = import ./.;
  sources = import ./nix/sources.nix;
  host_pkgs = (import sources.nixpkgs {});
in
with host_pkgs;
mkShell {
  buildInputs = [
    mypython
    python-language-server
    black
  ];
}
