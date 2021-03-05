let
  defaults = import ./defaults.nix;
in
{ compiler ? defaults.compiler

, sources ? defaults.sources

, haskell-nix ? defaults.haskell-nix

, pkgs ? defaults.pkgs

}:
let
  unstable = import sources.nixpkgs-unstable {};
  hsPkgs = (import ../default.nix { inherit compiler; } );
in
{
  inherit (pkgs) gdb;
  inherit (unstable) hlint ghcid;
  lldb = pkgs.lldb_9;
  possibly = hsPkgs.possibly.components.all;
  ghcide = (import sources.ghcide-nix {})."ghcide-${compiler}";
}
