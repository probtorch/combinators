let
  defaults = import ./nix/defaults.nix;
in
{ compiler ? defaults.compiler
}:
let
  project-binary = "proto-combinators";
  hsPkgs = (import ./default.nix { inherit compiler; } );
in
{
  "${project-binary}" = hsPkgs."${project-binary}".components.exes."${project-binary}";
}
