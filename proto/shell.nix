let
  defaults = import ./nix/defaults.nix;
in
{ compiler ? defaults.compiler

, sources ? defaults.sources

, haskell-nix ? defaults.haskell-nix

, pkgs ? defaults.pkgs

}:
let
  development = import ./nix/development.nix {inherit compiler sources haskell-nix pkgs;};
  hsPkgs = import ./default.nix { inherit compiler; };
in
hsPkgs.shellFor {
  # Include only the *local* packages of your project.
  packages = ps: [ps.proto-combinators];

  # Builds a Hoogle documentation index of all dependencies,
  # and provides a "hoogle" command to search the index.
  withHoogle = false;

  # Some you may need to get some other way.
  # buildInputs = with pkgs.haskellPackages;
  buildInputs = with development; [gdb lldb hlint ghcid ghcide];

  # probably unnessecary
  # ==================================
  # shellHook = ''
  #   export LD_LIBRARY_PATH=${lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH
  #   export LANG=en_US.UTF-8
  # '';
  # LOCALE_ARCHIVE =
  #   if stdenv.isLinux
  #   then "${glibcLocales}/lib/locale/locale-archive"
  #   else "";

  # Prevents cabal from choosing alternate plans, so that
  # *all* dependencies are provided by Nix.
  exactDeps = true;
}
