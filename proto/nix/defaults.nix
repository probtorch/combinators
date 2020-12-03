rec {
  # standardize compiler across default.nix, release.nix, shell.nix
  compiler = "ghc865";

  # pin with niv
  sources = import ./sources.nix;

  # haskell.nix uses niv under-the-hood
  haskell-nix = (import sources.haskell-nix {sourcesOverride = sources;});

  # let's use as much shiny stuff as we can
  nixpkgs-src = haskell-nix.sources.nixpkgs-2003;

  # import nixpkgs with overlays
  pkgs = import nixpkgs-src (haskell-nix.nixpkgsArgs // {
    overlays = haskell-nix.overlays ++ [
      (import ./overlays/llvm9.nix)
    ];
  });
}
