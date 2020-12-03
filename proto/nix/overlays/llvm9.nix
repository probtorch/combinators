final: prev: rec {
  llvm_9 = (prev.llvm_9.override { debugVersion = true; }).overrideAttrs(_: { doCheck = false; });
  llvm-config = llvm_9;
  # These do not seem to be used:
  # ================================== #
  # haskell-nix = prev.haskell-nix // {
  #   "${compiler}" = prev.haskell.packages."${compiler}".override {
  #     overrides = hfinal: hprev: {
  #       llvm-hs = hprev.callPackage (import "${sources.llvm-hs}/llvm-hs") {llvm-config = llvm_9;};
  #       llvm-hs-pure = hprev.callPackage (import "${sources.llvm-hs}/llvm-hs-pure") {};
  #     };
  #   };
  # };
}
