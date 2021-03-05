let
  pyEnv = import ./. {};
  sources = import ./nix/sources.nix;
  inherit (import sources.nixpkgs {}) mkShell gnused;
in
mkShell rec {
  buildInputs = [
    pyEnv
    gnused
  ];
  shellHook = ''
    cp pyrightconfig.json{,.bk}
    VENV_NAME="$(echo ${pyEnv} | sed -E 's/\/nix\/store\/(.*)-env/\1/')"
    sed -E -i "s/(\"venv\": \")\w+-python3-[23].[0-9].[0-9]-env/\1$VENV_NAME/" pyrightconfig.json
  '';
}
