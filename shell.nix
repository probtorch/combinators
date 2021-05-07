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
    export PYTHONPATH="$(git rev-parse --show-toplevel):$PYTHONPATH"
    if [ -f pyrightconfig.json ]; then
      cp pyrightconfig.json{,.bk}
      VENV_NAME="$(echo ${pyEnv} | sed -E 's/\/nix\/store\/(.*)-env/\1/')"
      echo "detected pyrightconfig.json, patching venv with $VENV_NAME"
      sed -E -i "s/(\"venv\": \")\w+-python3-[23].[0-9].[0-9]-env/\1$VENV_NAME/" pyrightconfig.json
    fi
  '';
}
