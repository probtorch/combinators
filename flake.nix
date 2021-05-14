{
  description = "Inference Combinators";

  inputs = rec {
    flake-utils.url = "github:numtide/flake-utils";
    mach-nix.url = "github:DavHau/mach-nix";
    probtorch.url = "github:probtorch/probtorch/combinators-dev";
    probtorch.flake = false;
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix, probtorch}:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system:
      let
        inherit (mach-nix.lib.${system}) mkPython;
        with-jupyter = false;
        dev-profile = false;
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (pkgs) lib mkShell;
      in rec {
        # FIXME: make this buildPythonApplication and move this to devShell only
        packages.combinatorsPy = mkPython { # Package {
          #src = ./.;
          requirements = lib.strings.concatStringsSep "\n" ([
            (builtins.readFile ./requirements.txt)
          ] ++ (lib.optionals with-jupyter ["jupyterlab"])
            ++ (lib.optionals dev-profile ["filprofiler"]))
            ;
          packagesExtra = [
            probtorch
          ];

          providers = {
            _default = "nixpkgs,wheel";
            torch = "wheel";
            tqdm = "wheel";
            torchvision = "wheel";
            protobuf = "wheel"; # ...otherwise infinite recursion.
          };
        };
        defaultPackage = packages.combinatorsPy;
        devShell = mkShell {
          buildInputs = [
            packages.combinatorsPy
            pkgs.gnused
          ];
          # applying patch: https://github.com/microsoft/pyright/issues/565
          shellHook = ''
            export PYTHONPATH="$(git rev-parse --show-toplevel):$PYTHONPATH"
            VENV_NAME="$(echo ${packages.combinatorsPy} | sed -E 's/\/nix\/store\/(.*)-env/\1/')-env"

            if [ -f pyrightconfig.json ]; then
              DETECTED=$(\grep -oP "[^\"]\w+-python3-[23].[0-9].[0-9](-env)?" pyrightconfig.json)
              if [ "$DETECTED" != "$VENV_NAME" ]; then
                cp pyrightconfig.json{,.bk}
                echo "detected venv $DETECTED in pyrightconfig.json, patching venv with $VENV_NAME"
                sed -E -i "s/(\"venv\": \")\w+-python3-[23].[0-9].[0-9](-env)?/\1$VENV_NAME/" pyrightconfig.json
              else
                echo "detected matching venv $DETECTED in pyrightconfig.json"
              fi
            else
              echo "did not detect pyrightconfig.json, generating new venv with $VENV_NAME"
              cat > pyrightconfig.json << EOF
            {
              "pythonVersion": "3.8",
              "pythonPlatform": "Linux",
              "venv": "$VENV_NAME",
              "venvPath": "/nix/store"
            }
            EOF
            fi
          '';
        };
      });
}
