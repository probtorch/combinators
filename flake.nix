{
  description = "Inference Combinators";

  inputs = rec {
    flake-utils.url = "github:numtide/flake-utils";
    devshell.url = "github:numtide/devshell";
    nixpkgs.url = "github:NixOS/nixpkgs/29b0d4d0b600f8f5dd0b86e3362a33d4181938f9";

    mach-nix.url = "github:DavHau/mach-nix";
    mach-nix.inputs.flake-utils.follows = "flake-utils";
    mach-nix.inputs.nixpkgs.follows = "nixpkgs";

    probtorch.url = "github:stites/probtorch/combinators-dev";
    probtorch.flake = false;
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix, probtorch, devshell }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system:
      let
        inherit (mach-nix.lib.${system}) mkPython;
        with-jupyter = false;
        dev-profile = false;
        with-simulator = false;
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ devshell.overlay ];
        };
        inherit (pkgs) lib mkShell;
      in rec {
        # FIXME: make this buildPythonApplication and move this to devShell only
        packages.combinatorsPy = mkPython { # Package {
          #src = ./.;
          requirements = (builtins.readFile ./requirements.txt) + lib.strings.concatStringsSep "\n" ([
          ] ++ (lib.optionals with-simulator ["imageio" "scikit-image"])
            ++ (lib.optionals with-jupyter ["jupyterlab"])
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
        devShell = pkgs.devshell.mkShell {
          packages = with pkgs; [
            packages.combinatorsPy
            gnused
            watchexec
            rsync
          ];
          # applying patch: https://github.com/microsoft/pyright/issues/565
          bash.extra = ''export PYTHONPATH="$(git rev-parse --show-toplevel):$PYTHONPATH"'';
          bash.interactive = ''
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
          '' + (lib.optionalString true ''
            temp_dir=$(mktemp -d)
            cat <<'EOF' >"$temp_dir/.zshrc"
            if [ -e ~/.zshrc ]; then . ~/.zshrc; fi
            if [ -e ~/.config/zsh/.zshrc ]; then . ~/.config/zsh/.zshrc; fi
            menu
            EOF
            ZDOTDIR=$temp_dir zsh -i
          '');
          commands = let
            watchexec = "${pkgs.watchexec}/bin/watchexec";
            mk = category: {name, command}:
              { inherit category name;
                command = command + "\n";
              };
            smoke-test = mk "smoke tests";
            watcher = mk "watchers";

          in [
            (smoke-test {
              name = "smoke-annealing";
              command = "make RUN_FLAGS=\"--iterations 10\" ex/annealing";
            })
            (watcher {
              name = "watch-annealing-dev";
              command = "${watchexec} -e py '(cd ./experiments/annealing/ && echo \"=========================\" && python ./main.py --pdb)'";
            })
            (watcher {
              name = "watch-short";
              command = ''${watchexec} -e py "make RUN_FLAGS='--iterations 10' ex/$1"'';
            })
            {
              category = "deployment";
              name = "rsync-dev";
              command = ''
                ${pkgs.rsync}/bin/rsync -Paurzvh . $1""
              '';
            }
            (watcher {
              name = "watch-rsync";
              command = ''${watchexec} -e py "rsync-dev $1"'';
            })
          ];
        };
      });
}
