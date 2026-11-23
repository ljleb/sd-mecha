{
  description = "sd-mecha development environment";

  inputs.nixpkgs.url = "https://releases.nixos.org/nixpkgs/nixpkgs-26.11pre1038038.421eebfd0ec7/nixexprs.tar.xz";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python311;
        in {
          default = pkgs.mkShell {
            packages = [ python ];

            shellHook = ''
              sd_mecha_setup_venv() {
                local root venv venv_python nixos=
                if [[ -e /run/current-system/nixos-version ]]; then
                  nixos=1
                fi

                ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
                  local library_path="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]}"
                  if [[ -n "$nixos" && -d /run/opengl-driver/lib ]]; then
                    library_path="$library_path:/run/opengl-driver/lib"
                  fi
                  export LD_LIBRARY_PATH="$library_path''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                ''}

                root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
                venv="$root/venv"
                venv_python="$venv/bin/python"

                if [[ ! -f "$venv/pyvenv.cfg" ]]; then
                  printf 'Creating virtual environment at %s\n' "$venv"
                  "${python}/bin/python3.11" -m venv "$venv" || return
                elif [[ -n "$nixos" && ! "$venv_python" -ef "${python}/bin/python3.11" ]]; then
                  printf 'Move %s aside to recreate it with the Python pinned by Nix.\n' "$venv" >&2
                  return 1
                fi

                if ! "$venv_python" -m pip --version >/dev/null 2>&1; then
                  "$venv_python" -m ensurepip --default-pip || return
                fi

                "$venv_python" -m pip install --editable "$root" \
                  --requirement "$root/requirements-dev.txt" &&
                source "$venv/bin/activate"
              }

              if sd_mecha_setup_venv; then
                printf '%s\n' 'sd-mecha dev shell ready' 'Run tests with: pytest'
              fi
              unset -f sd_mecha_setup_venv
            '';
          };
        });
    };
}
