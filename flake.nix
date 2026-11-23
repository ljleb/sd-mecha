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

            shellHook = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            '' + ''
              sd_mecha_setup_venv() {
                local root venv venv_python
                root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
                venv="$root/venv"
                venv_python="$venv/bin/python"

                if [[ ! -f "$venv/pyvenv.cfg" ]]; then
                  printf 'Creating %s and installing locked dependencies.\n' "$venv"
                  "${python}/bin/python3.11" -m venv "$venv" || return
                elif [[ ! "$venv_python" -ef "${python}/bin/python3.11" ]]; then
                  printf 'Move %s aside to recreate it with the Python pinned by Nix.\n' "$venv" >&2
                  return 1
                fi

                if ! "$venv_python" -m pip --version >/dev/null 2>&1; then
                  "$venv_python" -m ensurepip --default-pip || return
                fi

                "$venv_python" -m pip --isolated --disable-pip-version-check install --quiet \
                  --index-url https://pypi.org/simple --require-hashes --only-binary=:all: \
                  --requirement "$root/requirements-nix.txt" &&
                "$venv_python" -m pip --isolated --disable-pip-version-check install --quiet \
                  --no-index --no-deps --no-build-isolation --editable "$root" &&
                source "$venv/bin/activate"
              }
              sd_mecha_setup_venv
              unset -f sd_mecha_setup_venv
            '';
          };
        });
    };
}
