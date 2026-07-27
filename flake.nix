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
          python = pkgs.python311.withPackages (ps: [ ps.pip ]);
        in {
          default = pkgs.mkShell {
            packages = [ python ];
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];

            shellHook = ''
              if [ -f .nix-venv/bin/activate ]; then
                source .nix-venv/bin/activate
              else
                echo "Create the project virtualenv with:"
                echo "  python -m venv .nix-venv && .nix-venv/bin/pip install -e . -r requirements-dev.txt"
              fi

              echo "sd-mecha dev shell ready"
              echo "Run tests with: pytest"
            '';
          };
        });
    };
}
