{ pkgs }:
	let
		libs = with pkgs; [
			stdenv.cc.cc.lib
			zlib
			openssl
			libffi
			glibc
		];

		libPath = pkgs.lib.makeLibraryPath libs;

	in pkgs.mkShell {
		name = "default shell";

		nativeBuildInputs = with pkgs; [
			hatch
			uv
			just
		];

		env.NIX_LD = pkgs.stdenv.cc.bintools.dynamicLinker;
		env.NIX_LD_LIBRARY_PATH = libPath;

		# Unset PYTHONPATH to prevent nix packages from interfering with
		# hatch virtual environments. The PYTHONPATH set by pythonEnv includes
		# nix store paths that can shadow packages installed in hatch venvs.
		#shellHook = ''
		#	unset PYTHONPATH
		#'';
	}
