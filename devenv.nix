{ pkgs, lib, config, inputs, ... }:
{
	languages.python = {
		enable = true;
		uv.enable = true;
	};

	packages = with pkgs; [
		hatch
		pre-commit
	];

	env.LD_LIBRARY_PATH = "${lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]}";
}
