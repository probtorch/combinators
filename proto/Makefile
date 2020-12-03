##
# haskell.nix helpers
#
# @file
# @version 0.1

package = proto-combinators
binary = proto-combinators

exe:
	nix-build -A $(package).components.exes.$(binary)

release:
	nix-build ./release.nix

development:
	nix-build ./nix/development.nix

# end
