ENV["PYTHON"] = "/usr/local/bin/micromamba/envs/quantum/bin/python"
using Pkg
Pkg.update()
Pkg.precompile()
Pkg.activate(".")
Pkg.develop(path="./QuantumMAMBO")
Pkg.instantiate()
Pkg.precompile()
