ENV["PYTHON"] = "/usr/local/bin/micromamba/envs/quantum/bin/python"
using Pkg
Pkg.add("Arpack")
Pkg.activate(".")
Pkg.instantiate()
import QuantumMAMBO as QM
Pkg.add("PythonCall")
using PythonCall
pyscf = pyimport("pyscf")
scipy = pyimport("scipy")
np = pyimport("numpy")
juliacall = pyimport("juliacall")

data_file_path = "fcidumps_catalysts/fcidump.2_co2_6-311++G__"
fcidump_output_file_path = "fcidumps_mf/fcidump.2_co2_6-311++G__mf"


# Define the directory path
directory_path = "fcidumps_tbt_combined"


# Get a list of files in the directory
files = readdir(directory_path)


# Iterate through each file
for file in files
  # Construct the full file path
  file_path = joinpath(directory_path, file)
  mf_tag = "_DF"
  fcidump_output_file_path = "fcidumps_optimized/$file$mf_tag"


  # Check if the path is a file (and not a directory)
  if isfile(file_path)
    # Load FCIDUMP file
    # OBT, TBT in chemist notation
    (
      one_body_tensor,
      two_body_tensor,
      core_energy,
      num_orbitals,
      num_spin_orbitals,
      num_electrons,
      two_S,
      two_Sz,
      orb_sym,
      extra_attributes,
    ) = QM.load_tensors_from_fcidump(data_file_path=file_path)


    println(typeof(one_body_tensor))
    println(typeof(two_body_tensor))



    # Get F_OP structured H from the obt and tbt.
    # Everything in spatial orbitals
    one_body_dummy = zeros(Float64, num_orbitals, num_orbitals)
    H = QM.F_OP(two_body_tensor, false) + QM.F_OP(one_body_dummy, false)
    println("F_OP obtained.")


    # Get the planted solutions
    # OB is set to false so that the one-body tensor is not included in the optimization
    # It will be added in the end for a fixed frame
    # This returns F_FRAG structuerd F_DFB in spatial orbital
    @time F_DFB = QM.MF_planted(H, method="DF", OB=false)
    println("Planted solutions obtained.")
    println(F_DFB.TECH)


    # Get ferm_op(F_OP) from F_DFB(F_FRAG). The process is as follows:
    # to_OP: line 1 in fermionic.jl file
    # fermionic_frag_representer: line 85 in fermionic.jl as F_FRAG is set to CSA_SD in MF_planted above
    # (line 89 in fermionic.jl)F = cartan_SD_to_F_OP(C) : line 58 in fermionic.jl
    # (line 91 in fermionic.jl)return F_OP_rotation(U[1], F): line 366 in unitaries.jl
    # there the one body rotation and two-body rotation are done.
    @time ferm_op = QM.to_OP(F_DFB)




    # Get spatial obt and tbt from ferm_op(F_OP) obtained above
    # This also transforms the chemist notation into physicist notation
    one_body_tensor_2, two_body_tensor_2 = QM.F_OP_to_eri(ferm_op)


    println(scipy.linalg.norm(one_body_tensor_2))
    L2_1 = scipy.linalg.norm(one_body_tensor, ord=nothing) - scipy.linalg.norm(one_body_tensor_2, ord=nothing)
    L2_2 = scipy.linalg.norm(two_body_tensor, ord=nothing) - scipy.linalg.norm(two_body_tensor_2, ord=nothing)
    println(L2_1)
    println(L2_2)


    println(typeof(one_body_tensor_2))
    println(typeof(two_body_tensor_2))


    pyscf.tools.fcidump.from_integrals(filename=fcidump_output_file_path,
      h1e=np.array(one_body_tensor_2),
      h2e=np.array(two_body_tensor_2),
      # h2e=np.array(two_body_tensor_bliss),
      nmo=num_orbitals,
      nelec=num_electrons,
      nuc=core_energy,
      ms=two_S,
      orbsym=nothing, # "C1" point group symmetry is assumed, so orbsym will be written as [1,1,1,...,1]
      tol=1e-15, # All |h_ij| or |g_ijkl| less than tol will be discarded
      float_format=" %.16g"
    )
    println("LPBLISS-modified tensors written to FCIDUMP file.")
  end
end
