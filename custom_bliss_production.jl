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
Pkg.add("HDF5")
using HDF5

data_file_path = "fcidumps_catalysts/fcidump.2_co2_6-311++G__"
fcidump_output_file_path = "fcidumps_bliss/fcidump.H4_BLISS"
lpbliss_hdf5_output_loading_file_path = "fcidumps_bliss/H4_BLISS.h5"
lpbliss_hdf5_output_original = "fcidumps_bliss/H4_original_2body.h5"
# Define the directory path
directory_path = "fcidumps_original"


# Get a list of files in the directory
files = readdir(directory_path)


# Iterate through each file
for file in files
  # Construct the full file path
  file_path = joinpath(directory_path, file)
  mf_tag = "_BLISS"
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
    # Create a 6-dimensional array of Float64
    dims = (num_orbitals, num_orbitals, num_orbitals, num_orbitals, num_orbitals, num_orbitals) # Example dimensions
    A = rand(Float64, dims...)  # Random values as an example

    # Define a Hermitian-like property: A[i, j, k, l, m, n] = conj(A[j, i, l, k, n, m])
    function make_hermitian(A)
      dims = size(A)
      for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3], l in 1:dims[4], m in 1:dims[5], n in 1:dims[6]
        A[i, j, k, l, m, n] = conj(A[j, i, l, k, n, m])
      end
      return A
    end

    # three_body_tensor = make_hermitian(A)
    # println(typeof(three_body_tensor))




    # Get F_OP structured H from the obt and tbt.
    # Everything in spatial orbitals
    # one_body_dummy = zeros(Float64, num_orbitals, num_orbitals)
    # H = QM.F_OP(two_body_tensor, false) + QM.F_OP(one_body_tensor, false)

    H_orig = QM.eri_to_F_OP(one_body_tensor, two_body_tensor, core_energy, spin_orb=false)

    fid = h5open(lpbliss_hdf5_output_original, "cw")
    create_group(fid, "BLISS")
    BLISS_group = fid["BLISS"]
    println("Saving results of BLISS optimization to $lpbliss_hdf5_output_original")
    BLISS_group["ovec"] = zeros(H_orig.N, H_orig.N)
    BLISS_group["t1"] = 0
    BLISS_group["t2"] = 0
    BLISS_group["t3"] = 0
    BLISS_group["N"] = H_orig.N
    BLISS_group["Ne"] = H_orig.N
    create_group(fid, "BLISS_HAM")
    MOL_DATA = fid["BLISS_HAM"]
    MOL_DATA["h_const"] = H_orig.mbts[1]
    MOL_DATA["obt"] = H_orig.mbts[2]
    MOL_DATA["tbt"] = H_orig.mbts[3]
    MOL_DATA["threebt"] = H_orig.mbts[3]
    MOL_DATA["eta"] = num_electrons
    close(fid)

    @time begin
      H_bliss, K_operator = QM.bliss_linprog(H_orig,
        num_electrons,
        model="highs", # LP solver used by Optim; "highs" or "ipopt". Both give the same answer, while "highs" is faster.
        verbose=true,
        SAVELOAD=true,
        SAVENAME=lpbliss_hdf5_output_loading_file_path)
      println("BLISS optimization/operator retrieval complete.")
    end

    # obt_FOP = H_bliss.mbts[2]
    # tbt_FOP = H_bliss.mbts[3]
    # three_FOP = H_bliss.mbts[4]

    # are_equal = three_body_tensor == three_FOP
    #println(are_equal)


    # The obtained F_OPs are in spatial orbitals

    # Retrieve the tensors from the fermionic operator
    ######
    # Chemist notation to physicist notation
    one_body_tensor_bliss, two_body_tensor_bliss = QM.F_OP_to_eri(H_bliss)
    core_energy_bliss = H_bliss.mbts[1][1]
    # These tensors assume the Hamiltonian is in the form:
    # H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
    # ijkl refer to spatial orbitals
    println("Tensors retrieved from the fermionic operator.")


    # Compress tensors based on permutation symmetries, also verifying the symmetry
    ######
    two_body_tensor_bliss_compressed = QM.compress_tensors(one_body_tensor_bliss, two_body_tensor_bliss, num_orbitals)

    # Save the tensors to an FCIDUMP file
    ######
    # The tensors written assume the Hamiltonian is in the form:
    # H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
    # ijkl refer to spatial orbitals
    pyscf = pyimport("pyscf")
    np = pyimport("numpy")
    pyscf.tools.fcidump.from_integrals(filename=fcidump_output_file_path,
      h1e=np.array(one_body_tensor_bliss),
      h2e=two_body_tensor_bliss_compressed,
      # h2e=np.array(two_body_tensor_bliss),
      nmo=num_orbitals,
      nelec=num_electrons,
      nuc=core_energy_bliss,
      ms=two_S,
      orbsym=nothing, # "C1" point group symmetry is assumed, so orbsym will be written as [1,1,1,...,1]
      tol=1e-15, # All |h_ij| or |g_ijkl| less than tol will be discarded
      float_format=" %.16g"
    )
    println("LPBLISS-modified tensors written to FCIDUMP file.")

    num_lanczos_steps_whole_fock_space = 3 # Increase this for more accurate results
    num_lanczos_steps_subspace = 5 # Increase this for more accurate results
    pyscf_fci_max_cycle = 1000 # May want to reduce this for speed
    pyscf_fci_conv_tol = 1E-3 # Will probably want a convergence tolerance (conv_tol) around chemical accuracy (1e-3), 
    # or perhaps looser if the calculations are too slow.

    QM.eliminate_small_values!(one_body_tensor, 1e-8)
    QM.eliminate_small_values!(two_body_tensor, 1e-8)
    println("Small values eliminated.")

    (E_min_orig, E_max_orig, E_min_orig_subspace, E_max_orig_subspace) = QM.pyscf_full_ci(one_body_tensor,
      two_body_tensor,
      core_energy,
      num_electrons,
      pyscf_fci_max_cycle,
      pyscf_fci_conv_tol)

    delta_E_div_2_orig = (E_max_orig - E_min_orig) / 2

    QM.eliminate_small_values!(one_body_tensor_bliss, 1e-8)
    QM.eliminate_small_values!(two_body_tensor_bliss, 1e-8)
    println("Small values eliminated.")

    println("Core energy Bliss", core_energy_bliss)

    (E_min_bliss, E_max_bliss, E_min_bliss_subspace, E_max_bliss_subspace) = QM.pyscf_full_ci(one_body_tensor_bliss,
      two_body_tensor_bliss,
      core_energy_bliss,
      num_electrons,
      pyscf_fci_max_cycle,
      pyscf_fci_conv_tol)

    delta_E_div_2_bliss = (E_max_bliss - E_min_bliss) / 2

    println(E_min_orig)
    println(E_min_bliss)
    println(E_min_orig_subspace)
    println(E_min_bliss_subspace)
    println("ΔE/2, orig: ", delta_E_div_2_orig)
    println("ΔE/2, LPBLISS: ", delta_E_div_2_bliss)
  end
end