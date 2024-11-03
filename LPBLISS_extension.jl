ENV["PYTHON"] = "/usr/local/bin/micromamba/envs/quantum/bin/python"
using Pkg
Pkg.add("Arpack")
Pkg.activate(".")
Pkg.instantiate()
import QuantumMAMBO
Pkg.add("PythonCall")
using PythonCall
pyscf = pyimport("pyscf")
scipy = pyimport("scipy")
np = pyimport("numpy")
juliacall = pyimport("juliacall")


data_file_path = "fcidumps_original/fcidump.H2O_21_original"
lpbliss_hdf5_output_loading_file_path = "fcidumps_bliss/H2O_21_BLISSS.h5"
lpbliss_fcidump_output_file_path = "fcidumps_bliss/fcidump.H2O_21_BLISS"
# If lpbliss_hdf5_output_loading_file_path already exists, 
# bliss_linprog will load tensors from the h5 file and return the operator

# NOTE: The tensors in the lpbliss_hdf5_output_loading_file_path hdf5 file assume the Hamiltonian is in the form:
# H = E_0 + h_ij a†_i a_j + g_ijkl a†_i a_j a†_k a_l 
# NOTE: The tensors read from and written to the FCIDUMP files assume the Hamiltonian is in the form:
# H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j


# Read FCIDUMP file
######
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
) = QuantumMAMBO.load_tensors_from_fcidump(data_file_path=data_file_path)
# The tensors stored in the FCIDUMP file and returned by load_tensors_from_fcidump 
# are assumed to fit the following definition of the Hamiltonian:
# H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
# where i,j,k,l are indices for the spatial orbitals (NOT spin orbitals)
# The full g_ijkl tensor, not a permutation-symmetry-compressed version, is returned.
# "C1" point group symmetry is assumed

println("Number of orbitals: ", num_orbitals)
println("Number of spin orbitals: ", num_spin_orbitals)
println("Number of electrons: ", num_electrons)
println("Two S: ", two_S)
println("Two Sz: ", two_Sz)
println("Orbital symmetry: ", orb_sym)
println("Extra attributes: ", extra_attributes)



# Convert to QuantumMAMBO fermion operator
######
H_orig = QuantumMAMBO.eri_to_F_OP(one_body_tensor, two_body_tensor, core_energy, spin_orb=false)
# The tensors inside H_orig assume the Hamiltonian is in the form:
# H = E_0 + \sum_{ij} h_{ij} a_i^† a_j + \sum_{ijkl} g_ijkl a_i^† a_j a_k^† a_l
println("Fermionic operator generated.")
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

three_body_tensor = make_hermitian(A)
N = size(one_body_tensor)[1]

H_orig = QuantumMAMBO.eri_to_F_OP(one_body_tensor, two_body_tensor, core_energy, spin_orb=false)
H_orig = H_orig + QuantumMAMBO.F_OP(three_body_tensor, false)

# Run LPBLISS
######
@time begin
  H_bliss, K_operator = QuantumMAMBO.bliss_linprog_extension(H_orig,
    num_electrons,
    model="highs", # LP solver used by Optim; "highs" or "ipopt". Both give the same answer, while "highs" is faster.
    verbose=true,
    SAVELOAD=true,
    SAVENAME=lpbliss_hdf5_output_loading_file_path)
  println("BLISS optimization/operator retrieval complete.")
end