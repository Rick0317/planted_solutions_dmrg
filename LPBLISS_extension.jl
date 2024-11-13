#ENV["PYTHON"] = "/usr/local/bin/micromamba/envs/quantum/bin/python"
ENV["PYTHON"] = "/cptg/u4/rhuang/.conda/envs/bliss_env"
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
Pkg.add("HDF5")
using HDF5

data_file_path = "fcidumps_original/fcidump.H4_original"
lpbliss_hdf5_output_loading_file_path = "fcidumps_bliss/H4_BLISS.h5"
lpbliss_fcidump_output_file_path = "fcidumps_bliss/fcidump.H4_BLISS"
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
println("Core energy: ", core_energy)



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
    A[i, j, k, l, m, n] = conj(A[n, m, l, k, j, i])
  end
  return A
end

three_body_tensor = make_hermitian(A)
N = size(one_body_tensor)[1]

H_orig = QuantumMAMBO.eri_to_F_OP(one_body_tensor, two_body_tensor, core_energy, spin_orb=false)
H_orig = H_orig + QuantumMAMBO.F_OP(three_body_tensor, false)

lpbliss_hdf5_output_original = "fcidumps_bliss/H4_original.h5"

println("Original h_const:", H_orig.mbts[1])

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
MOL_DATA["threebt"] = H_orig.mbts[4]
MOL_DATA["eta"] = num_electrons
close(fid)

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

QuantumMAMBO.bliss_test(H_orig, H_bliss, num_electrons)