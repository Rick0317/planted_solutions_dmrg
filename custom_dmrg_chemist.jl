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
using ITensors, ITensorMPS
Pkg.add("HDF5")
using HDF5

# Define the directory path
directory_path = "fcidumps_original"


# Get a list of files in the directory
files = readdir(directory_path)

function prepare_electronic_states(states, N_electrons)
  for i in 1:N_electrons
    if isodd(i)
      states[i] = "Up"  # Odd-indexed sites have "Up" spins
    else
      states[i] = "Dn"  # Even-indexed sites have "Dn" spins
    end
  end
  return states
end

function apply_operator(ampo, h3e_val, p, q, r, s, t, u, op1, op2, op3, op4, op5, op6)
  return h3e_val,
  op1, p,
  op2, q,
  op3, r,
  op4, s,
  op5, t,
  op6, u

end

# Iterate through each file
for file in files
  # Construct the full file path
  file_path = joinpath(directory_path, file)
  mf_tag = "_DF"
  fcidump_output_file_path = "fcidumps_optimized/$file$mf_tag"
  lpbliss_hdf5_output_loading_file_path = "fcidumps_bliss/H4_BLISS.h5"


  fid = h5open(lpbliss_hdf5_output_loading_file_path, "cw")
  if haskey(fid, "BLISS")
    BLISS_group = fid["BLISS"]
    if haskey(BLISS_group, "ovec")
      println("Loading results for BLISS optimization from file: ", lpbliss_hdf5_output_loading_file_path)
      ovec = read(BLISS_group, "ovec")
      t1 = read(BLISS_group, "t1")
      t2 = read(BLISS_group, "t2")
      t3 = read(BLISS_group, "t3")
      N = read(BLISS_group, "N")
      t_opt = [t1, t2, t3]
      O = zeros(N, N)
      idx = 1
      for i = 1:N
        for j = 1:N
          O[i, j] = ovec[idx]
          idx += 1
        end
      end
      ham = fid["BLISS_HAM"]
      MOL_DATA = fid["BLISS_HAM"]
      h_const = MOL_DATA["h_const"]
      h1e = MOL_DATA["obt"]
      h2e = MOL_DATA["tbt"]
      h3e = MOL_DATA["threebt"]
      # ne = BLISS_group["Ne"]
      # println("The L1 cost of symmetry treated fermionic operator is: ", PAULI_L1(F_new))

      println("Loading over")

      println("H3e:", h3e)



      sites = siteinds("Electron", 2 * N; conserve_qns=true)

      ampo = OpSum()

      for i in 1:N
        for j in 1:N
          ampo += h1e[i, j],
          "Cdagup", 2 * i - 1,  # creation operator for orbital i with spin a
          "Cup", 2 * j - 1      # annihilation operator for orbital j with spin a
          ampo += h1e[i, j],
          "Cdagdn", 2 * i,  # creation operator for orbital i with spin a
          "Cdn", 2 * j      # annihilation operator for orbital j with spin a
        end
      end

      println("One-body terms added")

      for i in 1:N
        for j in 1:N
          for k in 1:N
            for l in 1:N

              p = 2 * i - 1
              r = 2 * k - 1
              s = 2 * l - 1
              q = 2 * j - 1

              ampo += h2e[i, j, k, l],
              "Cdagup", p,  # creation operator for orbital i with spin a
              "Cup", q,       # annihilation operator for orbital j with spin a
              "Cdagup", r,  # creation operator for orbital k with spin b
              "Cup", s      # annihilation operator for orbital l with spin b


              p = 2 * i - 1
              r = 2 * k
              s = 2 * l
              q = 2 * j - 1
              ampo += h2e[i, j, k, l],
              "Cdagup", p,  # creation operator for orbital i with spin a
              "Cup", q,       # annihilation operator for orbital j with spin a
              "Cdagdn", r,  # creation operator for orbital k with spin b
              "Cdn", s      # annihilation operator for orbital l with spin b


              p = 2 * i
              r = 2 * k - 1
              s = 2 * l - 1
              q = 2 * j

              ampo += h2e[i, j, k, l],
              "Cdagdn", p,  # creation operator for orbital i with spin a
              "Cdn", q,       # annihilation operator for orbital j with spin a
              "Cdagup", r,  # creation operator for orbital k with spin b
              "Cup", s      # annihilation operator for orbital l with spin b


              p = 2 * i
              r = 2 * k
              s = 2 * l
              q = 2 * j

              ampo += h2e[i, j, k, l],
              "Cdagdn", p,  # creation operator for orbital i with spin a
              "Cdn", q,       # annihilation operator for orbital j with spin a
              "Cdagdn", r,  # creation operator for orbital k with spin b
              "Cdn", s      # annihilation operator for orbital l with spin b

            end
          end
        end
      end

      println("Two-body terms added")

      for i in 1:N
        for j in 1:N
          for k in 1:N
            for l in 1:N
              for m in 1:N
                for n in 1:N
                  h3e_val = h3e[i, j, k, l, m, n]

                  # Case 1: All spins up
                  p, q, r, s, t, u = 2 * i - 1, 2 * j - 1, 2 * k - 1, 2 * l - 1, 2 * m - 1, 2 * n - 1
                  ampo += apply_operator(ampo, h3e_val, p, q, r, s, t, u,
                    "Cdagup", "Cup", "Cdagup", "Cup", "Cdagup", "Cup")

                  # Case 2: Mixed spins
                  p, q, r, s, t, u = 2 * i - 1, 2 * j - 1, 2 * k, 2 * l, 2 * m - 1, 2 * n - 1
                  ampo += apply_operator(ampo, h3e_val, p, q, r, s, t, u,
                    "Cdagup", "Cup", "Cdagdn", "Cdn", "Cdagup", "Cup")

                  # Case 3: Mixed spins (another configuration)
                  p, q, r, s, t, u = 2 * i - 1, 2 * j - 1, 2 * k - 1, 2 * l - 1, 2 * m, 2 * n
                  ampo += apply_operator(ampo, h3e_val, p, q, r, s, t, u,
                    "Cdagup", "Cup", "Cdagup", "Cup", "Cdagdn", "Cdn")

                  # Case 4: Other configurations (continue similar pattern)
                  p, q, r, s, t, u = 2 * i, 2 * j, 2 * k - 1, 2 * l - 1, 2 * m - 1, 2 * n - 1
                  ampo += apply_operator(ampo, h3e_val, p, q, r, s, t, u,
                    "Cdagdn", "Cdn", "Cdagup", "Cup", "Cdagup", "Cup")

                  # Case 5: Other configurations (continue similar pattern)
                  p, q, r, s, t, u = 2 * i, 2 * j, 2 * k, 2 * l, 2 * m - 1, 2 * n - 1
                  ampo += apply_operator(ampo, h3e_val, p, q, r, s, t, u,
                    "Cdagdn", "Cdn", "Cdagdn", "Cdn", "Cdagup", "Cup")

                  # Case 6: Other configurations (continue similar pattern)
                  p, q, r, s, t, u = 2 * i, 2 * j, 2 * k - 1, 2 * l - 1, 2 * m, 2 * n
                  ampo += apply_operator(ampo, h3e_val, p, q, r, s, t, u,
                    "Cdagdn", "Cdn", "Cdagup", "Cup", "Cdagdn", "Cdn")


                  # Case 7: Other configurations (continue similar pattern)
                  p, q, r, s, t, u = 2 * i - 1, 2 * j - 1, 2 * k, 2 * l, 2 * m, 2 * n
                  ampo += apply_operator(ampo, h3e_val, p, q, r, s, t, u,
                    "Cdagup", "Cup", "Cdagdn", "Cdn", "Cdagdn", "Cdn")

                  # Case 8: Other configurations (continue similar pattern)
                  p, q, r, s, t, u = 2 * i, 2 * j, 2 * k, 2 * l, 2 * m, 2 * n
                  ampo += apply_operator(ampo, h3e_val, p, q, r, s, t, u,
                    "Cdagdn", "Cdn", "Cdagdn", "Cdn", "Cdagdn", "Cdn")

                  # Additional cases can be added here similarly as needed.
                end
              end
            end
          end
        end
      end

      println("Three-body terms added")


      # print(ampo)

      H = MPO(ampo, sites)


      #print(H)

      nsweeps = 10 # number of sweeps is 5
      maxdim = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400] # gradually increase states kept
      noise = [1E-20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      cutoff = [0] # desired truncation error

      states = ["Emp" for n = 1:2*N]
      ne = 4
      states = prepare_electronic_states(states, ne)
      # psi0 = productMPS(sites, states)
      psi0 = randomMPS(sites, states)
      # psi0 = MPS(sites, states)
      print(ne)
      println("Total QN MPS:", flux(psi0))

      # psi0 = randomMPS(sites; linkdims=2)
      print(h_const[1])
      energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise)
      #print(energy + t * ne^3)
      close(fid)
    end
  end
end