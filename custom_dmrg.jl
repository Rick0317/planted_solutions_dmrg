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

# Define the directory path
directory_path = "fcidumps_optimized"


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


    ampo = AutoMPO()
    N = num_orbitals
    println("Number of orbitals: ", N)
    sites = siteinds("Electron", 2 * N, conserve_qns=true, conserve_sz=true)

    for i in 1:N
      for j in 1:N
        ampo += one_body_tensor[i, j],
        "Cdagup", 2 * i - 1,  # creation operator for orbital i with spin a
        "Cup", 2 * j - 1      # annihilation operator for orbital j with spin a
        ampo += one_body_tensor[i, j],
        "Cdagdn", 2 * i,  # creation operator for orbital i with spin a
        "Cdn", 2 * j      # annihilation operator for orbital j with spin a

      end
    end

    for i in 1:N
      for j in 1:N
        for k in 1:N
          for l in 1:N

            p = 2 * i - 1
            r = 2 * k - 1
            s = 2 * l - 1
            q = 2 * j - 1

            ampo += 0.5 * two_body_tensor[i, j, k, l],
            "Cdagup", p,  # creation operator for orbital i with spin a
            "Cdagup", r,  # creation operator for orbital k with spin b
            "Cup", s,      # annihilation operator for orbital l with spin b
            "Cup", q       # annihilation operator for orbital j with spin a



            p = 2 * i - 1
            r = 2 * k
            s = 2 * l
            q = 2 * j - 1
            ampo += 0.5 * two_body_tensor[i, j, k, l],
            "Cdagup", p,  # creation operator for orbital i with spin a
            "Cdagdn", r,  # creation operator for orbital k with spin b
            "Cdn", s,      # annihilation operator for orbital l with spin b
            "Cup", q       # annihilation operator for orbital j with spin a


            p = 2 * i
            r = 2 * k - 1
            s = 2 * l - 1
            q = 2 * j
            ampo += 0.5 * two_body_tensor[i, j, k, l],
            "Cdagdn", p,  # creation operator for orbital i with spin a
            "Cdagup", r,  # creation operator for orbital k with spin b
            "Cup", s,      # annihilation operator for orbital l with spin b
            "Cdn", q       # annihilation operator for orbital j with spin a


            p = 2 * i
            r = 2 * k
            s = 2 * l
            q = 2 * j

            ampo += 0.5 * two_body_tensor[i, j, k, l],
            "Cdagdn", p,  # creation operator for orbital i with spin a
            "Cdagdn", r,  # creation operator for orbital k with spin b
            "Cdn", s,      # annihilation operator for orbital l with spin b
            "Cdn", q       # annihilation operator for orbital j with spin a


          end
        end
      end
    end



    # for i in 1:N
    #   for j in 1:N
    #     for k in 1:N
    #       terms = [
    #         ("Cdagup", "Cup", "Cdagup", "Cup", "Cdagup", "Cup"),
    #         ("Cdagup", "Cup", "Cdagup", "Cup", "Cdagdn", "Cdn"),
    #         ("Cdagup", "Cup", "Cdagdn", "Cdn", "Cdagup", "Cup"),
    #         ("Cdagup", "Cup", "Cdagdn", "Cdn", "Cdagdn", "Cdn"),
    #         ("Cdagdn", "Cup", "Cdagup", "Cup", "Cdagup", "Cup"),
    #         ("Cdagdn", "Cdn", "Cdagup", "Cup", "Cdagdn", "Cdn"),
    #         ("Cdagdn", "Cdn", "Cdagdn", "Cup", "Cdagup", "Cup"),
    #         ("Cdagdn", "Cdn", "Cdagdn", "Cdn", "Cdagdn", "Cdn")
    #       ]
    #       for t in terms
    #         ampo += 

    #       end 


    #     end
    #   end
    # end

    # t = 10

    # for i in 1:N
    #   for j in 1:N
    #     for k in 1:N
    #       # Three-body number operators using creation and annihilation operators
    #       ampo += -t, "Cdagup", 2 * i - 1, "Cup", 2 * i - 1, "Cdagup", 2 * j - 1, "Cup", 2 * j - 1, "Cdagup", 2 * k - 1, "Cup", 2 * k - 1
    #       ampo += -t, "Cdagup", 2 * i - 1, "Cup", 2 * i - 1, "Cdagup", 2 * j - 1, "Cup", 2 * j - 1, "Cdagdn", 2 * k, "Cdn", 2 * k
    #       ampo += -t, "Cdagup", 2 * i - 1, "Cup", 2 * i - 1, "Cdagdn", 2 * j, "Cdn", 2 * j, "Cdagup", 2 * k - 1, "Cup", 2 * k - 1
    #       ampo += -t, "Cdagup", 2 * i - 1, "Cup", 2 * i - 1, "Cdagdn", 2 * j, "Cdn", 2 * j, "Cdagdn", 2 * k, "Cdn", 2 * k
    #       ampo += -t, "Cdagdn", 2 * i, "Cdn", 2 * i, "Cdagup", 2 * j - 1, "Cup", 2 * j - 1, "Cdagup", 2 * k - 1, "Cup", 2 * k - 1
    #       ampo += -t, "Cdagdn", 2 * i, "Cdn", 2 * i, "Cdagup", 2 * j - 1, "Cup", 2 * j - 1, "Cdagdn", 2 * k, "Cdn", 2 * k
    #       ampo += -t, "Cdagdn", 2 * i, "Cdn", 2 * i, "Cdagdn", 2 * j, "Cdn", 2 * j, "Cdagup", 2 * k - 1, "Cup", 2 * k - 1
    #       ampo += -t, "Cdagdn", 2 * i, "Cdn", 2 * i, "Cdagdn", 2 * j, "Cdn", 2 * j, "Cdagdn", 2 * k, "Cdn", 2 * k
    #     end
    #   end
    # end

    # print(ampo)

    H = MPO(ampo, sites)


    #print(H)

    nsweeps = 10 # number of sweeps is 5
    maxdim = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400] # gradually increase states kept
    # maxdim = [100 for n in 1:10]
    noise = [1E-4, 1E-4, 1E-5, 1E-5, 1E-6, 1E-6, 1E-7, 1E-7, 1E-8, 0.0]
    # noise = [0, 0, 0, 0, 0]
    cutoff = [1E-15] # desired truncation error

    states = ["Emp" for n = 1:2*N]
    states = prepare_electronic_states(states, num_electrons)
    # psi0 = productMPS(sites, states)
    psi0 = randomMPS(sites, states)
    print(num_electrons)

    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise)
    # print(energy + t * num_electrons^3)
  end
end