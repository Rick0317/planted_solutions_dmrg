# ENV["PYTHON"] = "/usr/local/bin/micromamba/envs/quantum/bin/python"
# using Pkg
# Pkg.add("Arpack")
# Pkg.activate(".")
# Pkg.instantiate()
# Pkg.add("PythonCall")
# using PythonCall
# pyscf = pyimport("pyscf")
# scipy = pyimport("scipy")
# np = pyimport("numpy")
# juliacall = pyimport("juliacall")
using ITensors, ITensorMPS


let
  N = 4  # Number of sites
  sites = siteinds("Fermion", 2 * N)  # Double the number of sites for spin-up and spin-down

  # Set up the AutoMPO for the Hubbard model
  U = 4.0  # On-site interaction strength
  t = 1
  ampo = AutoMPO()


  ### Artificial One-body term ###
  dims = (N, N) # Example dimensions
  A = rand(Float64, dims...)  # Random values as an example

  # Define a Hermitian-like property: A[i, j, k, l, m, n] = conj(A[j, i, l, k, n, m])
  function make_hermitian(A)
    dims = size(A)
    for i in 1:dims[1], j in 1:dims[2]
      A[i, j] = conj(A[j, i])
    end
    return A
  end

  one_body_tensor = make_hermitian(A)
  println(typeof(one_body_tensor))

  for i in 1:N
    for j in 1:N
      ampo += one_body_tensor[i, j], "Cdag", 2 * i - 1, "C", 2 * j - 1  # Spin-up
      ampo += one_body_tensor[i, j], "Cdag", 2 * i, "C", 2 * j  # Spin-down
    end
  end


  ### Artificial Two-body term ###
  # We have zero αβ/βα interactions.
  # Sz is preserved


  dims = (N, N, N, N) # Example dimensions
  A = rand(Float64, dims...)  # Random values as an example

  # Define a Hermitian-like property: A[i, j, k, l, m, n] = conj(A[j, i, l, k, n, m])
  function make_hermitian(A)
    dims = size(A)
    for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3], l in 1:dims[4]
      A[i, j, k, l] = conj(A[j, i, l, k])
    end
    return A
  end

  two_body_tensor = make_hermitian(A)
  println(typeof(two_body_tensor))

  for i in 1:N
    for j in 1:N
      for k in 1:N
        for l in 1:N
          if i != k && j != l
            ampo += two_body_tensor[i, j, k, l], "Cdag", 2 * i - 1, "Cdag", 2 * k - 1, "C", 2 * j - 1, "C", 2 * l - 1  # Spin-up
            ampo += two_body_tensor[i, j, k, l], "Cdag", 2 * i, "Cdag", 2 * k, "C", 2 * j, "C", 2 * l  # Spin-down
          end
        end
      end
    end
  end

  ### Artificial Three-body term ###
  do_three = false

  if do_three
    dims = (N, N, N, N, N, N) # Example dimensions
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
    println(typeof(three_body_tensor))

    for i in 1:N
      for j in 1:N
        for k in 1:N
          for l in 1:N
            for m in 1:N
              for n in 1:N
                if i != k && j != l && k != m && l != n && i != m && j != n
                  ampo += three_body_tensor[i, j, k, l, m, n], "Cdag", 2 * i - 1, "Cdag", 2 * k - 1, "Cdag", 2 * m - 1, "C", 2 * j - 1, "C", 2 * l - 1, "C", 2 * n - 1  # Spin-up
                  ampo += three_body_tensor[i, j, k, l, m, n], "Cdag", 2 * i, "Cdag", 2 * k, "Cdag", 2 * m, "C", 2 * j, "C", 2 * l, "C", 2 * n  # Spin-down
                end
              end
            end
          end
        end
      end
    end
  end

  # # Add hopping terms for both spin-up and spin-down electrons
  # for j in 1:N-1
  #   ampo += -t, "Cdag", 2 * j - 1, "C", 2 * (j + 1) - 1  # Spin-up hopping (odd sites)
  #   ampo += -t, "Cdag", 2 * (j + 1) - 1, "C", 2 * j - 1  # Spin-up hopping reverse
  #   ampo += -t, "Cdag", 2 * j, "C", 2 * (j + 1)  # Spin-down hopping (even sites)
  #   ampo += -t, "Cdag", 2 * (j + 1), "C", 2 * j  # Spin-down hopping reverse
  # end

  # # Add on-site interaction terms: U * n_up * n_down
  # for j in 1:N
  #   ampo += U, "N", 2 * j - 1, "N", 2 * j  # n_up * n_down at each site
  # end

  # # Add on-site interaction terms: U * n_up * n_down
  # for j in 1:N-2
  #   ampo += U, "N", 2 * j - 1, "N", 2 * j, "N", 2 * j + 1, "N", 2 * j + 2  # n_up * n_down at each site
  # end

  # println(ampo)

  H = MPO(ampo, sites)

  print(H)

  nsweeps = 10 # number of sweeps is 5
  maxdim = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400] # gradually increase states kept
  cutoff = [1E-10] # desired truncation error

  psi0 = randomMPS(sites; linkdims=2)

  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

  return
end
