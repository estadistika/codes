using DataFrames
using Distributions
using Plots
using Random
Random.seed!(111)

function simulate(n = 1000, w0 = -.3, w1 = -.7, alpha = 1 / 5)
    x = rand(Uniform(-1, 1), n)

    w = [w0; w1]
    A = [ones(size(x)) x]
    
    y = A * w + rand(Normal(0, alpha), size(x))
    return (y, x)
end

x, y = simulate()
println(x[1:10])
println(y[1:10])

p = plot(x, y, markershape = :circle, linealpha = 0)
display(p)
# n = 1000; α = 1 / 5;
# w₀ = -.3; w₁ = -.7;
# w = [w₀; w₁];

# x = rand(Uniform(-1, 1), n);
# A = [ones(size(x)) x];
# y = A * w + rand(Normal(0, α), size(x));

# p = plot(;x = x, y = y)
# draw(SVG(5inch, 4inch), p)

# β = (1 / 2)^2
# μ = zeros(2)
# Σ = Matrix{Float64}(I, 2, 2) * β

# w = rand(MvNormal(μ, Σ), 1)

# Σ⁻¹ = α * A'A + β * I;
# Σ = (Σ⁻¹)^(-1);
# μ = α * Σ * A' * y;

# df = DataFrame(x = x, y = y);
# for i in 1:1000
#     w = rand(MvNormal(μ, Σ))
#     df[Symbol("y" * string(i))] = A * w
# end