using Flux
using RDatasets
using Measures
using StatPlots
using PlotThemes
using Turing
pyplot()
theme(:orange)

iris = dataset("datasets", "iris");

@df iris scatter(:SepalLength, :SepalWidth, group = :Species,
    xlabel = "Length", ylabel = "Width", markersize = 5,
    markeralpha = 0.75, markerstrokewidth = 0, linealpha = 0, 
    m = (0.5, [:cross :hex :star7], 12),
    margin = 5mm)

inputs = convert(Array, iris[1:4]);
labels = map(x -> x == "setosa" ? 0 : x == "versicolor" ? 1 : 2, iris[end]);

function weights(theta::AbstractVector)
    W0 = reshape(theta[ 1:20], 5, 4); b0 = reshape(theta[21:25], 5)
    W1 = reshape(theta[26:45], 4, 5); b1 = reshape(theta[46:49], 4)
    WO = reshape(theta[50:61], 3, 4); bO = reshape(theta[61:63], 3)    

    return W0, b0, W1, b1, WO, bO
end

function feedforward(inp::Array{Float64, 2}, theta::AbstractVector)
    W0, b0, W1, b1, W2, b2 = weights(theta)
    model = Chain(
        Dense(W0, b0, tanh),
        Dense(W1, b1, tanh),
        Dense(W2, b2, σ),
        softmax
    )

    return model(inp)
end

alpha = 0.09;              
sigma = sqrt(1.0 / alpha); 

@model bayesnn(inp, lab) = begin
    theta ~ MvNormal(zeros(63), sigma .* ones(63))
    
    preds = feedforward(inp, theta)
    for i = 1:length(lab)
        lab[i] ~ Categorical(preds[:, i])
    end
end

steps = 5000
chain = sample(bayesnn(Array(inputs'), labels), HMC(steps, 0.05, 4));


##########
using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, params
using Base.Iterators: repeated
using RDatasets

iris = dataset("datasets", "iris");

inputs = convert(Array, iris[1:4]);
labels = map(x -> x == "setosa" ? 0 : x == "versicolor" ? 1 : 2, iris[end]);

X = Array(inputs');
Y = onehotbatch(labels, 0:2);

m = Chain(
    Dense(4, 5, σ),
    Dense(5, 3, σ),
    softmax
)

loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

data = repeated((X, Y), 2000)
evalcb = () -> @show(loss(X, Y))
opt = [ADAM(params(m)), SGD(params(m))]

Flux.train!(loss, data, opt, cb = throttle(evalcb, 10))


##########
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, params
using Base.Iterators: repeated

imgs = MNIST.images()
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...)

labels = MNIST.labels()
Y = onehotbatch(labels, 0:9)

m = Chain(
    Dense(28^2, 32, relu),
    Dense(32, 10),
    softmax
)
  
loss(x, y) = crossentropy(m(x), y)
  
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

data = repeated((X, Y), 200)
evalcb = () -> @show(loss(X, Y))
opt = ADAM(params(m))

Flux.train!(loss, data, opt, cb = throttle(evalcb, 10))

# W0, b0, W1, b1, W2, b2 = weights(reshape(theta, 63))
# preds = feedforward(Array(inputs'), reshape(theta, 63))

inputs

stack(iris)

N = 5000
ch = sample(bayes_nn(hcat(xs...), ts), HMC(N, 0.05, 4));

plot(iris[:SepalLength], iris[:SepalWidth], markershape = :circle, markersize = 5, linealpha = 0)


#########
using Flux
using Turing
using PyPlot
using Random

Random.seed!(1234)

N = 80; M = round(Int, N / 4); Random.seed!(1234)

x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M]))
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i = 1:M]))

xs = [xt1s; xt0s]; ts = [ones(2*M); zeros(2*M)]

# Plot data points
function plot_data()
    scatter(map(e -> e[1], xt1s), map(e -> e[2], xt1s), marker="x", color="blue")
    scatter(map(e -> e[1], xt0s), map(e -> e[2], xt0s), marker=".", color="red")
    xlim([-7, 7]); ylim([-7, 7])
end

plot_data()

function unpack(theta::AbstractVector)
    W₁ = reshape(theta[1:6], 3, 2);   b₁ = reshape(theta[7:9], 3)
    W₂ = reshape(theta[10:15], 2, 3); b₂ = reshape(theta[16:17], 2)
    Wₒ = reshape(theta[18:19], 1, 2); bₒ = reshape(theta[20:20], 1)   
    return W₁, b₁, W₂, b₂, Wₒ, bₒ
end

function nn_forward(x, theta::AbstractVector)
    W₁, b₁, W₂, b₂, Wₒ, bₒ = unpack(theta)
    nn = Chain(Dense(W₁, b₁, tanh),
               Dense(W₂, b₂, tanh),
               Dense(Wₒ, bₒ, σ),
               softmax)
    return nn(x)
end

alpha = 0.09            # regularizatin term
sig = sqrt(1.0 / alpha) # variance of the Gaussian prior

@model bayes_nn(xs, ts) = begin
    theta ~ MvNormal(zeros(20), sig .* ones(20))
    
    preds = nn_forward(xs, theta)
    for i = 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end

na2mat(na) = begin
    ncol = length(na); nrow = length(na[1])
    mat = Matrix{eltype(na[1])}(undef, nrow, ncol)
    for n = 1:ncol mat[:, n] = na[n] end
    mat
end
N = 5000
ch = sample(bayes_nn(hcat(xs...), ts), HMC(N, 0.05, 4))
theta = ch[:theta]

function nn_predict(x, W₁, b₁, W₂, b₂, Wₒ, bₒ, n_end)
    mean([nn_forward(x, W₁[i], b₁[i], W₂[i], b₂[i], wₒ[i], bₒ[i])[1] for i in 1:10:n_end])
end

function nn_predict(x, theta)
    mean([nn_forward(x, theta[i])[1] for i in 1:10:n_end])
end

plot_data()

n_end = 1500
xs = collect(range(-6,stop=6,length=25))
ys = collect(range(-6,stop=6,length=25))
Z = [nn_predict([x, y], theta)[1] for x=xs, y=ys]
contour(xs, ys, Z)