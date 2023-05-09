using Documenter
using FractalAnimation

makedocs(sitename="FractalAnimation.jl")

deploydocs(
    repo = "github.com/ErrolSeders/FractalAnimation.jl.git", push_preview=true 
)