using ColorTypes: RGB, N0f8
using Images
using Plots: Animation, plot!, heatmap
using Printf: @sprintf

import Plots: frame

struct ImageWrapper
    img::Matrix{RGB}
end

function frame(anim::Animation, iw::ImageWrapper)
    i = length(anim.frames) + 1
    filename = @sprintf("%010d.png", i)
    save(joinpath(anim.dir,filename), iw.img)
    push!(anim.frames, filename)
end

function gen_animation(images::Vector)
    anim = Animation()
    wrapped_images = ImageWrapper.(images)
    map((x)->frame(anim, x), wrapped_images)
    return anim
end

get_maxval(sets::Vector) = maximum(map((maximum ∘ collect ∘ Iterators.flatten), sets)) |> Integer

apply_colorscheme(csheme::ColorScheme, sets::Vector, maxval::Integer) :: Vector{Matrix{RGB{N0f8}}} = map((x) -> get(csheme, x, (0, maxval)) .|> RGB{N0f8}, sets)
apply_colorscheme(csheme::ColorScheme, set::AbstractArray, maxval::Integer) :: Matrix{RGB{N0f8}} = get(csheme, set, (0, maxval)) .|> RGB{N0f8} 

complex_euclidean_distance(u::Complex, v::Complex) = sqrt((u.re - v.re)^2 + (u.im - v.im)^2)

function find_location(val::Complex, plane::AbstractArray)
    distances = complex_euclidean_distance.(val, plane)
    return findmin(distances)[2]
end

map_points_to_plane(points::Vector, plane::AbstractArray) = map((point) -> find_location(point,plane), points) .|> Tuple .|> reverse

