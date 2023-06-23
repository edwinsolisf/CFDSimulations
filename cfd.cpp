/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
/*
This is a Computational Fluid Dynamics Simulation using the Lattice Boltzmann Method
For this simulation we are using D2N9 (2 dimensions, 9 neighbors) with bounce-back boundary conditions
For more information on the simulation equations,
check out https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods#Mathematical_equations_for_simulations

The initial conditions of the fluid are obtained from three images that specify their properties using the function
read_initial_condition_arrays. These images can be modified to simulate different cases
*/

#include <chrono>
#include <iostream>
#include <thread>

#include <arrayfire.h>

// Array Quantities
af::array ex;
af::array ey;
af::array wt;

af::array ex_T;
af::array ey_T;
af::array wt_T;

af::array ex_;
af::array ey_;

const float ex_vals[] = {
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0
};

const float ey_vals[] = {
    1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0
};

const float wt_vals[] = {
    1.0f / 36.0f,  4.0f / 36.0f, 1.0f / 36.0f,
    4.0f / 36.0f, 16.0f / 36.0f, 4.0f / 36.0f,
    1.0f / 36.0f,  4.0f / 36.0f, 1.0f / 36.0f
};

const int oppos_vals[] = {
    8, 7, 6, 5, 4, 3, 2, 1, 0
};


struct InitialConditions
{
    af::array ux;
    af::array uy;
    af::array set_boundaries;
};

struct SimulationQuantities
{
    af::array ux;
    af::array uy;
    af::array rho;
    af::array sigma;
    af::array f;
    af::array feq;
    af::array set_boundaries;
    size_t grid_width;
    size_t grid_height;
    float density;
    float velocity;
    float reynolds;

    SimulationQuantities(size_t grid_width_, size_t grid_height_, float density_, float velocity_, float reynolds_number_)
        : grid_width(grid_width_), grid_height(grid_height_),
          density(density_), velocity(velocity_), reynolds(reynolds_number_)
    {   
    }
};

InitialConditions read_initial_condition_arrays(size_t grid_width, size_t grid_height, float velocity,
                                                const char* ux_image_filename,
                                                const char* uy_image_filename,
                                                const char* boundaries_filename)
{
    InitialConditions vals;

    try
    {
        vals.ux = af::loadImage(ux_image_filename, true);
    }
    catch(const af::exception& e)
    {
        std::cerr << e.what() << std::endl;
        vals.ux = af::constant(0, grid_width, grid_height, 3);
    }

    auto ux_dim = vals.ux.dims();
    if (ux_dim[0] != grid_width || ux_dim[1] != grid_height)
    {
        std::cerr << "Fluid flow ux image has dimensions different to the simulation" << std::endl;
        throw std::runtime_error{"Fluid flow ux image has dimensions different to the simulation"};
    }

    try
    {
        vals.uy = af::loadImage(uy_image_filename, true);
    }
    catch(const af::exception& e)
    {
        std::cerr << e.what() << std::endl;
        vals.uy = af::constant(0, grid_width, grid_height, 3);
    }
    
    auto uy_dim = vals.uy.dims();
    if (uy_dim[0] != grid_width || uy_dim[1] != grid_height)
    {
        std::cerr << "Fluid flow uy image has dimensions different to the simulation" << std::endl;
        throw std::runtime_error{"Fluid flow uy image has dimensions different to the simulation"};
    }

    try
    {
        vals.set_boundaries = af::loadImage(boundaries_filename, false);
    }
    catch(const af::exception& e)
    {
        std::cerr << e.what() << std::endl;
        vals.set_boundaries = af::constant(0, grid_width, grid_height, 3);
    }
    
    auto b_dim = vals.set_boundaries.dims();
    if (b_dim[0] != grid_width || b_dim[1] != grid_height)
    {
        std::cerr << "Fluid boundary image has dimensions different to the simulation" << std::endl;
        throw std::runtime_error{"Fluid boundary image has dimensions different to the simulation"};
    }

    vals.ux = (vals.ux(af::span, af::span, 0).T() - vals.ux(af::span, af::span, 2).T()) * velocity / 255.f;
    vals.uy = (vals.uy(af::span, af::span, 0).T() - vals.uy(af::span, af::span, 2).T()) * velocity / 255.f;
    vals.set_boundaries = vals.set_boundaries.T() > 0;

    return vals;
}

/**
 * Creates all the af::arrays that will be used throughout the program
 * 
 * In this function, the initial conditions and boundary conditions of the simulation are set
 * 
*/
void initialize(SimulationQuantities& simulation,
                const char* ux_image_filename, const char* uy_image_filename, const char* boundary_image_filename)
{
    auto& ux = simulation.ux;
    auto& uy = simulation.uy;
    auto& rho = simulation.rho;
    auto& sigma = simulation.sigma;
    auto& f = simulation.f;
    auto& feq = simulation.feq;
    auto& set_boundaries = simulation.set_boundaries;
    auto density = simulation.density;
    auto velocity = simulation.velocity;
    auto xcount = simulation.grid_width;
    auto ycount = simulation.grid_height;

    ex = af::array(1, 1, 9, ex_vals);
    ey = af::array(1, 1, 9, ey_vals);
    wt = af::array(1, 1, 9, wt_vals);

    ex_T = af::array(1, 9, ex_vals);
    ey_T = af::array(1, 9, ey_vals);
    wt_T = af::moddims(wt, af::dim4(1,9));

    rho = af::constant(density, xcount, ycount, f32);
    sigma = af::constant(0, xcount, ycount, f32);
    ux = af::constant(0, xcount, ycount, f32);
    uy = af::constant(0, xcount, ycount, f32);

    // This initializes the velocity field
    auto vals = read_initial_condition_arrays(xcount, ycount, velocity,
                                              ux_image_filename,
                                              uy_image_filename,
                                              boundary_image_filename);
    ux = vals.ux;
    uy = vals.uy;
    set_boundaries = vals.set_boundaries;

    f   = af::constant(0, xcount, ycount, 9, f32);

    ex_ = af::tile(ex, xcount, ycount, 1);
    ey_ = af::tile(ey, xcount, ycount, 1);

    // Initialization of the distribution function
    auto edotu = ex_ * ux + ey_ * uy;
    auto udotu = ux * ux + uy * uy;

    feq = rho * wt * ((edotu * edotu * 4.5f) - (udotu * 1.5f) + (edotu * 3.0f) + 1.0f);
    f = feq;
}

void collide_stream(SimulationQuantities& simulation)
{
    auto& ux = simulation.ux;
    auto& uy = simulation.uy;
    auto& rho = simulation.rho;
    auto& sigma = simulation.sigma;
    auto& f = simulation.f;
    auto& feq = simulation.feq;
    const auto& set_boundaries = simulation.set_boundaries;
    auto density = simulation.density;
    auto velocity = simulation.velocity;
    auto xcount = simulation.grid_width;
    auto ycount = simulation.grid_height;
    auto reynolds = simulation.reynolds;

    const float viscosity = velocity * std::sqrt(static_cast<float>(xcount * ycount)) / reynolds;
    const float tau = 0.5f + 3.0f * viscosity;
    const float csky = 0.16f;

    auto edotu = ex_ * ux + ey_ * uy;
    auto udotu = ux * ux + uy * uy;

    // Compute the new distribution function
    feq = rho * wt * (edotu * edotu * 4.5f - udotu * 1.5f + edotu * 3.0f + 1.0f);

    auto taut = af::sqrt(sigma * (csky * csky * 18.0f * 0.25f) + (tau * tau * 0.25f)) - (tau * 0.5f);

    // Compute the shifted distribution functions
    auto fplus = f - (f - feq) / (taut + tau);


    // Compute new particle distribution according to the corresponding D2N9 weights
    for (int i = 0; i < 9; ++i)
    {
        int xshift = static_cast<int>(ex_vals[i]);
        int yshift = static_cast<int>(ey_vals[i]);

        fplus(af::span, af::span, i) = af::shift(fplus(af::span, af::span, i), xshift, yshift);
    }

    // Keep the boundary conditions at the borders the same
    // fplus.row(0)          = fnew.row(0);
    // fplus.row(xcount - 1) = fnew.row(xcount - 1);
    // fplus.col(0)          = fnew.col(0);
    // fplus.col(ycount - 1) = fnew.col(ycount - 1);
    fplus = af::select(set_boundaries, f, fplus);

    // Update the particle distribution
    f = fplus;

    // Computing u dot e at the each of the boundaries
    af::array ux_top = ux.rows(0, 2);
    ux_top = af::moddims(af::tile(ux_top, af::dim4(1, 3)).T(), af::dim4(ycount , 9));
    af::array ux_bot = ux.rows(xcount - 3, xcount - 1);
    ux_bot = af::moddims(af::tile(ux_bot, af::dim4(1, 3)).T(), af::dim4(ycount , 9));

    af::array uy_top = uy.rows(0, 2);
    uy_top = af::moddims(af::tile(uy_top, af::dim4(1, 3)).T(), af::dim4(ycount , 9));
    af::array uy_bot = uy.rows(xcount - 3, xcount - 1);
    uy_bot = af::moddims(af::tile(uy_bot, af::dim4(1, 3)).T(), af::dim4(ycount , 9));
    
    auto ux_lft = af::tile(ux.cols(0, 2), af::dim4(1, 3));
    auto uy_lft = af::tile(uy.cols(0, 2), af::dim4(1, 3));
    auto ux_rht = af::tile(ux.cols(ycount - 3, ycount - 1), af::dim4(1, 3));
    auto uy_rht = af::tile(uy.cols(ycount - 3, ycount - 1), af::dim4(1, 3));

    auto ubdoute_top = ux_top * ex_T + uy_top * ey_T;
    auto ubdoute_bot = ux_bot * ex_T + uy_bot * ey_T;
    auto ubdoute_lft = ux_lft * ex_T + uy_lft * ey_T;
    auto ubdoute_rht = ux_rht * ex_T + uy_rht * ey_T;

    // Computing bounce-back boundary conditions
    auto fnew_top = af::moddims(fplus.row(1)         , af::dim4(ycount, 9)) - 6.0 * density * wt_T * ubdoute_top;
    auto fnew_bot = af::moddims(fplus.row(xcount - 2), af::dim4(ycount, 9)) - 6.0 * density * wt_T * ubdoute_bot;
    auto fnew_lft = af::moddims(fplus.col(1         ), af::dim4(xcount, 9)) - 6.0 * density * wt_T * ubdoute_lft;
    auto fnew_rht = af::moddims(fplus.col(ycount - 2), af::dim4(xcount, 9)) - 6.0 * density * wt_T * ubdoute_rht;

    // Update the values near the boundaries with the correct bounce-back boundary
    for (int i = 0; i < 9; ++i)
    {
        int xshift = static_cast<int>(ex_vals[i]);
        int yshift = static_cast<int>(ey_vals[i]);
        if (xshift == 1)
            f(         1,   af::span, oppos_vals[i]) = fnew_top(af::span, i);
        if (xshift == -1)
            f(xcount - 2,   af::span, oppos_vals[i]) = fnew_bot(af::span, i);
        if (yshift == 1)
            f(  af::span,          1, oppos_vals[i]) = fnew_lft(af::span, i);
        if (yshift == -1)
            f(  af::span, ycount - 2, oppos_vals[i]) = fnew_rht(af::span, i);
    }
}

/**
 * Updates the velocity field, density and strain at each point in the grid
*/
void update(SimulationQuantities& simulation)
{
    auto& ux = simulation.ux;
    auto& uy = simulation.uy;
    auto& rho = simulation.rho;
    auto& sigma = simulation.sigma;
    auto& f = simulation.f;
    auto& feq = simulation.feq;

    auto xcount = simulation.grid_width;
    auto ycount = simulation.grid_height;

    auto f_tile = af::tile(f, af::dim4(1,1,1,3));
    auto e_tile = af::join(3, af::constant(1, xcount, ycount, 9), ex_, ey_);
    auto result = af::sum(f_tile * e_tile, 2);

    rho = result(af::span, af::span, af::span, 0);
    result /= rho;
    ux = result(af::span, af::span, af::span, 1);
    uy = result(af::span, af::span, af::span, 2);

    // Above code equivalent to 
    // rho = af::sum(f, 2);
    // ux = af::sum(f * ex_) / rho;
    // uy = af::sum(f * ey_) / rho;

    auto product = f - feq;
    auto temp = af::tile(product, af::dim4(1,1,1,3));

    temp(af::span, af::span, af::span, 0) *= ex_ * ex_;
    temp(af::span, af::span, af::span, 1) *= ey_ * ex_;
    temp(af::span, af::span, af::span, 2) *= ey_ * ey_;
    temp = af::sum(temp, 2);
    temp *= temp;

    sigma = af::sqrt(temp(af::span, af::span, af::span, 0) +
                     temp(af::span, af::span, af::span, 1) * 2 +
                     temp(af::span, af::span, af::span, 2));
    
    // Above code equivalent to
    // auto xx = af::sum(product * ex_ * ex_, 2);
    // auto xy = af::sum(product * ex_ * ey_, 2);
    // auto yy = af::sum(product * ey_ * ey_, 2);

    // sigma = af::sqrt(xx * xx + xy * xy * 2 + yy * yy);
}

af::array generate_image(size_t width, size_t height, const SimulationQuantities& simulation)
{
    const auto& ux = simulation.ux;
    const auto& uy = simulation.uy;
    const auto& boundaries = simulation.set_boundaries;
    auto  velocity = simulation.velocity;

    float image_scale = static_cast<float>(width) / static_cast<float>(simulation.grid_width);

    // Relative Flow speed at each cell
    auto val = af::sqrt(ux * ux + uy * uy) / velocity;

    val = af::select(val == 0 && boundaries, -1, val);

    // Scaling and interpolating flow speed to the window size
    if (image_scale > 1.0)
        val = af::approx2(val, af::iota(width, af::dim4(1, height)) / image_scale, af::iota(height, af::dim4(1, width)).T() / image_scale);
    
    // Flip image
    val = val.T();

    auto image = af::constant(0, height, width, 3);
    auto image2 = image;

    // Add custom coloring
    image(af::span, af::span, 0) = val * 2;
    image(af::span, af::span, 1) = val * 2;
    image(af::span, af::span, 2) = 1.0 - val * 2;

    image2(af::span, af::span, 0) = 1;
    image2(af::span, af::span, 1) = -2*val + 2;
    image2(af::span, af::span, 2) = 0;
    
    image = af::select(af::tile(val, 1, 1, 3) > 0.5, image2, image);
    image = af::select(af::tile(val, 1, 1, 3) >= 0, image, 0);

    return image;
}

int main(int argc, char** argv)
{
    int device = argc > 1 ? std::atoi(argv[1]) : 0;
    af::setDevice(device);
    af::info();

    // Simulation Parameters
    const size_t len = 128;
    const size_t grid_width = len;
    const size_t grid_height = len;
    float scale = 1.0f;
    
    // Forge window initialization
    int height = static_cast<int>(grid_width * scale);
    int width  = static_cast<int>(grid_height * scale);
    af::Window window(height, width, "Driven Cavity Flow");

    int frame_count = 0;
    int max_frames = 20000;
    int simulation_frames = 100;
    float total_time = 0;
    float total_time2 = 0;

    double avga = 0;
    double avga2 = 0;
    double avgb = 0;
    double avgb2 = 0;
    
    // Simulation code
    const float density = 2.7f;
    const float velocity = 0.35f;
    const float reynolds = 1e5f;

    const char*           ux_image = "../../default_ux.bmp";
    const char*           uy_image = "../../default_uy.bmp";
    const char* set_boundary_image = "../../default_boundary.bmp";
    
    // const char*           ux_image = "../../left_tesla_ux.bmp";
    // const char*           uy_image = "../../left_tesla_uy.bmp";
    // const char* set_boundary_image = "../../left_tesla_boundary.bmp";

    // const char*           ux_image = "../../right_tesla_ux.bmp";
    // const char*           uy_image = "../../right_tesla_uy.bmp";
    // const char* set_boundary_image = "../../right_tesla_boundary.bmp";

    auto simulation = SimulationQuantities(grid_width, grid_height, density, velocity, reynolds);
    
    initialize(simulation, ux_image, uy_image, set_boundary_image);

    while(!window.close() && frame_count != max_frames)
    {

        af::sync();
        auto begin = std::chrono::high_resolution_clock::now();

        collide_stream(simulation);
        
        af::sync();
        auto middle = std::chrono::high_resolution_clock::now();
        
        update(simulation);
        
        af::sync();
        auto end = std::chrono::high_resolution_clock::now();

        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(middle - begin).count();
        auto dur2 = std::chrono::duration_cast<std::chrono::microseconds>(end - middle).count();
        auto total = dur + dur2;

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        avga += dur;
        avga2 += dur * dur;
        avgb += dur2;
        avgb2 += dur2 * dur2;

        total_time += duration;
        total_time2 += duration * duration;

        if (frame_count % simulation_frames == 0)
        {
            auto image = generate_image(width, height, simulation);

            // Display colored image
            window.image(image);

            float avg_time = total_time / (float) simulation_frames;
            float stdv_time = std::sqrt(total_time2 *simulation_frames - total_time * total_time) / (float)simulation_frames;

            std::cout << "Average Simulation Step Time: (" << avg_time << " +/- " << stdv_time
                    << ") us; Total simulation time: " << total_time << " us; Simulation Frames: " << simulation_frames
                    << std::endl;

            total_time = 0;
            total_time2 = 0;
        }
        
        frame_count++;
    }

    std::cout << "First Part: (" << avga / frame_count << " +- " << std::sqrt(avga2 * frame_count - avga * avga) / frame_count
              << ") us; Second Part: (" << avgb / frame_count << " +- " << std::sqrt(avgb2 * frame_count - avgb * avgb) / frame_count
              << ") us; Total: (" << (avga + avgb) / frame_count << " +- " << std::sqrt(avga2 * frame_count - avga * avga + avgb2 * frame_count - avgb * avgb) / frame_count
              << ") us" << std::endl;

    return 0;
}