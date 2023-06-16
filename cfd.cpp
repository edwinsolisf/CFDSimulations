/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// This is a Computational Fluid Dynamics Simulation using the Lattice Boltzmann Method
// For this simulation we are using D2N9 (2 dimensions, 9 neighbors) with bounce-back boundary conditions
// For more information on the simulation equations,
// check out https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods#Mathematical_equations_for_simulations

#include <chrono>
#include <iostream>
#include <thread>

#include <arrayfire.h>

// Simulation Parameters
const size_t len = 128;
const size_t ycount = len;
const size_t xcount = len;

// Fluid Parameters
const float density = 2.7;
const float velocity = 0.05;
const float reynolds = 1e5;
const float viscosity = velocity * std::sqrt(xcount * ycount) / reynolds;

// Array Quantities
af::array ex;
af::array ey;
af::array wt;

af::array ex_T;
af::array ey_T;
af::array wt_T;

af::array rho;
af::array sigma;
af::array ux;
af::array uy;

af::array feq;
af::array f;
af::array fnew;

af::array ex_;
af::array ey_;

const float ex_vals[] = {
    0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0
};

const float ey_vals[] = {
    0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0
};

const float wt_vals[] = {
    16.0 / 36.0, 4.0 /  36.0, 4.0 / 36.0, 4.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

const int oppos_vals[] = {
    0, 3, 4, 1, 2, 7, 8, 5, 6
};

/**
 * Creates all the af::arrays that will be used throughout the program
 * 
 * In this function, the initial conditions and boundary conditions of the simulation are set
 * 
*/
void initialize()
{
    ex = af::array(1, 1, 9, ex_vals);
    ey = af::array(1, 1, 9, ey_vals);
    wt = af::array(1, 1, 9, wt_vals);

    wt_T = af::moddims(wt, af::dim4(1,9));

    rho = af::constant(density, xcount, ycount, f32);
    sigma = af::constant(0, xcount, ycount, f32);
    ux = af::constant(0, xcount, ycount, f32);
    uy = af::constant(0, xcount, ycount, f32);

    // This initializes the velocity field
    ux(af::span, 0) = velocity;
    uy(af::span, 0) = 0.1;
    ux(af::span, ycount - 1) = -velocity;
    uy(af::span, ycount - 1) = -0.1;
    ux(0, af::span) = 0.1;
    uy(0, af::span) = -velocity;
    ux(xcount - 1, af::span) = -0.1;
    uy(xcount - 1, af::span) = velocity;

    feq = af::constant(0, xcount, ycount, 9);  
    f   = af::constant(0, xcount, ycount, 9);  
    fnew = af::constant(0, xcount, ycount, 9);

    ex_ = af::tile(ex, xcount, ycount, 1);
    ey_ = af::tile(ey, xcount, ycount, 1);

    // Initialization of the distribution function
    auto edotu = ex_ * ux + ey_ * uy;
    auto udotu = ux * ux + uy * uy;

    feq = rho * wt * ((edotu * edotu * 4.5) - (udotu * 1.5) + (edotu * 3.0) + 1.0);
    f = feq;
    fnew = feq;
}

void collide_stream()
{
    const float tau = 0.5 + 3.0 * viscosity;
    const float csky = 0.16;

    auto edotu = ex_ * ux + ey_ * uy;
    auto udotu = ux * ux + uy * uy;

    // Compute the new distribution function
    feq = rho * wt * (edotu * edotu * 4.5 - udotu * 1.5 + edotu * 3.0 + 1.0);

    auto taut = af::sqrt(sigma * (csky * csky * 18.0 * 0.25) + (tau * tau * 0.25)) - (tau * 0.5);

    // Compute the shifted distribution functions
    auto fplus = f - (f - feq) / (taut + tau);

    af::array ux_top = ux.rows(0, 2).T();
    af::array ux_bottom = ux.rows(xcount - 3, xcount - 1).T();

    af::array uy_top = uy.rows(0, 2).T();
    af::array uy_bottom = uy.rows(xcount - 3, xcount - 1).T();

    auto ubdoute_top = af::array(ycount, 9);
    auto ubdoute_bot = af::array(ycount, 9);
    auto ubdoute_lft = af::array(xcount, 9);
    auto ubdoute_rht = af::array(xcount, 9);
    
    auto ux_pad = af::pad(ux, af::dim4(2,2,0,0), af::dim4(2,2,0,0), af::borderType::AF_PAD_SYM);

    // Compute new particle distribution according to the corresponding D2N9 weights
    for (int i = 0; i < 9; ++i)
    {
        int xshift = ex_vals[i];
        int yshift = ey_vals[i];

        fplus(af::span, af::span, i) = af::shift(fplus(af::span, af::span, i), xshift, yshift);
        
        // Computing u dot e at the each of the boundaries
        // ubdoute_top.col(i) = af::pad(ux_top, af::)
        ubdoute_top.col(i) = ux_top.col(1-xshift)        * ex_vals[i] + uy_top.col(1-xshift)        * ey_vals[i];
        ubdoute_bot.col(i) = ux_bottom.col(1-xshift)     * ex_vals[i] + uy_bottom.col(1-xshift)     * ey_vals[i];
        ubdoute_lft.col(i) = ux.col(1-yshift)            * ex_vals[i] + uy.col(1-yshift)            * ey_vals[i];
        ubdoute_rht.col(i) = ux.col(ycount - 2 - yshift) * ex_vals[i] + uy.col(ycount - 2 - yshift) * ey_vals[i];
    }

    // Keep the boundary conditions at the borders the same
    fplus.row(0)          = fnew.row(0);
    fplus.row(xcount - 1) = fnew.row(xcount - 1);
    fplus.col(0)          = fnew.col(0);
    fplus.col(ycount - 1) = fnew.col(ycount - 1);

    // Update the particle distribution
    fnew = fplus;

    // Computing bounce-back boundary conditions
    auto fnew_top = af::moddims(fplus(         1,  af::span, af::span), af::dim4(ycount, 9)) - 6.0 * density * wt_T * ubdoute_top;
    auto fnew_bot = af::moddims(fplus(xcount - 2,  af::span, af::span), af::dim4(ycount, 9)) - 6.0 * density * wt_T * ubdoute_bot;
    auto fnew_lft = af::moddims(fplus( af::span,          1, af::span), af::dim4(xcount, 9)) - 6.0 * density * wt_T * ubdoute_lft;
    auto fnew_rht = af::moddims(fplus( af::span, ycount - 2, af::span), af::dim4(xcount, 9)) - 6.0 * density * wt_T * ubdoute_rht;

    // Sets the values near the boundaries with the correct bounce-back boundary
    for (int i = 0; i < 9; ++i)
    {
        int xshift = ex_vals[i];
        int yshift = ey_vals[i];
        if (xshift == 1)
            fnew(         1,   af::span, oppos_vals[i]) = fnew_top(af::span, i);
        if (xshift == -1)
            fnew(xcount - 2,   af::span, oppos_vals[i]) = fnew_bot(af::span, i);
        if (yshift == 1)
            fnew(  af::span,          1, oppos_vals[i]) = fnew_lft(af::span, i);
        if (yshift == -1)
            fnew(  af::span, ycount - 2, oppos_vals[i]) = fnew_rht(af::span, i);
    }
}

/**
 * Updates the velocity field, density and strain at each point in the grid
*/
void update()
{
    f = fnew;

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

    auto product = fnew - feq;
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

int main(int argc, char** argv)
{
    int frame_count = 0;
    int max_frames = 20000;
    int simulation_frames = 100;
    float scale = 1.0f;
    float total_time = 0;
    float total_time2 = 0;

    double avga = 0;
    double avga2 = 0;
    double avgb = 0;
    double avgb2 = 0;

    // Forge window initialization
    af::Window window(ycount * scale, xcount * scale, "Hello world");

    // Simulation code

    initialize();

    while(!window.close() && frame_count != max_frames)
    {
        frame_count++;

        af::sync();
        auto begin = std::chrono::high_resolution_clock::now();

        collide_stream();
        
        af::sync();
        auto middle = std::chrono::high_resolution_clock::now();
        
        update();
        
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
            // Relative Flow speed at each cell
            auto val = af::sqrt(ux * ux + uy * uy)/velocity;

            // Scaling and interpolating flow speed to the window size
            if (scale > 1.0)
                val = af::approx2(val, af::iota(xcount * scale, af::dim4(1, ycount * scale)) / scale, af::iota(ycount * scale, af::dim4(1, xcount * scale)).T() / scale);
            
            // Flip image
            val = val.T();

            auto image = af::constant(0, ycount * scale, xcount * scale, 3);
            auto image2 = image;

            // Add custom coloring
            image(af::span, af::span, 0) = val * 2;
            image(af::span, af::span, 1) = val * 2;
            image(af::span, af::span, 2) = 1.0 - val * 2;
        
            image2(af::span, af::span, 0) = 1;
            image2(af::span, af::span, 1) = -2*val + 2;
            image2(af::span, af::span, 2) = 0;
            
            image = af::select(af::tile(val, 1, 1, 3) > 0.5, image2, image);

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
    }

    std::cout << "First Part: (" << avga / frame_count << " +- " << std::sqrt(avga2 * frame_count - avga * avga) / frame_count
              << ") us; Second Part: (" << avgb / frame_count << " +- " << std::sqrt(avgb2 * frame_count - avgb * avgb) / frame_count
              << ") us; Total: (" << (avga + avgb) / frame_count << " +- " << std::sqrt(avga2 * frame_count - avga * avga + avgb2 * frame_count - avgb * avgb) / frame_count
              << ") us" << std::endl;

    return 0;
}