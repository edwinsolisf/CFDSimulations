/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>

#include <array>
#include <chrono>
#include <iostream>

struct Simulation
{
    af::array posX;
    af::array posY;
    af::array velX;
    af::array velY;
    af::array density;
    af::array aX;
    af::array aY;

    uint32_t particle_count;
    uint32_t box_count;

    double base_density;
    double viscosity;
    double pressure_constant;
    double time_step;
    double restitution;
    double gravity;
    double radius;

    double min_x;
    double max_x;
    double min_y;
    double max_y;
};

Simulation create_simulation(uint32_t particle_count, uint32_t box_count, double time_step, double restitution,
                             double base_density, double temperature, double viscosity, double smooth_radius, double gravity,
                             std::array<double, 2> range_x, std::array<double, 2> range_y)
{
    Simulation sim;

    auto& posX    = sim.posX;
    auto& posY    = sim.posY;
    auto& density = sim.density;
    auto& velX    = sim.velX;
    auto& velY    = sim.velY;
    auto& aX      = sim.aX;
    auto& aY      = sim.aY;

    auto& min_x    = sim.min_x;
    auto& max_x    = sim.max_x;
    auto& min_y    = sim.min_y;
    auto& max_y    = sim.max_y;

    sim.particle_count = particle_count;
    sim.restitution = restitution;
    sim.base_density = base_density;
    sim.box_count = box_count;
    sim.radius = smooth_radius;
    sim.pressure_constant = temperature;
    sim.viscosity = viscosity;
    sim.gravity = gravity;
    sim.time_step = time_step;
    
    min_x = range_x[0];
    max_x = range_x[1];
    min_y = range_y[0];
    max_y = range_y[1]; 

    posX = af::randu(particle_count) * (max_x - min_x) * 1.0 / 4.f + (max_x - min_x) * (3./8.) + min_x;
    posY = af::randu(particle_count) * (max_y - min_y) * 1.0 / 4.f + (max_x - min_x) * (3./4.) + min_y;

    velX = af::constant(0, particle_count);
    velY = af::constant(0, particle_count);

    density = af::constant(base_density, particle_count);
    aX = af::constant(0, particle_count);
    aY = af::constant(0, particle_count);

    return sim;
};


void compute_acceleration(Simulation& sim)
{
    auto& posX    = sim.posX;
    auto& posY    = sim.posY;
    auto& density = sim.density;
    auto& velX    = sim.velX;
    auto& velY    = sim.velY;
    auto& aX      = sim.aX;
    auto& aY      = sim.aY;
    auto& h       = sim.radius;

    auto g        = sim.gravity;
    auto min_x    = sim.min_x;
    auto max_x    = sim.max_x;
    auto min_y    = sim.min_y;
    auto max_y    = sim.max_y;

    auto density0  = sim.base_density;
    auto viscosity = sim.viscosity;
    auto box_count = sim.box_count;
    auto particle_count = sim.particle_count;
    auto pressure_const = sim.pressure_constant;

    const double normPoly = 4. / (af::Pi * std::pow(h, 8));
    const double normSpiky = 10. / (af::Pi * std::pow(h,5));
    const double normVisc = 10. / (3. * af::Pi * std::pow(h, 2));
    const double m = density0 * af::Pi * std::pow(h, 2);

    uint32_t avg_count = particle_count / box_count;

    auto indices = af::iota(particle_count);
    af::array newPosX, newPosY;
    af::array temp(particle_count);

    af::sort(newPosX, indices, posX);
    temp(indices) = posY;
    af::sort(newPosY, indices, temp);

    auto boxPosX = af::unwrap(newPosX, avg_count, 1, avg_count, 1);
    boxPosX = af::reorder(boxPosX, 0, 2, 1);

    auto boxPosY = af::unwrap(newPosY, avg_count, 1, avg_count, 1);
    boxPosY = af::reorder(boxPosY, 0, 2, 1);

    auto boxIndices = af::unwrap(indices, avg_count, 1, avg_count, 1);

    auto tbPosX = af::tile(boxPosX, af::dim4(1, avg_count));
    auto tbPosY = af::tile(boxPosY, af::dim4(1, avg_count));

    auto dx = tbPosX - tbPosX.T();
    auto dy = tbPosY - tbPosY.T();
    auto dr2 = dx * dx + dy * dy;
    auto valid = dr2 < (h * h);
    auto sparse_valid = valid.as(f32) - af::identity(avg_count, avg_count, f32);
    auto other_indices = af::where(af::flat(sparse_valid));
    auto pair_count = other_indices.dims()[0];

    if (other_indices.isempty())
    {
        aX = af::constant(0, particle_count);
        aY = af::constant(-g, particle_count);
        
        return;
    }

    auto row_indices = other_indices / particle_count;
    auto sum_sparse = [&row_indices, particle_count](const af::array& arr)
    {
        af::array nnz_indices;
        af::array nnz_vals;

        af::sumByKey(nnz_indices, nnz_vals, row_indices, arr);

        af::array all_vals = af::constant(0, particle_count);
        all_vals(nnz_indices) = nnz_vals;

        return all_vals;
    };

    auto box_indices = af::lookup(af::flat(boxIndices), other_indices);

    dx = af::lookup(af::flat(dx), other_indices);
    dy = af::lookup(af::flat(dy), other_indices);
    dr2 = af::lookup(af::flat(dr2), other_indices);
    auto dr = af::sqrt(dr2);

    auto boxVelX = af::unwrap(velX(indices), avg_count, 1, avg_count, 1);
    boxVelX = af::reorder(boxVelX, 0, 2, 1);
    boxVelX = af::tile(boxVelX, af::dim4(1, avg_count));
    auto delta_vx = boxVelX - boxVelX.T();
    delta_vx = af::lookup(af::flat(delta_vx), other_indices);

    auto boxVelY = af::unwrap(velY(indices), avg_count, 1, avg_count, 1);
    boxVelY = af::reorder(boxVelY, 0, 2, 1);
    boxVelX = af::tile(boxVelY, af::dim4(1, avg_count));
    auto delta_vy = boxVelY - boxVelY.T();
    delta_vy = af::lookup(af::flat(delta_vy), other_indices);

    density = sum_sparse(m * normPoly * af::pow(h * h - dr2, 3));
    density += af::constant(m * normPoly * std::pow(h, 6), particle_count);
    density = density(indices);

    auto boxRho = af::unwrap(density(indices), avg_count, 1, avg_count, 1);
    boxRho = af::reorder(boxRho, 0, 2, 1);
    boxRho = af::tile(boxRho, af::dim4(1, avg_count));
    auto rho = af::lookup(af::flat(boxRho), other_indices);
    auto rho_T = af::lookup(af::flat(boxRho.T()), other_indices);

    auto delta_h2r2 = h * h - dr2;
    auto kernelSpikyprime = normSpiky * -3 * af::pow(h - dr, 2);
    auto kernelPolyprime = -6 * normPoly * dr * af::pow(delta_h2r2, 2);
    auto kernelViscprime2 = normVisc * 9 * (h - dr) / (2 * std::pow(h, 3));

    auto gradP = kernelSpikyprime * pressure_const * (rho + rho_T - density0 * 2) / (rho * rho_T);

    auto kernelPolyprime2 = normPoly * (24 * dr2 * delta_h2r2 - 6 * af::pow(delta_h2r2, 2));
    auto delta_r_delta_v_frac = (dx * delta_vx + dy * delta_vy) / dr;
        
    auto rhodt = kernelPolyprime * delta_r_delta_v_frac;
    auto rhogradx = -kernelPolyprime * dx / dr;
    auto rhogrady = -kernelPolyprime * dy / dr;
    
    auto rhodtgradx = (delta_r_delta_v_frac * dx * (kernelPolyprime / dr - kernelPolyprime2) - kernelPolyprime * delta_vx) / dr;
    auto rhodtgrady = (delta_r_delta_v_frac * dy * (kernelPolyprime / dr - kernelPolyprime2) - kernelPolyprime * delta_vy) / dr;
    
    auto avg_density = density0;
    auto graddivx = (rhodt * rhogradx / avg_density - rhodtgradx) / avg_density;
    auto graddivy = (rhodt * rhogrady / avg_density - rhodtgrady) / avg_density;

    af::replace(graddivx, !af::isNaN(graddivx), -6 * normPoly * std::pow(h, 4) * delta_vx / avg_density);
    af::replace(graddivy, !af::isNaN(graddivy), -6 * normPoly * std::pow(h, 4) * delta_vy / avg_density);

    auto ax = (gradP * dx + viscosity * delta_vx * kernelViscprime2 / (rho * rho_T) + graddivx * viscosity / (3 * avg_density)) * m;
    auto ay = (gradP * dy + viscosity * delta_vy * kernelViscprime2 / (rho * rho_T) + graddivy * viscosity / (3 * avg_density)) * m;

    aX = sum_sparse(ax);
    aY = sum_sparse(ay) - g;

    // auto diffRho = af::sum(af::dense(af::sparse(particle_count, particle_count, rhodt * m, other_row_indices, other_col_indices)), 1);
    // af::replace(diffRho, !af::isNaN(diffRho), 0);
    // density += diffRho(indices) * sim.time_step;

    aX = aX(indices);
    aY = aY(indices);
}

void update(Simulation& sim)
{
    auto& posX    = sim.posX;
    auto& posY    = sim.posY;
    auto& velX    = sim.velX;
    auto& velY    = sim.velY;
    auto& aX      = sim.aX;
    auto& aY      = sim.aY;

    auto min_x = sim.min_x;
    auto max_x = sim.max_x;
    auto min_y = sim.min_y;
    auto max_y = sim.max_y;
    auto dt    = sim.time_step;

    auto restitution    = sim.restitution;
    auto particle_count = sim.particle_count;
    auto pressure_const = sim.pressure_constant;

    velX += aX * dt / 2.;
    velY += aY * dt / 2.;

    // Test for collision with funnel

    double xlftp = min_x + (max_x - min_x) * 5.0 / 16.0;
    double ylftp = min_y + (max_y - min_y) * 3.0 / 4.0; 
    double xlfbt = min_x + (max_x - min_x) * 7.0 / 16.0;
    double ylfbt = min_y + (max_y - min_y) * 1.0 / 2.0; 
    double xrhtp = min_x + (max_x - min_x) * 11.0 / 16.0;
    double yrhtp = min_y + (max_y - min_y) * 3.0 / 4.0;
    double xrhbt = min_x + (max_x - min_x) * 9.0 / 16.0;
    double yrhbt = min_y + (max_y - min_y) * 1.0 / 2.0;

    auto tlf = (xlftp - posX) * (-velY) - (ylftp - posY) * (-velX);
    auto ulf = (xlftp - posX) * (ylftp - ylfbt) - (ylftp - posY) * (xlftp - xlfbt);

    auto trh = (xrhtp - posX) * (-velY) - (yrhtp - posY) * (-velX);
    auto urh = (xrhtp - posX) * (yrhtp - yrhbt) - (yrhtp - posY) * (xrhtp - xrhbt);

    auto max_lf = (xlftp - xlfbt) * (-velY) - (ylftp - ylfbt) * (-velX);
    auto max_rh = (xrhtp - xrhbt) * (-velY) - (yrhtp - yrhbt) * (-velX);

    // af::eval(tlf, ulf, trh, urh);
    // af::eval(max_lf, max_rh);

    auto m = -(ylftp - ylfbt) / (xlftp - xlfbt);
    auto dot_lf = (velX + velY * 1.0 / m) / (1 + std::pow(1.0 / m, 2));
    auto dot_rh = (velX - velY * 1.0 / m) / (1 + std::pow(1.0 / m, 2));

    auto condLf = !((tlf / max_lf) >= 0 && (tlf / max_lf) <= 1 && (ulf / max_lf) >= 0 && (ulf / max_lf) <= dt);
    auto condRh = !((trh / max_rh) >= 0 && (trh / max_rh) <= 1 && (urh / max_rh) >= 0 && (urh / max_rh) <= dt);

    af::replace(velX, condLf, velX - (1 + restitution) * dot_lf * 1.0);
    af::replace(velY, condLf, velY - (1 + restitution) * dot_lf * 1.0 / m);
    af::replace(velX, condRh, velX - (1 + restitution) * dot_rh * 1.0);
    af::replace(velY, condRh, velY - (1 + restitution) * dot_rh * -1.0 / m);

    // Move position 
    posX += velX * dt;
    posY += velY * dt;

    // Check for wall collision
    auto condXmin = posX > min_x;
    auto condXmax = posX < max_x;

    auto condYmin = posY > min_y;
    auto condYmax = posY < max_y;

    af::replace(posX, condXmin, min_x + 0.001);
    af::replace(posX, condXmax, max_x - 0.001);
    af::replace(posY, condYmin, min_y + 0.001);
    af::replace(posY, condYmax, max_y - 0.001);
    af::replace(velX, condXmin && condXmax, -velX * restitution + af::randu(particle_count) * 0.5e0 * std::sqrt(pressure_const));
    af::replace(velY, condYmin && condYmax, -velY * restitution + af::randu(particle_count) * 0.5e0 * std::sqrt(pressure_const));
    
    // Leap Frog Velocity
    velX += aX * dt / 2.;
    velY += aY * dt / 2.;
}

af::array generate_image(uint32_t width, uint32_t height, const Simulation& sim)
{
    const auto& posX = sim.posX;
    const auto& posY = sim.posY;

    auto particle_count = sim.particle_count;
    auto min_x = sim.min_x;
    auto max_x = sim.max_x;
    auto min_y = sim.min_y;
    auto max_y = sim.max_y;

    auto indices = af::sort(af::floor(posX * (width - 1) / (max_x - min_x)) * height + af::floor(posY * (height - 1) / (max_y - min_y))).as(s32);
    af::array valid_indices, valid_indices_count;

    af::sumByKey(valid_indices, valid_indices_count, indices, af::constant(1, particle_count, s32), 0);

    auto col_indices = (valid_indices % height).as(s32);

    af::array rows, row_count;
    af::sumByKey(rows, row_count, (valid_indices / height).as(s32), af::constant(1, valid_indices.dims()[0], s32));

    af::array row_indices = af::constant(0, width + 1, s32);
    row_indices(rows + 1) = row_count;
    row_indices = af::accum(row_indices);

    af::array sparse_image = af::sparse(width, height, af::constant(1, valid_indices.dims()[0]), row_indices, col_indices);

    auto tmp = af::dense(sparse_image);

    auto image = af::constant(0, width, height, 3);

    image(af::span, af::span, 0) = af::flip(tmp,1);

    double xlftp = min_x + (max_x - min_x) * 5.0 / 16.0;
    double ylftp = min_y + (max_y - min_y) * 3.0 / 4.0; 
    double xlfbt = min_x + (max_x - min_x) * 7.0 / 16.0;
    double ylfbt = min_y + (max_y - min_y) * 1.0 / 2.0; 
    double xrhtp = min_x + (max_x - min_x) * 11.0 / 16.0;
    double yrhtp = min_y + (max_y - min_y) * 3.0 / 4.0;
    double xrhbt = min_x + (max_x - min_x) * 9.0 / 16.0;
    double yrhbt = min_y + (max_y - min_y) * 1.0 / 2.0;

    uint32_t wlf = std::abs(xlftp - xlfbt) * width / (max_x - min_x);
    uint32_t hlf = std::abs(ylftp - ylfbt) * height / (max_y - min_y);

    auto t = af::iota(wlf + hlf) / (wlf + hlf - 1);
    auto wlfindices = -af::floor(t * wlf) + xlfbt * width / (max_x - min_x);
    auto hlfindices = -af::floor(t * hlf) + ylfbt * height / (max_y - min_y);

    auto wrhindices = af::floor(t * wlf) + xrhbt * width / (max_x - min_x);
    auto hrhindices = -af::floor(t * hlf) + yrhbt * height / (max_y - min_y);

    image(wlfindices + hlfindices * width + width * height) = 1;
    image(wrhindices + hrhindices * width + width * height) = 1;

    return image.T();
}

void smh_cfd_demo()
{
    int width = 800;
    int height = 600;
    double scale = 2.0f;
    
    // Forge window initialization
    int w_height = static_cast<int>(width * scale);
    int w_width  = static_cast<int>(height * scale);
    
    af::Window window(w_height, w_width, "Smothered Particle Hydrodynamics");

    int frame_count = 0;
    int max_frames = 20000;
    int simulation_frames = 20;
    double total_time = 0;
    double total_time2 = 0;

    uint32_t particle_count = 2000;
    uint32_t box_count = 20;
    double restitution = 0.5f;
    double fluid_density = 1000;
    double viscosity = 0.1e0;
    double temperature = 1.0e0f;
    double h = 0.025;
    double g = 1.;
    double dt = 0.005f;

    Simulation sim = create_simulation(particle_count, box_count, dt, restitution, fluid_density, temperature, viscosity, h, g,
                                    {0.f, 10.f}, {0.f, 10.f});

    while (!window.close() && frame_count < max_frames)
    {
        af::sync();
        auto begin = std::chrono::high_resolution_clock::now();

        compute_acceleration(sim);
        af::sync();

        auto middle = std::chrono::high_resolution_clock::now();

        update(sim);
        af::sync();

        auto end = std::chrono::high_resolution_clock::now();

        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        total_time += time;
        total_time2 += time * time;

        // Every number of `simulation_frames` display the last computed frame to the screen
        if (frame_count % simulation_frames == 0)
        {
            auto image = generate_image(width, height, sim);

            // Rescale image to window size
            if (scale != 1.0)
            {
                image = af::scale(image, scale, scale);
            }

            // Display colored image
            window.image(image);

            double avg_time = total_time / (double) simulation_frames;
            double stdv_time = std::sqrt(total_time2 *simulation_frames - total_time * total_time) / (double)simulation_frames;

            std::cout << "Average Simulation Step Time: (" << avg_time << " +/- " << stdv_time
                    << ") us; Total simulation time: " << total_time << " us; Simulation Frames: " << simulation_frames
                    << std::endl;

            total_time = 0;
            total_time2 = 0;
        }
        
        frame_count++;
    }
}

int main(int argc, char** argv)
{
    int device = argc > 1 ? std::atoi(argv[1]) : 0;

    try
    {
        af::setDevice(device);
        af::info();

        std::cout << "** ArrayFire Smothered Particle CFD Simulation Demo\n\n";

        smh_cfd_demo();
    }
    catch(const af::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}