#include <array>
#include <chrono>
#include <thread>
#include <iostream>

#include <arrayfire.h>

std::array<af::array, 3> grad2(const af::array& vals, double delta_x, double delta_y)
{
    std::array<af::array, 3> results;
    af::array& dx2 = results[0];
    af::array& dxy = results[1];
    af::array& dy2 = results[2];

    auto right = af::shift(vals, 1, 0);
    auto left = af::shift(vals, -1, 0);
    auto up = af::shift(vals, 0, 1);
    auto down = af::shift(vals, 0, -1);
    auto& center = vals;

    double vals[] = {
        0.0, 0.0, 0.0,
        1.0, -2.0, 1.0,
        0.0, 0.0, 1.0
    };

    // auto filter = af::array(3, 3, vals);
    // dx2 = af::convolve2(vals, filter);
    // dy2 = af::convolve2(vals, filter.T());

    return results;
}

std::pair<af::array, af::array> gradient(const af::array& vals, double delta_x, double delta_y)
{
    af::array dx;
    af::array dy;

    af::grad(dx, dy, vals);

    return { dx / delta_x , dy / delta_y }; 
}

af::array divergence(const af::array& x_vals, const af::array& y_vals, double delta_x, double delta_y)
{
    af::array dx_x;
    af::array dy_x; 

    af::array dx_y;
    af::array dy_y;

    af::grad(dx_x, dy_x, x_vals);
    af::grad(dx_y, dy_y, y_vals);

    return (dx_x / delta_x) + (dy_y / delta_y);
}

af::array laplacian(const af::array& vals, double delta_x, double delta_y)
{
    af::array dx;
    af::array dy; 

    af::array dx2;
    af::array dy2;
    af::array dxdy;

    af::grad(dx, dy, vals);
    af::grad(dx2, dxdy, dx);
    af::grad(dy2, dxdy, dy);

    return dx2 / (delta_x * delta_x) + dy2 / (delta_y * delta_y);
}

std::pair<af::array, af::array> dot_gradient(const af::array& ax_vals, const af::array& ay_vals, const af::array& bx_vals, const af::array& by_vals, double delta_x, double delta_y)
{
    af::array dx_bx;
    af::array dy_bx; 

    af::array dx_by;
    af::array dy_by;

    af::grad(dx_bx, dy_bx, bx_vals);
    af::grad(dx_by, dy_by, by_vals);

    // return { ax_vals * (dx_bx / delta_x + dy_bx / delta_y) , ay_vals * (dx_by / delta_x + dy_by / delta_y) };
    return { ax_vals * dx_bx / delta_x + ay_vals * dy_bx / delta_y , ax_vals * dx_by / delta_x + ay_vals * dy_by / delta_y };
}

std::pair<af::array, af::array> div_grad(const af::array& x_vals, const af::array& y_vals, double delta_x, double delta_y)
{
    af::array dx_x;
    af::array dy_x; 

    af::array dx_y;
    af::array dy_y;

    af::grad(dx_x, dy_x, x_vals);
    af::grad(dx_y, dy_y, y_vals);

    af::array dx2_x;
    af::array dxy_x;
    
    af::array dxy_y;
    af::array dy2_y;
    
    af::grad(dx2_x, dxy_x, dx_x);
    af::grad(dxy_y, dy2_y, dy_y);

    // return (dx_x / delta_x) + (dy_y / delta_y);
    return { dx2_x / (delta_x * delta_x) + dxy_y / (delta_x * delta_y) , dxy_x / (delta_x * delta_y) + dy2_y / (delta_y * delta_y) };
}

af::array average(const af::array& array)
{
    auto dim = array.dims();
    auto width = dim[0];
    auto height = dim[1];
    // auto no_borders = array;

    // no_borders.col(0) = 0;
    // no_borders.col(width - 1) = 0;
    // no_borders.row(0) = 0;
    // no_borders.row(height - 1) = 0;

    // auto total = af::constant(0, width, height);
    // for (int i = 1; i < 9; ++i)
    // {
    //     int xshift = i % 3 - 1;
    //     int yshift = (i / 3) % 3 - 1;

    //     total += af::shift(no_borders, xshift, yshift);
    // }
    // total += array;

    // auto weights = af::constant(9, width, height);
    // weights.row(0) = 6;
    // weights.row(height - 1) = 6;
    // weights.col(0) = 6;
    // weights.col(width - 1) = 6;
    // weights(0, 0) = 4;
    // weights(0, height - 1) = 4;
    // weights(width - 1, 0) = 4;
    // weights(width - 1, height - 1) = 4;

    // return total / weights;
    // af::array result = af::convolve2(array, af::constant(1.0/9.0, 3, 3)) / af::convolve2(af::constant(1, width, height), af::constant(1.0/9.0, 3, 3));
    double weights[] = {
        1./36., 4./36., 1./36.,
        4./36., 16./36., 4./36.,
        1./36., 4./36., 1./36.,
    };
    auto filter = af::array(3, 3, weights);
    auto base = af::convolve2(af::constant(1, width, height), filter);
    auto result = af::convolve2(array, filter) / base;
    
    return result;
}

af::array boundary(const af::array& array)
{
    double vals[] = {
        0.25, 0.0, -0.25,
        0.5, 0.0, -0.5,
        0.25, 0.0, -0.25
    };

    af::array dx, dy;

    auto result = af::convolve2(array, af::array(3, 3, vals).T());
    
    // auto l = af::shift(result, 0, -1);
    // auto r = af::shift(result, 0, 1);

    // return average(result);
    auto temp = result + 0.5;
    // return af::exp(-temp * temp * 10);
    return result;
    // return (l + r - 2 * result);
}

int main(int argc, char** argv)
{
    int device = argc > 2 ? std::atoi(argv[1]) : 0;

    af::setDevice(device);

    int frames = 10000;
    int size = 64;

    af::Window window(size, size);
    af::Window window2(size, size);

    double visc = 0.1;
    double pressure = 0.01;
    double dt = 1e-3;
    double delta_x = 0.1;
    double delta_y = 0.1;
    double g = 1;
    double min_rho = 0.001;


    af::array ux = af::constant(0, size, size, f64);
    af::array uy = af::constant(0, size, size, f64);
    // af::array rho = af::select(af::iota(af::dim4(1,size), size) > size / 2, af::constant(1, size, size, f64), af::constant(0, size, size, f64));
    af::array rho = af::constant(1, size, size, f64);
    // rho = af::select(af::iota(af::dim4(1,size), size) > size / 3, af::constant(3, size, size, f64), rho);

    for (int i = 0; i < size; ++i)
    {
        if (i > size / 2)
        {
            rho.col(i) = min_rho;
        }
        // else if (i > size / 3.0)
        // {
        //     rho.col(i) = 0.5;
        // }

        // if (i > size / 2)
        //     ux.row(i) = -1;
        // else
        //     ux.row(i) = 1;
    }

    // for (int j = 0; j < size; ++j)
    // {
    //     if (j > size / 2.0 || j < size / 3.0)
    //         ux.col(j) = 0;
    // }

    for (int i = 0; i < frames && !window.close(); ++i)
    {
        auto avg_rho = average(rho);
        auto avg_ux = average(rho * ux) / avg_rho;
        auto avg_uy = average(rho * uy) / avg_rho;

        avg_rho = af::select(af::isNaN(avg_rho), min_rho, avg_rho);
        avg_ux = af::select(af::isNaN(avg_ux), 0, avg_ux);
        avg_uy = af::select(af::isNaN(avg_uy), 0, avg_uy);

        auto rho_grad = gradient(avg_rho, delta_x, delta_y);
        // auto rho_grad = gradient(rho, delta_x, delta_y);
        auto rho_dx = rho_grad.first;
        auto rho_dy = rho_grad.second;

        auto convec = dot_gradient(avg_ux, avg_uy, avg_ux, avg_uy, delta_x, delta_y);
        // auto convec = dot_gradient(ux, uy, ux, uy, delta_x, delta_y);
        auto convec_x = convec.first;
        auto convec_y = convec.second;

        auto div_grad_u = div_grad(avg_ux, avg_ux, delta_x, delta_y);
        auto dg_ux = div_grad_u.first;
        auto dg_uy = div_grad_u.second;

        // auto ax = visc * average(laplacian(ux, delta_x, delta_y)) - (pressure * rho_dx) / avg_rho - convec_x + (dg_ux * visc / 3.0);
        // auto ay = visc * average(laplacian(uy, delta_x, delta_y)) - (pressure * rho_dy) / avg_rho - convec_y + (dg_uy * visc / 3.0) - g;
        auto ax = visc * laplacian(avg_ux, delta_x, delta_y) - (pressure * rho_dx) / (avg_rho) - convec_x;
        auto ay = visc * laplacian(avg_uy, delta_x, delta_y) - (pressure * rho_dy) / (avg_rho) - convec_y + g;
        auto rhodt = -(avg_ux * rho_dx + avg_uy * rho_dy);
        // auto rhodt = -divergence(avg_rho * avg_ux, avg_rho * avg_uy, delta_x, delta_y);
        // auto rhodt = - (avg_rho * divergence(avg_ux, avg_uy, delta_x, delta_y) + avg_ux * rho_dx + avg_uy * rho_dy);
        auto grad_rhodt = gradient(rhodt, delta_x, delta_y);
        auto grad_rhodtx = grad_rhodt.first;
        auto grad_rhodty = grad_rhodt.second;
        auto rhodt2 = - pressure * laplacian(avg_rho, delta_x, delta_y) - rho_dx * convec_x - rho_dy * (convec_y + g)
                        - grad_rhodtx * avg_ux - grad_rhodty * avg_uy;
        // auto rhodt = -(ux * rho_dx + uy * rho_dy);

        auto a_gradu = dot_gradient(ax, ay, avg_ux, avg_uy, delta_x, delta_y);
        auto agux = a_gradu.first;
        auto aguy = a_gradu.second;
        auto u_grada = dot_gradient(avg_ux, avg_uy, ax, ay, delta_x, delta_y);
        auto ugax = u_grada.first;
        auto ugay = u_grada.second;
        auto avg_rho2 = average(rho * rho);
        auto ax2 = -agux - ugax + visc * laplacian(ax, delta_x, delta_y) - pressure * grad_rhodtx / avg_rho + pressure * rhodt * rho_dx / avg_rho2;
        auto ay2 = -aguy - ugay + visc * laplacian(ay, delta_x, delta_y) - pressure * grad_rhodty / avg_rho + pressure * rhodt * rho_dy / avg_rho2;
    
        ax = af::select(af::isNaN(ax), 0, ax);
        ay = af::select(af::isNaN(ay), 0, ay);
        rhodt = af::select(af::isNaN(rhodt), 0, rhodt);
        rhodt2 = af::select(af::isNaN(rhodt2), 0, rhodt2);

        ux = ux + ax * dt + ax2 * (dt * dt / 2.0);
        uy = uy + ay * dt + ay2 * (dt * dt / 2.0);
        // rhodt = rhodt + rhodt2 * dt;
        rho = rho + rhodt * dt + rhodt2 * (dt * dt / 2.0);


        ux = af::select(af::isNaN(ux), 0, ux);
        uy = af::select(af::isNaN(uy), 0, uy);
        rho = af::select(af::isNaN(rho), min_rho, rho);
        rho = af::select(rho < min_rho, min_rho, rho);

        // std::cout << af::mean<double>(divergence(ux, uy, delta_x, delta_y)) << std::endl;

        rho.col(0) = 0;
        rho.col(1) = 0;
        // rho.col(size - 1) = 0;

        ux.row(0) = 0.;
        uy.row(0) = 1.e-1;

        // ux.row(1) = af::select(ux.row(1) > 0, -af::array(ux.row(1)), ux.row(1));

        ux.row(size - 1) = 0;
        uy.row(size - 1) = 1.e-1;
    
        // ux.row(size - 2) = af::select(ux.row(size - 2) > 0, ux.row(size - 2), -af::array(ux.row(size - 2)));

        // uy.col(size - 1) = 0;
        // uy.col(size - 2) = af::select(uy.col(size - 2) < 0, uy.col(size - 2), -af::array(uy.col(size - 2)));

        // rho.col(0) = 2;
        // rho.col(size - 1) = 2;

        auto image = af::constant(0, size, size, 3, f32);
        // if (true)
        {
            // image = af::tile(af::abs(divergence(ux, uy, delta_x, delta_y)), af::dim4(1, 1, 3)) * 1.0f;
            auto temp = divergence(ux, uy, delta_x, delta_y) * rho;
            // auto temp = boundary(rho);
            temp = temp.T();
            image(af::span, af::span, 0) = -af::select(temp < 0, temp, 0);
            image(af::span, af::span, 1) = af::select(temp > 0, temp, 0);
            image(af::span, af::span, 2) = 0;

            window2.image(image);
        }
        //else
        {
            auto image2 = af::constant(0, size, size, 3);
            image(af::span, af::span, 0) = rho.T() * 2;
            image(af::span, af::span, 1) = rho.T() * 2;
            image(af::span, af::span, 2) = 1.0 - rho.T() * 2;
        
            image2(af::span, af::span, 0) = 1;
            image2(af::span, af::span, 1) = -2*rho.T() + 2;
            image2(af::span, af::span, 2) = 0;
                
            image = af::select(af::tile(rho.T(), 1, 1, 3) > 0.5, image2, image);
        }
        

        window.image(image);

        using namespace std::chrono_literals;
        // std::this_thread::sleep_for(50ms);
    }

    return 0;
}