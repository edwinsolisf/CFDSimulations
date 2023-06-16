#include <chrono>
#include <cmath>
#include <thread>
#include <iostream>

#include <arrayfire.h>

af::array posX;
af::array posY;
af::array velX;
af::array velY;
af::array density;
af::array aX;
af::array aY;

template<typename T>
constexpr T const_pow(T val, size_t power)
{
    return power ? val * const_pow(val, power - 1) : T{1};
}

constexpr uint32_t particle_count = 1000;
constexpr double x_min = 0;
constexpr double x_max = 10.;
constexpr double y_min = 0;
constexpr double y_max = 10.;
constexpr double pi = 3.141592653589793;
constexpr double viscosity = 2;
constexpr double pressure_const = 50;
constexpr double dt = 0.005;
constexpr double g = 1;
constexpr double restitution = 0.5;
constexpr double density0 = 1000;
uint32_t width = 800;
uint32_t height = 600;
double h = 0.05;
double normPoly = 4. / (pi * const_pow(h, 8));
double normSpiky = 10. / (pi * const_pow(h,5));
double normVisc = 10. / (3. * pi * const_pow(h, 2));
double m = density0 / (const_pow(h, 6) * normPoly);

auto time_point = std::chrono::high_resolution_clock::now();

void profile_begin()
{
    time_point = std::chrono::high_resolution_clock::now();
}

void profile_end()
{
    auto begin = time_point;
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";
}

void update_constants()
{
    normPoly = 4. / (pi * const_pow(h, 8));
    normSpiky = 10. / (pi * const_pow(h,5));
    normVisc = 10. / (3. * pi * const_pow(h, 2));
    m = density0 / (const_pow(h, 6) * normPoly);
}

void initialize()
{
    // posX = (af::randu(particle_count)) * (x_max - x_min) / 2.0;
    posX = af::iota(particle_count) * (x_max - x_min) / particle_count;
    posY = af::constant(y_max / 2, particle_count);
    // posY = (af::randu(particle_count)) * (y_max - y_min) / 2.0;

    velX = af::constant(0, particle_count);
    velY = af::constant(0, particle_count);
    density = af::constant(density0, particle_count);
    aX = af::constant(0, particle_count);
    aY = af::constant(0, particle_count);

    // af_print(posX);
    // af_print(posY);
}


void distance()
{
    af::array x = af::tile(posX, af::dim4(1, particle_count));
    af::array y = af::tile(posY, af::dim4(1, particle_count));
    af::array vx = af::tile(velX, af::dim4(1, particle_count));
    af::array vy = af::tile(velY, af::dim4(1, particle_count));

    af::array dx = x - x.T();
    af::array dy = y - y.T();

    af::array dr2 = dx * dx + dy * dy;
    af::array valid = dr2 < (h * h);

    auto sparse_valid = valid.as(f32) - af::identity(particle_count, particle_count, f32);
    af::array col_indices, row_indices;
    size_t pair_count = 0;

    af::array delta_vx = vx - vx.T();
    af::array delta_vy = vy - vy.T();

    if (af::anyTrue<bool>(sparse_valid != 0))
    {
        valid = af::select(af::identity(particle_count, particle_count) == 1, 0, valid);
        sparse_valid = af::sparse(valid.as(f32));
        col_indices = af::sparseGetColIdx(sparse_valid);
        row_indices = af::sparseGetRowIdx(sparse_valid);
        pair_count = col_indices.dims()[0];
        // std::cout << "modified" << std::endl;
        // std::cout << pair_count << '\n';
        // af_print(sparse_valid);
    }
    else
    {
        aX = af::constant(0, particle_count);
        aY = af::constant(-g, particle_count);

        return;
    }


    auto get_values = [&valid, pair_count](af::array& arr)
    {
        arr = af::select(valid, arr + af::randu(particle_count) * 0.000001, 0);
        // if (af::allTrue<bool>(af::iszero(arr)))
        //     arr = af::constant(0, pair_count);
        // else
        //     arr = af::sparseGetValues(af::sparse(arr));
        arr = af::sparseGetValues(af::sparse(arr));
        arr.eval();
    };

    // profile_begin();

    get_values(dx);
    get_values(dy);
    get_values(delta_vx);
    get_values(delta_vy);
    get_values(dr2);

    density = af::constant(m * normPoly * const_pow(h, 6), particle_count);
    density += af::sum(af::dense(af::sparse(particle_count, particle_count, m * normPoly * af::pow(h * h - dr2, 3), row_indices, col_indices)), 1);

    density.eval();

    auto dr = af::sqrt(dr2);
    auto rho = af::tile(density, af::dim4(1, particle_count));
    auto rho_T = rho.T();

    get_values(rho);
    get_values(rho_T);

    // profile_end();

    auto delta_h2r2 = h * h - dr2;
    auto kernelSpikyprime = normSpiky * -3 * af::pow(h - dr, 2);
    auto kernelPolyprime = -6 * normPoly * dr * af::pow(delta_h2r2, 2);
    auto kernelViscprime2 = normVisc * 9 * (h - dr) / (2 * std::pow(h, 3));

    // auto pressure_i = rho * pressure_const;
    // auto pressure_j = pressure_i.T();
    // auto gradP = kernelSpikyprime * (pressure_i / (sim.density[pi] ** 2) + pressure_j / (sim.density[pj] ** 2));
    // auto gradP = kernelSpikyprime * pressure_const * (1.0 / rho + 1.0 / rho.T());

    auto gradP = kernelSpikyprime * pressure_const * (rho + rho_T - density0 * 2) / (rho * rho_T);

    auto ax = (gradP * dx + viscosity * delta_vx * kernelViscprime2 / (rho * rho_T)) * m;
    auto ay = (gradP * dy + viscosity * delta_vy * kernelViscprime2 / (rho * rho_T)) * m;


    // profile_begin();
    aX = af::sum(af::dense(af::sparse(particle_count, particle_count, ax, row_indices, col_indices)), 1);
    aY = af::sum(af::dense(af::sparse(particle_count, particle_count, ay, row_indices, col_indices)), 1) - g;
    // profile_end();

    // aX = af::constant(0, particle_count);
    // aY = af::constant(-g, particle_count);

    if (pair_count > std::pow(particle_count * 0.1, 2))
    {
        h = std::sqrt(static_cast<double>(pair_count) / static_cast<double>(particle_count * particle_count)) * 0.1;
        std::cout << "h: " << h << "\n";
    }
    else if (pair_count < 15)
    {
        h *= 2;
        std::cout << "h: " << h << "\n";
    }
    update_constants();

    aX.eval();
    aY.eval();
}

void update()
{
    velX += aX * dt / 2.;
    velY += aY * dt / 2.;

    posX += velX * dt;
    posY += velY * dt;

    auto condXmin = posX > x_min;
    auto condXmax = posX < x_max;

    auto condYmin = posY > y_min;
    auto condYmax = posY < y_max;
    posX = af::select(condXmin, posX, x_min + 0.001);
    posX = af::select(condXmax, posX, x_max - 0.001);
    velX = af::select(condXmin, velX, -velX * restitution + af::randu(particle_count) * 0.01);
    velX = af::select(condXmax, velX, -velX * restitution + af::randu(particle_count) * 0.01);
    
    posY = af::select(condYmin, posY, y_min + 0.001);
    posY = af::select(condYmax, posY, y_max - 0.001);
    velY = af::select(condYmin, velY, -velY * restitution + af::randu(particle_count) * 0.01);
    velY = af::select(condYmax, velY, -velY * restitution + af::randu(particle_count) * 0.01);
    
    velX += aX * dt / 2.;
    velY += aY * dt / 2.;
}

af::array get_image()
{
    auto indices = af::sort(af::floor(posX * (width - 1) / (x_max - x_min)) * height + af::floor(posY * (height - 1) / (y_max - y_min))).as(s32);
    af::array valid_indices, valid_indices_count;
    // af_print(indices);
    af::sumByKey(valid_indices, valid_indices_count, indices, af::constant(1, particle_count, s32), 0);

    auto col_indices = (valid_indices % height).as(s32);
    // af_print(col_indices);
    // af_print((valid_indices / height).as(s32));
    af::array rows, row_count;
    af::sumByKey(rows, row_count, (valid_indices / height).as(s32), af::constant(1, valid_indices.dims()[0], s32));

    auto rows_array = std::vector<int>(rows.dims()[0]);
    auto row_count_array = std::vector<int>(rows.dims()[0]);
    auto row_index_array = std::vector<int>(width + 1, 0);

    rows.host(rows_array.data());
    row_count.host(row_count_array.data());

    // af_print(posX);
    // af_print(posY);
    // af_print(rows);
    // af_print(row_count);

    row_index_array[0] = 0;
    for (size_t i = 0; i < rows_array.size(); ++i)
    {
        auto row = rows_array[i];
        row_index_array[row + 1] = row_count_array[i];
    }

    auto row_indices = af::array(width + 1, row_index_array.data());
    row_indices = af::accum(row_indices);
    row_indices.eval();
    col_indices.eval();

    auto sparse_image = af::sparse(width, height, af::constant(1, valid_indices.dims()[0]), row_indices, col_indices);
    // af_print(af::dense(sparse_image));
    auto tmp = af::dense(sparse_image);
    tmp.eval();

    auto image = af::constant(0, width, height, 3);
    image(af::span, af::span, 0) = af::flip(tmp,1);
    // image(af::span, af::span, 1) = 1.f;
    // image(af::span, af::span, 2) = 1.f;

    image = image.T();
    image.eval();

    return image;
}

int main(int argc, char** argv)
{
    af::setDevice(0);

    af::Window window(width, height);

    // auto tmp = af::sparse(af::iota({4, 4}));
    auto tmp = af::sparse(af::identity(4, 4));
    auto v1 = af::sparseGetValues(tmp);   
    auto v2 = af::sparseGetRowIdx(tmp);
    auto v3 = af::sparseGetColIdx(tmp);

    float vals[] = {
        1.0, 0.0, 2.0, 3.0,
        4.0, 0.0, 0.0, 0.0,
        0.0, 5.0, 6.0, 0.0
    };

    uint32_t count = 10;
    auto reverse_iota = count - 1 - af::iota(count, 1, u32);
    auto iota = af::iota(count, 1, u32);
    auto result = af::array();

    af::sort(result, iota, reverse_iota);
    af_print(reverse_iota);
    af_print(iota);
    af_print(result);

    auto arr = af::array(4, 3, vals);
    af_print(arr);
    af_print(af::sparse(arr));
    // af_print(af::sqrt(af::sparse(arr)));

    initialize();

    int i = 0;

    while (!window.close())
    {
        distance();
        update();

        if (i++ / 10)
        {
            auto image = get_image();
            window.image(image);
            i = 0;
        }

        using namespace std::chrono_literals;
        // std::this_thread::sleep_for(16ms);
    }

    return 0;
}