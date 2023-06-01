#include <iostream>

#include <arrayfire.h>
#include <chrono>
#include <thread>

size_t ycount = 128;
size_t xcount = 128;
float density = 2.7;
float velocity = 0.05;
float reynolds = 1e3;
float viscosity = velocity * std::sqrt(xcount * ycount) / reynolds;

af::array ex;
af::array ey;
af::array wt;

af::array rho;
af::array sigma;
af::array ux;
af::array uy;

af::array feq;
af::array f;
af::array fnew;

af::array ex_;
af::array ey_;
af::array ux_;
af::array uy_;
af::array wt_;
af::array rho_;

float ex_vals[] = {
    0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0
};

float ey_vals[] = {
    0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0
};

float wt_vals[] = {
    16.0 / 36.0, 4.0 /  36.0, 4.0 / 36.0, 4.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

int oppos_vals[] = {
    0, 3, 4, 1, 2, 7, 8, 5, 6
};

void initialize()
{


    ex = af::array(1, 1, 9, ex_vals);
    ey = af::array(1, 1, 9, ey_vals);
    wt = af::array(1, 1, 9, wt_vals);

    rho = af::constant(density, xcount, ycount, f32);
    sigma = af::constant(0, xcount, ycount, f32);
    ux = af::constant(0, xcount, ycount, f32);
    uy = af::constant(0, xcount, ycount, f32);

    ux(af::span, 0) = velocity;
    // ux(af::span, ycount - 1) = -velocity;
    // uy(0, af::span) = velocity;
    // uy(xcount - 1, af::span) = -velocity;
    // ux = af::select(af::iota(af::dim4(xcount, ycount)) < xcount * ycount / 2,
    //     af::constant(velocity, xcount, ycount), af::constant(0, xcount, ycount));

    feq = af::constant(0, xcount, ycount, 9);  
    f   = af::constant(0, xcount, ycount, 9);  
    fnew = af::constant(0, xcount, ycount, 9);

    ex_ = af::tile(ex, xcount, ycount, 1);
    ey_ = af::tile(ey, xcount, ycount, 1);
    wt_ = af::tile(wt, xcount, ycount, 1);

    ux_ = af::tile(ux, 1, 1, 9);
    uy_ = af::tile(uy, 1, 1, 9);
    rho_ = af::tile(rho, 1, 1, 9);

    auto edotu = ex_ * ux_ + ey_ * uy_;
    auto udotu = ux_ * ux_ + uy_ * uy_;

    feq = rho_ * wt_ * ((edotu * edotu * 4.5) - (udotu * 1.5) + (edotu * 3.0) + 1.0);
    f = feq;
    fnew = feq;
    // af_print(wt_);
    // af_print(feq);
    // af_print(ux * ux + uy * uy);
    // af_print(edotu);
}

void collide_stream()
{
    float tau = 0.5 + 3.0 * viscosity;
    float csky = 0.16;

    rho_ = af::tile(rho, 1, 1, 9);
    ux_ = af::tile(ux, 1, 1, 9);
    uy_ = af::tile(uy, 1, 1, 9);

    auto edotu = ex_ * ux_ + ey_ * uy_;
    auto udotu = ux_ * ux_ + uy_ * uy_;

    feq = rho_ * wt_ * (edotu * edotu * 4.5 - udotu * 1.5 + edotu * 3.0 + 1.0);

    auto taut = (af::sqrt(sigma * csky * csky * 18.0 + tau * tau) - tau) * 0.5;
    auto tau_eff = af::tile(taut + tau, 1, 1, 9);
    auto fplus = f - (f - feq) / tau_eff;

    af::array order_fplus = fplus;
    af::array order_ux = af::tile(ux, 1, 1, 9);
    af::array order_uy = af::tile(uy, 1, 1, 9);

    for (int i = 0; i < 9; ++i)
    {
        int xshift = ex_vals[i];
        int yshift = ey_vals[i];

        order_fplus(af::span, af::span, i) = af::shift(fplus(af::span, af::span, i), xshift, yshift);

        order_ux(af::span, af::span, i) = af::shift(ux, xshift, yshift);
        order_uy(af::span, af::span, i) = af::shift(uy, xshift, yshift);
    }

    order_fplus(0, af::span, af::span) = fnew(0, af::span, af::span);
    order_fplus(xcount - 1, af::span, af::span) = fnew(xcount - 1, af::span, af::span);
    order_fplus(af::span, 0, af::span) = fnew(af::span, 0, af::span);
    order_fplus(af::span, ycount - 1, af::span) = fnew(af::span, ycount - 1, af::span);

    fnew = order_fplus;

    // auto ubdoute = order_ux * ex_ + order_uy * ey_;
    // auto temp = order_fplus - 6.0 * density * wt_ * ubdoute;

    // for (int i = 0; i < 9; ++i)
    // {
    //     int xshift = ex(i).scalar<float>();
    //     int yshift = ey(i).scalar<float>();
    //     if (xshift == 1)
    //         fnew(1, af::span, oppos_vals[i]) = temp(1, af::span, i);
    //     if (xshift == -1)
    //         fnew(xcount - 2, af::span, oppos_vals[i]) = temp(xcount - 2, af::span, i);
    //     if (yshift == 1)
    //         fnew(af::span, 1, oppos_vals[i]) = temp(af::span, 1, i);
    //     if (yshift == -1)
    //         fnew(af::span, ycount - 2, oppos_vals[i]) = temp(af::span, ycount - 2, i);
    // }

    // fnew(0, af::span, af::span) = order_fplus(0, af::span, af::span);
    // fnew(xcount - 1, af::span, af::span) = order_fplus(xcount - 1, af::span, af::span);
    // fnew(af::span, 0, af::span) = order_fplus(af::span, 0, af::span);
    // fnew(af::span, ycount - 1, af::span) = order_fplus(af::span, ycount - 1, af::span);
}

void update()
{
    f = fnew;
    rho = af::sum(fnew, 2);

    af::array velx, vely;
    velx = af::sum(fnew * ex_, 2);
    vely = af::sum(fnew * ey_, 2);

    ux = velx / rho;
    uy = vely / rho;

    auto product = fnew - feq;
    auto temp = af::tile(product, af::dim4(1,1,1,3));
    temp(af::span, af::span, af::span, 0) *= ex_ * ex_;
    temp(af::span, af::span, af::span, 1) *= ey_ * ex_;
    temp(af::span, af::span, af::span, 2) *= ey_ * ey_;
    // auto xx = af::sum(product * ex_ * ex_, 2);
    // auto xy = af::sum(product * ex_ * ey_, 2);
    // auto yy = af::sum(product * ey_ * ey_, 2);
    // auto xxsum = product * ex_ * ex_;
    // auto xysum = product * ey_ * ex_;
    // auto yysum = product * ey_ * ey_;

    // af::eval(xxsum, yysum, xysum);
    
    // auto temp = af::join(3, xxsum, xysum);
    // temp = af::join(3, temp, yysum);

    temp = af::sum(temp, 2);
    temp *= temp;

    sigma = af::sqrt(temp(af::span, af::span, af::span, 0) +
                     temp(af::span, af::span, af::span, 1) * 2 +
                     temp(af::span, af::span, af::span, 2));

    // auto xx = af::sum(xxsum, 2);
    // auto xy = af::sum(xysum, 2);
    // auto yy = af::sum(yysum, 2);

    // sigma = af::sqrt(xx * xx + xy * xy * 2 + yy * yy);
}

int main(int argc, char** argv)
{
    int steps = 4;

    initialize();

    af::Window window(xcount, ycount, "Hello world");
    int max_frames = 10000;
    int frame_count = 0;
    int frame = 0;
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
        // af_print(feq);
        // af_print(f);
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(middle - begin).count();
        auto dur2 = std::chrono::duration_cast<std::chrono::microseconds>(end - middle).count();
        auto total = dur + dur2;

        std::cout << "First part: " << dur << " us; Second part: " << dur2 << " us; Total: " << total << " us" << std::endl;

        // af_print(sigma);
        // std::cout << 1.0e6/(float)std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;

        if (frame % 100 == 0)
        {
            auto val = af::sqrt(ux * ux + uy * uy)/velocity;
            auto image = af::constant(0, xcount, ycount, 3);
            auto image2 = image;

            image(af::span, af::span, 0) = val.T() * 2;
            image(af::span, af::span, 1) = val.T() * 2;
            image(af::span, af::span, 2) = 1.0 - val.T() * 2;
        
            image2(af::span, af::span, 0) = 1;
            image2(af::span, af::span, 1) = -2*val.T() + 2;
            image2(af::span, af::span, 2) = 0;
            
            image = af::select(af::tile(val.T(), 1, 1, 3) > 0.5, image2, image);

            window.image(image);
            frame = 0;
        }
        frame++;
        
        using namespace std::chrono_literals;
        // std::this_thread::sleep_for(100ms);
    }

    return 0;
}