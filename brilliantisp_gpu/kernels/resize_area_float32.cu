typedef long long npy_intp;

extern "C" __global__ void area_f32_fast(const float *in, float *out,
                              npy_intp batch, npy_intp h, npy_intp w,
                              npy_intp oh, npy_intp ow,
                              npy_intp iy, npy_intp ix,
                              const float *norm) {
    npy_intp idx = (npy_intp)blockIdx.x * blockDim.x + threadIdx.x;
    npy_intp n_out = batch * oh * ow;
    if (idx >= n_out)
        return;
    npy_intp b = idx / (oh * ow);
    npy_intp rem = idx - b * oh * ow;
    npy_intp oy = rem / ow;
    npy_intp ox = rem - oy * ow;
    const float *src = in + b * h * w + oy * iy * w + ox * ix;
    float result;
    if (iy == 2 && ix == 2) {
        float v00 = src[0], v01 = src[1];
        float v10 = src[w], v11 = src[w + 1];
        if (ox < (ow / 4) * 4)
            result = ((v00 + v01) + (v10 + v11)) * 0.25f;
        else
            result = (((v00 + v01) + v10) + v11) * 0.25f;
    } else {
        float sum = 0.0f;
        npy_intp n = iy * ix;
        npy_intp k = 0;
        for (; k <= n - 4; k += 4) {
            float v0 = src[(k / ix) * w + (k % ix)];
            float v1 = src[((k + 1) / ix) * w + ((k + 1) % ix)];
            float v2 = src[((k + 2) / ix) * w + ((k + 2) % ix)];
            float v3 = src[((k + 3) / ix) * w + ((k + 3) % ix)];
            float group = ((v0 + v1) + v2) + v3;
            sum = sum + group;
        }
        for (; k < n; ++k)
            sum = sum + src[(k / ix) * w + (k % ix)];
        result = sum * norm[0];
    }
    out[idx] = result;
}

extern "C" __global__ void area_f32_generic(const float *in, float *out,
                                 npy_intp batch, npy_intp h, npy_intp w,
                                 npy_intp oh, npy_intp ow,
                                 const npy_intp *y_off, const npy_intp *y_idx,
                                 const float *y_w,
                                 const npy_intp *x_off, const npy_intp *x_idx,
                                 const float *x_w) {
    npy_intp idx = (npy_intp)blockIdx.x * blockDim.x + threadIdx.x;
    npy_intp n_out = batch * oh * ow;
    if (idx >= n_out)
        return;
    npy_intp b = idx / (oh * ow);
    npy_intp rem = idx - b * oh * ow;
    npy_intp oy = rem / ow;
    npy_intp ox = rem - oy * ow;
    const float *src = in + b * h * w;
    float total = 0.0f;
    for (npy_intp yt = y_off[oy]; yt < y_off[oy + 1]; ++yt) {
        float hsum = 0.0f;
        const float *row = src + y_idx[yt] * w;
        for (npy_intp xt = x_off[ox]; xt < x_off[ox + 1]; ++xt)
            hsum = hsum + row[x_idx[xt]] * x_w[xt];
        total = total + hsum * y_w[yt];
    }
    out[idx] = total;
}
