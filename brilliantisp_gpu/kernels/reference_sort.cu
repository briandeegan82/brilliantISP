typedef long long npy_intp;

#define REF_QS_STACK 128
#define REF_SMALL_QUICKSORT 15
#define REF_DEPTH_KEEP (-2147483647 - 1)

__device__ __forceinline__ int ref_less(double a, double b) { return a < b; }

__device__ __forceinline__ void ref_swap(npy_intp *a, npy_intp *b) {
    npy_intp t = *a; *a = *b; *b = t;
}

__device__ void ref_aheapsort(const double *v, npy_intp *tosort, npy_intp n) {
    npy_intp *a = tosort - 1;
    npy_intp i, j, l, tmp;
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && ref_less(v[a[j]], v[a[j + 1]])) {
                j += 1;
            }
            if (ref_less(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }
    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && ref_less(v[a[j]], v[a[j + 1]])) {
                j++;
            }
            if (ref_less(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }
}

__device__ void ref_aquicksort(const double *v, npy_intp *tosort,
                               npy_intp num, int depth_override) {
    double vp;
    npy_intp *pl = tosort;
    npy_intp *pr = tosort + num - 1;
    npy_intp *stack[REF_QS_STACK];
    npy_intp **sptr = stack;
    npy_intp *pm, *pi, *pj, *pk, vi;
    int depth[REF_QS_STACK];
    int *psdepth = depth;
    int cdepth;
    {
        npy_intp t = num;
        int m = 0;
        while (t > 1) { t >>= 1; ++m; }
        cdepth = m * 2;
    }
    if (depth_override != REF_DEPTH_KEEP) {
        cdepth = depth_override;
    }

    for (;;) {
        if (cdepth < 0) {
            ref_aheapsort(v, pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > REF_SMALL_QUICKSORT) {
            pm = pl + ((pr - pl) >> 1);
            if (ref_less(v[*pm], v[*pl])) {
                ref_swap(pm, pl);
            }
            if (ref_less(v[*pr], v[*pm])) {
                ref_swap(pr, pm);
            }
            if (ref_less(v[*pm], v[*pl])) {
                ref_swap(pm, pl);
            }
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            ref_swap(pm, pj);
            for (;;) {
                do {
                    ++pi;
                } while (ref_less(v[*pi], vp));
                do {
                    --pj;
                } while (ref_less(vp, v[*pj]));
                if (pi >= pj) {
                    break;
                }
                ref_swap(pi, pj);
            }
            pk = pr - 1;
            ref_swap(pi, pk);
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && ref_less(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }
}

extern "C" __global__ void ref_init_index(npy_intp *idx, npy_intp n) {
    npy_intp i = (npy_intp)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        idx[i] = i;
    }
}

extern "C" __global__ void ref_argsort(const double *v, npy_intp *tosort,
                                       npy_intp num, int depth_override) {
    ref_aquicksort(v, tosort, num, depth_override);
}

extern "C" __global__ void ref_fused_project(const double *rows,
                                             const double *unit,
                                             npy_intp n, double *out) {
    npy_intp i = (npy_intp)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fma(rows[3 * i + 2], unit[2],
                     fma(rows[3 * i + 1], unit[1], rows[3 * i] * unit[0]));
    }
}
