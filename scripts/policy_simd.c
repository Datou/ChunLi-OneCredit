#include <wasm_simd128.h>

/* Build with clang --target=wasm32 -O3 -msimd128 -nostdlib -Wl,--no-entry
 * -Wl,--export=dense -Wl,--export=__heap_base -Wl,--export-memory
 * scripts/policy_simd.c -o web/models/policy_simd.wasm */
void dense(const float *input, const float *weights, const float *bias,
           float *output, int width, int channels, int cells, int relu) {
    for (int channel = 0; channel < channels; channel++) {
        const float *row = weights + channel * width;
        for (int cell = 0; cell < cells; cell++) {
            const float *patch = input + cell * width;
            v128_t sums = wasm_f32x4_splat(0.0f);
            int i = 0;
            for (; i + 3 < width; i += 4) {
                v128_t a = wasm_v128_load(patch + i);
                v128_t b = wasm_v128_load(row + i);
                sums = wasm_f32x4_add(sums, wasm_f32x4_mul(a, b));
            }
            float sum = bias[channel] + wasm_f32x4_extract_lane(sums, 0)
                + wasm_f32x4_extract_lane(sums, 1) + wasm_f32x4_extract_lane(sums, 2)
                + wasm_f32x4_extract_lane(sums, 3);
            for (; i < width; i++) sum += patch[i] * row[i];
            output[channel * cells + cell] = relu && sum < 0.0f ? 0.0f : sum;
        }
    }
}
