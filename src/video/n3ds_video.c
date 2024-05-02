/*
 * This file is part of Moonlight Embedded.
 *
 * Copyright (C) 2015 Iwan Timmer
 *
 * Moonlight is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Moonlight is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Moonlight; if not, see <http://www.gnu.org/licenses/>.
 */

#include "ffmpeg.h"
#include "video.h"
#include "vshader_shbin.h"

#include "../util.h"

#include <3ds.h>
#include <stdbool.h>
#include <unistd.h>

#define SLICES_PER_FRAME 1
#define N3DS_BUFFER_FRAMES 1

static void *ffmpeg_buffer;
static size_t ffmpeg_buffer_size;
static int image_width, image_height, surface_width, surface_height, pixel_size;
static u8 *img_buffer;

static int offset_lut_size;
static int *dest_offset_lut;
static int *src_offset_lut;

static int offset_lut_size_3d;
static int *dest_offset_lut_3d;
static int *src_offset_lut_3d_l;
static int *src_offset_lut_3d_r;

static int offset_lut_size_ds_bottom;
static int *dest_offset_lut_ds_bottom;
static int *src_offset_lut_ds_top;
static int *src_offset_lut_ds_bottom;
bool enable_dual_display = false;

u64 perf_frame_target_ticks = SYSCLOCK_ARM11 * ((double)(1.0 / 60.0));
u64 perf_decode_ticks = 0;
u64 perf_fbcopy_ticks = 0;

static inline int get_dest_offset(int x, int y, int dest_height) {
    return dest_height - y - 1 + dest_height * x;
}

static inline int get_source_offset(int x, int y, int src_width, int src_height,
                                    int dest_width, int dest_height) {
    return (x * src_width / dest_width) +
           (y * src_height / dest_height) * src_width;
}

static inline int get_source_offset_3d_l(int x, int y, int src_width,
                                         int src_height, int dest_width,
                                         int dest_height) {
    return (x * (src_width / 2) / dest_width) +
           (y * src_height / dest_height) * src_width;
}

static inline int get_source_offset_3d_r(int x, int y, int src_width,
                                         int src_height, int dest_width,
                                         int dest_height) {
    return ((x * (src_width / 2) / dest_width) + (src_width / 2)) +
           (y * src_height / dest_height) * src_width;
}

static inline int get_source_offset_ds_top(int x, int y, int src_width,
                                           int src_height, int dest_width,
                                           int dest_height) {
    return (x * src_width / dest_width) +
           (y * (src_height / 2) / dest_height) * src_width;
}

static inline int get_source_offset_ds_bottom(int x, int y, int src_width,
                                              int src_height, int dest_width,
                                              int dest_height) {
    return (x * src_width / dest_width) +
           ((y * (src_height / 2) / dest_height) + (src_height / 2)) *
               src_width;
}

static inline void ensure_3d_enabled() {
    if (!gfxIs3D()) {
        gfxSetWide(false);
        gfxSet3D(true);
    }
}

static inline void ensure_3d_disabled() {
    if (gfxIs3D()) {
        gfxSet3D(false);
    }
    if (surface_width == GSP_SCREEN_HEIGHT_TOP_2X) {
        gfxSetWide(true);
    }
}

static int n3ds_setup(int videoFormat, int width, int height, int redrawRate,
                      void *context, int drFlags) {
    if (ffmpeg_init(videoFormat, width, height, 0, N3DS_BUFFER_FRAMES,
                    SLICES_PER_FRAME) < 0) {
        fprintf(stderr, "Couldn't initialize video decoding\n");
        return -1;
    }

    ensure_buf_size(&ffmpeg_buffer, &ffmpeg_buffer_size,
                    INITIAL_DECODER_BUFFER_SIZE + AV_INPUT_BUFFER_PADDING_SIZE);

    if (y2rInit()) {
        fprintf(stderr, "Failed to initialize Y2R\n");
        return -1;
    }
    Y2RU_ConversionParams y2r_parameters;
    y2r_parameters.input_format = INPUT_YUV420_INDIV_8;
    y2r_parameters.output_format = OUTPUT_RGB_16_565;
    y2r_parameters.rotation = ROTATION_NONE;
    y2r_parameters.block_alignment = BLOCK_LINE;
    y2r_parameters.input_line_width = width;
    y2r_parameters.input_lines = height;
    y2r_parameters.standard_coefficient = COEFFICIENT_ITU_R_BT_709_SCALING;
    y2r_parameters.alpha = 0xFF;
    int status = Y2RU_SetConversionParams(&y2r_parameters);
    if (status) {
        fprintf(stderr, "Failed to set Y2RU params\n");
        return -1;
    }

    surface_height = GSP_SCREEN_WIDTH;
    if (width > GSP_SCREEN_HEIGHT_TOP) {
        surface_width = GSP_SCREEN_HEIGHT_TOP_2X;
    } else {
        surface_width = GSP_SCREEN_HEIGHT_TOP;
    }

    GSPGPU_FramebufferFormat px_fmt = gfxGetScreenFormat(GFX_TOP);
    image_width = width;
    image_height = height;
    pixel_size = gspGetBytesPerPixel(px_fmt);

    img_buffer = linearAlloc(width * height * pixel_size);
    if (!img_buffer) {
        fprintf(stderr, "Out of memory!\n");
        return -1;
    }

    return init_px_to_framebuffer(surface_width, surface_height, image_width,
                                  image_height, pixel_size);
}

static void n3ds_cleanup() {
    ffmpeg_destroy();
    y2rExit();
    linearFree(img_buffer);
    deinit_px_to_framebuffer();
}

static inline int init_px_to_framebuffer_2d(int dest_width, int dest_height,
                                            int src_width, int src_height,
                                            int px_size) {
    // Generate LUTs so we don't have to calculate pixel rotation while
    // streaming.
    offset_lut_size = dest_width * dest_height;
    src_offset_lut = malloc(sizeof(int) * offset_lut_size);
    if (!src_offset_lut) {
        fprintf(stderr, "Out of memory!\n");
        return -1;
    }
    dest_offset_lut = malloc(sizeof(int) * offset_lut_size);
    if (!dest_offset_lut) {
        fprintf(stderr, "Out of memory!\n");
        return -1;
    }

    int i = 0;
    for (int y = 0; y < dest_height; ++y) {
        for (int x = 0; x < dest_width; ++x) {
            src_offset_lut[i] =
                px_size * get_source_offset(x, y, src_width, src_height,
                                            dest_width, dest_height);
            dest_offset_lut[i] = px_size * get_dest_offset(x, y, dest_height);
            i++;
        }
    }
    return 0;
}

static inline int init_px_to_framebuffer_3d(int dest_width, int dest_height,
                                            int src_width, int src_height,
                                            int px_size) {
    // Generate LUTs so we don't have to calculate pixel rotation while
    // streaming.
    offset_lut_size_3d = dest_width * dest_height;
    src_offset_lut_3d_l = malloc(sizeof(int) * offset_lut_size_3d);
    if (!src_offset_lut_3d_l) {
        fprintf(stderr, "Out of memory!\n");
        return -1;
    }
    src_offset_lut_3d_r = malloc(sizeof(int) * offset_lut_size_3d);
    if (!src_offset_lut_3d_r) {
        fprintf(stderr, "Out of memory!\n");
        return -1;
    }
    dest_offset_lut_3d = malloc(sizeof(int) * offset_lut_size_3d);
    if (!dest_offset_lut_3d) {
        fprintf(stderr, "Out of memory!\n");
        return -1;
    }

    int i = 0;
    for (int y = 0; y < dest_height; ++y) {
        for (int x = 0; x < dest_width; ++x) {
            src_offset_lut_3d_l[i] =
                px_size * get_source_offset_3d_l(x, y, src_width, src_height,
                                                 dest_width, dest_height);
            src_offset_lut_3d_r[i] =
                px_size * get_source_offset_3d_r(x, y, src_width, src_height,
                                                 dest_width, dest_height);
            dest_offset_lut_3d[i] =
                px_size * get_dest_offset(x, y, dest_height);
            i++;
        }
    }
    return 0;
}

static inline int init_px_to_framebuffer_ds(int dest_width, int dest_height,
                                            int src_width, int src_height,
                                            int px_size) {
    // Generate LUTs so we don't have to calculate pixel rotation while
    // streaming.
    offset_lut_size_ds_bottom = GSP_SCREEN_HEIGHT_BOTTOM * dest_height;
    src_offset_lut_ds_top = malloc(sizeof(int) * offset_lut_size);
    if (!src_offset_lut_ds_top) {
        fprintf(stderr, "Out of memory!\n");
        return -1;
    }
    src_offset_lut_ds_bottom = malloc(sizeof(int) * offset_lut_size_ds_bottom);
    if (!src_offset_lut_ds_bottom) {
        fprintf(stderr, "Out of memory!\n");
        return -1;
    }
    dest_offset_lut_ds_bottom = malloc(sizeof(int) * offset_lut_size_ds_bottom);
    if (!dest_offset_lut_ds_bottom) {
        fprintf(stderr, "Out of memory!\n");
        return -1;
    }

    int i = 0;
    for (int y = 0; y < dest_height; ++y) {
        for (int x = 0; x < dest_width; ++x) {
            src_offset_lut_ds_top[i] =
                px_size * get_source_offset_ds_top(x, y, src_width, src_height,
                                                   dest_width, dest_height);
            i++;
        }
    }

    i = 0;
    for (int y = 0; y < dest_height; ++y) {
        for (int x = 0; x < GSP_SCREEN_HEIGHT_BOTTOM; ++x) {
            src_offset_lut_ds_bottom[i] =
                px_size * get_source_offset_ds_bottom(
                              x, y, src_width, src_height,
                              GSP_SCREEN_HEIGHT_BOTTOM, dest_height);
            dest_offset_lut_ds_bottom[i] =
                px_size * get_dest_offset(x, y, dest_height);
            i++;
        }
    }
    return 0;
}

int init_px_to_framebuffer(int dest_width, int dest_height, int src_width,
                           int src_height, int px_size) {
    surface_width = dest_width;
    surface_height = dest_height;
    image_width = src_width;
    image_height = src_height;
    int ret = init_px_to_framebuffer_2d(dest_width, dest_height, src_width,
                                        src_height, px_size);
    if (ret == 0) {
        ret = init_px_to_framebuffer_3d(GSP_SCREEN_HEIGHT_TOP, dest_height,
                                        src_width, src_height, px_size);
    }
    if (ret == 0) {
        ret = init_px_to_framebuffer_ds(dest_width, dest_height, src_width,
                                        src_height, px_size);
    }
    return ret;
}

void deinit_px_to_framebuffer() {
    if (src_offset_lut)
        free(src_offset_lut);
    if (src_offset_lut_3d_l)
        free(src_offset_lut_3d_l);
    if (src_offset_lut_3d_r)
        free(src_offset_lut_3d_r);
    if (src_offset_lut_ds_top)
        free(src_offset_lut_ds_top);
    if (src_offset_lut_ds_bottom)
        free(src_offset_lut_ds_bottom);
    if (dest_offset_lut)
        free(dest_offset_lut);
    if (dest_offset_lut_3d)
        free(dest_offset_lut_3d);
    if (dest_offset_lut_ds_bottom)
        free(dest_offset_lut_ds_bottom);
}

static inline void draw_perf_counters() {
    const size_t px_size = 2;

    // Use a line going across the first scanline (left) for the perf counters.
    u8 *dest = gfxGetFramebuffer(GFX_BOTTOM, GFX_LEFT, NULL, NULL);

    // Clear to black
    memset(dest, 0, GSP_SCREEN_WIDTH * 3);

    // Display frame target in the middle of the screen.
    double perf_tick_divisor = ((double)GSP_SCREEN_WIDTH) / ((double)(perf_frame_target_ticks * 2));
    u32 perf_px = 0;
    u32 perf_tmp_height = 0;

    #define PERF_DRAW(ticks, r, g, b) \
        perf_tmp_height = perf_tick_divisor * ((double)(ticks)); \
        do { \
            if (perf_px > GSP_SCREEN_WIDTH) break; \
            const u32 color = (r << 16) | (g << 8) | b; \
            memcpy(dest + (perf_px * 3), &color, 3); \
            perf_px++; \
        } while (perf_tmp_height-- > 0);

    PERF_DRAW(perf_decode_ticks, 255, 0, 0);
    PERF_DRAW(perf_fbcopy_ticks, 0, 0, 255);

    // Draw two green pixels at the center
    perf_px = (GSP_SCREEN_WIDTH / 2) - 1;
    PERF_DRAW(0, 0, 255, 0);
    PERF_DRAW(0, 0, 255, 0);
}

static u32 *cmdlist = NULL;
static void *vramFb = NULL;
static void *vramTex = NULL;

#define CMDLIST_SZ 0x800

// TODO: Stole this from somewhere...
static inline void write24(u8* p, u32 val) {
	p[0] = val;
	p[1] = val >> 8;
	p[2] = val >> 16;
}

static inline void write_px_to_framebuffer_2D(uint8_t *__restrict source, uint8_t *__restrict scratch, int px_size) {
    // TODO: Do nothing when GPU right is lost. Currently hangs when going to the home menu.

    u64 start_ticks = svcGetSystemTick();
    u8 *__restrict dest = gfxGetFramebuffer(GFX_TOP, GFX_LEFT, NULL, NULL);

    // NOTE: At 800x480, we can display the _width_ natively, but the height needs to be downsampled.
    // MVD is incapable of downsampling, so we have to do it on the GPU.

    if (!cmdlist) {
        cmdlist = linearAlloc(CMDLIST_SZ * 4);

        // TODO: Don't hardcode this.
        vramFb = vramAlloc(800 * 240 * 2);
        // Needs to be able to hold an 800x480
        vramTex = vramAlloc(MOON_CTR_VIDEO_TEX_W * MOON_CTR_VIDEO_TEX_H * 2);
    }

    // TODO: If we can use rotation from the decoder, we can do a 2x downscale using display transfer and
    // skip P3D. Not necessary because PICA is significantly faster than the decoder.

    // Tile the source image into the scratch buffer.
    GX_DisplayTransfer(
        source, GX_BUFFER_DIM(MOON_CTR_VIDEO_TEX_W, MOON_CTR_VIDEO_TEX_H),
        vramTex, GX_BUFFER_DIM(MOON_CTR_VIDEO_TEX_W, MOON_CTR_VIDEO_TEX_H),
        GX_TRANSFER_FLIP_VERT(1) | GX_TRANSFER_OUT_TILED(1) |
        GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGB565) |
        GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB565)
    );

    // While the transfer is running, create a temporary command list to rotate the framebuffer into source
    GPUCMD_SetBuffer(cmdlist, CMDLIST_SZ, 0);

    // TODO: Verify this mitigates rounding errors due to f24 precision issues.

    #define C GPUCMD_AddWrite

    C(GPUREG_FRAMEBUFFER_INVALIDATE, 1);
    C(GPUREG_COLORBUFFER_LOC, osConvertVirtToPhys(vramFb) >> 3);
    C(GPUREG_DEPTHBUFFER_LOC, 0);
    C(GPUREG_RENDERBUF_DIM, (1 << 24) | ((surface_width - 1) << 12) | surface_height);
    C(GPUREG_FRAMEBUFFER_DIM, (1 << 24) | ((surface_width - 1) << 12) | surface_height);
    C(GPUREG_FRAMEBUFFER_BLOCK32, 0);

    C(GPUREG_DEPTH_COLOR_MASK, 0xF << 8); // Write RGBA, no depth
    C(GPUREG_EARLYDEPTH_TEST1, 0);
    C(GPUREG_EARLYDEPTH_TEST2, 0);
    C(GPUREG_COLORBUFFER_FORMAT, GPU_RGB565 << 16);
    C(GPUREG_COLORBUFFER_READ, 0x0); // Buffer is uninitialized and should not be read.
    C(GPUREG_COLORBUFFER_WRITE, 0xF);
    C(GPUREG_DEPTHBUFFER_READ, 0); // No depth buffer
    C(GPUREG_DEPTHBUFFER_WRITE, 0);

    C(GPUREG_VIEWPORT_XY, 0);

    C(GPUREG_VIEWPORT_WIDTH, f32tof24(surface_height / 2));
    C(GPUREG_VIEWPORT_INVW, f32tof31(2.0 / ((double)surface_height)) << 1);
    C(GPUREG_VIEWPORT_HEIGHT, f32tof24(surface_width / 2));
    C(GPUREG_VIEWPORT_INVH, f32tof31(2.0 / ((double)surface_width)) << 1);

    C(GPUREG_SCISSORTEST_MODE, 0);
    C(GPUREG_SCISSORTEST_POS, 0);
    C(GPUREG_SCISSORTEST_DIM, 0);

    C(GPUREG_DEPTHMAP_ENABLE, 1);
    C(GPUREG_DEPTHMAP_SCALE, f32tof24(-1.0));
    C(GPUREG_DEPTHMAP_OFFSET, 0);
    C(GPUREG_STENCIL_TEST, 0);
    C(GPUREG_FRAGOP_ALPHA_TEST, 0);
    C(GPUREG_LOGIC_OP, 3);
    C(GPUREG_COLOR_OPERATION, 0x00E40000);

    // Texturing
    C(GPUREG_TEXUNIT0_TYPE, GPU_RGB565);
    C(GPUREG_TEXUNIT0_DIM, MOON_CTR_VIDEO_TEX_H | (MOON_CTR_VIDEO_TEX_W << 16));
    C(GPUREG_TEXUNIT0_ADDR1, osConvertVirtToPhys(vramTex) >> 3);
    C(GPUREG_TEXUNIT0_PARAM, GPU_NEAREST | (GPU_LINEAR << 1)); // Linear min and mag filter

    // Shading
    // GPUCMD_AddMaskedWrite(GPUREG_SH_OUTATTR_CLOCK, 0x2, 1 << 8); // No Z, Yes texcoord0
    C(GPUREG_TEXUNIT_CONFIG, 1 | (1 << 12) | (1 << 16)); // Activate texture 0, clear texture cache

    C(GPUREG_TEXENV0_SOURCE, 0x003003); // Texture 0
    C(GPUREG_TEXENV0_OPERAND, 0); // Source Color
    C(GPUREG_TEXENV0_COMBINER, 0); // Replace
    C(GPUREG_TEXENV0_SCALE, 0); // No Scale

    C(GPUREG_TEXENV1_SOURCE, 0x003003); // Texture 0
    C(GPUREG_TEXENV1_OPERAND, 0); // Source Color
    C(GPUREG_TEXENV1_COMBINER, 0); // Replace
    C(GPUREG_TEXENV1_SCALE, 0); // No Scale

    C(GPUREG_TEXENV2_SOURCE, 0x003003); // Texture 0
    C(GPUREG_TEXENV2_OPERAND, 0); // Source Color
    C(GPUREG_TEXENV2_COMBINER, 0); // Replace
    C(GPUREG_TEXENV2_SCALE, 0); // No Scale

    C(GPUREG_TEXENV3_SOURCE, 0x003003); // Texture 0
    C(GPUREG_TEXENV3_OPERAND, 0); // Source Color
    C(GPUREG_TEXENV3_COMBINER, 0); // Replace
    C(GPUREG_TEXENV3_SCALE, 0); // No Scale

    C(GPUREG_TEXENV4_SOURCE, 0x003003); // Texture 0
    C(GPUREG_TEXENV4_OPERAND, 0); // Source Color
    C(GPUREG_TEXENV4_COMBINER, 0); // Replace
    C(GPUREG_TEXENV4_SCALE, 0); // No Scale

    C(GPUREG_TEXENV5_SOURCE, 0x003003); // Texture 0
    C(GPUREG_TEXENV5_OPERAND, 0); // Source Color
    C(GPUREG_TEXENV5_COMBINER, 0); // Replace
    C(GPUREG_TEXENV5_SCALE, 0); // No Scale

    // Attribute buffers
    C(GPUREG_ATTRIBBUFFERS_LOC, 0);
    C(GPUREG_ATTRIBBUFFERS_FORMAT_LOW, 0);
    C(GPUREG_ATTRIBBUFFERS_FORMAT_HIGH, (0xFFF << 16) | (1 << 28)); // Two fixed vertex attributes

    // Vertex Shader
    static DVLB_s* vshader_dvlb = NULL;
    static shaderProgram_s program;

    if (!vshader_dvlb) {
        vshader_dvlb = DVLB_ParseFile((u32*)vshader_shbin, vshader_shbin_size);
        shaderProgramInit(&program);
        shaderProgramSetVsh(&program, &vshader_dvlb->DVLE[0]);
    }

    shaderProgramUse(&program);

    C(GPUREG_VSH_NUM_ATTR, 1); // 2 attributes
    GPUCMD_AddMaskedWrite(GPUREG_VSH_INPUTBUFFER_CONFIG, 0xB, 1 | (0xA0 << 24)); // 2 attributes, no geometry shader
    C(GPUREG_VSH_ATTRIBUTES_PERMUTATION_LOW, 0x00000010);
    C(GPUREG_VSH_ATTRIBUTES_PERMUTATION_HIGH, 0);

    // Geometry Pipeline
    C(GPUREG_FACECULLING_CONFIG, 0);
    C(GPUREG_GEOSTAGE_CONFIG, 0);
    GPUCMD_AddMaskedWrite(GPUREG_PRIMITIVE_CONFIG, 2, (1 << 8) | 1); // 2 outmap registers, drawing triangle strip
    C(GPUREG_INDEXBUFFER_CONFIG, 0x80000000);
    C(GPUREG_RESTART_PRIMITIVE, 1);

    // Vertex Data
    GPUCMD_AddMaskedWrite(GPUREG_GEOSTAGE_CONFIG2, 1, 1);
	GPUCMD_AddMaskedWrite(GPUREG_START_DRAW_FUNC0, 1, 0);
    C(GPUREG_FIXEDATTRIB_INDEX, 0xF);

    union {
		u32 packed[3];
		struct { u8 x[3], y[3], z[3], w[3]; };
	} param;

    #define ATTR(X, Y, Z, W) \
        { \
            write24(param.x, f32tof24(X)); \
            write24(param.y, f32tof24(Y)); \
            write24(param.z, f32tof24(Z)); \
            write24(param.w, f32tof24(W)); \
            \
            u32 p = param.packed[0]; \
            param.packed[0] = param.packed[2]; \
            param.packed[2] = p; \
            GPUCMD_AddIncrementalWrites(GPUREG_FIXEDATTRIB_DATA0, param.packed, 3); \
        }

    float sw = image_width / 1024.0f;
    float sh = image_height / 512.0f;

    float hw = 2.0f / surface_height;
    float hh = 2.0f / surface_width;

    ATTR(1.0, -1.0, 0.0, 0.0); // TR
    ATTR(sw, -hh, 0.0, 0.0);

    ATTR(-1.0, -1.0, 0.0, 0.0); // TL
    ATTR(sw, sh, 0.0, 0.0);

    ATTR(1.0, 1.0, 0.0, 0.0); // BR
    ATTR(0.0, -hh, 0.0, 0.0);

    ATTR(-1.0, 1.0, 0.0, 0.0); // BL
    ATTR(0.0, sh, 0.0, 0.0);

    // End Geometry Pipeline
    GPUCMD_AddMaskedWrite(GPUREG_START_DRAW_FUNC0, 1, 1);
	GPUCMD_AddMaskedWrite(GPUREG_GEOSTAGE_CONFIG2, 1, 0);
	C(GPUREG_VTX_FUNC, 1);

    // Stop Command List
    GPUCMD_AddMaskedWrite(GPUREG_PRIMITIVE_CONFIG, 0x8, 0x00000000);
    C(GPUREG_FRAMEBUFFER_FLUSH, 1);
    C(GPUREG_FRAMEBUFFER_INVALIDATE, 1);

    #undef C

    gspWaitForEvent(GSPGPU_EVENT_PPF, 0);

    u32 *unused;
    u32 cmdlist_len;
    GPUCMD_Split(&unused, &cmdlist_len);
    GSPGPU_FlushDataCache(cmdlist, cmdlist_len);

    extern u32 __ctru_linear_heap;
	extern u32 __ctru_linear_heap_size;
	GX_FlushCacheRegions(cmdlist, cmdlist_len*4, (u32 *) __ctru_linear_heap, __ctru_linear_heap_size, NULL, 0);

    GX_ProcessCommandList(cmdlist, cmdlist_len * 4, 2);

    gspWaitForEvent(GSPGPU_EVENT_P3D, 0);

    // Copy into framebuffer, untiled

    GX_DisplayTransfer(
        vramFb, GX_BUFFER_DIM(surface_height, surface_width),
        dest, GX_BUFFER_DIM(surface_height, surface_width),
        GX_TRANSFER_OUT_TILED(0) |
        GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGB565) |
        GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB565) |
        GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO)
    );

    gspWaitForEvent(GSPGPU_EVENT_PPF, 0);

    perf_fbcopy_ticks = svcGetSystemTick() - start_ticks;

    draw_perf_counters();

    gfxSwapBuffers();
}

static inline void write_px_to_framebuffer_3D(uint8_t *source, int px_size) {
    u8 *dest = gfxGetFramebuffer(GFX_TOP, GFX_LEFT, NULL, NULL);
    for (int i = 0; i < offset_lut_size_3d; i++) {
        memcpy(dest + dest_offset_lut_3d[i], source + src_offset_lut_3d_l[i],
               px_size);
    }

    dest = gfxGetFramebuffer(GFX_TOP, GFX_RIGHT, NULL, NULL);
    for (int i = 0; i < offset_lut_size_3d; i++) {
        memcpy(dest + dest_offset_lut_3d[i], source + src_offset_lut_3d_r[i],
               px_size);
    }
    gfxScreenSwapBuffers(GFX_TOP, true);
}

static inline void write_px_to_framebuffer_DS(uint8_t *source, int px_size) {
    u8 *dest = gfxGetFramebuffer(GFX_TOP, GFX_LEFT, NULL, NULL);
    for (int i = 0; i < offset_lut_size; i++) {
        memcpy(dest + dest_offset_lut[i], source + src_offset_lut_ds_top[i],
               px_size);
    }

    dest = gfxGetFramebuffer(GFX_BOTTOM, GFX_LEFT, NULL, NULL);
    for (int i = 0; i < offset_lut_size_ds_bottom; i++) {
        memcpy(dest + dest_offset_lut_ds_bottom[i],
               source + src_offset_lut_ds_bottom[i], px_size);
    }
    gfxSwapBuffers();
}

void write_px_to_framebuffer(const uint8_t *source, uint8_t *scratch, int px_size) {
    if (enable_dual_display) {
        ensure_3d_disabled();
        write_px_to_framebuffer_DS(source, px_size);
    } else if (osGet3DSliderState() > 0.0) {
        ensure_3d_enabled();
        write_px_to_framebuffer_3D(source, px_size);
    } else {
        ensure_3d_disabled();
        write_px_to_framebuffer_2D(source, scratch, px_size);
    }
}

static inline int write_yuv_to_framebuffer(const u8 **source, int width,
                                           int height, int px_size) {
    Handle conversion_finish_event_handle;
    int status = 0;

    status = Y2RU_SetSendingY(source[0], width * height, width, 0);
    if (status) {
        fprintf(stderr, "Y2RU_SetSendingY failed\n");
        goto y2ru_failed;
    }

    status = Y2RU_SetSendingU(source[1], width * height / 4, width / 2, 0);
    if (status) {
        fprintf(stderr, "Y2RU_SetSendingU failed\n");
        goto y2ru_failed;
    }

    status = Y2RU_SetSendingV(source[2], width * height / 4, width / 2, 0);
    if (status) {
        fprintf(stderr, "Y2RU_SetSendingV failed\n");
        goto y2ru_failed;
    }

    status = Y2RU_SetReceiving(img_buffer, width * height * px_size, 8, 0);
    if (status) {
        fprintf(stderr, "Y2RU_SetReceiving failed\n");
        goto y2ru_failed;
    }

    status = Y2RU_StartConversion();
    if (status) {
        fprintf(stderr, "Y2RU_StartConversion failed\n");
        goto y2ru_failed;
    }

    status = Y2RU_GetTransferEndEvent(&conversion_finish_event_handle);
    if (status) {
        fprintf(stderr, "Y2RU_GetTransferEndEvent failed\n");
        goto y2ru_failed;
    }

    svcWaitSynchronization(conversion_finish_event_handle,
                           10000000); // Wait up to 10ms.
    svcCloseHandle(conversion_finish_event_handle);
    // TODO: Yeah I don't care about old 3ds, buy a new one please
    write_px_to_framebuffer(img_buffer, img_buffer, px_size);
    return DR_OK;

y2ru_failed:
    return -1;
}

static int n3ds_submit_decode_unit(PDECODE_UNIT decodeUnit) {
    PLENTRY entry = decodeUnit->bufferList;
    int length = 0;

    ensure_buf_size(&ffmpeg_buffer, &ffmpeg_buffer_size,
                    decodeUnit->fullLength + AV_INPUT_BUFFER_PADDING_SIZE);

    while (entry != NULL) {
        memcpy(ffmpeg_buffer + length, entry->data, entry->length);
        length += entry->length;
        entry = entry->next;
    }
    ffmpeg_decode(ffmpeg_buffer, length);

    AVFrame *frame = ffmpeg_get_frame(false);
    int status = write_yuv_to_framebuffer(frame->data, image_width,
                                          image_height, pixel_size);

    return status;
}

DECODER_RENDERER_CALLBACKS decoder_callbacks_n3ds = {
    .setup = n3ds_setup,
    .cleanup = n3ds_cleanup,
    .submitDecodeUnit = n3ds_submit_decode_unit,
    .capabilities =
        CAPABILITY_DIRECT_SUBMIT | CAPABILITY_REFERENCE_FRAME_INVALIDATION_AVC,
};
