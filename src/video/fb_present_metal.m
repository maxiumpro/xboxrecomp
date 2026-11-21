/**
 * The SDL window's CAMetalLayer present (macOS): the zero-copy half of
 * fb_present_sdl.c, used when the Metal backend draws (RECOMP_PB_BACKEND=metal,
 * not RECOMP_DEBUG=metal_present=readback).
 *
 * The window has no SDL_Renderer then. fb_present_sdl.c keeps the window, the
 * event loop and the three-slot hand-off; this file owns what the slots point
 * at and how a slot reaches the screen:
 *
 *   device, queue  made here at window creation, before the guest starts; the
 *                  Metal backend adopts both at its lazy init
 *                  (xbox_FramebufferWindowMetal), so a slot texture takes the
 *                  backend's blit and a present pass on the same queue runs
 *                  after the blit it shows (commit order, tracked hazards).
 *   slots          three BGRA8 textures, frame-sized (render.scale's host
 *                  size). The backend blits its present target into the back
 *                  slot on the GPU; a guest-bytes frame is uploaded into it.
 *   present        on the main thread: the layer's next drawable, cleared to
 *                  black, the slot drawn into nv2a_present_rect's rectangle
 *                  with one full-screen triangle, nearest or linear as the
 *                  SDL path filters (and as D3D11's scaling blit does).
 *   shot           RECOMP_WINDOW_SHOT: the drawable read back after the pass,
 *                  framebufferOnly being off only when shots are asked for.
 *   internal       present.surface=internal: the drawable stays the frame's
 *                  size, the slot is drawn into it 1:1, and the layer scales
 *                  it to the view (aspect fit, the present filter's sampler),
 *                  as gamescope scales D3D11's frame-sized swap chain.
 *   overscan       display.overscan > 0: a second pipeline (vs_crop) samples
 *                  nv2a_present_crop's part of the slot. At 0 the stock
 *                  pipeline draws, so nothing changes.
 *   overlay        enhancements builds: the lists fb_present_sdl.c took from
 *                  nv2a_overlay (the SDL loop stays the overlay's one reader)
 *                  drawn in the same pass after the frame, premultiplied, so
 *                  the drawable shot has them and the Metal frame dump (the
 *                  backend's render target) never does.
 *   shader         present.shader (enhancements builds): librashader's Metal
 *                  filter chain on this queue draws the slot into the same
 *                  rectangle in place of the triangle; the overlay follows
 *                  in a second pass. Built on the main thread when the
 *                  request's generation moves; "none" never loads it.
 *   pacing         RECOMP_TRACE=pacing: each present's drawable and command
 *                  buffer get a presented and a completed handler, and a
 *                  [PRESENT] line every 600 presents or 10 s says what
 *                  reached the screen (new, repeat, skipped, dropped), the
 *                  new images' interval and age, and the pass's GPU time.
 *
 * Manual retain/release, like nv2a_pb_metal.m. Every entry point is wrapped in
 * an autorelease pool: the main loop and the ack thread have none of their own.
 */
#if defined(__APPLE__)

#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CABase.h>       /* CACurrentMediaTime */
#import <QuartzCore/CAMetalLayer.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_metal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kernel/nv2a_backend_common.h"
#ifdef XBOXRECOMP_ENHANCE
#include "kernel/nv2a_overlay.h"
#include "kernel/nv2a_present_chain.h"
#define LIBRA_RUNTIME_METAL
#include "librashader.h"
#endif
#include "fb_present_metal.h"
#include "metal_compile.h"
#include "recomp_env.h"

#if __has_feature(objc_arc)
#error "fb_present_metal.m is manual retain/release: build it without -fobjc-arc"
#endif

static id<MTLDevice>               s_dev;
static id<MTLCommandQueue>         s_queue;
static id<MTLRenderPipelineState>  s_pso;
static id<MTLRenderPipelineState>  s_pso_crop; /* display.overscan > 0 */
static id<MTLSamplerState>         s_smp[2];   /* nearest, linear */
static SDL_MetalView               s_view;
static CAMetalLayer               *s_layer;
static int                         s_unified;
static id<MTLTexture>              s_slot[3];
/* The last present pass that sampled each slot: an upload (a CPU write) into
 * a slot waits for it; a GPU blit needs no wait (tracked hazards). Written by
 * the main thread while the slot is front, read by the ack thread once the
 * hand-off's lock has made it back. */
static id<MTLCommandBuffer>        s_slot_cb[3];
static uint32_t                   *s_conv;     /* R5G6B5 -> BGRA8 rows */
static size_t                      s_conv_cap;
static int                         s_internal;  /* present.surface=internal */
static int                         s_layer_filter = -1;
#ifdef XBOXRECOMP_ENHANCE
static id<MTLRenderPipelineState>  s_pso_lines; /* present.shader: guest-line resolve */
static id<MTLRenderPipelineState>  s_pso_ovl;  /* the overlay, premultiplied */
static id<MTLTexture>              s_ovl_tex[1 + NV2A_OVL_IMAGES];
static uint64_t                    s_ovl_tex_gen[1 + NV2A_OVL_IMAGES];
/* Each layer's vertices and indices, copied in only when its gen moves.
 * Three sets a layer, so a new list never overwrites one a frame still in
 * flight reads; the last command buffer that used a set is waited on
 * before the set is written again. */
#define OVL_SETS 3
static struct ovl_set {
    id<MTLBuffer> vb, ib;
    NSUInteger vcap, icap;
    uint64_t gen;
    id<MTLCommandBuffer> cb;
} s_ovl_set[NV2A_OVL_COUNT][OVL_SETS];
static int s_ovl_cur[NV2A_OVL_COUNT];
#endif

static const char s_blit_msl[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct VO { float4 p [[position]]; float2 t; };\n"
    "static VO tri(uint i) {\n"
    "    VO o;\n"
    "    o.t = float2((i << 1) & 2, i & 2);\n"
    "    o.p = float4(o.t * float2(2, -2) + float2(-1, 1), 0, 1);\n"
    "    return o;\n"
    "}\n"
    "vertex VO vs_blit(uint i [[vertex_id]]) { return tri(i); }\n"
    "vertex VO vs_crop(uint i [[vertex_id]], constant float4 &crop [[buffer(0)]]) {\n"
    "    VO o = tri(i);\n"
    "    o.t = o.t * crop.xy + crop.zw;\n"
    "    return o;\n"
    "}\n"
    "fragment float4 fs_blit(VO i [[stage_in]], texture2d<float> t [[texture(0)]],\n"
    "                        sampler s [[sampler(0)]]) {\n"
    "    return float4(t.sample(s, i.t).rgb, 1);\n"
    "}\n"
    /* present.shader's resolve to the guest's lines: each row the mean of
     * its n host rows, at the whole host width (D3D11's s_lrs_res_src). */
    "fragment float4 fs_lines(VO i [[stage_in]], texture2d<float> t [[texture(0)]],\n"
    "                         constant uint &n [[buffer(0)]]) {\n"
    "    uint2 q = uint2(i.p.xy);\n"
    "    float3 c = 0;\n"
    "    for (uint k = 0; k < n; k++)\n"
    "        c += t.read(uint2(q.x, q.y * n + k)).rgb;\n"
    "    return float4(c / float(n), 1);\n"
    "}\n"
    /* The overlay: a list's vertices and indices as they are (20-byte
     * vertices, uint16 indices), each index read by vertex_id, so a command
     * is drawPrimitives over its index range. xf maps the list's output
     * pixels into the placement in clip space. */
    "struct OV { packed_float2 p; packed_float2 t; uint c; };\n"
    "struct OO { float4 p [[position]]; float2 t; float4 c; };\n"
    "vertex OO vs_ovl(uint i [[vertex_id]], const device OV *v [[buffer(0)]],\n"
    "                 const device ushort *ix [[buffer(1)]],\n"
    "                 constant float4 &xf [[buffer(2)]]) {\n"
    "    OV a = v[ix[i]];\n"
    "    OO o;\n"
    "    o.p = float4(float2(a.p) * xf.xy + xf.zw, 0, 1);\n"
    "    o.t = float2(a.t);\n"
    "    o.c = unpack_unorm4x8_to_float(a.c);\n"
    "    return o;\n"
    "}\n"
    "fragment float4 fs_ovl(OO i [[stage_in]], texture2d<float> t [[texture(0)]],\n"
    "                       sampler s [[sampler(0)]]) {\n"
    "    return t.sample(s, i.t) * i.c;\n"
    "}\n";

static int fail(char *err, size_t n, const char *what)
{
    snprintf(err, n, "%s", what);
    return 0;
}

int fbm_create(SDL_Window *win, int vsync, int shots, char *err, size_t errn)
{
    NSError *e = nil;
    id<MTLLibrary> lib;
    MTLRenderPipelineDescriptor *pd;
    MTLSamplerDescriptor *sd;
    int w = 0, h = 0, k;

    @autoreleasepool {
        s_dev = MTLCreateSystemDefaultDevice();
        if (!s_dev)
            return fail(err, errn, "no Metal device");
        s_queue = [s_dev newCommandQueue];
        if (!s_queue)
            return fail(err, errn, "no command queue");
        s_unified = [s_dev hasUnifiedMemory] ? 1 : 0;
        /* The backend's options (metal_compile.h): one language version and
         * math mode per process, and fs_blit's texture arguments compile in
         * milliseconds at the pin instead of about 100 ms more. */
        lib = metal_compile_library(s_dev, [NSString stringWithUTF8String:s_blit_msl],
                                    "the present library", &e);
        if (!lib)
            return fail(err, errn, "present shader failed to compile");
        pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction = [[lib newFunctionWithName:@"vs_blit"] autorelease];
        pd.fragmentFunction = [[lib newFunctionWithName:@"fs_blit"] autorelease];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        s_pso = [s_dev newRenderPipelineStateWithDescriptor:pd error:&e];
        pd.vertexFunction = [[lib newFunctionWithName:@"vs_crop"] autorelease];
        s_pso_crop = [s_dev newRenderPipelineStateWithDescriptor:pd error:&e];
#ifdef XBOXRECOMP_ENHANCE
        pd.vertexFunction = [[lib newFunctionWithName:@"vs_blit"] autorelease];
        pd.fragmentFunction = [[lib newFunctionWithName:@"fs_lines"] autorelease];
        s_pso_lines = [s_dev newRenderPipelineStateWithDescriptor:pd error:&e];
        if (!s_pso_lines)
            fprintf(stderr, "[SHADER] the guest-line resolve failed; the chain reads the host rows\n");
        pd.vertexFunction = [[lib newFunctionWithName:@"vs_ovl"] autorelease];
        pd.fragmentFunction = [[lib newFunctionWithName:@"fs_ovl"] autorelease];
        pd.colorAttachments[0].blendingEnabled = YES;
        pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
        pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        s_pso_ovl = [s_dev newRenderPipelineStateWithDescriptor:pd error:&e];
        if (!s_pso_ovl)
            fprintf(stderr, "[PRESENT] overlay pipeline failed; the layer draws no overlay\n");
#endif
        [pd release];
        [lib release];
        if (!s_pso || !s_pso_crop)
            return fail(err, errn, "present pipeline failed");
        for (k = 0; k < 2; k++) {
            sd = [[MTLSamplerDescriptor alloc] init];
            sd.minFilter = sd.magFilter = k ? MTLSamplerMinMagFilterLinear
                                             : MTLSamplerMinMagFilterNearest;
            sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
            s_smp[k] = [s_dev newSamplerStateWithDescriptor:sd];
            [sd release];
        }

        s_view = SDL_Metal_CreateView(win);
        if (!s_view)
            return fail(err, errn, SDL_GetError());
        s_layer = (CAMetalLayer *)SDL_Metal_GetLayer(s_view);
        if (!s_layer)
            return fail(err, errn, "SDL_Metal_GetLayer returned NULL");
        [s_layer retain];
        s_layer.device = s_dev;
        s_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        /* The SDL renderer this replaces draws SDR content as sRGB. */
        {
            CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
            s_layer.colorspace = cs;
            CGColorSpaceRelease(cs);
        }
        s_layer.framebufferOnly = shots ? NO : YES;
        s_layer.displaySyncEnabled = vsync ? YES : NO;
        /* RECOMP_DEBUG=metal_drawables=2 (KFC-2115, hub 20-nov): a probe of
         * frame age (flip to presented), 2 drawables instead of 3 */
        {
            long nd = recomp_env_int(RENV_METAL_DRAWABLES, 3);
            s_layer.maximumDrawableCount = nd == 2 ? 2 : 3;
            if (nd == 2)
                fprintf(stderr, "[PRESENT] metal_drawables=2: maximumDrawableCount 2
");
        }
        s_layer.allowsNextDrawableTimeout = YES;
        s_internal = nv2a_host_opts()->present_surface == NV2A_PRESENT_SURFACE_INTERNAL;
        if (s_internal) {
            /* Letterboxed like nv2a_present_rect's fit; fbm_present keeps
             * the drawable at the frame's size. */
            s_layer.contentsGravity = kCAGravityResizeAspect;
            w = (int)nv2a_host_window_w(640, 480, nv2a_host_render_scale());
            h = 480 * (int)nv2a_host_render_scale();
        } else {
            SDL_GetWindowSizeInPixels(win, &w, &h);
        }
        if (w > 0 && h > 0)
            s_layer.drawableSize = CGSizeMake(w, h);
    }
#ifdef XBOXRECOMP_ENHANCE
    /* Every backend presents through here on macOS, the CPU rasteriser's
     * frames too, so the menu's Shaders page is shown for all of them. */
    nv2a_present_chain_host("Metal");
#endif
    return 1;
}

/* present.surface=internal: the layer's scaling filter follows the present
 * filter (integer has no layer equivalent and scales linearly, as its
 * non-whole fallback does). */
static void layer_filter(int filter)
{
    NSString *f;
    if (filter == s_layer_filter)
        return;
    s_layer_filter = filter;
    f = filter == NV2A_PRESENT_NEAREST ? kCAFilterNearest : kCAFilterLinear;
    s_layer.magnificationFilter = f;
    s_layer.minificationFilter = f;
}

#ifdef XBOXRECOMP_ENHANCE
static void chain_free(void);
#endif

void fbm_destroy(void)
{
    int k;
    @autoreleasepool {
        for (k = 0; k < 3; k++) {
            if (s_slot_cb[k])
                [s_slot_cb[k] waitUntilCompleted];
            [s_slot_cb[k] release];
            s_slot_cb[k] = nil;
            [s_slot[k] release];
            s_slot[k] = nil;
        }
        [s_layer release];
        s_layer = nil;
        if (s_view)
            SDL_Metal_DestroyView(s_view);
        s_view = NULL;
        [s_pso release];
        s_pso = nil;
        [s_pso_crop release];
        s_pso_crop = nil;
#ifdef XBOXRECOMP_ENHANCE
        chain_free();
        [s_pso_lines release];
        s_pso_lines = nil;
        [s_pso_ovl release];
        s_pso_ovl = nil;
        for (k = 0; k <= NV2A_OVL_IMAGES; k++) {
            [s_ovl_tex[k] release];
            s_ovl_tex[k] = nil;
        }
        for (k = 0; k < NV2A_OVL_COUNT * OVL_SETS; k++) {
            struct ovl_set *o = &s_ovl_set[k / OVL_SETS][k % OVL_SETS];
            [o->vb release];
            [o->ib release];
            [o->cb release];
            memset(o, 0, sizeof *o);
        }
#endif
        [s_smp[0] release];
        [s_smp[1] release];
        s_smp[0] = s_smp[1] = nil;
        /* The device and queue stay: the backend holds its own references
         * and the process is about to exit. On the fallback path (setup
         * failed, the window goes back to the SDL renderer) that leaks the
         * device and queue made here for the run: the backend has not
         * started yet and creates its own. */
    }
}

void fbm_get(void **device, void **queue)
{
    *device = (void *)s_dev;
    *queue = (void *)s_queue;
}

/* Slot k at w x h, made or re-made at that size. Releasing the old texture
 * while a present pass reads it is fine: a committed command buffer retains
 * what it uses. */
void *fbm_slot(int k, uint32_t w, uint32_t h)
{
    @autoreleasepool {
        if (!s_slot[k] || [s_slot[k] width] != w || [s_slot[k] height] != h) {
            MTLTextureDescriptor *td = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                             width:w height:h mipmapped:NO];
            td.usage = MTLTextureUsageShaderRead;
            /* Shared: an upload needs replaceRegion, which a private texture
             * refuses, and a blit destination gains nothing from private.
             * Managed on a discrete GPU, as the backend's targets. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            td.storageMode = s_unified ? MTLStorageModeShared : MTLStorageModeManaged;
#pragma clang diagnostic pop
            [s_slot[k] release];
            s_slot[k] = [s_dev newTextureWithDescriptor:td];
        }
    }
    return (void *)s_slot[k];
}

/* A guest-bytes frame (bpp 4 = X8R8G8B8, which is BGRA8's byte order;
 * bpp 2 = R5G6B5) into slot k. */
int fbm_upload(int k, const uint8_t *src, uint32_t w, uint32_t h, uint32_t pitch,
               uint32_t bpp)
{
    id<MTLTexture> t;
    uint32_t x, y;

    @autoreleasepool {
        if (s_slot_cb[k]) {
            [s_slot_cb[k] waitUntilCompleted];
            [s_slot_cb[k] release];
            s_slot_cb[k] = nil;
        }
        t = (id<MTLTexture>)fbm_slot(k, w, h);
        if (!t)
            return 0;
        if (bpp == 4) {
            [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:src
                 bytesPerRow:pitch];
        } else {
            if (s_conv_cap < w) {
                uint32_t *n = (uint32_t *)realloc(s_conv, (size_t)w * 4);
                if (!n)
                    return 0;
                s_conv = n;
                s_conv_cap = w;
            }
            for (y = 0; y < h; y++) {
                const uint16_t *r = (const uint16_t *)(src + (size_t)y * pitch);
                for (x = 0; x < w; x++) {
                    uint32_t v = r[x];
                    uint32_t b = (v & 0x1F) << 3, g = ((v >> 5) & 0x3F) << 2,
                             rr = ((v >> 11) & 0x1F) << 3;
                    s_conv[x] = 0xFF000000u | (rr << 16) | (g << 8) | b;
                }
                [t replaceRegion:MTLRegionMake2D(0, y, w, 1) mipmapLevel:0 withBytes:s_conv
                     bytesPerRow:w * 4];
            }
        }
    }
    return 1;
}

static void save_shot(id<MTLBuffer> buf, uint32_t w, uint32_t h, const char *path,
                      uint32_t flip)
{
    SDL_Surface *surf = SDL_CreateSurface((int)w, (int)h, SDL_PIXELFORMAT_BGR24);
    const uint8_t *px = (const uint8_t *)[buf contents];
    uint32_t x, y;

    if (!surf)
        return;
    for (y = 0; y < h; y++) {
        uint8_t *o = (uint8_t *)surf->pixels + (size_t)y * surf->pitch;
        const uint8_t *in = px + (size_t)y * w * 4;
        for (x = 0; x < w; x++) {
            /* BGRA in the drawable; BGR24 is the bytes B, G, R. */
            o[x * 3 + 0] = in[x * 4 + 0];
            o[x * 3 + 1] = in[x * 4 + 1];
            o[x * 3 + 2] = in[x * 4 + 2];
        }
    }
    if (SDL_SaveBMP(surf, path))
        fprintf(stderr, "[PRESENT] window shot: %s (%ux%u, flip %u)\n", path, w, h, flip);
    else
        fprintf(stderr, "[PRESENT] window shot at flip %u failed: %s\n", flip, SDL_GetError());
    SDL_DestroySurface(surf);
}

/* Slot k (fw x fh, drawn at render scale n with hor+ pad pad) into the
 * window. 0 when no drawable came (timeout, display asleep, window hidden):
 * the caller keeps the frame and tries again. */
#ifdef XBOXRECOMP_ENHANCE
/* Texture k of the overlay (0 the atlas, 1.. the image slots), re-made when
 * its gen moves. */
static id<MTLTexture> ovl_texture(uint16_t k)
{
    const uint32_t *px;
    uint32_t w, h;
    uint64_t gen;
    MTLTextureDescriptor *td;
    if (k == 0) {
        const struct nv2a_ovl_atlas *a = nv2a_overlay_atlas();
        if (!a)
            return nil;
        px = a->px, w = a->w, h = a->h, gen = a->gen;
    } else {
        const struct nv2a_ovl_image *im = k <= NV2A_OVL_IMAGES ? nv2a_overlay_image(k) : NULL;
        if (!im || !im->px)
            return nil;
        px = im->px, w = im->w, h = im->h, gen = im->gen;
    }
    if (s_ovl_tex[k] && s_ovl_tex_gen[k] == gen)
        return s_ovl_tex[k];
    [s_ovl_tex[k] release];
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                            width:w height:h mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    s_ovl_tex[k] = [s_dev newTextureWithDescriptor:td];
    s_ovl_tex_gen[k] = gen;
    [s_ovl_tex[k] replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0
                      withBytes:px bytesPerRow:(NSUInteger)w * 4];
    return s_ovl_tex[k];
}

/* Grows b to hold n bytes, doubling from 4 KiB as D3D11's ovl_fill does. */
static int ovl_grow(id<MTLBuffer> *b, NSUInteger *cap, NSUInteger n)
{
    NSUInteger c;
    if (n <= *cap)
        return 1;
    for (c = *cap ? *cap : 4096; c < n; c *= 2)
        ;
    [*b release];
    *b = [s_dev newBufferWithLength:c options:MTLResourceStorageModeShared];
    *cap = *b ? c : 0;
    return *b != nil;
}

/* The set holding list l's current gen, filled when the gen moved; the
 * command buffer cb now uses it. */
static struct ovl_set *ovl_set(const struct nv2a_ovl_list *l, id<MTLCommandBuffer> cb)
{
    int y = (int)l->layer;
    struct ovl_set *o;
    if (y < 0 || y >= NV2A_OVL_COUNT)
        return NULL;
    o = &s_ovl_set[y][s_ovl_cur[y]];
    if (o->gen != l->gen) {
        s_ovl_cur[y] = (s_ovl_cur[y] + 1) % OVL_SETS;
        o = &s_ovl_set[y][s_ovl_cur[y]];
        if (o->cb && [o->cb status] < MTLCommandBufferStatusCompleted)
            [o->cb waitUntilCompleted];
        o->gen = 0;
        if (!ovl_grow(&o->vb, &o->vcap, l->nvtx * sizeof *l->vtx)
            || !ovl_grow(&o->ib, &o->icap, l->nidx * sizeof *l->idx))
            return NULL;
        memcpy([o->vb contents], l->vtx, l->nvtx * sizeof *l->vtx);
        memcpy([o->ib contents], l->idx, l->nidx * sizeof *l->idx);
        o->gen = l->gen;
    }
    [o->cb release];
    o->cb = [cb retain];
    return o;
}

/* The lists over the frame at r, in a drawable of dw x dh. */
static void ovl_draw(id<MTLRenderCommandEncoder> enc, id<MTLCommandBuffer> cb,
                     const struct nv2a_ovl_list *const *ls, int nls, struct nv2a_rect r,
                     int dw, int dh)
{
    int i;
    uint32_t c;
    MTLViewport vp = { 0, 0, dw, dh, 0.0, 1.0 };
    [enc setRenderPipelineState:s_pso_ovl];
    [enc setViewport:vp];
    [enc setFragmentSamplerState:s_smp[1] atIndex:0];
    for (i = 0; i < nls; i++) {
        const struct nv2a_ovl_list *l = ls[i];
        float sx = l->out_w ? (float)r.w / (float)l->out_w : 1.0f;
        float sy = l->out_h ? (float)r.h / (float)l->out_h : 1.0f;
        float xf[4] = { sx * 2.0f / dw, -sy * 2.0f / dh,
                        r.x * 2.0f / dw - 1.0f, 1.0f - r.y * 2.0f / dh };
        struct ovl_set *o = ovl_set(l, cb);
        if (o) {
            [enc setVertexBuffer:o->vb offset:0 atIndex:0];
            [enc setVertexBuffer:o->ib offset:0 atIndex:1];
            [enc setVertexBytes:xf length:sizeof xf atIndex:2];
            for (c = 0; c < l->ncmd; c++) {
                const struct nv2a_ovl_cmd *k = &l->cmd[c];
                id<MTLTexture> t = ovl_texture(k->tex);
                long x0 = r.x + (long)(k->clip[0] * sx), y0 = r.y + (long)(k->clip[1] * sy);
                long x1 = r.x + (long)(k->clip[0] * sx + k->clip[2] * sx + 0.999f);
                long y1 = r.y + (long)(k->clip[1] * sy + k->clip[3] * sy + 0.999f);
                MTLScissorRect sc;
                if (!t || !k->count)
                    continue;
                x0 = x0 < 0 ? 0 : x0 > dw ? dw : x0;
                y0 = y0 < 0 ? 0 : y0 > dh ? dh : y0;
                x1 = x1 < x0 ? x0 : x1 > dw ? dw : x1;
                y1 = y1 < y0 ? y0 : y1 > dh ? dh : y1;
                if (x1 == x0 || y1 == y0)
                    continue;
                sc.x = (NSUInteger)x0;
                sc.y = (NSUInteger)y0;
                sc.width = (NSUInteger)(x1 - x0);
                sc.height = (NSUInteger)(y1 - y0);
                [enc setScissorRect:sc];
                [enc setFragmentTexture:t atIndex:0];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:k->first
                        vertexCount:k->count];
            }
        }
    }
}
#endif /* XBOXRECOMP_ENHANCE */

#ifdef XBOXRECOMP_ENHANCE
/* present.shader's chain: the one built for s_chain_gen, or NULL (off, or
 * failed for that generation). Main thread only, like the rest. */
static libra_mtl_filter_chain_t s_chain;
static unsigned s_chain_gen;
static size_t s_chain_frames;
static PFN_libra_mtl_filter_chain_frame s_chain_ff;   /* with the chain */
static PFN_libra_mtl_filter_chain_set_param s_chain_sp;
static unsigned s_chain_pgen;       /* the parameter generation applied */
static int s_chain_ov_said;
static int s_chain_lines;           /* the preset draws per source line */
static int s_chain_said;            /* the chain's first frame said how it was fed */
/* The chain's input resolved to the guest's lines (nv2a_present_chain_source),
 * and its output at the rectangle's size, drawn 1:1 into the drawable after:
 * librashader gives the last pass the whole texture's size as OutputSize
 * and FinalViewportSize, not the viewport's, so drawn straight into a
 * letterboxed rectangle the mask was laid out for the whole drawable and
 * squeezed into it (D3D11's s_lrs_out). */
static id<MTLTexture> s_chain_src, s_chain_out;

/* *t at w x h (BGRA8, private, a render target that is read after), made
 * again when the size changed. A command buffer still in flight keeps the
 * old one. */
static id<MTLTexture> chain_tex(id<MTLTexture> *t, uint32_t w, uint32_t h)
{
    if (!*t || [*t width] != w || [*t height] != h) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                         width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        [*t release];
        *t = [s_dev newTextureWithDescriptor:td];
    }
    return *t;
}

static void chain_free(void)
{
    const struct nv2a_lrs *l;
    PFN_libra_mtl_filter_chain_free fr;
    if (!s_chain)
        return;
    l = nv2a_present_chain_lib();
    fr = l ? (PFN_libra_mtl_filter_chain_free)l->sym("libra_mtl_filter_chain_free") : NULL;
    if (fr)
        fr(&s_chain);
    s_chain = NULL;
    s_chain_ff = NULL;
    s_chain_sp = NULL;
    [s_chain_src release];
    [s_chain_out release];
    s_chain_src = s_chain_out = nil;
    nv2a_present_chain_params_clear();
}

/* nv2a_present_chain_params_apply's setter. librashader refuses a name the
 * chain lacks (UnknownShaderParameter), but the names come from the table
 * read off this chain's own preset, so none is unknown; an error here would
 * repeat on every change, so it is dropped unlogged. */
static void chain_set_param(void *ctx, const char *name, float v)
{
    libra_error_t e;
    char msg[256];
    if (!s_chain_sp)
        return;
    e = s_chain_sp((libra_mtl_filter_chain_t *)ctx, name, v);
    if (e)
        nv2a_present_chain_error(e, msg, sizeof msg);
}

/* The chain for the current request, (re)built when its generation moved:
 * NULL for "none", a missing library or a preset that failed (said once,
 * then not retried until the next request). */
static libra_mtl_filter_chain_t chain_get(void)
{
    unsigned gen = nv2a_present_chain_gen();
    const struct nv2a_lrs *l;
    PFN_libra_preset_create pc;
    PFN_libra_preset_free pf;
    PFN_libra_mtl_filter_chain_create cc;
    libra_shader_preset_t preset = NULL;
    libra_error_t e;
    filter_chain_mtl_opt_t co = { LIBRASHADER_CURRENT_VERSION, false };
    char path[1024], msg[512];
    int found;

    if (gen == s_chain_gen)
        return s_chain;
    chain_free();
    s_chain_gen = gen;
    s_chain_frames = 0;
    if (!nv2a_present_chain_wanted())
        return NULL;
    found = nv2a_present_chain_preset_lines(path, sizeof path, NULL, &s_chain_lines);
    s_chain_said = 0;
    l = found > 0 ? nv2a_present_chain_lib() : NULL;
    if (!l) {
        nv2a_present_chain_failed(gen);
        return NULL;
    }
    pc = (PFN_libra_preset_create)l->sym("libra_preset_create");
    pf = (PFN_libra_preset_free)l->sym("libra_preset_free");
    cc = (PFN_libra_mtl_filter_chain_create)l->sym("libra_mtl_filter_chain_create");
    s_chain_ff = (PFN_libra_mtl_filter_chain_frame)l->sym("libra_mtl_filter_chain_frame");
    s_chain_sp = (PFN_libra_mtl_filter_chain_set_param)l->sym("libra_mtl_filter_chain_set_param");
    if (!pc || !pf || !cc || !s_chain_ff) {
        fprintf(stderr, "[SHADER] present.shader off: the library has no Metal runtime\n");
        nv2a_present_chain_failed(gen);
        return NULL;
    }
    e = pc(path, &preset);
    /* The parameters before the chain: creating it consumes the preset. */
    if (!e)
        nv2a_present_chain_params_publish(&preset, gen);
    if (!e)
        e = cc(&preset, s_queue, &co, &s_chain);
    if (preset)
        pf(&preset);
    if (e) {
        nv2a_present_chain_error(e, msg, sizeof msg);
        fprintf(stderr, "[SHADER] present.shader off: %s: %s\n", path, msg);
        s_chain = NULL;
        nv2a_present_chain_params_clear();
        nv2a_present_chain_failed(gen);
        return NULL;
    }
    /* A new chain starts at the preset's values: apply every override. */
    s_chain_pgen = nv2a_present_chain_params_gen() - 1;
    fprintf(stderr, "[SHADER] present.shader: %s (Metal)\n", path);
    return s_chain;
}
#endif /* XBOXRECOMP_ENHANCE */

/* RECOMP_METAL_UPSCALE=metalfx: the composed frame (the slot: the guest's
 * size times render.scale, HUD included) is upscaled to the letterboxed
 * rectangle by MetalFX's spatial scaler instead of the linear blit, then
 * drawn 1:1 like present.shader's output. Off by default. The blit stays
 * when MetalFX is missing (before macOS 13, or a device it does not
 * support), when the rectangle is not larger than the frame, with
 * display.overscan (the scaler reads the whole frame) and with
 * present.shader. The slot is shared storage, so it is copied into a
 * private texture the scaler reads. Main thread only, like the rest. */
static int s_fx_want = -1;
static id s_fx;                         /* id<MTLFXSpatialScaler>, retained */
static id<MTLTexture> s_fx_in, s_fx_out;
static uint32_t s_fx_iw, s_fx_ih, s_fx_ow, s_fx_oh;
static int s_fx_ov_said;

static int fx_wanted(void)
{
    if (s_fx_want < 0) {
        const char *e = getenv("RECOMP_METAL_UPSCALE");
        s_fx_want = e && !strcmp(e, "metalfx");
    }
    return s_fx_want;
}

static void fx_off(const char *why)
{
    fprintf(stderr, "[METALFX] off: %s; the linear blit presents\n", why);
    s_fx_want = 0;
}

/* The scaler for iw x ih -> ow x oh with its two textures, made again when a
 * size changed (a command buffer in flight keeps the old ones); nil when
 * MetalFX cannot run here, which also turns the option off. */
static id fx_get(uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh)
{
    if (s_fx && s_fx_iw == iw && s_fx_ih == ih && s_fx_ow == ow && s_fx_oh == oh)
        return s_fx;
    [s_fx release];
    [s_fx_in release];
    [s_fx_out release];
    s_fx = nil;
    s_fx_in = s_fx_out = nil;
    if (@available(macOS 13.0, *)) {
        MTLFXSpatialScalerDescriptor *sd;
        id<MTLFXSpatialScaler> sc;
        MTLTextureDescriptor *td;

        if (![MTLFXSpatialScalerDescriptor supportsDevice:s_dev]) {
            fx_off("the device has no MetalFX spatial scaler");
            return nil;
        }
        sd = [[MTLFXSpatialScalerDescriptor alloc] init];
        sd.inputWidth = iw;
        sd.inputHeight = ih;
        sd.outputWidth = ow;
        sd.outputHeight = oh;
        sd.colorTextureFormat = MTLPixelFormatBGRA8Unorm;
        sd.outputTextureFormat = MTLPixelFormatBGRA8Unorm;
        /* The slot holds gamma-encoded SDR (the layer is sRGB). */
        sd.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
        sc = [sd newSpatialScalerWithDevice:s_dev];
        [sd release];
        if (!sc) {
            fx_off("the scaler could not be made");
            return nil;
        }
        td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                width:iw height:ih mipmapped:NO];
        td.storageMode = MTLStorageModePrivate;
        td.usage = sc.colorTextureUsage | MTLTextureUsageShaderRead;
        s_fx_in = [s_dev newTextureWithDescriptor:td];
        td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                width:ow height:oh mipmapped:NO];
        td.storageMode = MTLStorageModePrivate;
        td.usage = sc.outputTextureUsage | MTLTextureUsageShaderRead;
        s_fx_out = [s_dev newTextureWithDescriptor:td];
        if (!s_fx_in || !s_fx_out) {
            [sc release];
            fx_off("its textures could not be made");
            return nil;
        }
        s_fx = sc;
        s_fx_iw = iw;
        s_fx_ih = ih;
        s_fx_ow = ow;
        s_fx_oh = oh;
        fprintf(stderr, "[METALFX] spatial %ux%u -> %ux%u\n", iw, ih, ow, oh);
        return s_fx;
    }
    fx_off("MetalFX needs macOS 13");
    return nil;
}

double fbm_now(void)
{
    return CACurrentMediaTime();
}

/* RECOMP_TRACE=pacing (fbm_present's pace): what each present really put on
 * screen. One record a present, claimed on the main thread after
 * nextDrawable; the drawable's presented handler and the command buffer's
 * completed handler (any thread, possibly at once) each write only their own
 * field of it and then set their own done flag; the main thread reads a
 * record once both flags are set, in submit order, after the commit. A
 * full table skips the handlers for that present ("untracked"): nothing
 * waits, and a record is never reused while a handler may still write it.
 *
 *   skipped  frames published (hand-off seq) that no present was ever
 *            submitted with: the gap between consecutive submitted seqs.
 *   dropped  a present the system never showed (presentedTime 0).
 *   new      shown, with a seq past the last one shown.
 *   repeat   shown, with the seq already on screen.
 * Presents before the first frame (seq 0, the black paint) are not counted.
 * age is presentedTime - publish time, for new images only (the first time
 * a frame is on screen); the interval is between consecutive new images'
 * presentedTime. The GPU time is the whole present command buffer's
 * (GPUEndTime - GPUStartTime): MetalFX, the shader chain and a shot blit
 * included when they run. The age split (KFC-2115, hub 20-nov), p50/p95 of
 * each new image's age in five parts on the same clock: pickup, publish to
 * fbm_present's entry (the SDL main loop's turn); drawable, nextDrawable's
 * wait; encode, to the commit; gpu, the commit to the present pass's
 * GPUEndTime; display, GPUEndTime to presentedTime. */
#define PACE_RECS 64
#define PACE_SAMPLES 4096
static struct pace_rec {
    uint64_t seq, skip;
    double pub;
    double shown;               /* the presented handler's */
    double gpu;                 /* the completed handler's */
    double t_in, t_drw, t_cmt;  /* fbm_present's entry, nextDrawable back, commit */
    double gend;                /* the completed handler's GPUEndTime */
    int shown_done, gpu_done;   /* atomics: the handlers' release stores */
} s_pr[PACE_RECS];
static uint64_t s_pr_head, s_pr_tail;  /* main only: oldest live, next to claim */
static uint64_t s_pc_sub_seq;          /* the last seq submitted */
static uint64_t s_pc_shown_seq;        /* the last seq on screen */
static double s_pc_last_new;           /* presentedTime of the last new image */
static struct pace_win {        /* main only: the line being gathered */
    double t0;
    unsigned shown, fresh, repeat, dropped, untracked;
    uint64_t skipped;
    unsigned ni, na, ng, nq;
    float iv[PACE_SAMPLES], age[PACE_SAMPLES], gpu[PACE_SAMPLES];
    float split[5][PACE_SAMPLES];   /* nq each: the age split */
} s_pw;

static int pace_cmp(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}

/* p (0..1) of the sorted v[0..n) in ms with dec decimals, or "-" when empty. */
static const char *pace_pct(const float *v, unsigned n, double p, int dec, char *buf,
                            size_t bn)
{
    if (!n)
        return "-";
    snprintf(buf, bn, "%.*f", dec, (double)v[(size_t)(p * (n - 1) + 0.5)]);
    return buf;
}

/* The first present counted in a line (shown or untracked) starts its 10 s. */
static void pace_arm(void)
{
    if (!s_pw.shown && !s_pw.untracked)
        s_pw.t0 = CACurrentMediaTime();
}

static void pace_print(void)
{
    char a[16], b[16], c[16], d[16], e[16], f[16];
    double gmean = 0;
    unsigned i;

    qsort(s_pw.iv, s_pw.ni, sizeof *s_pw.iv, pace_cmp);
    qsort(s_pw.age, s_pw.na, sizeof *s_pw.age, pace_cmp);
    qsort(s_pw.gpu, s_pw.ng, sizeof *s_pw.gpu, pace_cmp);
    for (i = 0; i < s_pw.ng; i++)
        gmean += s_pw.gpu[i];
    if (s_pw.ng)
        gmean /= s_pw.ng;
    fprintf(stderr, "[PRESENT] shown %u: new %u repeat %u skipped %llu dropped %u"
            " \xc2\xb7 new-image interval p50 %s p95 %s ms \xc2\xb7 age p50 %s p95 %s ms"
            " \xc2\xb7 present pass gpu %.2f ms (p50 %s p95 %s)",
            s_pw.shown, s_pw.fresh, s_pw.repeat, (unsigned long long)s_pw.skipped,
            s_pw.dropped,
            pace_pct(s_pw.iv, s_pw.ni, 0.50, 1, a, sizeof a),
            pace_pct(s_pw.iv, s_pw.ni, 0.95, 1, b, sizeof b),
            pace_pct(s_pw.age, s_pw.na, 0.50, 1, c, sizeof c),
            pace_pct(s_pw.age, s_pw.na, 0.95, 1, d, sizeof d),
            gmean,
            pace_pct(s_pw.gpu, s_pw.ng, 0.50, 2, e, sizeof e),
            pace_pct(s_pw.gpu, s_pw.ng, 0.95, 2, f, sizeof f));
    if (s_pw.untracked)
        fprintf(stderr, " untracked %u", s_pw.untracked);
    if (s_pw.nq) {
        static const char *const nm[5] = { "pickup", "drawable", "encode", "gpu", "display" };
        fprintf(stderr, " \xc2\xb7 age split p50/p95 ms:");
        for (i = 0; i < 5; i++) {
            qsort(s_pw.split[i], s_pw.nq, sizeof s_pw.split[i][0], pace_cmp);
            fprintf(stderr, " %s %s/%s", nm[i],
                    pace_pct(s_pw.split[i], s_pw.nq, 0.50, 1, a, sizeof a),
                    pace_pct(s_pw.split[i], s_pw.nq, 0.95, 1, b, sizeof b));
        }
    }
    fputc('\n', stderr);
    /* The counters; the samples are overwritten, not read past the counts. */
    memset(&s_pw, 0, offsetof(struct pace_win, iv));
}

/* The records both handlers are done with, oldest first; a line every 600
 * shown or 10 s. Main thread, after the commit. */
static void pace_drain(void)
{
    while (s_pr_head != s_pr_tail) {
        struct pace_rec *r = &s_pr[s_pr_head % PACE_RECS];
        if (!__atomic_load_n(&r->shown_done, __ATOMIC_ACQUIRE)
                || !__atomic_load_n(&r->gpu_done, __ATOMIC_ACQUIRE))
            break;
        s_pr_head++;
        pace_arm();
        s_pw.shown++;
        s_pw.skipped += r->skip;
        if (r->gpu > 0 && s_pw.ng < PACE_SAMPLES)
            s_pw.gpu[s_pw.ng++] = (float)(r->gpu * 1e3);
        if (r->shown <= 0) {
            s_pw.dropped++;
        } else if (r->seq > s_pc_shown_seq) {
            s_pw.fresh++;
            if (s_pw.na < PACE_SAMPLES)
                s_pw.age[s_pw.na++] = (float)((r->shown - r->pub) * 1e3);
            if (r->gend > 0 && s_pw.nq < PACE_SAMPLES) {
                double t[6] = { r->pub, r->t_in, r->t_drw, r->t_cmt, r->gend, r->shown };
                int i;
                for (i = 0; i < 5; i++)
                    s_pw.split[i][s_pw.nq] = (float)((t[i + 1] - t[i]) * 1e3);
                s_pw.nq++;
            }
            if (s_pc_last_new > 0 && s_pw.ni < PACE_SAMPLES)
                s_pw.iv[s_pw.ni++] = (float)((r->shown - s_pc_last_new) * 1e3);
            s_pc_last_new = r->shown;
            s_pc_shown_seq = r->seq;
        } else {
            s_pw.repeat++;
        }
    }
    /* untracked counts too: a handler that never fires fills the table, and
     * the line keeps coming, saying so, instead of the trace going quiet. */
    if (s_pw.shown >= 600
            || ((s_pw.shown || s_pw.untracked) && CACurrentMediaTime() - s_pw.t0 >= 10.0))
        pace_print();
}

/* A record for this present's handlers, or NULL (before the first frame,
 * or the table full). */
static struct pace_rec *pace_claim(const struct fbm_pace_frame *pf)
{
    struct pace_rec *r;
    uint64_t skip;
    if (!pf->seq)
        return NULL;
    skip = pf->seq > s_pc_sub_seq + 1 ? pf->seq - s_pc_sub_seq - 1 : 0;
    if (pf->seq > s_pc_sub_seq)
        s_pc_sub_seq = pf->seq;
    if (s_pr_tail - s_pr_head >= PACE_RECS) {
        pace_arm();
        s_pw.untracked++;
        s_pw.skipped += skip;
        return NULL;
    }
    r = &s_pr[s_pr_tail++ % PACE_RECS];
    r->seq = pf->seq;
    r->pub = pf->pub;
    r->skip = skip;
    r->shown = r->gpu = r->gend = 0;
    __atomic_store_n(&r->shown_done, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&r->gpu_done, 0, __ATOMIC_RELAXED);
    return r;
}

/* Both handlers on cb and d, before presentDrawable and the commit. Without
 * addPresentedHandler (before macOS 10.15.4) the record counts as dropped. */
static void pace_handlers(struct pace_rec *r, id<CAMetalDrawable> d, id<MTLCommandBuffer> cb)
{
    if (@available(macOS 10.15.4, *)) {
        [d addPresentedHandler:^(id<MTLDrawable> dr) {
            r->shown = [dr presentedTime];
            __atomic_store_n(&r->shown_done, 1, __ATOMIC_RELEASE);
        }];
    } else {
        __atomic_store_n(&r->shown_done, 1, __ATOMIC_RELEASE);
    }
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        CFTimeInterval a = [c GPUStartTime], b = [c GPUEndTime];
        r->gpu = b > a ? b - a : 0;
        r->gend = b;
        __atomic_store_n(&r->gpu_done, 1, __ATOMIC_RELEASE);
    }];
}

int fbm_present(SDL_Window *win, int k, uint32_t fw, uint32_t fh, unsigned n,
                uint32_t pad, int filter, const char *shot_path, uint32_t flip,
                const struct nv2a_ovl_list *const *layers, int nlayers,
                const struct fbm_pace_frame *pace)
{
    int ok = 0;
    double t_in = pace ? CACurrentMediaTime() : 0, t_drw = 0;   /* the age split */

    @autoreleasepool {
        id<CAMetalDrawable> d;
        id<MTLCommandBuffer> cb;
        id<MTLRenderCommandEncoder> enc;
        MTLRenderPassDescriptor *rp;
        id<MTLBuffer> shot = nil;
        struct nv2a_rect r = {0, 0, 0, 0};
        int whole, dw = 0, dh = 0;
        unsigned ov = nv2a_host_opts()->overscan;
        CGSize cur = s_layer.drawableSize;
        int internal = s_internal;
        int fx = fx_wanted(), scaled = 0;
        struct pace_rec *pr = NULL;
#ifdef XBOXRECOMP_ENHANCE
        libra_mtl_filter_chain_t chain = NULL;
        int chained = 0;

        if (nv2a_present_chain_wanted() || s_chain)
            chain = chain_get();
        if (chain) {
            nv2a_present_chain_params_apply(&s_chain_pgen, chain_set_param, &s_chain);
            /* The chain draws its mask and scanlines per output pixel: it
             * must draw at the window's size. Drawn at the frame's and
             * scaled after (by the layer here, by gamescope on the Deck),
             * they are resampled into colour fringes and moire. */
            internal = 0;
        }
        /* The chain reads the whole slot: a crop has no place in it. */
        if (chain && ov) {
            if (!s_chain_ov_said)
                fprintf(stderr, "[SHADER] display.overscan=%u ignored while"
                        " present.shader runs\n", ov);
            s_chain_ov_said = 1;
            ov = 0;
        }
        if (chain)
            fx = 0;
#endif
        if (fx && ov) {
            if (!s_fx_ov_said)
                fprintf(stderr, "[METALFX] off while display.overscan=%u (the"
                        " scaler reads the whole frame)\n", ov);
            s_fx_ov_said = 1;
            fx = 0;
        }
        /* The scaler writes window pixels: the drawable must be the window's. */
        if (fx)
            internal = 0;

        /* Checked every present, not only on resize events: a move between
         * displays of another scale need not arrive as SIZE_CHANGED. Under
         * present.surface=internal the drawable is the frame instead (a
         * render.scale change from the menu resizes it). */
        if (internal) {
            layer_filter(filter);
            if (fw && fh) {
                dw = (int)fw;
                dh = (int)fh;
            } else {
                dw = (int)cur.width;
                dh = (int)cur.height;
            }
        } else {
            SDL_GetWindowSizeInPixels(win, &dw, &dh);
        }
        if (dw > 0 && dh > 0 && ((int)cur.width != dw || (int)cur.height != dh))
            s_layer.drawableSize = CGSizeMake(dw, dh);
        d = [s_layer nextDrawable];
        if (!d)
            return 0;
        if (pace)
            t_drw = CACurrentMediaTime();
        dw = (int)[d.texture width];
        dh = (int)[d.texture height];
        {   /* the pixels the frame really goes out at, once and on each change */
            static int said_w, said_h;
            if (dw != said_w || dh != said_h) {
                int ww = 0, wh = 0;
                SDL_GetWindowSizeInPixels(win, &ww, &wh);
                fprintf(stderr, "[PRESENT] drawable %dx%d (window %dx%d px, frame %ux%u%s)\n",
                        dw, dh, ww, wh, (unsigned)fw, (unsigned)fh, internal ? ", internal" : "");
                said_w = dw;
                said_h = dh;
            }
        }
        cb = [s_queue commandBuffer];
        rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = d.texture;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        enc = [cb renderCommandEncoderWithDescriptor:rp];
        if (s_slot[k] && fw && fh) {
            struct nv2a_crop c = {0};
            /* The slot is exactly the frame, so the texture size is fw x fh. */
            if (ov)
                nv2a_present_crop(fw, fh, fw, fh, n, pad, ov, filter, &c);
            whole = nv2a_present_rect(ov ? c.fit_w : fw, ov ? c.fit_h : fh, (uint32_t)dw,
                                      (uint32_t)dh, filter, &r);
#ifdef XBOXRECOMP_ENHANCE
            /* The chain fills what the stock present shows: under
             * internal the layer's fit, not an integer multiple. */
            if (chain)
                whole = nv2a_present_chain_rect(fw, fh, (uint32_t)dw, (uint32_t)dh, filter,
                                                s_internal, &r);
#endif
            if (internal) {
                /* 1:1: the layer does the scaling. A crop is zoomed into the
                 * whole drawable with the present filter's sampler, except
                 * integer, which stays nearest (D3D11 matches). */
                r.x = r.y = 0;
                r.w = dw;
                r.h = dh;
                whole = 1;
                if (!ov || filter == NV2A_PRESENT_INTEGER)
                    filter = NV2A_PRESENT_NEAREST;
            }
            if (fx && r.w > (int)fw && r.h > (int)fh
                && fx_get(fw, fh, (uint32_t)r.w, (uint32_t)r.h)) {
                /* This pass only clears; the copy and the scaler run after
                 * it, and the next pass draws the output 1:1 into the
                 * rectangle before the overlay (present.shader's shape). */
                id<MTLFXSpatialScaler> sc = (id<MTLFXSpatialScaler>)s_fx;
                id<MTLBlitCommandEncoder> b;
                MTLViewport vp = { r.x, r.y, r.w, r.h, 0.0, 1.0 };

                [enc endEncoding];
                b = [cb blitCommandEncoder];
                [b copyFromTexture:s_slot[k] toTexture:s_fx_in];
                [b endEncoding];
                sc.colorTexture = s_fx_in;
                sc.outputTexture = s_fx_out;
                sc.inputContentWidth = fw;
                sc.inputContentHeight = fh;
                [sc encodeToCommandBuffer:cb];
                rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
                enc = [cb renderCommandEncoderWithDescriptor:rp];
                [enc setRenderPipelineState:s_pso];
                [enc setViewport:vp];
                [enc setFragmentTexture:s_fx_out atIndex:0];
                [enc setFragmentSamplerState:s_smp[0] atIndex:0];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                scaled = 1;
            }
#ifdef XBOXRECOMP_ENHANCE
            if (chain && r.w > 0 && r.h > 0
                && chain_tex(&s_chain_out, (uint32_t)r.w, (uint32_t)r.h)) {
                /* This pass only clears; the chain draws after it into
                 * s_chain_out, which the next pass draws 1:1 into the
                 * rectangle before the overlay. */
                libra_viewport_t lv = { 0.0f, 0.0f, (uint32_t)r.w, (uint32_t)r.h };
                frame_mtl_opt_t fo;
                libra_error_t e;
                char msg[512];
                id<MTLTexture> src = s_slot[k];
                uint32_t sw, sh;
                int resolved = 0;

                memset(&fo, 0, sizeof fo);
                fo.version = LIBRASHADER_CURRENT_VERSION;
                fo.total_subframes = 1;
                fo.current_subframe = 1;
                fo.frames_per_second =
                    nv2a_present_chain_rate(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
                [enc endEncoding];
                enc = nil;
                nv2a_present_chain_source(fw, fh, n, s_chain_lines, &sw, &sh);
                if (sh < fh && s_pso_lines && chain_tex(&s_chain_src, sw, sh)) {
                    MTLRenderPassDescriptor *lp = [MTLRenderPassDescriptor renderPassDescriptor];
                    id<MTLRenderCommandEncoder> le;
                    uint32_t rows = fh / sh;
                    lp.colorAttachments[0].texture = s_chain_src;
                    lp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
                    lp.colorAttachments[0].storeAction = MTLStoreActionStore;
                    le = [cb renderCommandEncoderWithDescriptor:lp];
                    [le setRenderPipelineState:s_pso_lines];
                    [le setFragmentTexture:s_slot[k] atIndex:0];
                    [le setFragmentBytes:&rows length:sizeof rows atIndex:0];
                    [le drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                    [le endEncoding];
                    src = s_chain_src;
                    resolved = 1;
                }
                /* Said at the first frame, once the source is known: a
                 * height that is no multiple of the scale, or a resolve
                 * that did not build, feeds the host rows even to a CRT. */
                if (s_chain_lines && !s_chain_said) {
                    s_chain_said = 1;
                    if (resolved || n <= 1)
                        fprintf(stderr, "[SHADER] present.shader: fed the guest's lines"
                                " (%u of %u rows)\n", resolved ? sh : fh, fh);
                    else
                        fprintf(stderr, "[SHADER] present.shader: fed the host's %u rows (%s)\n",
                                fh, sh < fh ? "the guest-line resolve failed"
                                            : "not a multiple of render.scale");
                }
                e = s_chain_ff(&chain, cb, s_chain_frames++, src, s_chain_out, &lv, NULL, &fo);
                rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
                enc = [cb renderCommandEncoderWithDescriptor:rp];
                if (e) {
                    nv2a_present_chain_error(e, msg, sizeof msg);
                    fprintf(stderr, "[SHADER] present.shader off: frame: %s\n", msg);
                    nv2a_present_chain_failed(s_chain_gen);
                    chain_free();   /* the command buffer keeps what it uses */
                } else {
                    /* Same size, so nearest copies it texel for texel. */
                    MTLViewport vp = { r.x, r.y, r.w, r.h, 0.0, 1.0 };
                    [enc setRenderPipelineState:s_pso];
                    [enc setViewport:vp];
                    [enc setFragmentTexture:s_chain_out atIndex:0];
                    [enc setFragmentSamplerState:s_smp[0] atIndex:0];
                    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                }
                chained = 1;
            }
            if (!chained && !scaled && r.w > 0 && r.h > 0) {
#else
            if (!scaled && r.w > 0 && r.h > 0) {
#endif
                MTLViewport vp = { r.x, r.y, r.w, r.h, 0.0, 1.0 };
                int lin = filter == NV2A_PRESENT_LINEAR
                          || (filter == NV2A_PRESENT_INTEGER && !whole);
                if (ov) {
                    float crop[4] = { c.u_scale, c.v_scale, c.u_off, c.v_off };
                    [enc setRenderPipelineState:s_pso_crop];
                    [enc setVertexBytes:crop length:sizeof crop atIndex:0];
                } else {
                    [enc setRenderPipelineState:s_pso];
                }
                [enc setViewport:vp];
                [enc setFragmentTexture:s_slot[k] atIndex:0];
                [enc setFragmentSamplerState:s_smp[lin] atIndex:0];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            }
        }
#ifdef XBOXRECOMP_ENHANCE
        /* The overlay sits on the whole drawable, not the frame's
         * rectangle: that follows render.scale, display.filter and
         * present.shader, and the overlay's size and hit areas must follow
         * none of them (D3D11 places on its back buffer). */
        r.x = r.y = 0;
        r.w = dw;
        r.h = dh;
        nv2a_overlay_set_placement(r.x, r.y, r.w, r.h);
        /* Under internal the layer fits the drawable into the window after
         * this pass, so the pointer's window pixels map through that fit
         * (D3D11's stretch is the same call); else the two are one. */
        if (internal) {
            int ww = 0, wh = 0, v[4];
            SDL_GetWindowSizeInPixels(win, &ww, &wh);
            nv2a_overlay_view_fit(ww, wh, dw, dh, v);
            nv2a_overlay_set_view(v[0], v[1], v[2], v[3], dw, dh);
        } else {
            nv2a_overlay_set_view(0, 0, 0, 0, 0, 0);
        }
        if (nlayers && s_pso_ovl)
            ovl_draw(enc, cb, layers, nlayers, r, dw, dh);
#else
        (void)layers;
        (void)nlayers;
#endif
        [enc endEncoding];
        if (shot_path) {
            shot = [s_dev newBufferWithLength:(NSUInteger)dw * dh * 4
                                      options:MTLResourceStorageModeShared];
            if (shot) {
                id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
                [b copyFromTexture:d.texture sourceSlice:0 sourceLevel:0
                      sourceOrigin:MTLOriginMake(0, 0, 0)
                        sourceSize:MTLSizeMake((NSUInteger)dw, (NSUInteger)dh, 1)
                          toBuffer:shot destinationOffset:0
                 destinationBytesPerRow:(NSUInteger)dw * 4
               destinationBytesPerImage:(NSUInteger)dw * dh * 4];
                [b endEncoding];
            }
        }
        if (pace && (pr = pace_claim(pace))) {
            pr->t_in = t_in;
            pr->t_drw = t_drw;
            pr->t_cmt = CACurrentMediaTime();
            pace_handlers(pr, d, cb);   /* before presentDrawable and the commit */
        }
        [cb presentDrawable:d];
        [cb commit];
        [s_slot_cb[k] release];
        s_slot_cb[k] = [cb retain];
        if (shot) {
            [cb waitUntilCompleted];
            save_shot(shot, (uint32_t)dw, (uint32_t)dh, shot_path, flip);
            [shot release];
        }
        if (pace)
            pace_drain();   /* after the commit: no drawable is held across it */
        ok = 1;
    }
    return ok;
}

int fbm_window_level(SDL_Window *win, int top)
{
    NSWindow *w = (NSWindow *)SDL_GetPointerProperty(SDL_GetWindowProperties(win),
                                                     SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, NULL);
    NSWindowLevel want = top ? kCGMainMenuWindowLevel + 1 : NSNormalWindowLevel;

    if (!w || [w level] == want)
        return 0;
    [w setLevel:want];
    return 1;
}

/* SDL switches the display with CGDisplaySetDisplayMode (SDL_cocoamodes.m)
 * and sets the desktop's mode back when the window leaves fullscreen, is
 * minimised or destroyed, and when video quits. Apple's documentation is
 * understood to say a mode set that way lasts only for the app's lifetime,
 * reverting when the process ends; that is unverified: no macOS SDK header
 * was at hand when this was written. The file covers whatever is left
 * (a crash, a kill, the OS not reverting): it is written before the switch,
 * deleted once the desktop's mode is back, and read at the next start. */
#define DISPLAY_FMT "display=%u width=%lu height=%lu pixel_width=%lu pixel_height=%lu" \
                    " refresh=%lf native_pixel_width=%lu native_pixel_height=%lu\n"
static int s_display_saved;   /* the file holds this run's desktop mode */

static int display_file(char *out, size_t n, int make)
{
    const char *home = getenv("HOME");

    if (!home || !*home)
        return 0;
    if (make) {
        snprintf(out, n, "%s/Library/Application Support/xboxrecomp", home);
        mkdir(out, 0755);     /* most runs: EEXIST */
    }
    return snprintf(out, n, "%s/Library/Application Support/xboxrecomp/display-restore.txt",
                    home) < (int)n;
}

int fbm_display_save(SDL_Window *win, unsigned long npw, unsigned long nph)
{
    @autoreleasepool {
        NSWindow *w = (NSWindow *)SDL_GetPointerProperty(SDL_GetWindowProperties(win),
                                                         SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, NULL);
        NSNumber *num = [[[w screen] deviceDescription] objectForKey:@"NSScreenNumber"];
        CGDirectDisplayID disp = num ? (CGDirectDisplayID)[num unsignedIntValue] : CGMainDisplayID();
        CGDisplayModeRef m;
        char path[1024];
        FILE *f;
        int ok = 0;

        if (s_display_saved)
            return 1;
        if (!display_file(path, sizeof path, 1) || !(m = CGDisplayCopyDisplayMode(disp)))
            return 0;
        f = fopen(path, "w");
        if (f) {
            ok = fprintf(f, DISPLAY_FMT, (unsigned)disp, (unsigned long)CGDisplayModeGetWidth(m),
                         (unsigned long)CGDisplayModeGetHeight(m),
                         (unsigned long)CGDisplayModeGetPixelWidth(m),
                         (unsigned long)CGDisplayModeGetPixelHeight(m),
                         CGDisplayModeGetRefreshRate(m), npw, nph) > 0;
            ok = fclose(f) == 0 && ok;
            if (!ok)
                unlink(path);
        }
        CGDisplayModeRelease(m);
        s_display_saved = ok;
        return ok;
    }
}

void fbm_display_restore(const char *why)
{
    char path[1024];
    unsigned disp = 0;
    unsigned long w = 0, h = 0, pw = 0, ph = 0, npw = 0, nph = 0;
    double hz = 0;
    int got, done = 0;
    FILE *f;
    CGDisplayModeRef cur;

    s_display_saved = 0;
    if (!display_file(path, sizeof path, 0) || !(f = fopen(path, "r")))
        return;               /* nothing kept: the usual case */
    got = fscanf(f, DISPLAY_FMT, &disp, &w, &h, &pw, &ph, &hz, &npw, &nph);
    fclose(f);
    /* Gone before anything is tried: a restore that fails, or crashes,
     * is not tried again at every start. */
    unlink(path);
    if (got != 8) {
        fprintf(stderr, "[PRESENT] display mode: %s unreadable; deleted (%s)\n", path, why);
        return;
    }
    cur = CGDisplayCopyDisplayMode(disp);
    if (!cur) {
        fprintf(stderr, "[PRESENT] display mode: display %u is gone; %lux%lu (%lux%lu px)"
                " not put back (%s)\n", disp, w, h, pw, ph, why);
        return;
    }
    done = CGDisplayModeGetPixelWidth(cur) == pw && CGDisplayModeGetPixelHeight(cur) == ph;
    {
        /* Only what this toolkit set is undone: a display at any other
         * mode was changed by the user since, and that is theirs. */
        unsigned long cpw = CGDisplayModeGetPixelWidth(cur), cph = CGDisplayModeGetPixelHeight(cur);

        CGDisplayModeRelease(cur);
        if (!done && (cpw != npw || cph != nph)) {
            fprintf(stderr, "[PRESENT] display mode: display at %lux%lu px, not the native %lux%lu"
                    " a run set; left as it is (%s)\n", cpw, cph, npw, nph, why);
            return;
        }
    }
    if (!done) {
        /* With the duplicates the Displays settings hide, as SDL lists
         * them; every match is tried, since SDL_cocoamodes.m finds some
         * look-alikes that refuse to be set. For the session only: the
         * user's saved configuration is never written. */
        const void *k[] = { kCGDisplayShowDuplicateLowResolutionModes };
        const void *v[] = { kCFBooleanTrue };
        CFDictionaryRef opt = CFDictionaryCreate(NULL, k, v, 1, &kCFTypeDictionaryKeyCallBacks,
                                                 &kCFTypeDictionaryValueCallBacks);
        CFArrayRef all = CGDisplayCopyAllDisplayModes(disp, opt);
        CFIndex i, n = all ? CFArrayGetCount(all) : 0;

        for (i = 0; i < n && !done; i++) {
            CGDisplayModeRef m = (CGDisplayModeRef)CFArrayGetValueAtIndex(all, i);
            CGDisplayConfigRef cfg;

            if (CGDisplayModeGetWidth(m) != w || CGDisplayModeGetHeight(m) != h
                    || CGDisplayModeGetPixelWidth(m) != pw || CGDisplayModeGetPixelHeight(m) != ph
                    || fabs(CGDisplayModeGetRefreshRate(m) - hz) > 0.01)
                continue;
            if (CGBeginDisplayConfiguration(&cfg) != kCGErrorSuccess)
                break;
            if (CGConfigureDisplayWithDisplayMode(cfg, disp, m, NULL) != kCGErrorSuccess)
                CGCancelDisplayConfiguration(cfg);
            else
                done = CGCompleteDisplayConfiguration(cfg, kCGConfigureForSession) == kCGErrorSuccess;
        }
        if (all)
            CFRelease(all);
        if (opt)
            CFRelease(opt);
        if (!done) {
            fprintf(stderr, "[PRESENT] display mode: desktop %lux%lu (%lux%lu px) @%.0f not found"
                    " or refused; display left as it is (%s)\n", w, h, pw, ph, hz, why);
            return;
        }
        fprintf(stderr, "[PRESENT] display mode: desktop %lux%lu (%lux%lu px) @%.0f put back"
                " (%s)\n", w, h, pw, ph, hz, why);
        return;
    }
    fprintf(stderr, "[PRESENT] display mode: desktop %lux%lu (%lux%lu px) in place (%s)\n",
            w, h, pw, ph, why);
}

#endif /* __APPLE__ */
