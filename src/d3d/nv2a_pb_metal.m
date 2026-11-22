/**
 * A Metal backend for the pushbuffer executor (struct nv2a_pb_backend), the
 * macOS sibling of nv2a_pb_d3d11.c.
 *
 * The executor (src/kernel/nv2a_pb_exec.c) decodes the title's pushbuffer into
 * the state in nv2a_pb_state.h and calls the hooks of struct nv2a_pb_backend;
 * this file answers them with Metal instead of the CPU rasteriser:
 *
 *   on_clear  -> a render pass with loadAction Clear on the surface at
 *                color_offset (one MTLTexture per guest surface, grown to
 *                its largest clip); a clear of part of it (clip, clear
 *                rect) is a scissored draw of the clear colour
 *   on_draw   -> vertex-program batches through the program as MSL
 *                (d3d8_vsh_msl.c), pre-transformed (screen-space) ones through
 *                a pass-through vertex function; pixels through the register
 *                combiners as MSL (d3d8_combiners_msl.c) when they are
 *                programmed, else MODULATE. Four stages, textured from the
 *                shared decode (nv2a_backend_common.h).
 *   on_flip   -> commit and wait; with the CAMetalLayer window up
 *                (fb_present_metal.m, whose device and queue this backend
 *                adopts), blit the surface the walker picked for this flip
 *                (nv2a_pb_present_state) into the window's back slot and
 *                publish it: no CPU copy. Under metal_present=readback the
 *                texture is read back for the SDL renderer window
 *                (xbox_FramebufferWindowPresentPixels) as before.
 *   sync_guest -> write a surface back to guest memory when the executor
 *                dumps it (RECOMP_FB_DUMP and the report): the write-back is
 *                on demand (metal_writeback=lazy, default), since the title
 *                never reads its present surface; metal_writeback=always,
 *                metal_no_rtt or a read caught by metal_fb_guard write it at
 *                every flip.
 *
 * Opt-in: RECOMP_PB_BACKEND=metal (nv2a_pb_metal_register_from_env). The CPU
 * rasteriser stays the default and the oracle.
 *
 * Rules this file keeps (from the M1 plan):
 *  - Shaders are compiled at runtime (newLibraryWithSource) with fast math off,
 *    so float results match the CPU path's IEEE arithmetic.
 *  - Constant structs are float4/uint4 only on the MSL side; the C structs they
 *    mirror are checked with _Static_assert, as the HLSL ABI is in the D3D11
 *    backend.
 *  - Encoders begin lazily (first clear or draw on a surface) and end
 *    explicitly on a render-target change, before a readback and at the flip;
 *    every pass loads (loadAction Load) unless it starts with a clear.
 *  - Only the present surface is written back to guest memory, on demand
 *    (sync_guest, a texture decode that overlaps it, its eviction), or at
 *    every flip under metal_writeback=always.
 *  - Guest memory is touched on the executor's thread only, never in a
 *    completion handler; a write-back waits for the submission that last wrote the
 *    surface (rt_read_ready) and then copies.
 *
 * Depth/stencil and render-to-texture as the D3D11 backend (M2); visibility
 * tests (GET_REPORT) from Metal visibility results through the shared
 * tracker (M3, "Visibility tests" below). Near clipping needs nothing extra:
 * the vertex programs hand Metal clip-space positions (w kept, as for D3D11)
 * and the default MTLDepthClipModeClip clips at the near and far planes, as
 * D3D11's DepthClipEnable does. Not yet: swizzled render targets. The mip
 * LOD bias is a sample() argument (MsPsConsts.tex_bias), not a sampler
 * field, so every macOS takes the same path.
 *
 * Everything runs on the NV2A ack thread (or the smoke test's main thread).
 *
 * Off macOS this file compiles to nothing.
 */
#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stddef.h>
#include "recomp_env.h"
#include "mem_stats.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <signal.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#include "nv2a_pb_state.h"
#include "nv2a_2d.h"
#include "nv2a_backend_common.h"
#include "nv2a_stats.h"
#include "nv2a_tex_pack.h"
#include "nv2a_tex_dump.h"
#include "d3d8_msl.h"
#include "nv2a_basic_msl.h"
#include "metal_compile.h"
#include "shader_cache.h"
#ifdef XBOXRECOMP_ENHANCE
#include "nv2a_dbg_draw.h"
#endif
/* xbox_memory_layout.h pulls in the Win32 shim (BOOL, GetCurrentThread),
 * which collides with Objective-C and CoreServices: declare what is used. */
ptrdiff_t xbox_GetMemoryOffset(void);
/* kernel_path.c: <HDD root>/ShaderCache, which the preload list sits under. */
int xbox_path_shader_cache(char *out, size_t n);

#if __has_feature(objc_arc)
#error "nv2a_pb_metal.m is manual retain/release: build it without -fobjc-arc"
#endif

#define LOGP "[metal] "
/* Render targets at render.scale stay within Metal's 2D texture limit on
 * Apple GPUs (nv2a_host_size lowers the factor for a larger surface). */
#define METAL_MAX_DIM 16384u

extern void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch);
extern void xbox_FramebufferWindowPresent(uint32_t fb_va, uint32_t pitch);
extern int  xbox_FramebufferWindowRunning(void);
extern void xbox_FramebufferWindowPresentPixels(const void *px, uint32_t w, uint32_t h,
                                                uint32_t pitch, uint32_t bpp);
/* The CAMetalLayer window (fb_present_metal.m): its device and queue, which
 * this backend adopts so the slot textures and the present pass share them,
 * and the back slot a frame is blitted into. */
extern int   xbox_FramebufferWindowMetal(void **device, void **queue);
extern void *xbox_FramebufferWindowBackTexture(uint32_t w, uint32_t h, unsigned n,
                                               uint32_t pad);
extern void  xbox_FramebufferWindowPublishTexture(void);
static int s_layer;                  /* the window takes GPU slot textures */
static void guard_open(void);        /* metal_fb_guard: open the pages before we touch them */
static int wb_always(void);
static uint64_t s_window_presents;   /* frames sent to the window (slot or readback) */

/* ---- shader ABI ------------------------------------------------------------
 *
 * The MSL side declares only float4/uint4 members, so its layout is the C
 * struct's by construction: every member 16-aligned, no implicit padding.
 * The asserts pin the C structs (nv2a_backend_common.h) to that layout. */

/* VpConsts (MSL): float4 c[3]; uint4 u[2]; */
_Static_assert(sizeof(struct nv2a_vp_consts) == 80, "nv2a_vp_consts is not 5 x 16 bytes");
/* A program's c[] (MSL constant float4 *c, buffer 1) is v->c as it is. */
_Static_assert(sizeof(((struct nv2a_pb_vsh *)0)->c) == NV2A_VS_MAX_CONSTANTS * 16,
               "vertex constants are not 192 float4");
_Static_assert(offsetof(struct nv2a_vp_consts, u) == 48, "nv2a_vp_consts.u moved");

/* PsConsts (MSL): float4 c0[8], c1[8], fc0, fc1, fog_color; uint4 atest
 * (x alpha_ref as float bits, y alpha_func, z alpha_test_enable, w fog_enable);
 * uint4 alpha_only; float4 tex_scale[4]; uint4 tex_mode; float4 bump[4];
 * float4 bumpls[4] (BUMPENVMAP); uint4 clip_mode (CLIPPLANE). */
#define PS_AT(m, off) \
    _Static_assert(offsetof(struct nv2a_ps_consts, m) == (off), "nv2a_ps_consts." #m " moved")
PS_AT(c0, 0);
PS_AT(c1, 128);
PS_AT(fc0, 256);
PS_AT(fc1, 272);
PS_AT(fog_color, 288);
PS_AT(alpha_ref, 304);
PS_AT(alpha_func, 308);
PS_AT(alpha_test_enable, 312);
PS_AT(fog_enable, 316);
PS_AT(alpha_only, 320);
PS_AT(tex_scale, 336);
PS_AT(tex_mode, 400);
PS_AT(bump, 416);
PS_AT(bumpls, 480);
PS_AT(clip_mode, 544);
#undef PS_AT
_Static_assert(sizeof(struct nv2a_ps_consts) == 560, "nv2a_ps_consts size changed");

/* The MSL PsConsts: the shared constants plus Metal's own trailing float4
 * tex_bias (per stage mip LOD bias, passed to sample() as bias()). Kept out
 * of nv2a_ps_consts, whose size the D3D11 backend pins to NV2APSConstants. */
typedef struct {
    struct nv2a_ps_consts b;
    float tex_bias[4];
} MsPsConsts;
_Static_assert(offsetof(MsPsConsts, tex_bias) == 560 && sizeof(MsPsConsts) == 576,
               "MsPsConsts.tex_bias moved");
_Static_assert(sizeof(float) == 4 && sizeof(uint32_t) == 4, "alpha_ref is read as uint bits");

/* One expanded vertex of a pass-through batch: attributes 0, 3, 4, 9..12 as
 * float4, in this order (the MSL reads vb[vid * VTX_F4 + k]). */
#define VTX_F4 7
static const int s_vtx_attr[VTX_F4] = {0, 3, 4, 9, 10, 11, 12};

/* The basic library (fs_basic, vs_pass, the clears) and render.snap_2d's
 * fs_basic_snap, built when it first draws (fs_basic): nv2a_basic_msl.h. */
static const char s_msl[] = NV2A_MSL_BASIC;
static const char s_msl_snap[] = NV2A_MSL_BASIC_SNAP;
/* RECOMP_DEBUG=metal_nodiscard: fs_basic_nd and fs_basic_snap_nd, built on
 * first use (fs_basic_pick). */
static const char s_msl_nd[] = NV2A_MSL_BASIC_ND;
static const char s_msl_snap_nd[] = NV2A_MSL_BASIC_SNAP_ND;

/* ---- device ---------------------------------------------------------------- */

static id<MTLDevice>               s_dev;
static id<MTLCommandQueue>         s_queue;
static id<MTLFunction>             s_vs_pass, s_fs_basic;
static id<MTLFunction>             s_vs_clear, s_fs_clear;
static id<MTLRenderPipelineState>  s_clear_pso[2];   /* partial clears: without, with depth */
static id<MTLCommandBuffer>        s_cmd;      /* the frame's, begun lazily */
static id<MTLRenderCommandEncoder> s_enc;      /* open pass, or nil */
static int                         s_failed;
static int                         s_unified;  /* shared textures readable */
/* RECOMP_DEBUG=metal_rt_private=1 (unified memory only): colour targets in
 * Private storage, every CPU read or write of one through s_rtp_buf, a Shared
 * buffer grown on demand. s_stage_tex: the texture whose whole hw x hh the
 * staging holds (rt_managed_sync's blit), NULL when none. */
static int                         s_rtp;
static id<MTLBuffer>               s_rtp_buf;
static size_t                      s_stage_cap;
static const void                 *s_stage_tex;
static uint64_t                    s_stage_reads, s_stage_writes;
/* CPU writes into memory the GPU uses (KFC-1720 H1b step 3b). Each waits
 * for the allocation's last GPU use (cpu_write_wait), not for the frame:
 *  - s_rtp_buf: rt_managed_sync's staging copy writes it, stage_write's blit
 *    reads it; rt_seed fills it.
 *  - s_dbg_vb (enhancements' debug draws): dbg_present fills it every flip,
 *    the draw after reads it.
 *  - a presenter slot: a guest-bytes or readback frame is uploaded into the
 *    back slot (fbm_upload's replaceRegion), which waits there only for the
 *    presenter's own pass over it; s_win_blit_sub, the newest copy of a
 *    frame into a slot (layer mode), is waited for before the hand-off.
 * *_pend: the open s_cmd uses it; *_sub: the newest submission that did,
 * numbered when flush commits it, 0 none. Covered elsewhere: the vertex and
 * index rings (step 2, ring copies), the visibility slots and the tex_mag
 * buffers (step 3a, by the submission that writes them). Every other CPU
 * write into a texture, the texture cache's uploads (tex_make,
 * tex_make_cube: a changed texture gets a new MTLTexture), the 1x1 fallbacks
 * and rt_seed's replaceRegion, goes into a texture made just before it:
 * nothing to wait for. Constants go by setVertexBytes / setFragmentBytes,
 * copied at encode time. */
static int                         s_rtp_pend, s_dbg_vb_pend;
static uint64_t                    s_rtp_sub, s_dbg_vb_sub, s_win_blit_sub;
static int                         s_bc;       /* BC1 and BC3 textures sample (pack entries) */

/* Vertex and index rings, reset after every wait (flip or overflow). Two
 * copies of each (KFC-1720 H1b, step 2): the flip switches copies
 * (ring_switch), and a copy is written again only once the last submission
 * that bound it is done (its last_sub, "Submissions" below). s_vb, s_ib and
 * the cursors are the active copy's; s_vb and s_ib are not retained here. */
#define RING_COPIES 2
static struct {
    id<MTLBuffer> vb, ib;
    uint64_t      last_sub;   /* the newest submission that bound this copy, 0 none */
} s_ring[RING_COPIES];
static int           s_ring_cur;
static int           s_ring_guard;   /* the active copy waits its last_sub before its first write */
static int           s_ring_bound;   /* s_cmd encoded a draw from the active copy */
static id<MTLBuffer> s_vb, s_ib;
static NSUInteger    s_vb_size, s_vb_pos, s_ib_size, s_ib_pos;
static uint64_t      s_ring_switches, s_ring_cuts, s_ring_oversize;
static int           s_ring_small;   /* RECOMP_DEBUG=metal_ring_small is set */

/* RECOMP_DEBUG=metal_inflight=1|2 (KFC-1720 H1b step 4), read at init. 1, the
 * default: the flip commits the frame's command buffer and waits for it, as
 * before. 2: the flip commits it without waiting, and the CPU builds frame
 * N+1 while the GPU runs N. The bound is one frame back (flip_bound): at the
 * end of flip N the CPU waits for flip N-1's last submission (SUBW_FRAME)
 * before it goes on to N+1, so N+1 is built while at most N runs, whether or
 * not N+1 draws (the ring guard alone binds only frames that draw).
 * s_flip_sub: each flip's last submission, by flip count mod 4 (0 none).
 * s_ifl_*: frames with work committed and not done at the end of each flip,
 * before that wait, over the last 300 flips (2 at most; 3 is the bound
 * failing). s_occ_flip_early: reports the flip settled without the spin. */
static int      s_inflight = 1;
static uint64_t s_flip_sub[4];
static uint64_t s_ifl_sum, s_ifl_n, s_ifl_hist[4];
static int      s_ifl_max;
static uint64_t s_occ_flip_early;

#define VIS_SLOTS 8192

static uint64_t s_presents, s_clears, s_draws, s_draws_skipped, s_draws_prog,
                s_draws_culled, s_draws_tex, s_tex_uploads, s_tex_rehash,
                s_tex_changed, s_flushes, s_writebacks, s_cmd_errors,
                s_draws_stretch;   /* hor+: batches placed off-centre (nv2a_aspect_place) */
/* Draws RECOMP_DOF_SKIP dropped: a depth-of-field repaint (metal_on_draw). */
static uint64_t s_dof_skipped;

/* ---- render targets: one per guest surface --------------------------------- */

/* Depth/stencil buffers, one per zeta surface and size, as the D3D11
 * backend's: Depth32Float_Stencil8, so the test compares the same normalised
 * float z (depth units / zmax) the CPU path keeps, and the 8 stencil bits
 * are Z24S8's. Never read back to guest memory. */
#define DT_MAX 4
typedef struct {
    uint32_t       zeta, w, h;
    uint32_t       hw, hh;     /* host size: (w + 2 pad) x h at render.scale */
    uint32_t       pad;        /* hor+ padding, guest pixels a side (nv2a_host_pad) */
    unsigned       n;          /* the render.scale factor used */
    id<MTLTexture> tex;
    uint64_t       last_use;
    int            fresh;      /* contents undefined: the next pass clears */
    /* Sampled as a texture (zt_texture): where it lies, how it was drawn,
     * and its depth as the guest's bytes, converted on the GPU. */
    uint32_t       addr;       /* zeta resolved to a guest address */
    uint32_t       cpitch;     /* the colour pitch it was drawn with */
    uint32_t       zfmt;       /* SET_SURFACE_FORMAT zeta field: 1 Z16, 2 Z24S8 */
    uint32_t       use_flip;   /* s_presents at its last use */
    uint64_t       wseq;       /* s_zseq at its last depth or stencil write */
    uint64_t       mseq;       /* s_zseq of the last other write over its memory */
    uint64_t       zconv;      /* wseq ztex holds, 0 none */
    id<MTLTexture> zst;        /* tex's stencil (X32_Stencil8 view) */
    id<MTLTexture> ztex;       /* RGBA8: the Z24S8 bytes as A8R8G8B8 */
    /* RECOMP_W_EARLYZ: the depth stored as E = B (1/w - 1/zmax), tests
     * reversed (nv2a_w_earlyz_fast's comment). Set when made and at each
     * depth clear, from nv2a_w_earlyz_want(), as D3D11's. */
    int            earlyz;
} DepthTarget;

/* A post-process chain can draw more than 8 targets a frame (a glow chain
 * of small off-screen targets, the scene and the back buffers). With 8 the
 * LRU cycled them, every target was made again each frame, and an
 * off-screen one evicted between its draw and the pass that samples it
 * lost its pixels. 16 holds such a chain; it does not remove the limit. */
#define RT_MAX 16
typedef struct {
    uint32_t       offset;     /* color_offset, as the title set it */
    uint32_t       addr;       /* its guest address (nv2a_pb_dma_resolve) */
    uint32_t       w, h;       /* guest size: the cache key, the mapping */
    uint32_t       hw, hh;     /* host size: the texture, (w + 2 pad) * n x h * n */
    uint32_t       pad;        /* hor+ padding, guest pixels a side (nv2a_host_pad) */
    unsigned       n;          /* the render.scale factor used */
    id<MTLTexture> tex;
    uint64_t       last_use;
    DepthTarget   *dt;         /* the depth buffer its passes attach, or NULL */
    uint32_t       pitch;      /* the guest pitch it was made with */
    uint32_t       fmt;        /* the SET_SURFACE_FORMAT word it was drawn with */
    int            dirty;      /* drawn since its last write-back */
    int            presented;  /* ever picked by present_target */
    struct nv2a_rt_own own;    /* is the guest memory under it still its own */
    uint32_t       bar_flip[2]; /* hor+ band bars per third (nv2a_aspect_bar_mark) */
    int            copy;       /* made by a 2D blit (metal_on_blit), not drawn since */
    uint32_t       copy_flip;  /* the flip of the blit that last filled it */
    uint64_t       zseq;       /* s_zseq at its last pass, draw or blit fill (zt_find) */
    /* What a CPU read of its bytes (getBytes, or metal_rt_private's staging
     * copy) waits for (rt_read_ready, KFC-1720 H1b step 3a): the newest
     * submission that wrote tex (a pass, a blit), and the one with the copy
     * for the CPU rt_managed_sync queued (synchronizeResource, or the
     * staging blit), 0 none. wpend, spend: the open s_cmd has such a write,
     * such a copy, numbered when flush commits it. wbad: last_write_sub
     * ended in Error (sub_observe). */
    int            wpend, spend, wbad;
    uint64_t       last_write_sub, sync_sub;
    /* The newest submission that used tex at all (KFC-1720 H1b step 3b): a
     * write or sync copy above, or a read: a draw sampling it, a 2D blit or
     * the self copy from it, the flip's copy into the presenter's back slot.
     * upend: the open s_cmd has such a read (rt_read_mark), numbered when
     * flush commits it. A CPU write into tex would wait for it (cpu_write_wait);
     * none does today: every CPU write of a target's texels (rt_seed) goes
     * into a texture made just before it, which no submission has used. */
    int            upend;
    uint64_t       use_sub;
} RenderTarget;

/* The order of colour and depth writes, for a depth surface sampled as a
 * texture (zt_find): whichever wrote its memory last is what it holds. */
static uint64_t      s_zseq;
static RenderTarget  s_rt[RT_MAX];
static RenderTarget *s_enc_rt;   /* the target s_enc draws into */
static DepthTarget  *s_enc_dt;   /* and its depth attachment, or NULL */
static DepthTarget   s_dt[DT_MAX];
static uint64_t      s_tick;
static uint64_t      s_zclears, s_draws_z, s_rtt_binds, s_self_copies;
static uint64_t      s_stale_drops, s_retired, s_evict_lost;
static uint64_t      s_blits_gpu, s_blits_guest;   /* metal_on_blit */
static uint64_t      s_grows, s_partial_clears;   /* rt_grow, metal_on_clear */

/* Visibility tests (see "Visibility tests" below): the slot counting on the
 * open pass, opened by begin_pass and closed here. */
static void occ_pass_begin(void);
static void occ_pass_end(void);
static void occ_after_wait(void);
static id<MTLBuffer> s_vis;      /* visibility result slots, 8 bytes each */
static struct OccQuery *s_occ_q;     /* the query counting now, or NULL */
static int s_occ_slot = -1;          /* its slot on s_enc */
/* KFC-1720 H1b step 3a: flush, at each commit of s_cmd, gives its number to
 * what the command buffer writes for the CPU to read: the visibility slots
 * its passes count into (occ_committed) and the tex_mag trace's frame
 * (mag_committed); render targets are numbered in flush itself. */
static void occ_committed(uint64_t n);
static void mag_committed(uint64_t n);
static void cap_committed(uint64_t n);   /* metal_capture_at (step 4) */

/* RECOMP_TRACE=metal_prof: where on_flip's time goes, averaged over 300
 * flips: the wait for the frame's last command buffer, the GPU time of every
 * command buffer committed since the last flip, the host readback (scale > 1),
 * the guest write-back, the readback-mode window copy and hand-off, and the
 * layer-mode slot blit. Prints only. */
static int s_prof = -1;
#define PROF_CMDS 128
static id<MTLCommandBuffer> s_prof_cmd[PROF_CMDS];
static int s_prof_ncmd;
static struct {
    uint64_t n, flip, wait, gpu, hostrb, wb, winrb, winpush, blit, ncmd, hash;
} s_pa;
/* RECOMP_W_EARLYZ_ALT's in-run comparison: the GPU time of each flip summed
 * by the encoding its depth buffer was cleared in (s_prof_ez, set at each
 * depth clear), so old and new frames of one fight compare at one heat. */
static int s_prof_ez;
static uint64_t s_pa_ez_gpu[2], s_pa_ez_n[2];

static uint64_t s_decode_syncs;
static uint64_t s_pitch_alias_binds;   /* binds over a dirty target nv2a_rt_pitch_alias
                                          * decoded without a write-back */
static struct nv2a_pair_set s_pitch_alias_seen;   /* its RENV_METAL_TRACE pairs */

static uint64_t prof_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int prof_on(void)
{
    /* metal_gpu_blocks sets its split beside metal_prof's command-buffer
     * total, which it needs collected */
    if (s_prof < 0)
        s_prof = recomp_env(RENV_METAL_PROF) != NULL || recomp_env(RENV_METAL_GPU_BLOCKS) != NULL;
    return s_prof;
}

/* A flip-phase timer: no clock reads unless metal_prof is on. */
static uint64_t prof_t(void)
{
    return prof_on() ? prof_now() : 0;
}

static void prof_add(uint64_t *acc, uint64_t t0)
{
    if (prof_on())
        *acc += prof_now() - t0;
}

/* ---- RECOMP_TRACE=metal_passes and metal_gpu_blocks -------------------------
 *
 * On a tiler (Apple silicon) a render pass loads its attachments into tile
 * memory at its start and stores them at its end. A frame cut into more
 * passes than its targets need, a pass that loads what it then overwrites,
 * or a store nobody reads, costs GPU time at the passes' edges alone.
 * Kung Fu Chaos's fight costs the M4 ~8.8 ms of GPU a frame at 1920x1440
 * where a Vega 8 holds 16.7 ms frames at 2560x1920: about the same by pixel,
 * so these traces look for what the tiler gives away.
 *
 * Each site that ends or opens a pass names why (PT_WHY just before its
 * end_encoder, flush or begin_pass: those consume it; "untagged" shows a
 * site this list missed). Each frame's passes are kept (PtFrame) with
 * their attachments' load and store actions, their draws by kind
 * (nv2a_gpu_cat_draw) and the first reader of each attachment after them,
 * and judged one flip later, when the next frame has shown whether it
 * loads what this one stored or clears it unread. A store whose texture
 * the next frame never touches is "unknown", not "nobody". Readers are
 * attachments' textures by identity (the pointer, never used: rt_free can
 * release one mid-frame); a later texture at a freed one's address can
 * be taken for it, which this trace accepts.
 *
 * metal_gpu_blocks adds each pass's GPU time from timestamps sampled at
 * its stage boundaries (the only sampling point Apple GPUs offer: no split
 * between the draws of one pass): its share of the frame's busy time
 * (gb_sweep), its fragment and vertex stages' own spans beside it. A pass
 * with no draw, or with only stale samples, is counted, not timed. A pass
 * is put under its first draw's
 * kind; a pass whose draws are of several kinds is counted as mixed, with
 * its draws by kind in the metal_passes_at line, never split. Off, a site
 * costs the store of its reason and one cached int read. */
enum {
    PT_UNTAGGED, PT_TARGET, PT_RESUME, PT_DEPTH, PT_ZCLEAR, PT_CLEAR, PT_CLEAR_Q, PT_ZTEX,
    PT_SELF_COPY, PT_BLIT, PT_GROW, PT_SYNC, PT_REPORT, PT_QUERY_READ, PT_RING, PT_EVICT,
    PT_FLIP, PT_DEBUG, PT_HOST_PIXELS, PT_SCALE, PT_FOLD_READ, PT_NWHY
};
/* fold-read: metal_clear_fold's held clear is encoded before a draw samples
 * its target (rt_texture). */
static const char *const s_pt_why_name[PT_NWHY] = {
    "untagged", "target", "resume", "depth-attach", "zclear", "clear", "clear-in-query", "ztex",
    "self-copy", "blit", "grow", "readback", "report", "query-read", "ring-full", "evict",
    "flip", "debug-draw", "host-pixels", "scale", "fold-read"
};
static int s_pt_why;   /* the reason the next end_encoder or begin_pass runs for */
#define PT_WHY(r) (s_pt_why = (r))

/* Encoders: begin_pass's, a render encoder made elsewhere (rt_grow's
 * clear), compute, blit. */
enum { PTK_RENDER, PTK_AUX, PTK_COMPUTE, PTK_BLIT, PTK_N };
static const char *const s_ptk_name[PTK_N] = { "render", "render-aux", "compute", "blit" };
/* An attachment's first reader after the pass that stored it. */
enum { PTR_NONE, PTR_TEXTURE, PTR_LOAD, PTR_ZTEX, PTR_BLIT, PTR_FLIP, PTR_READBACK,
       PTR_CLEARED, PTR_UNKNOWN };
static const char *const s_ptr_name[] = {
    "nobody yet", "texture", "load", "ztex compute", "blit", "flip", "readback",
    "overwritten unread (a clear or whole copy)", "unknown (the next frame never touches it)"
};

#define PT_MAX   256            /* passes kept a frame */
#define GB_SLOTS (PT_MAX * 4)   /* timestamps a frame: 4 a render pass, 2 a compute or blit */
#define PT_W_MAX 64             /* attachment planes with a last writer */
typedef struct {
    const void *tex;            /* identity only */
    uint32_t addr, w, h, hw, hh, fmt;
    uint8_t  on, clear, store;  /* loadAction Clear (else Load); storeAction Store */
    uint8_t  rkind, rnext;      /* first reader (PTR_*), in the next frame */
    int16_t  rpass;             /* its pass, -1 none or past the table */
    uint16_t nread;
} PtAtt;
typedef struct {
    uint8_t  kind, open_why, close_why, cat, mixed, first_seen, first_full;
    uint8_t  after;             /* opened with no pass open: why the last one closed; 0xFF */
    uint8_t  reopen;            /* the same colour target loaded right after its pass closed:
                                 * that close's reason; 0xFF */
    uint8_t  dc;                /* metal_depth_dontcare opened it DontCare */
    uint8_t  fold;              /* metal_clear_fold: a held clear became its load actions */
    /* The shader of its first draw (pt_shader), for metal_passes_at: the
     * fragment function's FNV-1a (its pointer, the pipeline key's fs), the
     * stages sampling a texture, their host sizes, nv2a_snap_uv in play */
    uint8_t  sh_set, nstages, snap;
    uint32_t fs_id, tw[4], th[4];
    uint32_t draws, clears;
    uint32_t zdraws;            /* draws testing or writing depth or stencil */
    uint32_t cat_draws[NV2A_GCAT_N];
    PtAtt    att[3];            /* colour, depth, stencil */
    int      gb;                /* its first timestamp slot, -1 none */
    uint64_t g_open;            /* the GPU's clock when it was encoded: no sample is older */
    /* gb_resolve: its stages' valid samples in GPU ticks (0: none), and the
     * time the frame's sweep gave it alone (gb_sweep) */
    uint64_t sv, ev, sf, ef;    /* vertex, fragment (compute or blit: sf, ef) */
    uint64_t excl;
} PtPass;
/* RECOMP_DEBUG=metal_nodiscard / metal_fs_fastmath (KFC-0846): a frame's
 * draws (encoded); those that could take the variant without discard, and
 * of them the ones that can gain from hidden-surface removal (no blending,
 * no depth from the fragment: win, in either arm); those that took it, and
 * of them the ones that cannot gain (blended; the W variant's depth; both
 * if both); the draws with a fast-math function; the pipelines pipeline()
 * built. */
typedef struct {
    uint64_t draws, elig, taken, win, blend, fdepth, fast, pso;
} NdCount;
typedef struct {
    uint64_t flip;
    int      n, over;           /* passes kept, and passes past PT_MAX */
    int      nk[PTK_N];         /* encoders made, by kind */
    int      gb_n;              /* timestamp slots used */
    PtPass   p[PT_MAX];
    NdCount  nd;                /* metal_nodiscard's counts */
    int      nd_arm_b;          /* metal_ab_alt: the frame drew in arm B */
} PtFrame;
static PtFrame s_ptf[2];        /* by flip parity: this frame and the one being judged */
static int      s_pt_cur = -1;  /* the open render pass's index */
static uint64_t s_pt_cur_flip;
static int      s_pt_aux = -1;  /* the last compute or blit pass's index */
/* The last render pass closed: its colour texture, frame and reason. */
static struct { const void *tex; uint64_t flip; int why, valid; } s_pt_last;
static struct { const void *tex; uint8_t plane; uint64_t flip; int16_t pass; } s_ptw[PT_W_MAX];
static int s_ptw_n;
static struct {
    int      flips;
    uint64_t nk[PTK_N];
    int      maxk[PTK_N];
    uint64_t close[PT_NWHY], reopen[PT_NWHY];
    uint64_t empty, mixed, over;
    uint64_t sus_load, sus_store[3], unknown[3];
    /* metal_gpu_blocks */
    uint64_t cat_ticks[NV2A_GCAT_N], mixed_ticks, gb_passes, gb_missing;
    uint64_t gb_nodraw, gb_stale;    /* untimed: no draw and no sample; a sample older than the pass */
    uint64_t span_ticks, idle_ticks; /* the frames' first sample to last, and the gaps in it */
    uint64_t cmd_ns;            /* the command buffers' GPU time (prof_flip_end) */
    uint64_t dc_passes, dc_miss;     /* metal_depth_dontcare */
    /* metal_clear_fold: clears held, merged into a held one, not held (an
     * occlusion query open, the table full); held ones folded into the next
     * pass on their textures, or encoded as before: for a pass sharing part
     * of them, or at an end_encoder by its reason */
    uint64_t cf_held, cf_merged, cf_query, cf_full, cf_folded, cf_shared, cf_end[PT_NWHY];
    uint64_t fh_draws;               /* metal_flat_half: draws scissored */
    uint64_t cpu_n;                  /* metal_gpu_blocks: frames with a scene pass, */
    double   cpu_sum_ms, cpu_max_ms; /* its close to the flip's commit */
    NdCount  nd;                     /* metal_nodiscard: the frames' counts */
    uint64_t late_ticks;             /* metal_gpu_blocks' busy, the block's flips 31 on */
    int      arm_a, arm_b;           /* metal_ab_alt: frames by arm */
} s_pts;
static int  s_pt_on = -1, s_pt_log;
static int  s_pt_track;         /* readers kept: metal_passes, or metal_depth_dontcare's guess */
static int  s_dc_mode;          /* metal_depth_dontcare: 0 off, 1 on, 2 every other flip */
static int  s_fh_mode;          /* metal_flat_half: 0 off, 1 on, 2 every other flip */
static uint64_t s_scene_close_ns;   /* metal_gpu_blocks: the last scene or mixed pass's close; 0 none */
static MTLScissorRect s_draw_sr;    /* the open draw's scissor (metal_flat_half re-sets it) */
static int  s_dc_pass;          /* the open pass's depth and stencil are DontCare */
static int  s_dc_opener;        /* begin_pass is opened for a draw using neither */
/* RECOMP_DEBUG=metal_clear_fold=1 (probe, KFC-1105): a whole-target colour
 * clear (metal_on_clear) or a depth clear (metal_on_zclear) opens no pass.
 * Its descriptor is held, and the next pass on the same colour and depth
 * textures loads them with its Clears instead of Load. Anything that ends a
 * pass outside begin_pass (end_encoder: a blit, a compute, a commit, a
 * readback, an eviction), a pass on only part of its textures, a draw
 * sampling its target (rt_texture) or a free of its target or depth
 * buffer encodes it first, as it would have been encoded. Held ones never
 * share a texture. s_cf_mode is set by init. */
#define CF_MAX 4
typedef struct {
    MTLRenderPassDescriptor *rp;   /* retained; nil: a free entry */
    RenderTarget *rt;
    DepthTarget  *dt;
    const void   *ctex, *ztex;     /* its textures, identity only */
    int           why, had;        /* begin_pass's, for pt_open when encoded */
} CfHeld;
static CfHeld s_cf[CF_MAX];
static int  s_cf_n;             /* entries held */
static int  s_cf_mode;          /* metal_clear_fold */
static int  s_cf_call;          /* begin_pass is opened by a whole-target clear */
static int  s_cf_hold;          /* this end_encoder leaves the held clears held */
static long s_pt_at = -1;       /* metal_passes_at */
static int  s_gb_on = -1;       /* metal_gpu_blocks: sampling */
static id   s_gb_sb[2];         /* id<MTLCounterSampleBuffer> by flip parity */
static id   s_gb_cmd[2];        /* the last command buffer sampled into each */
static uint64_t s_gb_cal_ns, s_gb_cal_gpu;   /* a (host ns, GPU tick) pair */

/* RECOMP_DEBUG=metal_nodiscard, metal_fs_fastmath and metal_ab_alt
 * (KFC-0846), read once. s_nd_mode: 0 off, 1 the draws that cannot kill
 * take the fragment function without discard_fragment (nd_decide), 2 every
 * draw does (the negative control). s_ff_mode: 1 the combiners' fragment
 * libraries compile with fast math (not the W variant). *_alt: only on even
 * flips, as metal_depth_dontcare=alt. s_ab_blk: metal_ab_alt's block, the
 * fixed 300 flips of metal_gpu_blocks' window (pt_judge), arm A (both off)
 * on even blocks, arm B (as set) on odd ones; 0 off. */
static int s_nd_mode = -1, s_nd_alt, s_ff_mode, s_ff_alt, s_ab_blk;
/* RECOMP_DEBUG=metal_ab_null=1 (KFC-1105, the hub's null control): arm B's
 * combiner fragments are a second, separate compile of the same MSL (key[36]
 * bit 13, FsEntry.fast bit 1): new functions and pipelines, same shader. */
static int s_null_mode;
#define ND_BLOCK 300   /* pt_judge's window */

static void nd_env(void)
{
    const char *nd, *ff;
    long ab;
    if (s_nd_mode >= 0)
        return;
    nd = recomp_env(RENV_METAL_NODISCARD);
    ff = recomp_env(RENV_METAL_FS_FASTMATH);
    ab = recomp_env_int(RENV_METAL_AB_ALT, 0);
    s_nd_mode = !nd ? 0 : !strcasecmp(nd, "alt") ? 1 : !strcasecmp(nd, "altneg") ? 2
              : nd[0] == '1' ? 1 : nd[0] == '2' ? 2 : 0;
    s_nd_alt = nd && s_nd_mode && !strncasecmp(nd, "alt", 3);
    s_ff_mode = ff && (!strcasecmp(ff, "alt") || ff[0] == '1');
    s_ff_alt = s_ff_mode && !strcasecmp(ff, "alt");
    s_ab_blk = ab > 0 ? ND_BLOCK : 0;
    s_null_mode = recomp_env_on(RENV_METAL_AB_NULL) > 0;
    if (s_null_mode)
        fprintf(stderr, LOGP "metal_ab_null=1: arm B compiles the same combiner MSL again, as its own functions"
                " and pipelines (a null control for metal_ab_alt)\n");
    if (nd && !s_nd_mode && nd[0] != '0')
        fprintf(stderr, LOGP "metal_nodiscard=%s: not 1, 2, alt or altneg, off\n", nd);
    if (ff && !s_ff_mode && ff[0] != '0')
        fprintf(stderr, LOGP "metal_fs_fastmath=%s: not 1 or alt, off\n", ff);
    if (ab > 0 && ab != ND_BLOCK)
        fprintf(stderr, LOGP "metal_ab_alt=%ld: blocks of %d flips, metal_gpu_blocks' fixed"
                " window\n", ab, ND_BLOCK);
    if (s_nd_mode || s_ff_mode || s_ab_blk)
        fprintf(stderr, LOGP "KFC-0846: metal_nodiscard %s%s, metal_fs_fastmath %s%s;"
                " metal_ab_alt %s\n",
                s_nd_mode == 2 ? "2 (negative control: every draw)" : s_nd_mode ? "1" : "off",
                s_nd_alt ? " (alt: even flips only)" : "", s_ff_mode ? "1" : "off",
                s_ff_alt ? " (alt: even flips only)" : "",
                s_ab_blk ? "on: arm A (both off) on flips 0-299, arm B 300-599, ..." : "off");
}

/* Any of the three is set: pt_on, and the counts' line, follow. */
static int nd_any(void)
{
    nd_env();
    return s_nd_mode || s_ff_mode || s_ab_blk || s_null_mode;
}

/* metal_ab_alt: this flip is in arm B (always when metal_ab_alt is off). */
static int nd_arm_b(void)
{
    nd_env();
    return !s_ab_blk || (int)((s_presents / (uint64_t)s_ab_blk) & 1u);
}

/* metal_nodiscard for this flip: 0, 1 or 2. */
static int nd_now(void)
{
    nd_env();
    if (!s_nd_mode || !nd_arm_b() || (s_nd_alt && (s_presents & 1)))
        return 0;
    return s_nd_mode;
}

/* metal_fs_fastmath for this flip. */
static int ff_now(void)
{
    nd_env();
    return s_ff_mode && nd_arm_b() && !(s_ff_alt && (s_presents & 1));
}

/* metal_ab_null for this flip. */
static int null_now(void)
{
    nd_env();
    return s_null_mode && nd_arm_b();
}

/* The passes bookkeeping, for either trace and for metal_depth_dontcare,
 * which guesses from it (pt_dc_guess). */
static int pt_on(void)
{
    if (s_pt_on < 0) {
        const char *dc = recomp_env(RENV_METAL_DEPTH_DONTCARE);
        s_dc_mode = !dc ? 0 : !strcasecmp(dc, "alt") ? 2 : dc[0] == '1' ? 1 : 0;
        s_pt_log = recomp_env(RENV_METAL_PASSES) != NULL;
        s_pt_track = s_pt_log || s_dc_mode;
        const char *fh = recomp_env(RENV_METAL_FLAT_HALF);
        s_fh_mode = !fh ? 0 : !strcasecmp(fh, "alt") ? 2 : fh[0] == '1' ? 1 : 0;
        s_pt_on = s_pt_track || s_fh_mode || recomp_env(RENV_METAL_GPU_BLOCKS) != NULL
               || nd_any();   /* metal_nodiscard's counts, by judged frame */
        s_pt_at = s_pt_log ? recomp_env_int(RENV_METAL_PASSES_AT, -1) : -1;
        if (dc && !s_dc_mode)
            fprintf(stderr, LOGP "metal_depth_dontcare=%s: not 1 or alt, off\n", dc);
        if (fh && !s_fh_mode)
            fprintf(stderr, LOGP "metal_flat_half=%s: not 1 or alt, off\n", fh);
    }
    return s_pt_on;
}

/* Stage-boundary timestamps, made on the first pass after init (s_dev). 0
 * with a line when the device cannot: metal_passes still counts. */
static int gb_init(void)
{
    if (@available(macOS 11.0, iOS 14.0, *)) {
        id<MTLCounterSet> set = nil;
        MTLCounterSampleBufferDescriptor *d;
        NSError *err = nil;
        MTLTimestamp ct = 0, gt = 0;
        int i;
        if (![s_dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            fprintf(stderr, LOGP "metal_gpu_blocks: %s samples no counters at stage boundaries:"
                    " no GPU time by pass\n", [[s_dev name] UTF8String]);
            return 0;
        }
        for (id<MTLCounterSet> cs in [s_dev counterSets])
            if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp])
                set = cs;
        if (!set) {
            fprintf(stderr, LOGP "metal_gpu_blocks: no timestamp counter set: no GPU time by pass\n");
            return 0;
        }
        d = [[MTLCounterSampleBufferDescriptor alloc] init];
        d.counterSet = set;
        d.storageMode = MTLStorageModeShared;
        d.sampleCount = GB_SLOTS;
        for (i = 0; i < 2; i++) {
            s_gb_sb[i] = [s_dev newCounterSampleBufferWithDescriptor:d error:&err];
            if (!s_gb_sb[i]) {
                fprintf(stderr, LOGP "metal_gpu_blocks: no counter sample buffer (%s): no GPU"
                        " time by pass\n", err ? [[err localizedDescription] UTF8String] : "?");
                [s_gb_sb[0] release];
                s_gb_sb[0] = nil;
                [d release];
                return 0;
            }
        }
        [d release];
        /* The GPU's ticks are calibrated against prof_now() between two of
         * these (gb_window): no assumption about the tick's unit. */
        [s_dev sampleTimestamps:&ct gpuTimestamp:&gt];
        s_gb_cal_ns = prof_now();
        s_gb_cal_gpu = gt;
        fprintf(stderr, LOGP "metal_gpu_blocks: pass timestamps at stage boundaries, up to %d"
                " passes a frame\n", PT_MAX);
        return 1;
    }
    fprintf(stderr, LOGP "metal_gpu_blocks: needs macOS 11: no GPU time by pass\n");
    return 0;
}

static int gb_on(void)
{
    if (s_gb_on < 0)
        s_gb_on = s_dev && recomp_env(RENV_METAL_GPU_BLOCKS) ? gb_init() : 0;
    return s_gb_on;
}

/* The GPU's clock now, in the counter samples' ticks. A sample a pass's
 * stage never wrote keeps what the slot held from an older frame (a clear
 * pass with no draw writes none: 3586 ms "passes" on the M4), so a sample
 * older than the pass's encoding is not the pass's. */
static uint64_t gb_now(void)
{
    if (@available(macOS 11.0, iOS 14.0, *)) {
        MTLTimestamp ct = 0, gt = 0;
        [s_dev sampleTimestamps:&ct gpuTimestamp:&gt];
        return gt;
    }
    return 0;
}

static void gb_note_cmd(void)
{
    int k = (int)(s_presents & 1);
    if (s_gb_cmd[k] != s_cmd) {
        [s_gb_cmd[k] release];
        s_gb_cmd[k] = [s_cmd retain];
    }
}

/* n timestamp slots of this frame's buffer, -1 when they are used up. */
static int gb_take(int n)
{
    PtFrame *f = &s_ptf[s_presents & 1];
    int base;
    if (f->gb_n + n > GB_SLOTS)
        return -1;
    base = f->gb_n;
    f->gb_n += n;
    gb_note_cmd();
    return base;
}

static PtFrame *pt_frame(void)
{
    return &s_ptf[s_presents & 1];
}

/* A new pass record in this frame, NULL past PT_MAX (counted). */
static PtPass *pt_new(int kind, int why, int *idx)
{
    PtFrame *f = pt_frame();
    PtPass *p;
    f->nk[kind]++;
    if (f->n >= PT_MAX) {
        f->over++;
        *idx = -1;
        return NULL;
    }
    *idx = f->n;
    p = &f->p[f->n++];
    memset(p, 0, sizeof *p);
    p->kind = (uint8_t)kind;
    p->open_why = (uint8_t)why;
    p->close_why = PT_UNTAGGED;
    p->after = p->reopen = 0xFF;
    p->cat = NV2A_GCAT_OTHER;
    p->gb = -1;
    return p;
}

static int ptw_find(const void *tex, int plane)
{
    int i;
    for (i = 0; i < s_ptw_n; i++)
        if (s_ptw[i].tex == tex && s_ptw[i].plane == plane)
            return i;
    return -1;
}

/* The attachment record writer entry w names, while its frame is kept: this
 * one, the last, and at the flip's end the one being judged (s_presents has
 * moved on by then). */
static PtAtt *ptw_att(int w)
{
    uint64_t fl = s_ptw[w].flip;
    PtFrame *f = &s_ptf[fl & 1];
    if (fl > s_presents || fl + 2 < s_presents || f->flip != fl
            || s_ptw[w].pass < 0 || s_ptw[w].pass >= f->n)
        return NULL;
    return &f->p[s_ptw[w].pass].att[s_ptw[w].plane];
}

/* metal_depth_dontcare: the last pass opened DontCare, until a clear of
 * its depth; a load or read of that depth before then is a wrong guess. */
static struct { const void *tex; uint64_t flip; int pass, valid; } s_dc_last;

static void pt_dc_miss(const char *what)
{
    static int said;
    PtFrame *f = &s_ptf[s_dc_last.flip & 1];
    s_pts.dc_miss++;
    if (f->flip == s_dc_last.flip && s_dc_last.pass >= 0 && s_dc_last.pass < f->n)
        f->p[s_dc_last.pass].dc = 2;   /* its twin is not guessed again */
    if (said++ < 4)
        fprintf(stderr, LOGP "metal_depth_dontcare: wrong guess at flip %llu: %s after pass #%d"
                " of flip %llu opened its depth DontCare\n", (unsigned long long)s_presents,
                what, s_dc_last.pass, (unsigned long long)s_dc_last.flip);
    s_dc_last.valid = 0;
}

/* tex's plane (0 colour, 1 depth, 2 stencil) read by `kind`, in pass `by`
 * of this frame (-1: not a pass, or past the table). */
static void pt_read(const void *tex, int plane, int kind, int by)
{
    int w = ptw_find(tex, plane);
    PtAtt *a;
    if (plane && s_dc_last.valid && s_dc_last.tex == tex)
        pt_dc_miss("its depth read as a texture");
    if (w < 0 || !(a = ptw_att(w)))
        return;
    if (!a->rkind) {
        a->rkind = (uint8_t)kind;
        a->rpass = (int16_t)by;
        a->rnext = s_ptw[w].flip != s_presents;
    }
    if (a->nread < 0xFFFF)
        a->nread++;
}

/* Pass `by` writes tex's plane, loading it (clear 0) or not (clear 1): a
 * load reads the last writer's store, a clear before any reader leaves it
 * unread. */
static void pt_write(const void *tex, int plane, int clear, int by)
{
    int w = ptw_find(tex, plane);
    if (w >= 0) {
        PtAtt *a = ptw_att(w);
        if (a && !a->rkind) {
            a->rkind = clear ? PTR_CLEARED : PTR_LOAD;
            a->rpass = (int16_t)by;
            a->rnext = s_ptw[w].flip != s_presents;
        }
        if (a && !clear && a->nread < 0xFFFF)
            a->nread++;
    } else {
        if (s_ptw_n < PT_W_MAX) {
            w = s_ptw_n++;
        } else {   /* full: the oldest entry goes */
            int i;
            for (w = 0, i = 1; i < s_ptw_n; i++)
                if (s_ptw[i].flip < s_ptw[w].flip)
                    w = i;
        }
        s_ptw[w].tex = tex;
        s_ptw[w].plane = (uint8_t)plane;
    }
    s_ptw[w].flip = s_presents;
    s_ptw[w].pass = (int16_t)by;
}

/* clear: 0 Load, 1 Clear, 2 DontCare (metal_depth_dontcare); store: 1
 * Store, 0 DontCare. */
static void pt_att(PtAtt *a, const void *tex, uint32_t addr, uint32_t w, uint32_t h,
                   uint32_t hw, uint32_t hh, uint32_t fmt, int clear, int store)
{
    a->on = 1;
    a->tex = tex;
    a->addr = addr;
    a->w = w;
    a->h = h;
    a->hw = hw;
    a->hh = hh;
    a->fmt = fmt;
    a->clear = (uint8_t)clear;
    a->store = (uint8_t)store;
    a->rpass = -1;
}

static int pt_load(MTLLoadAction l)
{
    return l == MTLLoadActionClear ? 1 : l == MTLLoadActionDontCare ? 2 : 0;
}

/* RECOMP_DEBUG=metal_depth_dontcare: may the pass begin_pass opens on rt
 * and dt load and store its depth and stencil DontCare? Whether any draw
 * of it will use them, or a later pass read them before their next clear,
 * is not known when it opens; so the guess is its twin's, last frame's
 * pass at the same place on the same colour and depth target: the frame's
 * last pass (closed at the flip) with no depth or stencil draw, whose
 * depth and stencil the next frame cleared unread (or the twin was guessed
 * so itself and nothing proved it wrong). In Kung Fu Chaos's fight that is
 * the last full-screen pass over the 640x480 frame, #11 of 12. The caller
 * opens it for a draw using neither (s_dc_opener); a depth or stencil
 * draw in it ends it, a later read or load is counted (pt_dc_miss). */
static int pt_dc_guess(const RenderTarget *rt, const DepthTarget *dt)
{
    const PtFrame *f = pt_frame(), *pf = &s_ptf[(s_presents + 1) & 1];
    const PtPass *t;
    if (s_dc_mode == 2 && (s_presents & 1))
        return 0;   /* alt: odd flips as before */
    if (!s_presents || pf->flip != s_presents - 1 || f->n >= pf->n)
        return 0;
    t = &pf->p[f->n];
    return t->kind == PTK_RENDER && t->close_why == PT_FLIP && !t->zdraws && t->dc != 2
        /* the colour by its shape, not its texture: the frame's last pass
         * draws into the back buffer, which swaps with the front every flip */
        && t->att[0].w == rt->w && t->att[0].h == rt->h && t->att[0].fmt == rt->fmt
        && t->att[1].on
        && t->att[1].tex == (const void *)dt->tex
        && (t->dc == 1 || (t->att[1].rkind == PTR_CLEARED && t->att[2].rkind == PTR_CLEARED));
}

/* begin_pass, its descriptor filled: the pass's record, its writes, and
 * its timestamps (metal_gpu_blocks). `closed_now`: begin_pass itself
 * closed the pass before it. */
static void pt_open(const RenderTarget *rt, const DepthTarget *dt, MTLRenderPassDescriptor *rp,
                    int why, int closed_now)
{
    int idx, load = rp.colorAttachments[0].loadAction == MTLLoadActionLoad;
    int same = s_pt_last.valid && s_pt_last.flip == s_presents && s_pt_last.tex == (const void *)rt->tex;
    int zl = dt ? pt_load(rp.depthAttachment.loadAction) : 0;
    int sl = dt ? pt_load(rp.stencilAttachment.loadAction) : 0;
    PtPass *p = pt_new(PTK_RENDER, why, &idx);
    s_pt_cur = idx;
    s_pt_cur_flip = s_presents;
    if (load && same)
        s_pts.reopen[s_pt_last.why]++;
    if (p) {
        if (!closed_now && s_pt_last.valid && s_pt_last.flip == s_presents)
            p->after = (uint8_t)s_pt_last.why;
        if (load && same)
            p->reopen = (uint8_t)s_pt_last.why;
        pt_att(&p->att[0], rt->tex, rt->addr, rt->w, rt->h, rt->hw, rt->hh, rt->fmt, !load, 1);
        if (dt) {
            pt_att(&p->att[1], dt->tex, dt->addr, dt->w, dt->h, dt->hw, dt->hh, dt->zfmt, zl,
                   rp.depthAttachment.storeAction == MTLStoreActionStore);
            pt_att(&p->att[2], dt->tex, dt->addr, dt->w, dt->h, dt->hw, dt->hh, dt->zfmt, sl,
                   rp.stencilAttachment.storeAction == MTLStoreActionStore);
        }
    }
    /* metal_depth_dontcare's guess: a load of the depth it left alone
     * before its next clear proves it wrong. */
    if (dt && s_dc_last.valid && s_dc_last.tex == (const void *)dt->tex && !s_dc_pass) {
        if (zl == 1 && sl == 1)
            s_dc_last.valid = 0;   /* cleared: the guess held */
        else
            pt_dc_miss("its depth loaded");
    }
    pt_write(rt->tex, 0, !load, idx);
    if (dt && !s_dc_pass) {
        pt_write(dt->tex, 1, zl == 1, idx);
        pt_write(dt->tex, 2, sl == 1, idx);
    } else if (dt) {
        /* DontCare: it neither reads nor writes the depth in memory, so the
         * last pass that stored it stays its writer */
        static int said;
        s_pts.dc_passes++;
        s_dc_last.tex = dt->tex;
        s_dc_last.flip = s_presents;
        s_dc_last.pass = idx;
        s_dc_last.valid = 1;
        if (p)
            p->dc = 1;
        if (!said++)
            fprintf(stderr, LOGP "metal_depth_dontcare: flip %llu pass #%d on 0x%08X %ux%u opens"
                    " its depth and stencil (zeta 0x%08X) DontCare: last frame's pass there was"
                    " the last, with no depth or stencil draw, and the next frame cleared them"
                    " unread%s\n", (unsigned long long)s_presents, idx, rt->offset, rt->w, rt->h,
                    dt->zeta, s_dc_mode == 2 ? " (alt: even flips only)" : "");
    }
    if (p && gb_on()) {
        if (@available(macOS 11.0, iOS 14.0, *)) {
            int base = gb_take(4);
            if (base >= 0) {
                MTLRenderPassSampleBufferAttachmentDescriptor *a = rp.sampleBufferAttachments[0];
                a.sampleBuffer = (id<MTLCounterSampleBuffer>)s_gb_sb[s_presents & 1];
                a.startOfVertexSampleIndex = (NSUInteger)base;
                a.endOfVertexSampleIndex = (NSUInteger)base + 1;
                a.startOfFragmentSampleIndex = (NSUInteger)base + 2;
                a.endOfFragmentSampleIndex = (NSUInteger)base + 3;
                p->gb = base;
                p->g_open = gb_now();
            }
        }
    }
}

/* end_encoder, a render pass open: its close reason. */
static void pt_close(void)
{
    PtFrame *f = pt_frame();
    s_pts.close[s_pt_why]++;
    s_pt_last.valid = 1;
    s_pt_last.flip = s_presents;
    s_pt_last.why = s_pt_why;
    s_pt_last.tex = NULL;
    if (s_pt_cur >= 0 && s_pt_cur_flip == s_presents && s_pt_cur < f->n) {
        PtPass *p = &f->p[s_pt_cur];
        p->close_why = (uint8_t)s_pt_why;
        s_pt_last.tex = p->att[0].tex;
        /* metal_gpu_blocks: the CPU time from the last scene pass's close to
         * the flip's commit (flush) */
        if (s_gb_on > 0 && (p->cat == NV2A_GCAT_SCENE || p->mixed) && p->draws)
            s_scene_close_ns = prof_now();
    }
    s_pt_cur = -1;
}

static PtPass *pt_cur_pass(void)
{
    PtFrame *f = pt_frame();
    if (s_pt_cur < 0 || s_pt_cur_flip != s_presents || s_pt_cur >= f->n)
        return NULL;
    return &f->p[s_pt_cur];
}

/* A draw of kind cat in the open pass; full: the heuristic's "covers the
 * whole colour target opaquely" (pt_draw). */
static void pt_note_draw(int cat, int full)
{
    PtPass *p = pt_cur_pass();
    if (!p)
        return;
    if (!p->first_seen) {
        p->first_seen = 1;
        p->first_full = (uint8_t)full;
        p->cat = (uint8_t)cat;
    } else if (cat != p->cat) {
        p->mixed = 1;
    }
    p->draws++;
    p->cat_draws[cat]++;
}

/* metal_on_clear's partial clear, a draw under a scissor. */
static void pt_note_clear(void)
{
    PtPass *p = pt_cur_pass();
    if (!p)
        return;
    p->clears++;
    p->first_seen = 1;
}

/* A compute or blit pass (or another render encoder) about to be made;
 * its index (s_pt_aux) for the reads and writes that follow. */
static int pt_aux(int kind, int why, int cat)
{
    int idx;
    PtPass *p = pt_new(kind, why, &idx);
    if (p) {
        p->cat = (uint8_t)cat;
        p->close_why = (uint8_t)why;
    }
    s_pt_aux = idx;
    return idx;
}

/* pt_aux's compute encoder, with timestamps under metal_gpu_blocks. */
static id<MTLComputeCommandEncoder> pt_compute(int why, int cat)
{
    int idx = pt_aux(PTK_COMPUTE, why, cat);
    if (idx >= 0 && gb_on()) {
        if (@available(macOS 11.0, iOS 14.0, *)) {
            int base = gb_take(2);
            if (base >= 0) {
                MTLComputePassDescriptor *cd = [MTLComputePassDescriptor computePassDescriptor];
                MTLComputePassSampleBufferAttachmentDescriptor *a = cd.sampleBufferAttachments[0];
                a.sampleBuffer = (id<MTLCounterSampleBuffer>)s_gb_sb[s_presents & 1];
                a.startOfEncoderSampleIndex = (NSUInteger)base;
                a.endOfEncoderSampleIndex = (NSUInteger)base + 1;
                pt_frame()->p[idx].gb = base;
                pt_frame()->p[idx].g_open = gb_now();
                return [s_cmd computeCommandEncoderWithDescriptor:cd];
            }
        }
    }
    return [s_cmd computeCommandEncoder];
}

/* pt_aux's blit encoder, with timestamps under metal_gpu_blocks. */
static id<MTLBlitCommandEncoder> pt_blit(int why, int cat)
{
    int idx = pt_aux(PTK_BLIT, why, cat);
    if (idx >= 0 && gb_on()) {
        if (@available(macOS 11.0, iOS 14.0, *)) {
            int base = gb_take(2);
            if (base >= 0) {
                MTLBlitPassDescriptor *bd = [MTLBlitPassDescriptor blitPassDescriptor];
                MTLBlitPassSampleBufferAttachmentDescriptor *a = bd.sampleBufferAttachments[0];
                a.sampleBuffer = (id<MTLCounterSampleBuffer>)s_gb_sb[s_presents & 1];
                a.startOfEncoderSampleIndex = (NSUInteger)base;
                a.endOfEncoderSampleIndex = (NSUInteger)base + 1;
                pt_frame()->p[idx].gb = base;
                pt_frame()->p[idx].g_open = gb_now();
                return [s_cmd blitCommandEncoderWithDescriptor:bd];
            }
        }
    }
    return [s_cmd blitCommandEncoder];
}

static int gb_u64_cmp(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

/* The judged frame's GPU timeline from its passes' stage intervals, shared
 * out. On a tiler a pass's vertex stage runs ahead, under earlier passes'
 * fragments: on the M4 every start-of-vertex sample sat near the frame's
 * start, so start-of-vertex to end-of-fragment grew pass after pass (6.3,
 * 7.0, 9.9, 13.2 ms in a 14 ms frame) and their sum was several frames. So
 * each stretch of time goes to the fragment intervals (a compute or blit
 * encoder's, too) covering it, split evenly; a stretch no fragment covers
 * goes to the vertex intervals covering it; one nothing covers is idle.
 * The shares (excl) sum to the intervals' union, the frame's busy GPU time,
 * which is what the command buffers' total is set against. */
static void gb_sweep(PtFrame *f)
{
    static uint64_t b[PT_MAX * 4];
    int nb = 0, i, j;
    uint64_t busy = 0;
    for (i = 0; i < f->n; i++) {
        const PtPass *p = &f->p[i];
        if (p->sf) {
            b[nb++] = p->sf;
            b[nb++] = p->ef;
        }
        if (p->sv) {
            b[nb++] = p->sv;
            b[nb++] = p->ev;
        }
    }
    if (nb < 2)
        return;
    qsort(b, (size_t)nb, sizeof b[0], gb_u64_cmp);
    for (j = 0; j + 1 < nb; j++) {
        uint64_t s = b[j], e = b[j + 1];
        int nf = 0, nv = 0;
        if (e == s)
            continue;
        for (i = 0; i < f->n; i++) {
            const PtPass *p = &f->p[i];
            if (p->sf && p->sf <= s && p->ef >= e)
                nf++;
            else if (p->sv && p->sv <= s && p->ev >= e)
                nv++;
        }
        if (!nf && !nv)
            continue;   /* idle */
        busy += e - s;
        for (i = 0; i < f->n; i++) {
            PtPass *p = &f->p[i];
            if (nf ? p->sf && p->sf <= s && p->ef >= e : p->sv && p->sv <= s && p->ev >= e)
                p->excl += (e - s) / (uint64_t)(nf ? nf : nv);
        }
    }
    s_pts.span_ticks += b[nb - 1] - b[0];
    s_pts.idle_ticks += b[nb - 1] - b[0] - busy;
    for (i = 0; i < f->n; i++) {
        const PtPass *p = &f->p[i];
        if (!p->sf && !p->sv)
            continue;
        s_pts.gb_passes++;
        s_pts.cat_ticks[p->cat] += p->excl;
        if (p->mixed)
            s_pts.mixed_ticks += p->excl;
    }
}

/* The frame being judged's timestamps into its passes' stage intervals
 * (gb_sweep shares them out). Its command buffers are done (the flip after
 * it waited for later work on the same queue); the last one sampled into
 * is waited for if not. */
static void gb_resolve(PtFrame *f)
{
    int k = (int)(f->flip & 1), i;
    if (s_gb_on <= 0 || !f->gb_n)
        return;
    if (@available(macOS 11.0, iOS 14.0, *)) {
        @autoreleasepool {
            id<MTLCommandBuffer> c = (id<MTLCommandBuffer>)s_gb_cmd[k];
            NSData *d;
            const MTLCounterResultTimestamp *ts;
            if (c && [c status] < MTLCommandBufferStatusCompleted) {
                if ([c status] < MTLCommandBufferStatusCommitted) {
                    s_pts.gb_missing += (uint64_t)f->n;   /* never committed: nothing to read */
                    return;
                }
                [c waitUntilCompleted];
            }
            d = [(id<MTLCounterSampleBuffer>)s_gb_sb[k] resolveCounterRange:NSMakeRange(0, (NSUInteger)f->gb_n)];
            if (!d || [d length] < (NSUInteger)f->gb_n * sizeof *ts) {
                s_pts.gb_missing += (uint64_t)f->n;
                return;
            }
            ts = (const MTLCounterResultTimestamp *)[d bytes];
            {
                /* Should the GPU clock gb_now reads and the samples not
                 * share a base after all (every written sample "older" than
                 * its pass), a sample more than 50 M ticks (50 ms at the
                 * M4's 1 ns) older than the frame's newest is the stale one
                 * instead. */
                static int clock_bad;
                uint64_t g_end = gb_now(), newest = 0;
                int nz = 0, ok = 0;
                for (i = 0; i < f->n; i++) {
                    const PtPass *p = &f->p[i];
                    int j, n = p->kind == PTK_RENDER ? 4 : 2;
                    if (p->gb < 0 || (p->kind == PTK_RENDER && !p->draws && !p->clears))
                        continue;
                    for (j = 0; j < n; j++) {
                        uint64_t x = ts[p->gb + j].timestamp;
                        if (x == MTLCounterErrorValue || !x)
                            continue;
                        nz++;
                        if (x > newest)
                            newest = x;
                        if ((!p->g_open || x >= p->g_open) && x <= g_end)
                            ok++;
                    }
                }
                if (nz && !ok && !clock_bad) {
                    clock_bad = 1;
                    fprintf(stderr, LOGP "metal_gpu_blocks: no sample of flip %llu is newer than"
                            " its pass's encoding by sampleTimestamps: stale samples taken as those"
                            " 50 M ticks older than the frame's newest from here on\n",
                            (unsigned long long)f->flip);
                }
                for (i = 0; i < f->n; i++) {
                    PtPass *p = &f->p[i];
                    uint64_t v[4];
                    int j, n = p->kind == PTK_RENDER ? 4 : 2, stale = 0;
                    p->sv = p->ev = p->sf = p->ef = p->excl = 0;
                    if (p->gb < 0) {
                        if (p->kind != PTK_AUX)
                            s_pts.gb_missing++;
                        continue;
                    }
                    /* No draw: its stages may write no sample, or some; its
                     * load and store work shows in no stage's interval
                     * either. Counted, not timed. */
                    if (p->kind == PTK_RENDER && !p->draws && !p->clears) {
                        s_pts.gb_nodraw++;
                        continue;
                    }
                    for (j = 0; j < n; j++) {
                        v[j] = ts[p->gb + j].timestamp;
                        if (v[j] == MTLCounterErrorValue || !v[j]) {
                            v[j] = 0;   /* a stage the pass never ran */
                        } else if (clock_bad ? newest - v[j] > 50000000u
                                             : (p->g_open && v[j] < p->g_open) || v[j] > g_end) {
                            v[j] = 0;   /* left from an older frame: never written */
                            stale = 1;
                        }
                    }
                    /* A stage counts with both its samples, end after start. */
                    if (n == 4) {
                        if (v[0] && v[1] && v[1] >= v[0]) {
                            p->sv = v[0];
                            p->ev = v[1];
                        }
                        if (v[2] && v[3] && v[3] >= v[2]) {
                            p->sf = v[2];
                            p->ef = v[3];
                        }
                    } else if (v[0] && v[1] && v[1] >= v[0]) {
                        p->sf = v[0];
                        p->ef = v[1];
                    }
                    if (!p->sf && !p->sv) {
                        if (stale)
                            s_pts.gb_stale++;
                        else
                            s_pts.gb_missing++;
                    }
                }
            }
            gb_sweep(f);
        }
    }
}

/* host ns a GPU tick, over the window since the last call; 0 when the two
 * samples cannot tell. */
static double gb_window(void)
{
    double r = 0;
    if (@available(macOS 11.0, iOS 14.0, *)) {
        MTLTimestamp ct = 0, gt = 0;
        uint64_t now;
        [s_dev sampleTimestamps:&ct gpuTimestamp:&gt];
        now = prof_now();
        if (gt > s_gb_cal_gpu && now > s_gb_cal_ns)
            r = (double)(now - s_gb_cal_ns) / (double)(gt - s_gb_cal_gpu);
        s_gb_cal_ns = now;
        s_gb_cal_gpu = gt;
    }
    return r;
}

static const char *pt_cat_draws(const PtPass *p, char *b, size_t n)
{
    size_t at = 0;
    int c;
    b[0] = 0;
    for (c = 0; c < NV2A_GCAT_N && at < n; c++)
        if (p->cat_draws[c])
            at += (size_t)snprintf(b + at, n - at, "%s%s %u", at ? ", " : "",
                                   nv2a_gpu_cat_name(c), p->cat_draws[c]);
    return b;
}

static void pt_print_att(const char *what, const PtAtt *a, char *b, size_t n)
{
    char r[96];
    if (!a->on) {
        snprintf(b, n, " | %s none", what);
        return;
    }
    if (a->rkind == PTR_NONE || a->rkind == PTR_UNKNOWN || a->rkind == PTR_FLIP
            || a->rkind == PTR_READBACK || a->rpass < 0)
        snprintf(r, sizeof r, "%s", s_ptr_name[a->rkind]);
    else
        snprintf(r, sizeof r, "%s by %s#%d", s_ptr_name[a->rkind], a->rnext ? "next frame " : "",
                 a->rpass);
    snprintf(b, n, " | %s 0x%08X %ux%u host %ux%u %s fmt 0x%X x1 load=%s store=%s read: %s"
             " (%u reads)", what, a->addr, a->w, a->h, a->hw, a->hh,
             what[0] == 'c' ? "BGRA8Unorm" : "Depth32Float_Stencil8", a->fmt,
             a->clear == 1 ? "clear" : a->clear == 2 ? "dontcare" : "load",
             a->store ? "store" : "dontcare", r, a->nread);
}

/* metal_passes_at: the judged frame, a line a pass, then its suspects. */
static void pt_detail(const PtFrame *f, double nspt)
{
    static const char *const plane[3] = { "colour", "depth", "stencil" };
    char cd[160], at[3][320];
    int i, k, ns = 0;
    fprintf(stderr, LOGP "passes flip %llu: %d passes (%d render, %d render-aux, %d compute,"
            " %d blit), %d past the table\n", (unsigned long long)f->flip, f->n + f->over,
            f->nk[PTK_RENDER], f->nk[PTK_AUX], f->nk[PTK_COMPUTE], f->nk[PTK_BLIT], f->over);
    uint64_t t0 = UINT64_MAX;
    double ms = (nspt > 0 ? nspt : 1.0) / 1e6;   /* uncalibrated: a tick taken as a ns */
    for (i = 0; i < f->n; i++) {
        const PtPass *p = &f->p[i];
        if (p->sv && p->sv < t0)
            t0 = p->sv;
        if (p->sf && p->sf < t0)
            t0 = p->sf;
    }
    for (i = 0; i < f->n; i++) {
        const PtPass *p = &f->p[i];
        char gpu[160] = "", sh[96];
        /* excl: its share of the frame's busy time (gb_sweep); frag and
         * vtx: its stages' own end - start; "at": from the frame's first
         * sample, ms */
        if (p->sf || p->sv)
            snprintf(gpu, sizeof gpu, " gpu %.3f ms (frag %.3f at +%.3f, vtx %.3f at +%.3f)%s",
                     p->excl * ms, p->sf ? (p->ef - p->sf) * ms : 0.0,
                     p->sf ? (p->sf - t0) * ms : 0.0, p->sv ? (p->ev - p->sv) * ms : 0.0,
                     p->sv ? (p->sv - t0) * ms : 0.0, nspt > 0 ? "" : " uncalibrated");
        else if (p->gb >= 0)
            snprintf(gpu, sizeof gpu, " gpu untimed (%s)",
                     p->kind == PTK_RENDER && !p->draws && !p->clears ? "no draw"
                                                                       : "no valid sample");
        for (k = 0; k < 3; k++)
            pt_print_att(plane[k], &p->att[k], at[k], sizeof at[k]);
        /* the first draw's shader: fs, stages, host size of each, UV snap */
        sh[0] = 0;
        if (p->kind == PTK_RENDER && p->sh_set) {
            size_t sn = (size_t)snprintf(sh, sizeof sh, " | fs 0x%08X stages %u tex", p->fs_id,
                                         p->nstages);
            for (k = 0; k < 4; k++)
                if (p->tw[k] && sn < sizeof sh)
                    sn += (size_t)snprintf(sh + sn, sizeof sh - sn, " %ux%u", p->tw[k], p->th[k]);
            if (sn < sizeof sh)
                snprintf(sh + sn, sizeof sh - sn, " snap %u", p->snap);
        }
        fprintf(stderr, LOGP "pass %llu #%d %s open=%s%s%s close=%s draws %u (%s) clears %u"
                " (%u depth or stencil) kind %s%s%s%s%s%s%s%s%s\n", (unsigned long long)f->flip, i,
                s_ptk_name[p->kind], s_pt_why_name[p->open_why], p->after != 0xFF ? " after " : "",
                p->after != 0xFF ? s_pt_why_name[p->after] : "", s_pt_why_name[p->close_why],
                p->draws, pt_cat_draws(p, cd, sizeof cd), p->clears, p->zdraws,
                nv2a_gpu_cat_name(p->cat), p->mixed ? " (mixed)" : "",
                p->dc == 1 ? " depth-dontcare" : p->dc == 2 ? " depth-dontcare (wrong guess)" : "",
                p->fold ? " clear-folded" : "", gpu,
                p->kind == PTK_RENDER ? at[0] : "", p->kind == PTK_RENDER ? at[1] : "",
                p->kind == PTK_RENDER ? at[2] : "", sh);
    }
    for (i = 0; i < f->n; i++) {
        const PtPass *p = &f->p[i];
        if (p->kind != PTK_RENDER)
            continue;
        if (p->reopen != 0xFF) {
            fprintf(stderr, LOGP "suspect %llu #%d: its colour target loaded again right after"
                    " the last render pass on it closed for %s (a store and a load of the same"
                    " target)\n", (unsigned long long)f->flip, i, s_pt_why_name[p->reopen]);
            ns++;
        }
        if (!p->draws && !p->clears) {
            fprintf(stderr, LOGP "suspect %llu #%d: no draw (only its load actions ran)\n",
                    (unsigned long long)f->flip, i);
            ns++;
        }
        if (!p->att[0].clear && p->first_full) {
            fprintf(stderr, LOGP "suspect %llu #%d: colour load=load, its first draw an opaque"
                    " quad over the whole scissored target (heuristic)\n",
                    (unsigned long long)f->flip, i);
            ns++;
        }
        for (k = 0; k < 3; k++) {
            const PtAtt *a = &p->att[k];
            if (!a->on || !a->store)
                continue;
            if (a->rkind == PTR_CLEARED) {
                fprintf(stderr, LOGP "suspect %llu #%d: %s store=store, overwritten unread by"
                        " %s#%d\n",
                        (unsigned long long)f->flip, i, plane[k], a->rnext ? "next frame " : "",
                        a->rpass);
                ns++;
            } else if (a->rkind == PTR_UNKNOWN) {
                fprintf(stderr, LOGP "suspect? %llu #%d: %s store=store, %s\n",
                        (unsigned long long)f->flip, i, plane[k], s_ptr_name[a->rkind]);
                ns++;
            }
        }
    }
    fprintf(stderr, LOGP "passes flip %llu: %d suspects\n", (unsigned long long)f->flip, ns);
}

/* The judged frame into the window's sums; the window's lines every 300. */
static void pt_judge(PtFrame *f)
{
    int i, k;
    double nspt = 0;
    /* What the next frame never touched stays unknown. */
    for (i = 0; i < s_ptw_n; i++) {
        if (s_ptw[i].flip != f->flip)
            continue;
        {
            PtAtt *a = ptw_att(i);
            if (a && !a->rkind)
                a->rkind = PTR_UNKNOWN;
        }
        s_ptw[i--] = s_ptw[--s_ptw_n];
    }
    {   /* metal_nodiscard's line: the busy time of the block's flips 31 on */
        uint64_t b0 = 0, b1 = 0;
        for (k = 0; k < NV2A_GCAT_N; k++)
            b0 += s_pts.cat_ticks[k];
        gb_resolve(f);
        for (k = 0; k < NV2A_GCAT_N; k++)
            b1 += s_pts.cat_ticks[k];
        if (s_pts.flips >= 30)
            s_pts.late_ticks += b1 - b0;
    }
    s_pts.nd.draws += f->nd.draws;
    s_pts.nd.elig += f->nd.elig;
    s_pts.nd.taken += f->nd.taken;
    s_pts.nd.win += f->nd.win;
    s_pts.nd.blend += f->nd.blend;
    s_pts.nd.fdepth += f->nd.fdepth;
    s_pts.nd.fast += f->nd.fast;
    s_pts.nd.pso += f->nd.pso;
    if (f->nd_arm_b)
        s_pts.arm_b++;
    else
        s_pts.arm_a++;
    for (k = 0; k < PTK_N; k++) {
        s_pts.nk[k] += (uint64_t)f->nk[k];
        if (f->nk[k] > s_pts.maxk[k])
            s_pts.maxk[k] = f->nk[k];
    }
    s_pts.over += (uint64_t)f->over;
    for (i = 0; i < f->n; i++) {
        const PtPass *p = &f->p[i];
        if (p->kind != PTK_RENDER)
            continue;
        if (!p->draws && !p->clears)
            s_pts.empty++;
        if (p->mixed)
            s_pts.mixed++;
        if (!p->att[0].clear && p->first_full)
            s_pts.sus_load++;
        for (k = 0; k < 3; k++) {
            if (!p->att[k].on || !p->att[k].store)
                continue;
            if (p->att[k].rkind == PTR_CLEARED)
                s_pts.sus_store[k]++;
            else if (p->att[k].rkind == PTR_UNKNOWN)
                s_pts.unknown[k]++;
        }
    }
    if (s_pt_log && s_pt_at >= 0 && f->flip == (uint64_t)s_pt_at) {
        /* restarts the calibration window: the next line's is shorter */
        nspt = s_gb_on > 0 ? gb_window() : 0;
        pt_detail(f, nspt);
    }
    if (++s_pts.flips < 300)
        return;
    {
        double n = (double)s_pts.flips, gscale = 0;
        char line[1024], arm[48] = "";
        size_t at = 0;
        /* metal_ab_alt: the block's arm, at the end of its lines */
        if (nd_any() && s_ab_blk) {
            if (s_pts.arm_a && s_pts.arm_b)
                snprintf(arm, sizeof arm, " arm mixed (A %d, B %d)", s_pts.arm_a, s_pts.arm_b);
            else
                snprintf(arm, sizeof arm, " arm %s", s_pts.arm_b ? "B" : "A");
        }
        if (s_pt_log) {
            fprintf(stderr, LOGP "passes over %d flips: render %.2f/frame (max %d), render-aux"
                    " %.2f (max %d), compute %.2f (max %d), blit %.2f (max %d); %.2f render"
                    " passes/frame with no draw, %.2f of several kinds; %llu passes past the table\n",
                    s_pts.flips, s_pts.nk[PTK_RENDER] / n, s_pts.maxk[PTK_RENDER],
                    s_pts.nk[PTK_AUX] / n, s_pts.maxk[PTK_AUX], s_pts.nk[PTK_COMPUTE] / n,
                    s_pts.maxk[PTK_COMPUTE], s_pts.nk[PTK_BLIT] / n, s_pts.maxk[PTK_BLIT],
                    s_pts.empty / n, s_pts.mixed / n, (unsigned long long)s_pts.over);
            at = 0;
            line[0] = 0;
            for (k = 0; k < PT_NWHY && at < sizeof line; k++)
                if (s_pts.close[k])
                    at += (size_t)snprintf(line + at, sizeof line - at, " %s %.2f",
                                           s_pt_why_name[k], s_pts.close[k] / n);
            fprintf(stderr, LOGP "passes closed by (/frame):%s\n", line);
            at = 0;
            line[0] = 0;
            for (k = 0; k < PT_NWHY && at < sizeof line; k++)
                if (s_pts.reopen[k])
                    at += (size_t)snprintf(line + at, sizeof line - at, " %s %.2f",
                                           s_pt_why_name[k], s_pts.reopen[k] / n);
            fprintf(stderr, LOGP "same colour target reopened with Load after (/frame):%s\n",
                    at ? line : " none");
            fprintf(stderr, LOGP "suspects (/frame): load over an opaque whole-target first draw"
                    " %.2f (heuristic); stored and cleared unread: colour %.2f depth %.2f stencil"
                    " %.2f; stored, untouched by the next frame (unknown): colour %.2f depth %.2f"
                    " stencil %.2f\n", s_pts.sus_load / n, s_pts.sus_store[0] / n,
                    s_pts.sus_store[1] / n, s_pts.sus_store[2] / n, s_pts.unknown[0] / n,
                    s_pts.unknown[1] / n, s_pts.unknown[2] / n);
        }
        if (s_gb_on > 0) {
            double r = gb_window(), sum = 0, scale = r > 0 ? r : 1.0;
            gscale = scale;
            at = 0;
            line[0] = 0;
            for (k = 0; k < NV2A_GCAT_N; k++) {
                double ms = s_pts.cat_ticks[k] * scale / n / 1e6;
                sum += ms;
                at += (size_t)snprintf(line + at, sizeof line - at, "%s%s %.3f", k ? " + " : "",
                                       nv2a_gpu_cat_name(k), ms);
            }
            /* Each pass's share of its frame's busy time (gb_sweep): the
             * blocks sum to the busy time, set against the command buffers'
             * GPUStartTime-GPUEndTime, which also holds the idle gaps and
             * the untimed work (passes with no draw, render-aux encoders). */
            fprintf(stderr, LOGP "gpu blocks over %d flips (ms/frame): %s = %.3f busy (mixed"
                    " passes %.3f of it), %.3f idle in a %.3f span from the first sample to the"
                    " last; command buffers %.3f, busy - buffers %+.3f; %s %.3f ns/tick; %llu"
                    " passes timed, untimed: %llu with no draw, %llu with only stale samples,"
                    " %llu without timestamps%s\n", s_pts.flips, line, sum,
                    s_pts.mixed_ticks * scale / n / 1e6, s_pts.idle_ticks * scale / n / 1e6,
                    s_pts.span_ticks * scale / n / 1e6, s_pts.cmd_ns / n / 1e6,
                    sum - s_pts.cmd_ns / n / 1e6, r > 0 ? "calibrated" : "UNCALIBRATED, taken as",
                    scale, (unsigned long long)s_pts.gb_passes,
                    (unsigned long long)s_pts.gb_nodraw, (unsigned long long)s_pts.gb_stale,
                    (unsigned long long)s_pts.gb_missing, arm);
        }
        /* metal_nodiscard (KFC-0846): the block's draws, and its busy time
         * without its first 30 flips (new pipelines after an arm change) */
        if (s_gb_on > 0 || nd_any()) {
            char late[64] = " no GPU time (metal_gpu_blocks off)";
            if (s_gb_on > 0 && s_pts.flips > 30)
                snprintf(late, sizeof late, " %.3f busy over flips 31-%d (ms/frame)",
                         s_pts.late_ticks * gscale / (double)(s_pts.flips - 30) / 1e6,
                         s_pts.flips);
            fprintf(stderr, LOGP "nodiscard over %d flips: %llu draws, %llu eligible (%llu of"
                    " them can gain: opaque, no depth from the fragment), %llu took no discard"
                    " (%llu blended, %llu with depth from the fragment), %llu with fast math;"
                    " %llu pipelines built;%s%s\n", s_pts.flips,
                    (unsigned long long)s_pts.nd.draws, (unsigned long long)s_pts.nd.elig,
                    (unsigned long long)s_pts.nd.win, (unsigned long long)s_pts.nd.taken,
                    (unsigned long long)s_pts.nd.blend, (unsigned long long)s_pts.nd.fdepth,
                    (unsigned long long)s_pts.nd.fast, (unsigned long long)s_pts.nd.pso,
                    late, arm);
        }
        if (s_gb_on > 0 && s_pts.cpu_n)
            fprintf(stderr, LOGP "cpu after scene over %d flips: mean %.3f ms, max %.3f ms (scene"
                    " pass close to flip commit)\n", s_pts.flips,
                    s_pts.cpu_sum_ms / (double)s_pts.cpu_n, s_pts.cpu_max_ms);
        if (s_fh_mode)
            fprintf(stderr, LOGP "metal_flat_half over %d flips: %llu draws clamped%s\n",
                    s_pts.flips, (unsigned long long)s_pts.fh_draws,
                    s_fh_mode == 2 ? " (alt: even flips only)" : "");
        if (s_dc_mode)
            fprintf(stderr, LOGP "metal_depth_dontcare over %d flips: %llu passes opened depth"
                    " DontCare, %llu wrong guesses%s\n", s_pts.flips,
                    (unsigned long long)s_pts.dc_passes, (unsigned long long)s_pts.dc_miss,
                    s_dc_mode == 2 ? " (alt: even flips only)" : "");
        if (s_cf_mode) {
            at = 0;
            line[0] = 0;
            for (k = 0; k < PT_NWHY && at < sizeof line; k++)
                if (s_pts.cf_end[k])
                    at += (size_t)snprintf(line + at, sizeof line - at, " %s %llu",
                                           s_pt_why_name[k], (unsigned long long)s_pts.cf_end[k]);
            fprintf(stderr, LOGP "metal_clear_fold over %d flips: %llu whole-target clears held"
                    " (%llu more merged into a held one), %llu folded into the next pass on their"
                    " textures; encoded as before: %llu for a pass on part of their textures, at an"
                    " end of pass for:%s; not held: %llu with an occlusion query open, %llu with %d"
                    " held already\n", s_pts.flips, (unsigned long long)s_pts.cf_held,
                    (unsigned long long)s_pts.cf_merged, (unsigned long long)s_pts.cf_folded,
                    (unsigned long long)s_pts.cf_shared, at ? line : " none",
                    (unsigned long long)s_pts.cf_query, (unsigned long long)s_pts.cf_full, CF_MAX);
        }
        memset(&s_pts, 0, sizeof s_pts);
    }
}

/* prof_flip_end, after s_presents moved to the next frame: the frame
 * before the one just shown is judged (this one has shown its loads and
 * clears), and its table starts the next frame. */
static void pt_flip(void)
{
    PtFrame *f = &s_ptf[s_presents & 1];
    if (s_presents >= 2 && f->flip == s_presents - 2)
        pt_judge(f);
    f->flip = s_presents;
    f->n = f->over = f->gb_n = 0;
    memset(f->nk, 0, sizeof f->nk);
    memset(&f->nd, 0, sizeof f->nd);
    f->nd_arm_b = s_ab_blk && nd_arm_b();   /* metal_ab_alt: this frame's arm */
    s_pt_cur = -1;
    s_pt_aux = -1;
    s_pt_last.valid = 0;
}

/* metal_shaders: what the first use of a shader, pipeline or texture costs
 * the ack thread, per flip. The clocks are read always: a build is
 * rare and costs milliseconds, so two clock reads around it are noise. */
static int s_sh_trace = -1;
static struct { uint64_t lib_ns, pso_ns, tex_ns; unsigned lib, pso, tex; } s_sh_flip;

static int sh_trace(void)
{
    if (s_sh_trace < 0)
        s_sh_trace = recomp_env(RENV_METAL_SHADERS) != NULL;
    return s_sh_trace;
}

/* The preload list ("The preload list" below): init starts the preload;
 * compile_fn and pipeline() record what they build. */
static void preload_start(void);
static void pl_note_library(id<MTLFunction> fn, const char *what, const char *name,
                            const char *src);
static void pl_note_pso(id<MTLFunction> vs, id<MTLFunction> fs, const uint32_t key[6]);

/* Each render encoder made, so a binding remembered against one is never
 * taken for another's (an address can come back after a release). */
static uint64_t s_enc_serial;
/* What vertex buffer 1 of the encoder s_vc1.enc holds: c[] at generation
 * s_vc1.gen while ok. */
static struct { uint64_t enc; uint32_t gen; int ok; } s_vc1;
static uint64_t s_c_uploads;   /* c[] writes to buffer 1; "consts" in the flip trace */

static void cf_encode(int i, int close_why);   /* metal_clear_fold, after ensure_cmd */

static void end_encoder(void)
{
    int why = s_pt_why;   /* metal_clear_fold: the held clears close for it */
    if (s_enc) {
        if (pt_on())
            pt_close();   /* metal_passes: why it closes (s_pt_why) */
        occ_pass_end();
        [s_enc endEncoding];
        [s_enc release];
        s_enc = nil;
    }
    s_pt_why = PT_UNTAGGED;   /* consumed, whether a pass was open or not */
    s_dc_pass = 0;
    s_enc_rt = NULL;
    s_enc_dt = NULL;
    /* RECOMP_DEBUG=metal_clear_fold: whatever ends a pass here (not
     * begin_pass's own end) runs after the held clears: they are encoded
     * now, each closed for this reason. Never set with the switch off. */
    if (s_cf_n && !s_cf_hold) {
        int i;
        for (i = 0; i < CF_MAX; i++)
            if (s_cf[i].rp) {
                s_pts.cf_end[why]++;
                cf_encode(i, why);
            }
    }
}

/* ---- Submissions (KFC-1720 H1b, step 2) -------------------------------------
 *
 * Every command buffer committed on s_queue gets a number, 1 up, in commit
 * order (sub_commit): the frame's (flush) and metal_rt_private's seed blit
 * (stage_write). Its completed handler, added before the commit, stores the
 * final status, then the number, into the entry's atomics and then signals
 * the entry's semaphore; it takes no lock and needs no other thread. The
 * entry keeps the command buffer retained until the backend's thread
 * observes the completion (sub_wait, sub_reap), which consumes that signal:
 * exactly once per number, so an observed entry's status is the handler's.
 * The metal_gpu_spin kernel's queue is not numbered. */
#define SUB_TABLE 256u
typedef struct {
    id<MTLCommandBuffer> cb;     /* retained until observed; nil: observed or free */
    uint64_t             n;      /* its number */
    int                  bad;    /* observed with status Error */
    dispatch_semaphore_t sem;    /* signalled once per number, by the handler */
    _Atomic uint64_t     done;   /* n once the handler ran */
    _Atomic int          err;    /* status Error, stored before done */
} SubEntry;
static SubEntry s_sub[SUB_TABLE];
static uint64_t s_sub_last;     /* the newest number given out, 0 none */
static uint64_t s_sub_floor;    /* every number up to this one is observed */
static _Atomic uint64_t s_sub_errs;   /* ended in Error, counted by the handler */
/* Why a sub_wait blocked: a flush(1)'s own wait (the serial flip's), the
 * seed blit's, a ring copy's reuse, a full table, a GET_REPORT in
 * RECOMP_METAL_OCC=sync (step 3a), a CPU read of a render target's bytes
 * (step 3a); CPU writes (step 3b): rt_seed's fill of s_rtp_buf, the debug
 * draws' s_dbg_vb, a frame uploaded into a presenter slot, the tex_mag
 * trace's read and zeroing of a frame's buffer; step 4: a ring copy's first
 * write after the flip (the ring guard, SUBW_GUARD; SUBW_RING keeps the ring
 * cuts and flush(1)'s own "every earlier one"), metal_inflight=2's wait one
 * frame back at the end of each flip (SUBW_FRAME). */
enum { SUBW_FLUSH, SUBW_STAGE, SUBW_RING, SUBW_FULL, SUBW_REPORT, SUBW_READ,
       SUBW_RTP, SUBW_DBG, SUBW_SLOT, SUBW_MAG, SUBW_GUARD, SUBW_FRAME, SUBW_N };
static uint64_t s_subw_blocked[SUBW_N], s_subw_ns[SUBW_N];
static uint64_t s_rb_skipped;   /* CPU reads skipped: the write's submission ended in Error */

/* The handler has stored (done == n) or is about to: take its signal and
 * let the command buffer go. A submission in Error marks the render targets
 * it was the last write of (wbad), which outlives the entry's reuse. */
static int sub_observe(SubEntry *e)
{
    dispatch_semaphore_wait(e->sem, DISPATCH_TIME_FOREVER);
    e->bad = atomic_load_explicit(&e->err, memory_order_acquire) != 0;
    [e->cb release];
    e->cb = nil;
    if (e->bad) {
        int i;
        for (i = 0; i < RT_MAX; i++)
            if (s_rt[i].tex && s_rt[i].last_write_sub == e->n)
                s_rt[i].wbad = 1;
    }
    return e->bad ? -1 : 1;
}

/* Observe, oldest first, every submission the handler has finished with. */
static void sub_reap(void)
{
    while (s_sub_floor < s_sub_last) {
        uint64_t k = s_sub_floor + 1;
        SubEntry *e = &s_sub[k % SUB_TABLE];
        if (e->n == k && e->cb) {
            if (atomic_load_explicit(&e->done, memory_order_acquire) != k)
                break;
            sub_observe(e);
        }
        s_sub_floor = k;
    }
}

/* Block until submission n is complete and its handler has published it:
 * 1 done, -1 done with status Error. 0 and numbers never given out are
 * done. Only the backend's thread calls it. */
static int sub_wait(uint64_t n, int why)
{
    SubEntry *e;
    uint64_t t0;
    int r;
    if (!n || n <= s_sub_floor || n > s_sub_last)
        return 1;
    e = &s_sub[n % SUB_TABLE];
    if (e->n != n)
        return 1;                     /* observed, the entry reused since */
    if (!e->cb)
        return e->bad ? -1 : 1;       /* observed */
    if (atomic_load_explicit(&e->done, memory_order_acquire) == n
        || [e->cb status] >= MTLCommandBufferStatusCompleted)
        return sub_observe(e);        /* the GPU is done: at most the handler to come */
    t0 = prof_now();
    [e->cb waitUntilCompleted];
    r = sub_observe(e);               /* the handler's store, not the wait, decides */
    s_subw_blocked[why]++;
    s_subw_ns[why] += prof_now() - t0;
    return r;
}

/* Every submission up to n, not only n: a ring copy is free once all that
 * bound it are done, whatever the queue's completion order. */
static void sub_wait_upto(uint64_t n, int why)
{
    if (n > s_sub_last)
        n = s_sub_last;
    while (s_sub_floor < n) {
        sub_wait(s_sub_floor + 1, why);
        sub_reap();
    }
}

/* sub_wait without the wait: 1 when submission n is done (observed now if
 * its handler has run), -1 done with status Error while its entry still
 * says so, 0 not yet. 0 and numbers never given out are done. */
static int sub_poll(uint64_t n)
{
    SubEntry *e;
    if (!n || n > s_sub_last)
        return 1;
    e = &s_sub[n % SUB_TABLE];
    if (e->n != n)
        return 1;                     /* observed, the entry reused since */
    if (!e->cb)
        return e->bad ? -1 : 1;       /* observed */
    if (atomic_load_explicit(&e->done, memory_order_acquire) == n)
        return sub_observe(e);
    return 0;
}

/* Every submission up to n is done (observed now where its handler ran). */
static int sub_done_upto(uint64_t n)
{
    sub_reap();
    return n <= s_sub_floor;
}

/* Number cb, register its completed handler, then commit it. */
static uint64_t sub_commit(id<MTLCommandBuffer> cb)
{
    uint64_t n;
    SubEntry *e;
    dispatch_semaphore_t sem;
    sub_reap();
    n = s_sub_last + 1;
    e = &s_sub[n % SUB_TABLE];
    if (e->cb)                        /* SUB_TABLE numbers back and not observed */
        sub_wait(e->n, SUBW_FULL);
    if (!e->sem)
        e->sem = dispatch_semaphore_create(0);
    sem = e->sem;
    e->n = n;
    e->bad = 0;
    e->cb = [cb retain];
    atomic_store_explicit(&e->err, 0, memory_order_relaxed);
    atomic_store_explicit(&e->done, 0, memory_order_relaxed);
    s_sub_last = n;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        int bad = [c status] == MTLCommandBufferStatusError;
        if (bad)
            atomic_fetch_add_explicit(&s_sub_errs, 1, memory_order_relaxed);
        atomic_store_explicit(&e->err, bad, memory_order_relaxed);
        atomic_store_explicit(&e->done, n, memory_order_release);
        dispatch_semaphore_signal(sem);
    }];
    [cb commit];
    return n;
}

/* End the open pass, commit the frame's command buffer and (wait) block
 * until the GPU is done with it. After a wait the rings are free again. */
static void flush(int wait)
{
    int flipping = s_pt_why == PT_FLIP;   /* end_encoder consumes it */
    uint64_t n;
    end_encoder();
    if (!s_cmd) {
        if (wait) {
            /* Nothing open, but a commit nobody waited for (a GET_REPORT's,
             * the layer copy's) may still read the active ring copy: its
             * last submission first, then the rings are free. */
            sub_wait_upto(s_ring[s_ring_cur].last_sub, SUBW_RING);
            s_ring_guard = 0;
            s_vb_pos = 0;
            s_ib_pos = 0;
            s_flushes++;
        }
        return;
    }
    n = sub_commit(s_cmd);
    if (s_ring_bound) {
        s_ring[s_ring_cur].last_sub = n;
        s_ring_bound = 0;
    }
    /* What this command buffer writes for the CPU to read takes its number
     * (KFC-1720 H1b step 3a): the targets it draws or copies into, their
     * sync blits, the visibility slots it counts into, the tex_mag frame.
     * And what it uses that the CPU rewrites (step 3b): any target it reads
     * or writes, s_rtp_buf, s_dbg_vb. */
    {
        int i;
        for (i = 0; i < RT_MAX; i++) {
            if (s_rt[i].wpend || s_rt[i].spend || s_rt[i].upend) {
                s_rt[i].use_sub = n;
                s_rt[i].upend = 0;
            }
            if (s_rt[i].wpend) {
                s_rt[i].last_write_sub = n;
                s_rt[i].wpend = 0;
                s_rt[i].wbad = 0;
            }
            if (s_rt[i].spend) {
                s_rt[i].sync_sub = n;
                s_rt[i].spend = 0;
            }
        }
    }
    if (s_rtp_pend) {
        s_rtp_sub = n;
        s_rtp_pend = 0;
    }
    if (s_dbg_vb_pend) {
        s_dbg_vb_sub = n;
        s_dbg_vb_pend = 0;
    }
    occ_committed(n);
    mag_committed(n);
    cap_committed(n);
    if (flipping && s_gb_on > 0 && s_scene_close_ns) {
        double ms = (prof_now() - s_scene_close_ns) / 1e6;
        s_pts.cpu_n++;
        s_pts.cpu_sum_ms += ms;
        if (ms > s_pts.cpu_max_ms)
            s_pts.cpu_max_ms = ms;
        s_scene_close_ns = 0;
    }
    if (wait) {
        if (sub_wait(n, SUBW_FLUSH) < 0) {
            s_cmd_errors++;
            if (s_cmd_errors <= 4)
                fprintf(stderr, LOGP "command buffer error: %s\n",
                        [[[s_cmd error] localizedDescription] UTF8String]);
        }
        sub_wait_upto(n, SUBW_RING);   /* and every earlier one: the rings are free */
        s_vb_pos = 0;
        s_ib_pos = 0;
    }
    if (prof_on() && s_prof_ncmd < PROF_CMDS)
        s_prof_cmd[s_prof_ncmd++] = [s_cmd retain];
    [s_cmd release];
    s_cmd = nil;
    s_flushes++;
    if (wait)
        occ_after_wait();
}

static void ensure_cmd(void)
{
    if (!s_cmd)
        s_cmd = [[s_queue commandBuffer] retain];
}

/* metal_clear_fold: from's Clears onto to, where to loads (over: wherever). */
static void cf_apply(MTLRenderPassDescriptor *to, MTLRenderPassDescriptor *from, int over)
{
    MTLRenderPassColorAttachmentDescriptor *tc = to.colorAttachments[0];
    MTLRenderPassColorAttachmentDescriptor *fc = from.colorAttachments[0];
    if (fc.loadAction == MTLLoadActionClear && (over || tc.loadAction == MTLLoadActionLoad)) {
        tc.loadAction = MTLLoadActionClear;
        tc.clearColor = fc.clearColor;
    }
    if (from.depthAttachment.texture && from.depthAttachment.loadAction == MTLLoadActionClear
            && (over || to.depthAttachment.loadAction == MTLLoadActionLoad)) {
        to.depthAttachment.loadAction = MTLLoadActionClear;
        to.depthAttachment.clearDepth = from.depthAttachment.clearDepth;
    }
    if (from.stencilAttachment.texture && from.stencilAttachment.loadAction == MTLLoadActionClear
            && (over || to.stencilAttachment.loadAction == MTLLoadActionLoad)) {
        to.stencilAttachment.loadAction = MTLLoadActionClear;
        to.stencilAttachment.clearStencil = from.stencilAttachment.clearStencil;
    }
}

/* The held clear h is on exactly rt's colour and dt's depth (or none). */
static int cf_same(const CfHeld *h, const RenderTarget *rt, const DepthTarget *dt)
{
    return h->rt == rt && h->ctex == (const void *)rt->tex && h->dt == dt
        && (!dt || h->ztex == (const void *)dt->tex);
}

/* ... or on one of them. */
static int cf_shares(const CfHeld *h, const RenderTarget *rt, const DepthTarget *dt)
{
    return h->ctex == (const void *)rt->tex || (dt && h->ztex == (const void *)dt->tex);
}

/* The held clear i encoded as begin_pass would have encoded it (pt_open,
 * the occlusion query's slot) and closed for close_why. No pass is open. */
static void cf_encode(int i, int close_why)
{
    CfHeld *h = &s_cf[i];
    MTLRenderPassDescriptor *rp = h->rp;
    int hold = s_cf_hold;
    if (!rp)
        return;
    h->rp = nil;
    s_cf_n--;
    @autoreleasepool {
        ensure_cmd();
        if (pt_on())
            pt_open(h->rt, h->dt, rp, h->why, h->had);
        s_enc = [[s_cmd renderCommandEncoderWithDescriptor:rp] retain];
        s_enc_serial++;
        s_enc_rt = h->rt;
        s_enc_dt = h->dt;
        occ_pass_begin();
        s_cf_hold = 1;   /* the other held ones stay: the caller has them */
        PT_WHY(close_why);
        end_encoder();
        s_cf_hold = hold;
    }
    [rp release];
}

/* The held clear i encoded now, the open pass (another target's) ending
 * first for `why`. */
static void cf_spill(int i, int why)
{
    int hold = s_cf_hold;
    if (s_enc) {
        s_cf_hold = 1;
        PT_WHY(why);
        end_encoder();
        s_cf_hold = hold;
    }
    s_pts.cf_end[why]++;
    cf_encode(i, why);
}

/* Before rt (its colour) or dt goes, or is read: its held clears encoded. */
static void cf_drop(const RenderTarget *rt, const DepthTarget *dt, int why)
{
    int i;
    for (i = 0; i < CF_MAX && s_cf_n; i++)
        if (s_cf[i].rp && ((rt && s_cf[i].rt == rt) || (dt && s_cf[i].dt == dt)))
            cf_spill(i, why);
}

/* Before a CPU write into memory the GPU uses (KFC-1720 H1b step 3b): when
 * the open s_cmd uses it (*pend), s_cmd is committed first (`pt` tags its
 * pass end, metal_passes), which numbers *sub; then every submission up to
 * *sub, its last use, is waited for, not the frame. Up to, not only *sub:
 * earlier ones may have used it too, and two reads are not ordered against
 * each other. A wait that blocks is counted under `why`. */
static void cpu_write_wait(const int *pend, const uint64_t *sub, int pt, int why)
{
    if (*pend) {
        PT_WHY(pt);
        flush(0);
    }
    sub_wait_upto(*sub, why);
}

/* A GPU read of rt's texture encoded in s_cmd (use_sub). Only the targets
 * in s_rt: the self copy and the depth views rt_texture also serves are not
 * numbered. */
static void rt_read_mark(RenderTarget *rt)
{
    if (rt >= s_rt && rt < s_rt + RT_MAX)
        rt->upend = 1;
}

/* Open a pass on rt: with `clear`, loadAction Clear to c; else Load. The
 * pass attaches rt->dt when it has one; zmask (CLEAR_SURFACE's bit 0 depth,
 * bit 1 stencil) clears it to z / st, and a fresh buffer clears to the far
 * plane and stencil 0, where the CPU path starts its own. */
static void begin_pass(RenderTarget *rt, int clear, const double c[4],
                       unsigned zmask, float z, uint32_t st)
{
    MTLRenderPassDescriptor *rp;
    DepthTarget *dt = rt->dt;
    int why = s_pt_why, had = s_enc != nil;   /* metal_passes: end_encoder consumes it */
    int fold = 0;   /* metal_clear_fold: a held clear became this pass's load actions */
    s_cf_hold = 1;   /* metal_clear_fold: the held clears are decided below */
    end_encoder();
    s_cf_hold = 0;
    ensure_cmd();
    rt->dirty = 1;
    rt->wpend = 1;   /* the pass writes it: s_cmd's number is its last write */
    nv2a_rt_own_drawn(&rt->own, (uint32_t)s_presents);
    rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = rt->tex;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (clear) {
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(c[0], c[1], c[2], c[3]);
    } else {
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
    }
    rt->zseq = ++s_zseq;
    if (dt) {
        int zc = (zmask & 1u) || dt->fresh, sc = (zmask & 2u) || dt->fresh;
        if (zc || sc)
            dt->wseq = ++s_zseq;   /* cleared: see zt_texture */
        rp.depthAttachment.texture = dt->tex;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        rp.depthAttachment.loadAction = zc ? MTLLoadActionClear : MTLLoadActionLoad;
        /* a fresh buffer's far plane: 1, or E's far end, 0, in
         * RECOMP_W_EARLYZ's mode (metal_on_zclear passes z encoded) */
        rp.depthAttachment.clearDepth = (zmask & 1u) ? z : dt->earlyz ? 0.0 : 1.0;
        rp.stencilAttachment.texture = dt->tex;
        rp.stencilAttachment.storeAction = MTLStoreActionStore;
        rp.stencilAttachment.loadAction = sc ? MTLLoadActionClear : MTLLoadActionLoad;
        rp.stencilAttachment.clearStencil = (zmask & 2u) ? (st & 0xFFu) : 0u;
        dt->fresh = 0;
    }
    rp.visibilityResultBuffer = s_vis;
    /* RECOMP_DEBUG=metal_clear_fold (s_cf): a held clear on exactly these
     * textures takes this clear's Clears (a clear: nothing is opened) or
     * gives this pass its own where it loads; one on part of them is
     * encoded first. A whole-target clear is then held, unless an
     * occlusion query is open or the table is full. Off: s_cf_n is 0. */
    if (s_cf_mode) {
        int i, merged = 0;
        for (i = 0; i < CF_MAX && s_cf_n; i++) {
            CfHeld *h = &s_cf[i];
            if (!h->rp)
                continue;
            if (cf_same(h, rt, dt)) {
                if (s_cf_call) {
                    cf_apply(h->rp, rp, 1);   /* the later clear's actions win */
                    merged = 1;
                } else {
                    cf_apply(rp, h->rp, 0);
                    [h->rp release];
                    h->rp = nil;
                    s_cf_n--;
                    fold = 1;
                    s_pts.cf_folded++;
                }
            } else if (cf_shares(h, rt, dt)) {
                s_pts.cf_shared++;
                cf_encode(i, why);
            }
        }
        if (merged) {
            s_pts.cf_merged++;
            return;
        }
        if (s_cf_call) {
            for (i = 0; i < CF_MAX && s_cf[i].rp; i++)
                ;
            if (s_occ_q) {
                s_pts.cf_query++;
            } else if (i == CF_MAX) {
                s_pts.cf_full++;
            } else {
                CfHeld *h = &s_cf[i];
                h->rp = [rp retain];
                h->rt = rt;
                h->dt = dt;
                h->ctex = (const void *)rt->tex;
                h->ztex = dt ? (const void *)dt->tex : NULL;
                h->why = why;
                h->had = had;
                s_cf_n++;
                s_pts.cf_held++;
                return;
            }
        }
    }
    /* RECOMP_DEBUG=metal_depth_dontcare (pt_dc_guess): neither loaded into
     * tile memory nor stored back, when this pass loads both (a held
     * clear's Clears count). s_dc_opener is only ever set with the switch
     * on. */
    if (dt && s_dc_opener && rp.depthAttachment.loadAction == MTLLoadActionLoad
            && rp.stencilAttachment.loadAction == MTLLoadActionLoad && pt_dc_guess(rt, dt)) {
        rp.depthAttachment.loadAction = MTLLoadActionDontCare;
        rp.depthAttachment.storeAction = MTLStoreActionDontCare;
        rp.stencilAttachment.loadAction = MTLLoadActionDontCare;
        rp.stencilAttachment.storeAction = MTLStoreActionDontCare;
        s_dc_pass = 1;
    }
    if (pt_on()) {
        pt_open(rt, dt, rp, why, had);   /* metal_passes, and the timestamps of metal_gpu_blocks */
        if (fold) {
            PtPass *p = pt_cur_pass();
            if (p)
                p->fold = 1;
        }
    }
    s_enc = [[s_cmd renderCommandEncoderWithDescriptor:rp] retain];
    s_enc_serial++;
    s_enc_rt = rt;
    s_enc_dt = dt;
    occ_pass_begin();
}

static void begin_encoder(RenderTarget *rt, int clear, const double c[4])
{
    begin_pass(rt, clear, c, 0, 1.0f, 0);
}

static int init(void)
{
    static int tried;
    NSError *err = nil;
    id<MTLLibrary> lib;

    if (tried)
        return !s_failed;
    tried = 1;
    s_failed = 1;

    @autoreleasepool {
        void *wdev = NULL, *wq = NULL;
        /* The layer window was made before the guest started, with its own
         * device and queue: one device so its slots take this backend's
         * blits, one queue so a present pass runs after the blit it shows. */
        if (xbox_FramebufferWindowMetal(&wdev, &wq) && wdev && wq) {
            s_dev = [(id<MTLDevice>)wdev retain];
            s_queue = [(id<MTLCommandQueue>)wq retain];
            s_layer = 1;
        } else {
            s_dev = MTLCreateSystemDefaultDevice();
            if (s_dev)
                s_queue = [s_dev newCommandQueue];
        }
        if (!s_dev || !s_queue) {
            fprintf(stderr, LOGP "no Metal device\n");
            return 0;
        }
        s_unified = [s_dev hasUnifiedMemory] ? 1 : 0;
        s_rtp = s_unified && recomp_env_on(RENV_METAL_RT_PRIVATE);
        if (s_rtp)
            fprintf(stderr, LOGP "metal_rt_private=1: color render targets in Private storage,"
                    " CPU reads/writes through a staging blit\n");
        s_inflight = recomp_env_int(RENV_METAL_INFLIGHT, 1) >= 2 ? 2 : 1;
        if (s_inflight > 1)
            fprintf(stderr, LOGP "metal_inflight=2: the flip commits the frame without waiting;"
                    " at the end of flip N the CPU waits for flip N-1's work\n");
        s_cf_mode = recomp_env_on(RENV_METAL_CLEAR_FOLD);
        if (s_cf_mode)
            fprintf(stderr, LOGP "KFC-1105: metal_clear_fold=1: a whole-target colour clear or a"
                    " depth clear opens no pass; it becomes the Clear load actions of the next pass"
                    " on the same colour and depth textures, and is encoded as before ahead of"
                    " anything else that ends a pass, reads its target or shares only part of"
                    " them\n");
        /* Every Mac GPU samples BC; an iPad before the A-series GPUs that
         * report it does not, and its pack BC3 entries go up from the guest. */
#if TARGET_OS_OSX
        s_bc = 1;
#endif
        if (@available(macOS 11.0, iOS 16.4, *))
            s_bc = [s_dev supportsBCTextureCompression] ? 1 : 0;

        /* The shared options (metal_compile.h): fast math off, safe math,
         * the MSL 3.0 pin. */
        lib = metal_compile_library(s_dev, [NSString stringWithUTF8String:s_msl],
                                    "the base library", &err);
        if (!lib) {
            fprintf(stderr, LOGP "MSL compile failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            return 0;
        }
        s_vs_pass = [lib newFunctionWithName:@"vs_pass"];
        s_fs_basic = [lib newFunctionWithName:@"fs_basic"];
        s_vs_clear = [lib newFunctionWithName:@"vs_clear"];
        s_fs_clear = [lib newFunctionWithName:@"fs_clear"];
        [lib release];
        if (!s_vs_pass || !s_fs_basic || !s_vs_clear || !s_fs_clear) {
            fprintf(stderr, LOGP "MSL functions missing\n");
            return 0;
        }

        /* The vertex ring is bounded by the 16-bit vertex range (65536
         * vertices are 1 MB per float4 stream); the index ring by the cap,
         * so any batch the walker accepts fits: three 32-bit list entries
         * per index at most. */
        s_vb_size = 16u << 20;
        s_ib_size = NV_MAX_INDICES * 3u * 4u;
        int rings_ok = 1;
        {
            /* RECOMP_DEBUG=metal_ring_small=<KB> (test hook): both rings of
             * both copies at most that size, so draws cut the ring often;
             * a draw larger than a ring is skipped (it changes the picture) */
            long kb = recomp_env_int(RENV_METAL_RING_SMALL, 0);
            int k;
            if (kb > 0) {
                NSUInteger b = (NSUInteger)kb << 10;
                s_ring_small = 1;
                if (b < s_vb_size)
                    s_vb_size = b;
                if (b < s_ib_size)
                    s_ib_size = b;
                fprintf(stderr, LOGP "metal_ring_small=%ld: vertex rings %lu B, index rings"
                        " %lu B\n", kb, (unsigned long)s_vb_size, (unsigned long)s_ib_size);
            }
            for (k = 0; k < RING_COPIES; k++) {
                s_ring[k].vb = [s_dev newBufferWithLength:s_vb_size
                                                  options:MTLResourceStorageModeShared];
                s_ring[k].ib = [s_dev newBufferWithLength:s_ib_size
                                                  options:MTLResourceStorageModeShared];
                if (!s_ring[k].vb || !s_ring[k].ib)
                    rings_ok = 0;
            }
        }
        s_ring_cur = 0;
        s_vb = s_ring[0].vb;
        s_ib = s_ring[0].ib;
        s_vis = [s_dev newBufferWithLength:VIS_SLOTS * 8u
                                   options:MTLResourceStorageModeShared];
        if (!rings_ok || !s_vis) {
            fprintf(stderr, LOGP "buffer creation failed\n");
            return 0;
        }
        fprintf(stderr, LOGP "device up: %s (unified memory %d), fast math off; present %s,"
                " write-back %s\n", [[s_dev name] UTF8String], s_unified,
                s_layer ? "CAMetalLayer slots" : "texture readback",
                wb_always() ? "every flip" : "on demand");
    }
    s_failed = 0;
    preload_start();
    return 1;
}

/* ---- surfaces --------------------------------------------------------------- */

static void rt_sync(RenderTarget *rt, uint32_t pitch);
static void rt_free(RenderTarget *rt);

/* ---- render-target ownership (nv2a_backend_common.h) -------------------------
 *
 * A target is the title's surface only while the guest memory under it holds
 * what the backend last knew there. A title that frees a target's memory and
 * loads something else into it (a menu leaves 512x512 targets behind
 * whose megabytes the next level refills with textures and tables) would otherwise
 * get the old image sampled in place of its texels, or written back over
 * them. A stale target is dropped without a write-back. A/B:
 * RECOMP_DEBUG=rt_alias_check=0. */
static int alias_check_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = recomp_env(RENV_RT_ALIAS_CHECK);
        on = !(e && e[0] == '0');
    }
    return on;
}

static void rt_own_reset(RenderTarget *rt)
{
    uint64_t t;
    if (!alias_check_on() || !rt->pitch)
        return;
    guard_open();   /* our own read, not a title read */
    t = prof_t();
    nv2a_rt_own_reset(&rt->own, (const uint8_t *)xbox_GetMemoryOffset(), rt->addr,
                      rt->pitch, rt->h, (uint32_t)s_presents);
    prof_add(&s_pa.hash, t);
}

/* Drop rt when the title has rewritten its memory; 1 if it was dropped.
 * `why` names the caller for the log: bind, sync, evict, overlap, reuse. */
static int rt_drop_if_stale(RenderTarget *rt, const char *why)
{
    static int said;
    uint64_t t;
    int stale;

    if (!alias_check_on() || !rt->tex || !rt->pitch)
        return 0;
    guard_open();
    t = prof_t();
    stale = nv2a_rt_own_stale(&rt->own, (const uint8_t *)xbox_GetMemoryOffset(), rt->addr,
                              rt->pitch, rt->h, (uint32_t)s_presents);
    prof_add(&s_pa.hash, t);
    if (!stale)
        return 0;
    s_stale_drops++;
    if (said < 20 || recomp_env(RENV_METAL_TRACE)) {
        said++;
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target dropped (%s): the title"
                " rewrote its %u bytes; last drawn at flip %u, now %llu%s\n",
                rt->offset, rt->w, rt->h, why, rt->pitch * rt->h, rt->own.draw_flip,
                (unsigned long long)s_presents, said == 20 ? " (further drops: the"
                " present summary)" : "");
    }
    rt->dirty = 0;   /* the title's bytes win: no write-back */
    rt_free(rt);
    return 1;
}

static void rt_free(RenderTarget *rt)
{
    /* A presented target's bytes are what a new target at that surface seeds
     * from, as when every flip wrote it back. The eviction in surface() has
     * already written back any target it frees. */
    if (rt->dirty && rt->presented && rt_drop_if_stale(rt, "evict"))
        return;
    if (rt->dirty && rt->presented)
        rt_sync(rt, rt->pitch);
    cf_drop(rt, NULL, PT_EVICT);   /* metal_clear_fold: its held clear, as before */
    if (s_enc_rt == rt) {
        PT_WHY(PT_EVICT);
        end_encoder();
    }
    [rt->tex release];
    memset(rt, 0, sizeof *rt);
}

/* The depth buffer for the zeta surface now set, at w x h; made on first
 * use (fresh), the least recently used one giving way when all are taken. */
static int zt_on(void);   /* depth surfaces as textures, below rt_texture */

/* A depth buffer's textures released, its slot zeroed. */
static void dt_free(DepthTarget *d)
{
    [d->ztex release];
    [d->zst release];
    [d->tex release];
    memset(d, 0, sizeof *d);
}

/* Another write over guest memory at [addr, addr + bytes), at seq: a depth
 * surface there no longer holds what the memory reads (zt_find), unless it
 * is written again after it. */
static void dt_mem_written(uint32_t addr, uint32_t bytes, uint64_t seq)
{
    int i;
    for (i = 0; i < DT_MAX; i++) {
        DepthTarget *d = &s_dt[i];
        if (d->tex && seq > d->mseq && nv2a_rt_overlap(d->addr, d->cpitch * d->h, addr, bytes))
            d->mseq = seq;
    }
}

static DepthTarget *depth_target(uint32_t w, uint32_t h)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    uint32_t zeta = nv2a_pb_vsh_state()->zeta_offset;
    DepthTarget *e = NULL, *lru = &s_dt[0];
    MTLTextureDescriptor *td;
    int i;

    for (i = 0; i < DT_MAX; i++) {
        DepthTarget *d = &s_dt[i];
        if (d->tex && d->zeta == zeta && d->w == w && d->h == h) {
            d->last_use = s_tick;
            d->use_flip = (uint32_t)s_presents;
            d->cpitch = g->pitch;
            d->zfmt = (g->format >> 4) & 0xF;
            return d;
        }
        if (!d->tex && !e)
            e = d;
        if (d->last_use < lru->last_use)
            lru = d;
    }
    if (!e) {
        e = lru;
        cf_drop(NULL, e, PT_EVICT);   /* metal_clear_fold: its held clear, as before */
        if (s_enc_dt == e) {
            PT_WHY(PT_EVICT);
            end_encoder();
        }
        for (i = 0; i < RT_MAX; i++)
            if (s_rt[i].dt == e)
                s_rt[i].dt = NULL;
        dt_free(e);
    }
    e->pad = nv2a_host_pad(w, h);
    e->n = nv2a_host_size(w + 2u * e->pad, h, nv2a_host_scale_for(w, h), METAL_MAX_DIM,
                          &e->hw, &e->hh);
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                            width:e->hw height:e->hh mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    /* Read by zt_texture's compute pass: the depth as it is, the stencil
     * through an X32_Stencil8 view. */
    if (zt_on())
        td.usage |= MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    td.storageMode = MTLStorageModePrivate;
    e->tex = [s_dev newTextureWithDescriptor:td];
    if (!e->tex)
        return NULL;
    e->zeta = zeta;
    e->addr = nv2a_pb_dma_resolve(zeta);
    e->w = w;
    e->h = h;
    e->last_use = s_tick;
    e->use_flip = (uint32_t)s_presents;
    e->cpitch = g->pitch;
    e->zfmt = (g->format >> 4) & 0xF;
    e->wseq = ++s_zseq;
    e->fresh = 1;
    e->earlyz = nv2a_w_earlyz_want();   /* begin_pass's fresh clear reads it */
    if (e->hw != w)
        fprintf(stderr, LOGP "zeta 0x%08X: %ux%u depth buffer (%ux%u host)\n",
                zeta, w, h, e->hw, e->hh);
    else
        fprintf(stderr, LOGP "zeta 0x%08X: %ux%u depth buffer\n", zeta, w, h);
    return e;
}

/* The earlyz of the depth buffer depth_target(w, h) will give, without
 * making or evicting one: metal_on_draw picks its fragment function before
 * it attaches the buffer, and RECOMP_W_EARLYZ's fast path draws with the
 * stock one. A buffer not made yet will take nv2a_w_earlyz_want(). */
static int dt_earlyz_peek(uint32_t w, uint32_t h)
{
    uint32_t zeta = nv2a_pb_vsh_state()->zeta_offset;
    int i;
    for (i = 0; i < DT_MAX; i++) {
        const DepthTarget *d = &s_dt[i];
        if (d->tex && d->zeta == zeta && d->w == w && d->h == h)
            return d->earlyz;
    }
    return nv2a_w_earlyz_want();
}

/* A BGRA8 colour texture of hw x hh host pixels, as every target is made. */
static id<MTLTexture> rt_texture_new(uint32_t hw, uint32_t hh)
{
    MTLTextureDescriptor *td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                           width:hw height:hh
                                                       mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    /* Managed only for a discrete GPU (Intel Macs); deprecated on 27. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    td.storageMode = s_unified ? MTLStorageModeShared : MTLStorageModeManaged;
#pragma clang diagnostic pop
    if (s_rtp)
        td.storageMode = MTLStorageModePrivate;
    return [s_dev newTextureWithDescriptor:td];
}

/* metal_rt_private: s_rtp_buf at least `need` bytes. Nothing in flight uses
 * the old one: every staging blit is waited for before the next. */
static int stage_fit(size_t need)
{
    if (s_stage_cap >= need)
        return 1;
    [s_rtp_buf release];   /* a command buffer using it keeps its own reference */
    s_rtp_buf = [s_dev newBufferWithLength:need options:MTLResourceStorageModeShared];
    s_stage_cap = s_rtp_buf ? need : 0;
    s_stage_tex = NULL;
    s_rtp_pend = 0;   /* a new buffer: no submission uses it */
    s_rtp_sub = 0;
    return s_rtp_buf != nil;
}

/* getBytes of rt->tex's region r, rows bpr apart at dst. metal_rt_private:
 * from the staging copy rt_managed_sync queued and the caller's wait
 * completed (every read site syncs first); none of this texture: skipped. */
static void rt_get(const RenderTarget *rt, void *dst, NSUInteger bpr, MTLRegion r)
{
    const uint8_t *s;
    NSUInteger y;

    if (!s_rtp) {
        [rt->tex getBytes:dst bytesPerRow:bpr fromRegion:r mipmapLevel:0];
        return;
    }
    if (s_stage_tex != (const void *)rt->tex) {
        static int said;
        if (!said++)
            fprintf(stderr, LOGP "metal_rt_private: surface 0x%08X read with no staged copy,"
                    " skipped\n", rt->offset);
        return;
    }
    s = (const uint8_t *)[s_rtp_buf contents]
        + ((size_t)r.origin.y * rt->hw + r.origin.x) * 4u;
    for (y = 0; y < r.size.height; y++)
        memcpy((uint8_t *)dst + y * bpr, s + y * (size_t)rt->hw * 4u, r.size.width * 4u);
}

/* metal_rt_private: rt_seed's pixels (rw x rh, tight rows) are in s_rtp_buf;
 * blit them into tex on a command buffer of its own, committed and waited
 * for now. It is enqueued ahead of the frame's s_cmd (not committed yet), so
 * every pass that uses tex, and rt_grow's later blit of the old contents,
 * runs after it; and s_rtp_buf is free again on return. */
static void stage_write(id<MTLTexture> tex, uint32_t rw, uint32_t rh)
{
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [s_queue commandBuffer];
        id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b copyFromBuffer:s_rtp_buf sourceOffset:0 sourceBytesPerRow:(NSUInteger)rw * 4u
              sourceBytesPerImage:(NSUInteger)rw * 4u * rh sourceSize:MTLSizeMake(rw, rh, 1)
                toTexture:tex destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
        [b endEncoding];
        s_rtp_sub = sub_commit(cb);   /* it reads s_rtp_buf (step 3b) */
        sub_wait(s_rtp_sub, SUBW_STAGE);
    }
    s_stage_writes++;
}

/* tex (w x h guest, hw x hh host) from the guest's bytes at addr when the
 * surface is 32-bit, as the CPU path draws over whatever memory holds; 0 when
 * it is not (the texture is left as made). tex is always one rt_texture_new
 * made just before (surface, rt_grow): no submission has used it, so its
 * replaceRegion waits for nothing; s_rtp_buf may be in use (step 3b). */
static int rt_seed(id<MTLTexture> tex, uint32_t addr, uint32_t pitch, uint32_t w,
                    uint32_t h, uint32_t pad, uint32_t hw, uint32_t hh)
{
    const uint8_t *src = (const uint8_t *)xbox_GetMemoryOffset() + addr;

    if (pitch < w * 4u)
        return 0;
    guard_open();   /* our own read of the seed is not a title read */
    if (s_rtp) {
        /* Private: the same pixels into s_rtp_buf, then a blit (stage_write). */
        uint8_t *st;
        uint32_t y;
        if (!stage_fit((size_t)hw * hh * 4u))
            return 0;
        /* A staging copy may still be writing it: its submission first. */
        cpu_write_wait(&s_rtp_pend, &s_rtp_sub, PT_SYNC, SUBW_RTP);
        st = (uint8_t *)[s_rtp_buf contents];
        s_stage_tex = NULL;   /* it holds the seed now */
        if (hw == w) {
            for (y = 0; y < h; y++)
                memcpy(st + (size_t)y * w * 4u, src + (size_t)y * pitch, (size_t)w * 4u);
            stage_write(tex, w, h);
        } else {
            unsigned n = hw / (w + 2u * pad);
            memset(st, 0, (size_t)hw * hh * 4u);
            nv2a_upscale_nearest32(src, pitch, w, h, n, st + (size_t)pad * n * 4u, hw * 4u);
            stage_write(tex, hw, hh);
        }
        return 1;
    }
    if (hw == w) {
        [tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0
                 withBytes:src bytesPerRow:pitch];
    } else {
        /* Scaled or padded: the guest bytes, each pixel an n x n block,
         * centred; the hor+ pads start black (calloc). */
        unsigned n = hw / (w + 2u * pad);
        uint8_t *up = (uint8_t *)calloc((size_t)hw * hh, 4u);
        if (up) {
            nv2a_upscale_nearest32(src, pitch, w, h, n, up + (size_t)pad * n * 4u, hw * 4u);
            [tex replaceRegion:MTLRegionMake2D(0, 0, hw, hh) mipmapLevel:0
                     withBytes:up bytesPerRow:hw * 4u];
            free(up);
        }
    }
    return 1;
}

/* rt made w x h (guest pixels), what it holds kept at its top left: the
 * D3D11 backend's rt_grow. The new area is seeded from the guest bytes, as a
 * new target is; replaceRegion runs now and the blit of the old contents
 * later on the GPU, so the blit wins where both write. A blit cannot
 * rescale: when the larger size takes another host factor (past
 * METAL_MAX_DIM), rt is written back first and the whole texture seeded. */
static int rt_grow(RenderTarget *rt, uint32_t w, uint32_t h)
{
    static int said;
    id<MTLTexture> tex;
    uint32_t hw, hh, pad;
    unsigned n;
    int keep, enc = 0;   /* enc: s_cmd writes the new texture (clear pass, blit) */

    pad = nv2a_host_pad(w, h);
    n = nv2a_host_size(w + 2u * pad, h, nv2a_host_scale_for(w, h), METAL_MAX_DIM, &hw, &hh);
    keep = rt->pad == pad && rt->hw == (rt->w + 2u * pad) * n && rt->hh == rt->h * n;
    if (!keep)
        rt_sync(rt, rt->pitch);
    /* An encoder must not outlive its attachment's swap, and the passes
     * below need none open. */
    PT_WHY(PT_GROW);
    end_encoder();
    tex = rt_texture_new(hw, hh);
    if (!tex)
        return 0;
    if (!rt_seed(tex, rt->addr, rt->pitch, w, h, pad, hw, hh)) {
        /* A new texture's contents are undefined: a 16-bit surface's margin
         * starts black, as D3D11's does. */
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        id<MTLRenderCommandEncoder> e;
        rp.colorAttachments[0].texture = tex;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        ensure_cmd();
        if (pt_on())
            (void)pt_aux(PTK_AUX, PT_GROW, NV2A_GCAT_OTHER);   /* metal_passes: counted, not timed */
        e = [s_cmd renderCommandEncoderWithDescriptor:rp];
        [e endEncoding];
        enc = 1;
    }
    if (keep) {
        id<MTLBlitCommandEncoder> b;
        ensure_cmd();
        b = pt_on() ? pt_blit(PT_GROW, NV2A_GCAT_OTHER) : [s_cmd blitCommandEncoder];
        [b copyFromTexture:rt->tex sourceSlice:0 sourceLevel:0
              sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(rt->hw, rt->hh, 1)
                 toTexture:tex destinationSlice:0 destinationLevel:0
         destinationOrigin:MTLOriginMake(0, 0, 0)];
        [b endEncoding];
        enc = 1;
    }
    if (said++ < 20)
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target grown to %ux%u\n",
                rt->offset, rt->w, rt->h, w, h);
    [rt->tex release];   /* an encoded command keeps its own reference */
    rt->tex = tex;
    /* A new texture: the seed is the CPU's (or stage_write's, waited for);
     * a clear pass or the blit of the old contents is s_cmd's. */
    rt->wbad = 0;
    rt->wpend |= enc;
    rt->w = w;
    rt->h = h;
    rt->hw = hw;
    rt->hh = hh;
    rt->pad = pad;
    rt->n = n;
    /* The cached depth buffer is the old size; begin_pass would attach it to
     * the larger colour texture. The next depth draw attaches one at the
     * grown size (depth_target), fresh. */
    rt->dt = NULL;
    rt_own_reset(rt);
    s_grows++;
    return 1;
}

/* The render target for the surface at color_offset, at least the size of
 * its clip extent (the D3D11 backend's rule), made on first use. A new one
 * starts from the guest's bytes (rt_seed). */
static RenderTarget *surface(void)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    uint32_t w = g->clip_x + g->clip_w, h = g->clip_y + g->clip_h;
    RenderTarget *free_rt = NULL, *lru = &s_rt[0];
    uint32_t hw, hh, pad;
    unsigned n;
    int i;

    if (!g->color_offset || !w || !h || w > 4096 || h > 4096)
        return NULL;
    s_tick++;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex && rt->offset == g->color_offset && rt->w == w && rt->h == h) {
            /* First use this flip of a target the title may have rewritten:
             * a stale one is made again from the title's bytes, which the
             * GPU would draw over. */
            if (rt_drop_if_stale(rt, "reuse"))
                break;
            rt->last_use = s_tick;
            rt->fmt = g->format;
            rt->copy = 0;
            return rt;
        }
    }
    /* The same surface under another clip extent: one target per surface,
     * as large as any clip it was drawn with (D3D11's rule, toolkit
     * 4773a01). A title can draw its 640x480 back buffer with clips of
     * 640x464, 623x401, 159x344 and more in one frame; a target per extent
     * made the overlap rule below write back and retire one at every clip
     * change (a GPU wait, and at render.scale > 1 the box filter's loss),
     * and a flip found no target of the frame's size. Draws map guest pixels
     * to NDC through rt->w/h, so a larger target keeps every position; a
     * clear covers only its clear rect or clip (metal_on_clear) and a flip
     * shows only the frame's extent (present_extent). Another pitch at the
     * same offset is a new allocation, not another clip: it gets a target of
     * its own and the overlap rule retires this one. */
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (!rt->tex || rt->offset != g->color_offset || rt->pitch != g->pitch)
            continue;
        if (rt_drop_if_stale(rt, "reuse"))
            break;
        if ((rt->w >= w && rt->h >= h)
                || rt_grow(rt, rt->w > w ? rt->w : w, rt->h > h ? rt->h : h)) {
            rt->last_use = s_tick;
            rt->fmt = g->format;
            rt->copy = 0;
            return rt;
        }
        break;
    }
    /* No two live targets share memory: one the new target overlaps is
     * written back first (the seed below then reads what was drawn) or,
     * if the title rewrote it, dropped. */
    if (alias_check_on()) {
        uint32_t addr = nv2a_pb_dma_resolve(g->color_offset);
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *rt = &s_rt[i];
            if (!rt->tex || !nv2a_rt_overlap(rt->addr, rt->pitch * rt->h, addr, g->pitch * h))
                continue;
            if (rt_drop_if_stale(rt, "overlap"))
                continue;
            rt_sync(rt, rt->pitch);
            if (s_retired++ < 20)
                fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target retired:"
                        " 0x%08X %ux%u overlaps it\n", rt->offset, rt->w, rt->h,
                        g->color_offset, w, h);
            rt_free(rt);
        }
    }
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (!rt->tex && !free_rt)
            free_rt = rt;
        if (rt->last_use < lru->last_use)
            lru = rt;
    }
    if (!free_rt) {
        free_rt = lru;
        /* What was drawn goes to guest memory first, presented or not: a
         * later pass that samples this surface decodes it from there
         * (tex_bind) or seeds a new target from it. A target the title
         * rewrote is dropped instead (its bytes win). */
        if (!rt_drop_if_stale(free_rt, "evict")) {
            rt_sync(free_rt, free_rt->pitch);
            if (free_rt->dirty && s_evict_lost++ < 20)
                fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target evicted for"
                        " 0x%08X with no write-back (pitch %u), its pixels lost\n",
                        free_rt->offset, free_rt->w, free_rt->h, g->color_offset,
                        free_rt->pitch);
            rt_free(free_rt);
        }
    }
    pad = nv2a_host_pad(w, h);
    n = nv2a_host_size(w + 2u * pad, h, nv2a_host_scale_for(w, h), METAL_MAX_DIM, &hw, &hh);
    free_rt->tex = rt_texture_new(hw, hh);
    if (!free_rt->tex)
        return NULL;
    free_rt->offset = g->color_offset;
    free_rt->addr = nv2a_pb_dma_resolve(g->color_offset);
    free_rt->w = w;
    free_rt->h = h;
    free_rt->hw = hw;
    free_rt->hh = hh;
    free_rt->pad = pad;
    free_rt->n = n;
    free_rt->pitch = g->pitch;
    free_rt->fmt = g->format;
    free_rt->last_use = s_tick;
    rt_own_reset(free_rt);   /* the bytes the seed reads, or would */
    rt_seed(free_rt->tex, free_rt->addr, g->pitch, w, h, pad, hw, hh);
    if (hw != w)
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target (%ux%u host)\n",
                g->color_offset, w, h, hw, hh);
    else
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target\n",
                g->color_offset, w, h);
    return free_rt;
}

static id<MTLDepthStencilState> depth_state(int depth, int stencil, int ez);

/* The partial-clear pipeline for a pass with (dt) or without a depth
 * attachment, made on first use; nil if it cannot be made. */
static id<MTLRenderPipelineState> clear_pso(int dt)
{
    static int failed[2];
    MTLRenderPipelineDescriptor *pd;
    NSError *err = nil;

    if (s_clear_pso[dt] || failed[dt])
        return s_clear_pso[dt];
    pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = s_vs_clear;
    pd.fragmentFunction = s_fs_clear;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pd.colorAttachments[0].writeMask = MTLColorWriteMaskAll;
    if (dt) {
        pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    }
    s_clear_pso[dt] = [s_dev newRenderPipelineStateWithDescriptor:pd error:&err];
    [pd release];
    if (!s_clear_pso[dt]) {
        failed[dt] = 1;
        fprintf(stderr, LOGP "partial-clear pipeline failed (%s): a partial clear clears"
                " the whole target\n", err ? [[err localizedDescription] UTF8String] : "?");
    }
    return s_clear_pso[dt];
}

/* A colour clear covers its clear rect, else its clip (nv2a_clear_box), as on
 * D3D11 and the CPU path: one target serves every clip of its surface
 * (rt_grow). A load action clears a whole attachment, so a box that is the
 * whole target keeps it, and a smaller one is a full-screen triangle of the
 * clear colour under a scissor, in a loading pass. */
static void metal_on_clear(uint32_t param)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    RenderTarget *rt;
    double c[4];
    uint32_t argb;
    uint32_t b[4];
    id<MTLRenderPipelineState> pso;

    if (!(param & 0xF0) || !init())
        return;
    @autoreleasepool {
        rt = surface();
        if (!rt || !nv2a_clear_box(g, rt->w, rt->h, b))
            return;
        argb = nv2a_clear_argb(g);   /* in the surface's format */
        c[0] = (double)((argb >> 16) & 0xFF) / 255.0;
        c[1] = (double)((argb >> 8) & 0xFF) / 255.0;
        c[2] = (double)(argb & 0xFF) / 255.0;
        c[3] = (double)(argb >> 24) / 255.0;
        s_clears++;
        if (b[0] == 0 && b[1] == 0 && b[2] == rt->w && b[3] == rt->h) {
            PT_WHY(PT_CLEAR);
            s_cf_call = s_cf_mode;   /* metal_clear_fold: it may be held */
            begin_encoder(rt, 1, c);
            s_cf_call = 0;
            return;
        }
        if (!s_enc || s_enc_rt != rt) {
            PT_WHY(s_enc ? PT_TARGET : PT_RESUME);
            begin_encoder(rt, 0, NULL);
        }
        pso = clear_pso(s_enc_dt ? 1 : 0);
        if (!pso) {
            PT_WHY(PT_CLEAR);
            begin_encoder(rt, 1, c);
            return;
        }
        {
            /* byte/255: BGRA8Unorm stores the bytes the CPU path writes. */
            float fc[4] = {(float)c[0], (float)c[1], (float)c[2], (float)c[3]};
            MTLViewport vp = {0.0, 0.0, (double)rt->hw, (double)rt->hh, 0.0, 1.0};
            MTLScissorRect sr;
            uint32_t hb[4];
            /* As a 3D draw's clip: an edge at 0 or w reaches the pads. */
            nv2a_aspect_box(b, rt->w, rt->h, rt->pad, rt->n, NV2A_PLACE_CENTRE, 0, 0, hb);
            sr.x = hb[0];
            sr.y = hb[1];
            sr.width = hb[2] - hb[0];
            sr.height = hb[3] - hb[1];
            [s_enc setRenderPipelineState:pso];
            if (s_enc_dt)
                [s_enc setDepthStencilState:depth_state(0, 0, 0)];
            [s_enc setCullMode:MTLCullModeNone];
            [s_enc setViewport:vp];
            [s_enc setScissorRect:sr];
            [s_enc setFragmentBytes:fc length:sizeof fc atIndex:0];
            /* A clear is not a zpass sample: no counting for this draw. Metal
             * sets a visibility offset once per pass, so counting cannot
             * resume at the same slot: the pass ends, and the next draw's
             * pass takes a new slot in the query (occ_pass_begin), as a
             * query spanning passes does. */
            if (s_occ_slot >= 0)
                [s_enc setVisibilityResultMode:MTLVisibilityResultModeDisabled
                                        offset:(NSUInteger)s_occ_slot * 8u];
            [s_enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            if (pt_on())
                pt_note_clear();
            /* The scissor stays: each draw sets its own (metal_on_draw). */
            if (s_occ_slot >= 0) {
                PT_WHY(PT_CLEAR_Q);
                end_encoder();
            }
        }
        rt->zseq = ++s_zseq;   /* zt_find */
        s_partial_clears++;
    }
}

/* CLEAR_SURFACE with the Z or stencil bit: the value scaled as the CPU path
 * scales it, by the zeta format's range; the stencil its low byte. A pass
 * that loads colour and clears the depth buffer. */
static void metal_on_zclear(uint32_t param)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    RenderTarget *rt;
    DepthTarget *dt;
    float z;

    if (!(param & 3u) || !init())
        return;
    @autoreleasepool {
        rt = surface();
        if (!rt)
            return;
        dt = depth_target(rt->w, rt->h);
        if (!dt)
            return;
        rt->dt = dt;
        z = nv2a_zclear_depth(g->format, v->zclear);
        /* RECOMP_W_EARLYZ, as D3D11's d3d11_on_zclear: each depth clear
         * decides the buffer's encoding (a stencil-only one keeps it), and
         * stores E of the clear depth in that mode. */
        if (param & 1u)
            dt->earlyz = nv2a_w_earlyz_want();
        s_prof_ez = dt->earlyz;   /* metal_prof: the mode this frame draws in */
        if (dt->earlyz)
            z = nv2a_w_earlyz_clear(z, nv2a_zmax(g->format), nv2a_w_earlyz_b());
        PT_WHY(PT_ZCLEAR);
        s_cf_call = s_cf_mode;   /* metal_clear_fold: it may be held */
        begin_pass(rt, 0, NULL, param & 3u, z, v->zclear & 0xFFu);
        s_cf_call = 0;
        if (param & 1u)
            s_zclears++;
    }
}

/* ---- vertices ---------------------------------------------------------------- */

/* One attribute of one vertex as float4, decoded the way the CPU path's
 * fetch_attr does. 0 if the stream has no data for it. */
static int fetch(const struct nv2a_pb_draw *d, const VertexAttr *a,
                 uint32_t index, float out[4])
{
    const uint8_t *p;

    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    if (d->inline_data) {
        size_t at = (size_t)a->offset + (size_t)index * a->stride;
        if (at + 4 > d->inline_bytes)
            return 0;
        p = (const uint8_t *)d->inline_buf + at;
    } else {
        if (!a->offset)
            return 0;
        p = (const uint8_t *)xbox_GetMemoryOffset() + a->offset
          + (size_t)index * a->stride;
    }
    return nv2a_vtx_decode(a->type, a->size, p, out);
}

/* A whole draw's room in the active ring copy, vertex and index bytes
 * together, 16-aligned, before anything is written. When either does not
 * fit, the frame so far is flushed and waited for (flush(1): the pass ends,
 * and the draw opens the next with Load, colour, depth and stencil kept) and
 * the active copy starts over; the other copy is not touched. 0 when the
 * draw can never fit a ring: skipped, counted, logged once. */
static int ring_reserve(NSUInteger vbytes, NSUInteger ibytes, float **vf, NSUInteger *vb_off,
                        void **ip, NSUInteger *ib_off)
{
    NSUInteger va, ia;
    if (vbytes > s_vb_size || ibytes > s_ib_size) {
        if (!s_ring_oversize++)
            fprintf(stderr, LOGP "a draw of %lu vertex and %lu index bytes exceeds the rings"
                    " (%lu and %lu): skipped (later ones counted, not logged)\n",
                    (unsigned long)vbytes, (unsigned long)ibytes,
                    (unsigned long)s_vb_size, (unsigned long)s_ib_size);
        return 0;
    }
    if (s_ring_guard) {
        /* the copy's first write since the flip: whatever read it last is
         * done (with metal_inflight=2 the flip's flip_bound has waited for
         * the frame that bound it: this blocks only if that one did not) */
        s_ring_guard = 0;
        sub_wait_upto(s_ring[s_ring_cur].last_sub, SUBW_GUARD);
    }
    va = (s_vb_pos + 15u) & ~(NSUInteger)15u;
    ia = (s_ib_pos + 15u) & ~(NSUInteger)15u;
    if (va + vbytes > s_vb_size || ia + ibytes > s_ib_size) {
        PT_WHY(PT_RING);
        s_ring_cuts++;
        flush(1);   /* waits everything so far; the active copy's cursors to 0 */
        va = 0;
        ia = 0;
    }
    s_vb_pos = va + vbytes;
    s_ib_pos = ia + ibytes;
    *vb_off = va;
    *ib_off = ia;
    *vf = (float *)((uint8_t *)[s_vb contents] + va);
    *ip = (uint8_t *)[s_ib contents] + ia;
    return 1;
}

/* At the flip, after its commit (and wait, metal_inflight=1): the other copy
 * becomes the active one, from its start; its first write waits for its last
 * submission, the frame before the one just committed (serial: long done;
 * metal_inflight=2: done by flip_bound at the end of this flip). Then the
 * completions so far are observed. */
static void ring_switch(void)
{
    s_ring_cur = (s_ring_cur + 1) % RING_COPIES;
    s_vb = s_ring[s_ring_cur].vb;
    s_ib = s_ring[s_ring_cur].ib;
    s_vb_pos = 0;
    s_ib_pos = 0;
    s_ring_guard = 1;   /* the flip calls it with nothing open: s_ring_bound is 0 */
    s_ring_switches++;
    sub_reap();
}

/* ---- pipelines (blend state is part of them in Metal) ----------------------- */

static MTLBlendOperation blend_op(uint32_t eq)
{
    switch (eq) {
    case 0x800A: return MTLBlendOperationSubtract;
    case 0x800B:
    case 0xF005: return MTLBlendOperationReverseSubtract;
    case 0x8007: return MTLBlendOperationMin;
    case 0x8008: return MTLBlendOperationMax;
    default:     return MTLBlendOperationAdd;
    }
}

static MTLBlendFactor blend_factor(uint32_t gl, int alpha)
{
    static const MTLBlendFactor map[NV2A_BF_UNKNOWN + 1] = {
        MTLBlendFactorZero, MTLBlendFactorOne,
        MTLBlendFactorSourceColor, MTLBlendFactorOneMinusSourceColor,
        MTLBlendFactorSourceAlpha, MTLBlendFactorOneMinusSourceAlpha,
        MTLBlendFactorDestinationAlpha, MTLBlendFactorOneMinusDestinationAlpha,
        MTLBlendFactorDestinationColor, MTLBlendFactorOneMinusDestinationColor,
        MTLBlendFactorSourceAlphaSaturated,
        /* As the D3D11 backend: one RGBA constant, which
         * nv2a_blend_constant() fills for _COLOR or _ALPHA use. */
        MTLBlendFactorBlendColor, MTLBlendFactorOneMinusBlendColor,
        MTLBlendFactorBlendColor, MTLBlendFactorOneMinusBlendColor,
        MTLBlendFactorOne,                      /* unknown */
    };
    uint32_t bf = nv2a_blend_from_gl(gl);
    return map[alpha ? nv2a_blend_for_alpha(bf) : bf];
}

/* A pipeline is its two functions plus the blend state. The functions are
 * cached for the life of the process (vs_cache, fs_cache), so their
 * pointers identify them. */
typedef struct {
    uint64_t vs, fs;
    uint32_t key[6];
} PsoKey;
#define PSO_MAX 512
static struct { PsoKey k; id<MTLRenderPipelineState> pso; } s_pso[PSO_MAX];
static int s_pso_n, s_pso_last;

/* The pipeline state for vs, fs and a pipeline() key (+1, nil on failure).
 * pipeline() and the preload both build here, so what the preload warms is
 * what the game asks for. Any thread: it touches only its arguments. */
static id<MTLRenderPipelineState> pso_build(id<MTLDevice> dev, id<MTLFunction> vs,
                                            id<MTLFunction> fs, const uint32_t key[6],
                                            NSError **err)
{
    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    MTLRenderPipelineColorAttachmentDescriptor *ca;
    id<MTLRenderPipelineState> pso;

    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    ca = pd.colorAttachments[0];
    ca.pixelFormat = MTLPixelFormatBGRA8Unorm;
    ca.blendingEnabled = key[0] ? YES : NO;
    if (key[0]) {
        ca.sourceRGBBlendFactor = blend_factor(key[1], 0);
        ca.destinationRGBBlendFactor = blend_factor(key[2], 0);
        ca.sourceAlphaBlendFactor = blend_factor(key[1], 1);
        ca.destinationAlphaBlendFactor = blend_factor(key[2], 1);
        ca.rgbBlendOperation = (MTLBlendOperation)key[4];
        ca.alphaBlendOperation = (MTLBlendOperation)key[4];
    }
    ca.writeMask = ((key[3] & 0x00FF0000u) ? 0 : MTLColorWriteMaskRed)
                 | ((key[3] & 0x0000FF00u) ? 0 : MTLColorWriteMaskGreen)
                 | ((key[3] & 0x000000FFu) ? 0 : MTLColorWriteMaskBlue)
                 | ((key[3] & 0xFF000000u) ? 0 : MTLColorWriteMaskAlpha);
    if (key[5]) {
        pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    }
    pso = [dev newRenderPipelineStateWithDescriptor:pd error:err];
    [pd release];
    return pso;
}

static id<MTLRenderPipelineState> pipeline(id<MTLFunction> vs, id<MTLFunction> fs)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    id<MTLRenderPipelineState> pso;
    NSError *err = nil;
    PsoKey pk;
    uint32_t *key = pk.key;
    int i;

    memset(&pk, 0, sizeof pk);
    pk.vs = (uint64_t)(uintptr_t)vs;
    pk.fs = (uint64_t)(uintptr_t)fs;
    key[0] = g->blend_enable ? 1 : 0;
    key[1] = key[0] ? g->blend_sfactor : 0;
    key[2] = key[0] ? g->blend_dfactor : 0;
    key[3] = g->color_keep;             /* SET_COLOR_MASK, inverted */
    key[4] = key[0] ? (uint32_t)blend_op(g->blend_equation) : 0;
    key[5] = s_enc_dt ? 1 : 0;          /* the open pass has a depth buffer */
    if (s_pso_last < s_pso_n && !memcmp(&s_pso[s_pso_last].k, &pk, sizeof pk))
        return s_pso[s_pso_last].pso;
    for (i = 0; i < s_pso_n; i++)
        if (!memcmp(&s_pso[i].k, &pk, sizeof pk)) {
            s_pso_last = i;
            return s_pso[i].pso;
        }

    {
        uint64_t t0 = prof_now(), ns;
        pso = pso_build(s_dev, vs, fs, key, &err);
        ns = prof_now() - t0;
        s_sh_flip.pso_ns += ns;
        s_sh_flip.pso++;
        if (s_pt_on > 0)
            s_ptf[s_presents & 1].nd.pso++;   /* metal_nodiscard's line */
        if (sh_trace())
            fprintf(stderr, LOGP "shader: pipeline %.2f ms (blend %u, depth %u), flip %llu\n",
                    ns / 1e6, key[0], key[5], (unsigned long long)s_presents);
    }
    if (!pso) {
        static int said;
        if (!said++)
            fprintf(stderr, LOGP "pipeline failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
        return nil;
    }
    pl_note_pso(vs, fs, key);
    if (s_pso_n < PSO_MAX) {
        i = s_pso_n++;
    } else {
        static unsigned next;
        i = (int)(next++ % PSO_MAX);
        [s_pso[i].pso release];   /* an encoded draw keeps its own reference */
    }
    s_pso[i].k = pk;
    s_pso[i].pso = pso;
    s_pso_last = i;
    return pso;
}

/* ---- textures ------------------------------------------------
 *
 * Decoded to A8R8G8B8 on the CPU by the shared decode (nv2a_tex_decode), the
 * same texels the CPU sampler and the D3D11 backend see, and uploaded as
 * BGRA8Unorm (the same bytes). Cache key and invalidation as the D3D11
 * backend: offset + format + size + pitch + palette + levels; rehashed once
 * per flip and re-uploaded if it changed. A changed texture gets a new
 * MTLTexture rather than a replaceRegion, so a draw already encoded this frame
 * keeps the texels it was encoded with. */

typedef struct nv2a_stage Stage;
static Stage s_stage[4];

#define TEX_MAX_TEXELS (2048u * 2048u)
#define TEX_CACHE 512   /* array bound; entries in use: tex_cache_n() */
typedef struct {
    uint32_t       offset, color, width, height, pitch, palette, levels;
    int            cube;        /* six faces in a cube texture (nv2a_stage_cube_bind) */
    uint32_t       bytes;       /* nv2a_tex_extent, for tex_cache_stale */
    uint32_t       gen, used;
    uint32_t       pack_gen;    /* nv2a_tex_pack_gen() it was uploaded under */
    /* What the sampler follows (nv2a_stage_lod): the entry's scale over the
     * guest texture, and its chain. 1 and the guest's levels for guest texels. */
    uint32_t       pack_scale, pack_levels;
    uint64_t       hash;
    int            live;
    id<MTLTexture> tex;
} TexEntry;

static TexEntry       s_tc[TEX_CACHE];

/* Entries in use: RECOMP_DEBUG=tex_cache=n, 1..TEX_CACHE. Default 256, this
 * backend's value since its first frames (D3D11 uses 512); neither has been
 * measured against the other. */
static uint32_t tex_cache_n(void)
{
    static long n = -1;
    if (n < 0) {
        n = recomp_env_int(RENV_TEX_CACHE, 256);
        if (n < 1) n = 1;
        if (n > TEX_CACHE) n = TEX_CACHE;
    }
    return (uint32_t)n;
}
static uint32_t       s_tc_clock;

/* A write-back just rewrote guest [addr, addr + bytes): the cache entries over
 * it rehash at their next bind even inside this flip. tex_bind hashes a
 * texture once a flip, so without this a target drawn, sampled, redrawn and
 * sampled again in one flip (Burnout 3's speed blur redraws the 16-bit bloom
 * target and composites it) kept the first decode, and the blur showed the
 * dark bright-pass instead of the scene. */
static void tex_cache_stale(uint32_t addr, uint32_t bytes)
{
    uint32_t n = tex_cache_n(), lo = addr & 0x07FFFFFFu, hi = lo + bytes, i;
    for (i = 0; i < n; i++) {
        TexEntry *c = &s_tc[i];
        uint32_t clo = c->offset & 0x07FFFFFFu;
        if (c->live && clo < hi && lo < clo + c->bytes)
            c->gen = nv2a_pb_gpu_state()->flips - 1u;
    }
}

static uint32_t      *s_texels;
static size_t         s_texels_cap;
static id<MTLTexture> s_black, s_magenta, s_black_cube;

static id<MTLTexture> solid_texture(uint32_t argb)
{
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:1 height:1
                                 mipmapped:NO];
    id<MTLTexture> t = [s_dev newTextureWithDescriptor:td];
    [t replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&argb
         bytesPerRow:4];
    return t;
}

/* A 1x1 cube map of argb: the fallback of a stage bound as a cube. */
static id<MTLTexture> solid_cube(uint32_t argb)
{
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        textureCubeDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm size:1 mipmapped:NO];
    id<MTLTexture> t = [s_dev newTextureWithDescriptor:td];
    uint32_t f;
    for (f = 0; f < 6; f++)
        [t replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 slice:f withBytes:&argb
             bytesPerRow:4 bytesPerImage:4];
    return t;
}

static int no_mips(void)
{
    static int off = -1;
    if (off < 0)
        off = recomp_env(RENV_METAL_NO_MIPS) != NULL;
    return off;
}

/* Per-batch CPU memos (RECOMP_DEBUG=metal_memo=0 turns them off for A/B),
 * as D3D11's d3d11_memo: each one gives what the full path would. */
static int memo_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = recomp_env(RENV_METAL_MEMO);
        on = !(e && *e == '0');
    }
    return on;
}

static void build_stages(void)
{
    extern uint32_t xbox_ContiguousAllocatedBytes(void);
    const uint32_t *regs = nv2a_pb_tex_regs();
    const uint8_t  *sets = nv2a_pb_tex_regs_set();
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    /* The stages are a function of the texture registers, the shader
     * program and the DMA resolve, whose answer moves only with the
     * contiguous arena's high-water mark: unchanged, the last decode
     * stands (D3D11's build_stages). */
    static struct { uint32_t gen, prog, contig; int set, ok; } memo;
    uint32_t gen = nv2a_pb_tex_regs_gen(), contig = xbox_ContiguousAllocatedBytes();
    int n;

    if (memo_on() && memo.ok && memo.gen == gen && memo.prog == v->shader_prog
        && memo.set == v->shader_set && memo.contig == contig)
        return;
    memo.gen = gen;
    memo.prog = v->shader_prog;
    memo.set = v->shader_set;
    memo.contig = contig;
    memo.ok = 1;
    for (n = 0; n < 4; n++) {
        Stage *t = &s_stage[n];
        nv2a_stage_decode(n, &regs[(0x40u * n) / 4], &sets[(0x40u * n) / 4],
                          v->shader_set, v->shader_prog, nv2a_pb_dma_resolve,
                          NV2A_STAGE_P8_PALETTE | NV2A_STAGE_PITCH_LINEAR
                          | (no_mips() ? 0u : NV2A_STAGE_MIPS), t);
        if (t->mode == 5)
            t->mode = 0;    /* CLIP_PLANE: no texture; the kill is clip_mode's */
    }
}

/* nv2a_stage.cube_rt for this draw's stages, against the live targets: a
 * cube whose faces lie under one is read as 2D (nv2a_stage_cube_bind), as
 * the D3D11 backend does, so the key, the shader and the bind agree. Set
 * every draw: the targets move without the stage registers. */
static void stages_cube_rt(void)
{
    int n, i;
    for (n = 0; n < 4; n++) {
        Stage *t = &s_stage[n];
        t->cube_rt = 0;
        for (i = 0; t->cube && i < RT_MAX && !t->cube_rt; i++)
            /* Only a live target: one the title has since overwritten
             * (a level's cube loaded over a menu's leftover target) is
             * dropped here, as rt_texture drops one at its own address,
             * or the cube would read 2D until the target was evicted. */
            t->cube_rt = s_rt[i].tex
                         && nv2a_stage_cube_over_rt(t, s_rt[i].addr, s_rt[i].pitch * s_rt[i].h)
                         && !rt_drop_if_stale(&s_rt[i], "cube");
    }
}

static uint64_t tex_hash_pal(const uint8_t *mem, const Stage *t, uint32_t bytes)
{
    return nv2a_stage_hash(mem, t, bytes);
}

/* A texture of `levels` levels from src (level 0 first), or nil. `bc`:
 * XRTP_BC_*, src is a pack entry's blocks (XRTP_FLAG_BC1, XRTP_FLAG_BC3),
 * uploaded as they are with a block-row pitch; otherwise A8R8G8B8 texels
 * into BGRA8. */
static id<MTLTexture> tex_make(uint32_t width, uint32_t height, uint32_t levels,
                               const uint32_t *src, int bc)
{
    MTLTextureDescriptor *td;
    id<MTLTexture> tex;
    const uint8_t *p = (const uint8_t *)src;
    uint32_t l;
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:bc == XRTP_BC_3 ? MTLPixelFormatBC3_RGBA
                                                                  : bc == XRTP_BC_1 ? MTLPixelFormatBC1_RGBA
                                                                  : MTLPixelFormatBGRA8Unorm
                                                            width:width
                                                           height:height
                                                        mipmapped:levels > 1];
    if (levels > 1)
        td.mipmapLevelCount = levels;
    td.usage = MTLTextureUsageShaderRead;
    tex = [s_dev newTextureWithDescriptor:td];
    if (!tex)
        return nil;
    for (l = 0; l < levels; l++) {
        uint32_t w = nv2a_tex_level_dim(width, l), h = nv2a_tex_level_dim(height, l);
        uint32_t row = bc ? (w + 3u) / 4u * (bc == XRTP_BC_1 ? 8u : 16u) : w * 4u;
        uint32_t rows = bc ? (h + 3u) / 4u : h;
        [tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:l withBytes:p
               bytesPerRow:row];
        p += (size_t)row * rows;
    }
    return tex;
}

/* A cube map of `size` square faces from src: six faces of A8R8G8B8 texels,
 * each with its `levels` levels, in the NV2A's face order (+X -X +Y -Y +Z
 * -Z, Metal's slices 0-5). */
static id<MTLTexture> tex_make_cube(uint32_t size, uint32_t levels, const uint32_t *src)
{
    MTLTextureDescriptor *td;
    id<MTLTexture> tex;
    const uint8_t *p = (const uint8_t *)src;
    uint32_t f, l;
    td = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                               size:size
                                                          mipmapped:levels > 1];
    if (levels > 1)
        td.mipmapLevelCount = levels;
    td.usage = MTLTextureUsageShaderRead;
    tex = [s_dev newTextureWithDescriptor:td];
    if (!tex)
        return nil;
    for (f = 0; f < 6; f++)
        for (l = 0; l < levels; l++) {
            uint32_t w = nv2a_tex_level_dim(size, l);
            [tex replaceRegion:MTLRegionMake2D(0, 0, w, w) mipmapLevel:l slice:f withBytes:p
                   bytesPerRow:w * 4u bytesPerImage:w * w * 4u];
            p += (size_t)w * w * 4u;
        }
    return tex;
}

/* `hash` is the entry's texel hash: with a texture pack loaded
 * (enhancements, textures.pack) an entry it holds is uploaded from the
 * pack's texels at the pack's size instead of the guest's. If the device
 * refuses a texture of the entry's size, the guest texels go up instead:
 * the pack only ever adds detail, never a black texture. */
static id<MTLTexture> tex_upload_body(const Stage *t, uint64_t hash, int cube);
/* What the last tex_upload left (TexEntry.pack_scale, .pack_levels). */
static uint32_t s_up_scale, s_up_levels;

static id<MTLTexture> tex_upload(const Stage *t, uint64_t hash, int cube)
{
    uint64_t t0 = prof_now();
    id<MTLTexture> tex = tex_upload_body(t, hash, cube);
    s_sh_flip.tex_ns += prof_now() - t0;
    s_sh_flip.tex++;
    return tex;
}

/* The guest texels of t into s_texels, or 0 when it cannot grow: `faces`
 * of them (6 for a cube map), each nv2a_stage_face_stride on from the
 * last, packed one after another. */
static int tex_decode_guest(const Stage *t, uint32_t faces)
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    size_t n = nv2a_tex_texels(t->width, t->height, t->levels);
    const uint32_t *pal = NULL;
    uint32_t f, stride = faces > 1 ? nv2a_stage_face_stride(t) : 0;
    if (n * faces > s_texels_cap) {
        uint32_t *p = (uint32_t *)realloc(s_texels, n * faces * 4);
        if (!p)
            return 0;
        s_texels = p;
        s_texels_cap = n * faces;
    }
    if (t->color == NV2A_TEX_P8)
        pal = (const uint32_t *)(mem + t->palette);
    for (f = 0; f < faces; f++)
        nv2a_tex_decode(mem + t->offset + f * stride, t->color, t->width, t->height, t->pitch,
                        t->levels, pal, pal ? t->pal_len : 0, s_texels + n * f);
    return 1;
}

static id<MTLTexture> tex_upload_body(const Stage *t, uint64_t hash, int cube)
{
    id<MTLTexture> tex = nil;

    s_up_scale = 1;
    s_up_levels = t->levels;
    if (cube) {
        /* No pack entry is a cube map: the six faces go up from the guest.
         * tex_dump writes face 0, the first texels decoded. */
        if (!tex_decode_guest(t, 6))
            return nil;
        if (nv2a_tex_dump_on())
            nv2a_tex_dump_upload(t, hash, s_texels);
        if (!(tex = tex_make_cube(t->width, t->levels, s_texels)))
            return nil;
        s_tex_uploads++;
        nv2a_stats_add(NV2A_STAT_TEX_UPLOADS, 1);
        return tex;
    }
    if (nv2a_tex_pack_count()) {
        struct nv2a_tex_pack_entry *pe = nv2a_tex_pack_find(hash, t->width, t->height);
        const uint32_t *src = NULL;
        if (pe && pe->bc && !s_bc) {
            static int said;
            if (!said++)
                fprintf(stderr, LOGP "textures.pack: this GPU has no BC textures; BC1 and BC3"
                                     " entries go up from the guest\n");
        } else if (pe) {
            src = nv2a_tex_pack_texels(pe);
        }
        if (src)
            tex = tex_make(pe->width, pe->height, pe->levels, src, pe->bc);
        if (tex) {
            s_up_scale = nv2a_stage_pack_scale(t->width, pe->width);
            s_up_levels = pe->levels;
            nv2a_tex_pack_release(pe);
        }
        /* A refused entry is a miss too, tagged so in the trace. */
        if (!tex)
            nv2a_tex_pack_miss(hash, t->width, t->height, t->color, t->levels);
        /* RECOMP_DEBUG=tex_dump writes the guest's texels, never the
         * pack's: under a hit they are decoded for it alone, once. */
        else if (nv2a_tex_dump_on())
            nv2a_tex_dump_upload(t, hash, nv2a_tex_dump_wants(t, hash)
                                          && tex_decode_guest(t, 1) ? s_texels : NULL);
    }
    if (!tex) {
        if (!tex_decode_guest(t, 1))
            return nil;
        if (nv2a_tex_dump_on())
            nv2a_tex_dump_upload(t, hash, s_texels);
        tex = tex_make(t->width, t->height, t->levels, s_texels, 0);
        if (!tex)
            return nil;
    }
    s_tex_uploads++;
    nv2a_stats_add(NV2A_STAT_TEX_UPLOADS, 1);
    return tex;
}

/* The entry tex_bind's last texture came from (NULL for magenta or black). */
static TexEntry *s_tex_last;

/* `cube`: t goes up as a cube map (nv2a_stage_cube_bind), its six faces
 * hashed and decoded together; tex_bind's fallbacks are 2D, so such a
 * stage gets s_black_cube for them. */
static id<MTLTexture> tex_bind_body(const Stage *t, int cube);

static id<MTLTexture> tex_bind(const Stage *t, int cube)
{
    id<MTLTexture> tex = tex_bind_body(t, cube);
    return cube && tex.textureType != MTLTextureTypeCube ? s_black_cube : tex;
}

static id<MTLTexture> tex_bind_body(const Stage *t, int cube)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint32_t bytes = 0, i;
    TexEntry *e = NULL, *victim = &s_tc[0];

    s_tex_last = NULL;
    if (t->valid && (uint64_t)t->width * t->height <= TEX_MAX_TEXELS)
        bytes = cube ? nv2a_stage_cube_extent(t)
                     : nv2a_tex_extent(t->color, t->width, t->height, t->pitch, t->levels);
    /* The guest bytes are about to be hashed and decoded: a render target
     * over them that rt_texture did not bind (another format, no_rtt, an
     * evicted name) is written back first, so the decode sees what was
     * drawn there. */
    if (bytes) {
        uint32_t lo = t->offset & 0x07FFFFFFu, hi = lo + bytes;
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *rt = &s_rt[i];
            uint32_t rlo = rt->addr & 0x07FFFFFFu;
            if (rt->tex && rt->dirty && rt->pitch
                    && lo < rlo + rt->pitch * rt->h && rlo < hi) {
                if (rt_drop_if_stale(rt, "sync"))
                    continue;
                /* A linear read at a narrower pitch than the target's
                 * restacks the target's rows, no picture of it on the
                 * console either: decode the guest bytes as they stand, as
                 * D3D11 does. KUF's front end reads its XDK supersample
                 * backbuffer through a half-pitch header every flip, and
                 * the write-back was a 4.9 MB readback per flip at scale 1. */
                if (nv2a_rt_pitch_alias(t->color, t->pitch, t->height, rt->fmt, rt->pitch)) {
                    s_pitch_alias_binds++;
                    if (recomp_env(RENV_METAL_TRACE)
                            && nv2a_pair_new(&s_pitch_alias_seen, rt->fmt, rt->pitch, t->color,
                                             t->pitch))
                        fprintf(stderr, LOGP "pitch alias: target 0x%08X fmt 0x%X pitch %u,"
                                " texture 0x%08X fmt 0x%02X %ux%u pitch %u decoded without"
                                " a write-back, flip %llu\n",
                                rt->offset, rt->fmt, rt->pitch, t->offset, t->color, t->width,
                                t->height, t->pitch, (unsigned long long)s_presents);
                    continue;
                }
                s_decode_syncs++;
                if (recomp_env(RENV_METAL_TRACE))
                    fprintf(stderr, LOGP "decode sync: target 0x%08X %ux%u (%u bytes) for"
                            " texture 0x%08X fmt 0x%02X %ux%u (%u bytes), flip %llu\n",
                            rt->offset, rt->w, rt->h, rt->pitch * rt->h, t->offset,
                            t->color, t->width, t->height, bytes,
                            (unsigned long long)s_presents);
                rt_sync(rt, rt->pitch);
            }
        }
    }
    if (!bytes) {
        static uint8_t said[256];
        if ((uint64_t)t->width * t->height > TEX_MAX_TEXELS)
            return s_black;
        if (!said[t->color & 0xFF]) {
            said[t->color & 0xFF] = 1;
            fprintf(stderr, LOGP "\x1b[1;35mtexture format 0x%02X (%ux%u at"
                    " 0x%08X) not decoded: drawn magenta\x1b[0m\n",
                    t->color, t->width, t->height, t->offset);
        }
        return s_magenta;
    }
    uint32_t n = tex_cache_n();
    for (i = 0; i < n; i++) {
        TexEntry *c = &s_tc[i];
        if (c->live && c->offset == t->offset && c->color == t->color
                && c->width == t->width && c->height == t->height
                && c->pitch == t->pitch && c->palette == t->palette
                && c->levels == t->levels && c->cube == cube) {
            e = c;
            break;
        }
        if (!c->live || (victim->live && c->used < victim->used))
            victim = c;
    }
    if (e) {
        /* A pack toggle (nv2a_tex_pack_gen) changes what the same texels
         * upload as: re-upload as if the hash had moved. */
        uint32_t pg = nv2a_tex_pack_gen();
        if (e->gen != g->flips || e->pack_gen != pg) {
            uint64_t h = tex_hash_pal(mem, t, bytes);
            s_tex_rehash++;
            e->gen = g->flips;
            if (h != e->hash || e->pack_gen != pg) {
                id<MTLTexture> nt = tex_upload(t, h, cube);
                s_tex_changed++;
                /* The hash moves only with the texels: a failed upload keeps
                 * the old texture and is tried again next flip. */
                if (!nt)
                    return s_black;
                e->hash = h;
                e->pack_gen = pg;
                e->pack_scale = s_up_scale;
                e->pack_levels = s_up_levels;
                [e->tex release];
                e->tex = nt;
            }
        }
    } else {
        id<MTLTexture> nt;
        uint64_t h = tex_hash_pal(mem, t, bytes);
        e = victim;
        [e->tex release];
        e->tex = nil;
        e->live = 0;
        nt = tex_upload(t, h, cube);
        if (!nt)
            return s_black;
        e->offset = t->offset;
        e->color = t->color;
        e->width = t->width;
        e->height = t->height;
        e->pitch = t->pitch;
        e->palette = t->palette;
        e->levels = t->levels;
        e->cube = cube;
        e->bytes = bytes;
        e->gen = g->flips;
        e->hash = h;
        e->pack_gen = nv2a_tex_pack_gen();
        e->pack_scale = s_up_scale;
        e->pack_levels = s_up_levels;
        e->tex = nt;
        e->live = 1;
    }
    e->used = ++s_tc_clock;
    s_tex_last = e;
    return e->tex;
}

/* NV2A wrap modes: 1 wrap, 2 mirror, 3 clamp to edge, 4 border, 5 clamp.
 * Border takes the stage's SET_TEXTURE_BORDER_COLOR (nv2a_stage_decode),
 * but a Metal sampler offers only three border colours: border_colour picks
 * the one the register names, and any other colour stays a clamp to edge
 * (logged once). Burnout 3's is 0x00000000, transparent black. */
static MTLSamplerAddressMode address_mode(uint32_t m, int border)
{
    switch (m) {
    case 1:  return MTLSamplerAddressModeRepeat;
    case 2:  return MTLSamplerAddressModeMirrorRepeat;
    case NV2A_ADDR_BORDER:
        if (border)
            return MTLSamplerAddressModeClampToBorderColor;
        return MTLSamplerAddressModeClampToEdge;
    default: return MTLSamplerAddressModeClampToEdge;
    }
}

/* MTLSamplerBorderColor + 1 for an A8R8G8B8 border Metal can give, else
 * 0. */
static int border_colour(uint32_t argb)
{
    static int said;
    switch (argb) {
    case 0x00000000u: return 1 + MTLSamplerBorderColorTransparentBlack;
    case 0xFF000000u: return 1 + MTLSamplerBorderColorOpaqueBlack;
    case 0xFFFFFFFFu: return 1 + MTLSamplerBorderColorOpaqueWhite;
    }
    if (!said) {
        said = 1;
        fprintf(stderr, LOGP "texture border colour 0x%08X has no Metal sampler"
                " colour: clamping to the edge instead\n", argb);
    }
    return 0;
}

/* SET_TEXTURE_FILTER as the D3D11 backend reads it (filter_linear,
 * filter_mip there). RECOMP_METAL_POINT=1 forces nearest texels. */
#define filter_linear nv2a_tex_filter_linear   /* nv2a_backend_common.h */
#define filter_mip    nv2a_tex_filter_mip
#define lod_bias      nv2a_tex_lod_bias

/* What each stage's bind took from its cache entry, by value: a later
 * stage's bind in the same draw can evict the entry at a small tex_cache.
 * The sampler's scale and chain, and the texel hash (0: no cache entry). */
static uint32_t s_stage_pack[4][2];
static uint64_t s_stage_hash[4];

static void stage_from_cache(int k)
{
    if (s_tex_last) {
        s_stage_pack[k][0] = s_tex_last->pack_scale;
        s_stage_pack[k][1] = s_tex_last->pack_levels;
        s_stage_hash[k] = s_tex_last->hash;
    }
}

/* ---- The trace's buffer: a frame's slots and what each draw id had bound.
 * Several frames, since each stays with the GPU until its command buffer is
 * done; the walker polls them at the flip and folds the ones that are, and
 * waits only when the next one is still out. */
#define MAG_FRAMES 4u
static struct {
    id<MTLBuffer>            buf;
    /* The newest submission with a draw that writes it (mag_committed), 0
     * none; at the flip the frame is out until that one is done. */
    uint64_t                 sub;
    struct nv2a_tex_mag_draw *rec;
    unsigned                 n;
} s_mag[MAG_FRAMES];
static unsigned s_mag_cur;
static int      s_mag_state;   /* 0 not tried, 1 ok, -1 failed */
static int      s_mag_bound;   /* the open s_cmd has a draw that writes s_mag[s_mag_cur] */

static void mag_committed(uint64_t n)
{
    if (s_mag_bound) {
        s_mag[s_mag_cur].sub = n;
        s_mag_bound = 0;
    }
}

static int mag_init(void)
{
    unsigned k;
    if (s_mag_state)
        return s_mag_state > 0;
    s_mag_state = 1;
    for (k = 0; k < MAG_FRAMES; k++) {
        s_mag[k].buf = [s_dev newBufferWithLength:NV2A_TEX_MAG_BYTES
                                          options:MTLResourceStorageModeShared];
        s_mag[k].rec = (struct nv2a_tex_mag_draw *)malloc(NV2A_TEX_MAG_DRAWS
                                                          * sizeof *s_mag[k].rec);
        if (!s_mag[k].buf || !s_mag[k].rec)
            s_mag_state = -1;
        else
            memset([s_mag[k].buf contents], 0, NV2A_TEX_MAG_BYTES);
    }
    if (s_mag_state < 0)
        fprintf(stderr, LOGP "tex_mag: no device buffer; the trace reads nothing\n");
    return s_mag_state > 0;
}

/* The next draw's id, and where in the frame's buffer it writes: no id in
 * the shader, the slots start at the buffer offset. */
static void mag_draw(void)
{
    uint64_t hash[NV2A_TEX_MAG_STAGES];
    uint32_t scale[NV2A_TEX_MAG_STAGES];
    unsigned k, id;
    for (k = 0; k < NV2A_TEX_MAG_STAGES; k++) {
        hash[k] = s_stage_hash[k];
        scale[k] = s_stage_pack[k][0];
    }
    id = nv2a_tex_mag_draw_id(s_mag[s_mag_cur].rec, &s_mag[s_mag_cur].n, hash, scale);
    [s_enc setFragmentBuffer:s_mag[s_mag_cur].buf
                      offset:(NSUInteger)id * NV2A_TEX_MAG_STAGES * 4u
                     atIndex:2];
    s_mag_bound = 1;
}

/* A pre-transformed batch's figure: positions in surface pixels (slot 0)
 * and each stage's texcoord (slot 9 + stage), as the vertex ring holds them. */
static void mag_pretransformed(const float *vf, unsigned nf4, const int *slot,
                               const uint32_t *idx, uint32_t nidx)
{
    unsigned rs = nv2a_host_render_scale(), k, a;
    for (k = 0; k < NV2A_TEX_MAG_STAGES; k++) {
        const Stage *t = &s_stage[k];
        int norm = nv2a_tex_size_from_format(t->color), kp = -1, kt = -1;
        uint32_t i;
        for (a = 0; a < nf4; a++) {
            if (slot[a] == 0)
                kp = (int)a;
            if (slot[a] == 9 + (int)k)
                kt = (int)a;
        }
        if (kp < 0 || kt < 0)
            continue;
        /* A few triangles of each: a 2D batch's pieces share their scale. */
        for (i = 0; s_stage_hash[k] && i + 3 <= nidx && i < 3u * 64u; i += 3) {
            float xy[3][2], uv[3][2];
            int c;
            for (c = 0; c < 3; c++) {
                const float *o = vf + (size_t)idx[i + (uint32_t)c] * nf4 * 4u;
                xy[c][0] = o[(size_t)kp * 4u];
                xy[c][1] = o[(size_t)kp * 4u + 1];
                uv[c][0] = o[(size_t)kt * 4u];
                uv[c][1] = o[(size_t)kt * 4u + 1];
                if (norm) {
                    uv[c][0] *= (float)t->width;
                    uv[c][1] *= (float)t->height;
                }
            }
            nv2a_tex_mag_note(s_stage_hash[k], nv2a_tex_mag_triangle((const float (*)[2])xy,
                                                                     (const float (*)[2])uv) * rs,
                              rs);
        }
    }
}

/* A frame goes with the newest submission that drew into it (its draws are
 * in that one or earlier ones, all on s_queue); each frame whose submissions
 * are done is folded, and the next one starts empty. */
static void mag_fold_frame(unsigned k)
{
    nv2a_tex_mag_fold(s_mag[k].rec, s_mag[k].n, (const uint32_t *)[s_mag[k].buf contents],
                      nv2a_host_render_scale());
    memset([s_mag[k].buf contents], 0, NV2A_TEX_MAG_BYTES);
    s_mag[k].n = 0;
    s_mag[k].sub = 0;
}

/* At the flip, once its command buffer is committed (s_mag_bound is 0). */
static void mag_flip(void)
{
    unsigned k, next = (s_mag_cur + 1u) % MAG_FRAMES;
    if (s_mag_state <= 0)
        return;
    for (k = 0; k < MAG_FRAMES; k++)
        if (s_mag[k].sub && sub_done_upto(s_mag[k].sub))
            mag_fold_frame(k);
    if (s_mag[next].sub) {
        sub_wait_upto(s_mag[next].sub, SUBW_MAG);   /* the CPU reads, then zeroes it */
        mag_fold_frame(next);
    }
    /* An empty frame stays current; one no submission drew into is too. */
    if (s_mag[s_mag_cur].sub || !s_mag[s_mag_cur].n)
        s_mag_cur = next;
}

#define SMP_MAX 128
static struct { uint64_t key; id<MTLSamplerState> s; } s_smp[SMP_MAX];
static int s_smp_n;

/* scale, entry_levels: the bound cache entry's (TexEntry.pack_scale,
 * .pack_levels), 1 and 1 for a render target or a black or magenta one: a
 * pack entry moves the clamps (nv2a_stage_lod). */
static id<MTLSamplerState> sampler(const Stage *t, int bump_source, uint32_t scale,
                                   uint32_t entry_levels)
{
    static int point = -1;
    MTLSamplerDescriptor *sd;
    id<MTLSamplerState> st;
    uint32_t u = t->addr_u & 0xF, v = t->addr_v & 0xF;
    int mn, mg, mip, guest_mip, i;
    unsigned an;
    uint64_t key;
    int bc = 0;
    struct nv2a_stage_lod lod;

    if (point < 0)
        point = recomp_env(RENV_METAL_POINT) != NULL;
    if (u == NV2A_ADDR_BORDER || v == NV2A_ADDR_BORDER)
        bc = border_colour(t->border);
    mn = !point && filter_linear(t->filter, 0);
    mg = !point && filter_linear(t->filter, 1);
    guest_mip = t->levels > 1 ? filter_mip(t->filter) : 0;
    lod = nv2a_stage_lod(guest_mip, t->lod_min, t->lod_max, scale, entry_levels);
    mip = lod.mip;
    /* render.aniso, on the D3D11 sampler's rule (nv2a_stage_aniso) and key
     * bit: a bump source in one draw and plain in the next gets its own. */
    an = nv2a_stage_aniso(t, guest_mip, mn, bump_source);
    key = u | v << 4 | (uint64_t)mn << 8 | (uint64_t)mg << 9 | (uint64_t)mip << 10
        | (uint64_t)an << 35 | (uint64_t)bc << 40 | lod.key;
    if (guest_mip)
        key |= (uint64_t)t->lod_min << 25 | (uint64_t)t->lod_max << 29;
    for (i = 0; i < s_smp_n; i++)
        if (s_smp[i].key == key)
            return s_smp[i].s;
    sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = mn ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.magFilter = mg ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.mipFilter = mip == 2 ? MTLSamplerMipFilterLinear
                 : mip == 1 ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterNotMipmapped;
    sd.sAddressMode = address_mode(u, bc);
    sd.tAddressMode = address_mode(v, bc);
    sd.rAddressMode = MTLSamplerAddressModeClampToEdge;
    if (bc)
        sd.borderColor = (MTLSamplerBorderColor)(bc - 1);
    sd.lodMinClamp = lod.min;
    sd.lodMaxClamp = lod.max;
    if (an) {
        /* D3D11_FILTER_ANISOTROPIC filters min, mag and mip linearly; set
         * the same here so both backends apply one filter. */
        sd.minFilter = MTLSamplerMinMagFilterLinear;
        sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.mipFilter = MTLSamplerMipFilterLinear;
        sd.maxAnisotropy = an;
    }
    st = [s_dev newSamplerStateWithDescriptor:sd];
    [sd release];
    if (!st)
        return s_smp_n ? s_smp[0].s : nil;
    if (s_smp_n < SMP_MAX) {
        i = s_smp_n++;
    } else {
        static unsigned next;
        i = (int)(next++ % SMP_MAX);
        [s_smp[i].s release];
    }
    s_smp[i].key = key;
    s_smp[i].s = st;
    return st;
}

/* ---- shaders (M2): vertex programs and register combiners ------------------
 *
 * As the D3D11 backend's vertex_shader and combiner_ps: the program is
 * parsed from the slots it starts at (d3d8_vsh_parse) and emitted as MSL
 * (d3d8_vsh_generate_msl, screen-space epilogue), keyed by
 * nv2a_vsh_program_hash; the combiner registers become NV2ACombinerState
 * (d3d8_combiners_from_regs) and MSL (d3d8_combiners_generate_msl), keyed by
 * the state's bytes. Each compiles once, with the shared options (metal_compile.h). A
 * source that fails to compile is logged with its text, counted
 * (s_compile_errors, in the present log) and remembered as bad. */

static uint64_t s_compile_errors, s_vs_compiled, s_fs_compiled;

/* metal_shader_nonce and metal_shader_dump, read once on the ack thread
 * (compile_fn, and preload_start before the preload is queued, so the
 * preload's thread only reads them). */
static const char *s_sh_nonce, *s_sh_dump;

static void sh_env(void)
{
    static int read;
    if (!read) {
        read = 1;
        s_sh_nonce = recomp_env(RENV_METAL_SHADER_NONCE);
        s_sh_dump = recomp_env(RENV_METAL_SHADER_DUMP);
    }
}

/* A new entry name is new AIR: both the source and the function caches
 * of macOS miss, as on a player's first launch. compile_fn and the preload
 * both rename here, so under a nonce the preload builds what the game will.
 * Returns the text to compile (autoreleased) and sets *name. */
static NSString *sh_nonce_rename(NSString *text, NSString **name)
{
    if (s_sh_nonce && *s_sh_nonce) {
        NSString *nn = [NSString stringWithFormat:@"%@_%s_%d", *name, s_sh_nonce,
                                                  (int)getpid()];
        text = [text stringByReplacingOccurrencesOfString:
                     [NSString stringWithFormat:@" %@(", *name]
                 withString:[NSString stringWithFormat:@" %@(", nn]];
        *name = nn;
    }
    return text;
}

/* fast: RECOMP_DEBUG=metal_fs_fastmath's options (metal_compile_library_fs_fast),
 * for a combiners' fragment library only; compile_fn is safe. */
static id<MTLFunction> compile_fn_opts(const char *src, NSString *name, const char *what,
                                       int fast)
{
    const char *dump;
    NSError *err = nil;
    NSString *text = [NSString stringWithUTF8String:src];
    NSString *orig = name;
    id<MTLLibrary> lib;
    id<MTLFunction> fn;
    uint64_t t0, ns;

    sh_env();
    dump = s_sh_dump;
    if (dump) {
        static int dump_said;
        char path[1024];
        FILE *f = NULL;
        int n = snprintf(path, sizeof path, "%s%s.metal", dump, what);
        if (n > 0 && (size_t)n < sizeof path) {
            for (char *c = path + strlen(dump); *c; c++)
                if (*c == ' ')
                    *c = '_';
            f = fopen(path, "w");
        }
        if (f) {
            fputs(src, f);
            fclose(f);
        } else if (!dump_said++) {
            fprintf(stderr, LOGP "metal_shader_dump: cannot write %s (missing directory or"
                    " path too long); no more dump warnings\n", n > 0 && (size_t)n < sizeof path
                    ? path : dump);
        }
    }
    text = sh_nonce_rename(text, &name);
    t0 = prof_now();
    lib = fast ? metal_compile_library_fs_fast(s_dev, text, what, &err)
               : metal_compile_library(s_dev, text, what, &err);
    ns = prof_now() - t0;
    s_sh_flip.lib_ns += ns;
    s_sh_flip.lib++;
    if (sh_trace())
        fprintf(stderr, LOGP "shader: %s library %.2f ms (%zu bytes), flip %llu\n",
                what, ns / 1e6, strlen(src), (unsigned long long)s_presents);
    if (!lib) {
        s_compile_errors++;
        if (s_compile_errors <= 4)
            fprintf(stderr, LOGP "\x1b[1;31m%s: MSL compile failed: %s\x1b[0m\n%s\n",
                    what, err ? [[err localizedDescription] UTF8String] : "?", src);
        return nil;
    }
    fn = [lib newFunctionWithName:name];
    [lib release];
    if (!fn) {
        s_compile_errors++;
        fprintf(stderr, LOGP "%s: no function %s\n", what, [name UTF8String]);
    } else {
        pl_note_library(fn, what, [orig UTF8String], src);   /* as generated */
    }
    return fn;
}

static id<MTLFunction> compile_fn(const char *src, NSString *name, const char *what)
{
    return compile_fn_opts(src, name, what, 0);
}

#define VS_MAX 512
typedef struct {
    uint32_t        hash;
    uint16_t        inputs;
    int             used, bad;
    struct nv2a_aspect_gate hp;   /* hor+: is this program 2D */
    id<MTLFunction> fn;
} VsEntry;
static VsEntry s_vs[VS_MAX];

static VsEntry *vs_cache(void)
{
    static NV2AVshProgram parsed;
    static char msl[65536];
    static struct { int valid; uint32_t dwords, start; VsEntry *e; } memo;
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    uint32_t hash;
    VsEntry *e = NULL;
    char what[32];
    int i;

    /* Programs only change by upload (prog_dwords) or a new start slot. */
    if (memo.valid && memo.dwords == v->prog_dwords && memo.start == v->start)
        return memo.e;
    hash = nv2a_vsh_program_hash(v, NULL);
    if (!hash)
        hash = 1;
    for (i = 0; i < VS_MAX; i++) {
        if (s_vs[i].used && s_vs[i].hash == hash) {
            e = &s_vs[i];
            break;
        }
    }
    if (!e) {
        for (i = 0; i < VS_MAX && s_vs[i].used; i++)
            ;
        if (i == VS_MAX)
            return NULL;   /* full: a title with this many programs wants LRU */
        e = &s_vs[i];
        e->used = 1;
        e->hash = hash;
        e->hp.two_d = -1;
        e->hp.n = 0;
        e->bad = 1;
        d3d8_vsh_parse(&v->prog[v->start][0], NV2A_VSH_SLOTS - (int)v->start, &parsed);
        snprintf(what, sizeof what, "program %08X", hash);
        if (d3d8_vsh_generate_msl(&parsed, D3D8_MSL_SCREEN_SPACE, msl, (int)sizeof msl) <= 0) {
            s_compile_errors++;
            fprintf(stderr, LOGP "%s: no MSL\n", what);
        } else if ((e->fn = compile_fn(msl, @"vs_main", what)) != nil) {
            e->inputs = parsed.inputs_read;
            e->bad = 0;
            s_vs_compiled++;
        }
    }
    memo.valid = 1;
    memo.dwords = v->prog_dwords;
    memo.start = v->start;
    memo.e = e->bad ? NULL : e;
    return memo.e;
}

#define FS_MAX 512
typedef struct {
    uint32_t          hash;
    int               bad;
    int               fast;   /* metal_fs_fastmath: built with fast math */
    NV2ACombinerState st;
    id<MTLFunction>   fn;
} FsEntry;
static FsEntry *s_fs;   /* FS_MAX, allocated on first use */
static int      s_fs_n;

static uint32_t fnv(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint32_t h = 0x811c9dc5u;
    while (n--) {
        h ^= *b++;
        h *= 0x01000193u;
    }
    return h;
}

/* The fragment function for a batch without combiners: fs_basic, or
 * while render.snap_2d is on fs_basic_snap (s_msl_snap), built the first
 * time it is asked for. fs_basic when it cannot be built. */
static id<MTLFunction> fs_basic(void)
{
    static int tried;
    static id<MTLFunction> fn;

    if (!nv2a_snap_2d_on(nv2a_host_render_scale()))
        return s_fs_basic;
    if (!tried) {
        tried = 1;
        fn = compile_fn(s_msl_snap, @"fs_basic_snap", "fs_basic_snap");
    }
    return fn ? fn : s_fs_basic;
}

/* RECOMP_DEBUG=metal_nodiscard (KFC-0846), nd_decide's per-draw choice:
 * s_fs_nd, the batch takes the functions without discard; s_fs_nd_elig,
 * none of its kills could fire (counted). fs_select sets s_fs_sel_nd and
 * s_fs_sel_fast to what the function it returned is. s_fs_basic_nd:
 * fs_basic_nd and fs_basic_snap_nd, built the first time one is asked for
 * (nil until then, or when it cannot be built). */
static int s_fs_nd, s_fs_nd_elig, s_fs_sel_nd, s_fs_sel_fast;
static id<MTLFunction> s_fs_basic_nd[2];

/* fs_basic(), or with s_fs_nd its variant without the kills (fs_basic()
 * when that cannot be built). Sets s_fs_sel_nd and s_fs_sel_fast. */
static id<MTLFunction> fs_basic_pick(void)
{
    static int tried[2];
    int snap;

    s_fs_sel_nd = 0;
    s_fs_sel_fast = 0;
    if (!s_fs_nd)
        return fs_basic();
    snap = nv2a_snap_2d_on(nv2a_host_render_scale()) ? 1 : 0;
    if (!tried[snap]) {
        tried[snap] = 1;
        s_fs_basic_nd[snap] = snap ? compile_fn(s_msl_snap_nd, @"fs_basic_snap_nd",
                                                "fs_basic_snap_nd")
                                   : compile_fn(s_msl_nd, @"fs_basic_nd", "fs_basic_nd");
    }
    if (!s_fs_basic_nd[snap])
        return fs_basic();
    s_fs_sel_nd = 1;
    return s_fs_basic_nd[snap];
}

/* fn is one of the functions for a batch without combiners. */
static int fs_is_basic(id<MTLFunction> fn)
{
    return fn == s_fs_basic || fn == fs_basic()
        || (s_fs_basic_nd[0] && fn == s_fs_basic_nd[0])
        || (s_fs_basic_nd[1] && fn == s_fs_basic_nd[1]);
}

/* The batch's metal_nodiscard choice, before fs_select: eligible when no
 * kill of its fragment function can fire, exactly: the alpha test off or
 * ALWAYS (D3DCMP 8, which leaves ok true), no CLIPPLANE stage (every
 * clip_mode 0, so nv2a_clip_kills is false), and for the combiners' W
 * variant (w with combiners) the clamp mode, where wkill is false
 * (nv2a_wdepth_consts_fill: w1.y is 1 only for cull). No path here sets
 * alpha to coverage. =2 takes the variant on every draw. */
static void nd_decide(const struct nv2a_pb_vsh *v, const struct nv2a_ps_consts *pc, int w)
{
    int nd = nd_now(), k;

    s_fs_nd_elig = !pc->alpha_test_enable || pc->alpha_func == 8u;
    for (k = 0; k < 4; k++)
        if (pc->clip_mode[k])
            s_fs_nd_elig = 0;
    if (w && v->rc_set && (v->rc_ctl & 0xFF) && ((v->zminmax_ctl >> 4) & 0xFu) == 0u)
        s_fs_nd_elig = 0;
    s_fs_nd = nd == 2 || (nd == 1 && s_fs_nd_elig);
}

/* fs_select returned the combiners' function: only that one declares
 * cube maps, so only then does a stage bind one (nv2a_stage_cube_bind). A
 * batch that falls back to fs_basic (no MSL, a compile failure, FS_MAX)
 * samples face 0 as 2D, as before. */
static int s_fs_comb;

/* RECOMP_DEBUG=metal_snap_lazy=1|alt (KFC-1340 V1): the combiners call
 * nv2a_snap_uv only on a stage whose tex_scale.z is not 0, the picture
 * unchanged; alt: even flips only, an in-run A/B. =2|alt2 (V2): the call
 * only on the stages of s_snap_mask and none at all elsewhere, no guard
 * for the compiler to keep the derivatives under. Returns 0 off, 1 V1, 2 V2
 * for this flip. */
static int snap_lazy_now(void)
{
    static int mode = -1;
    if (mode < 0) {
        const char *e = recomp_env(RENV_METAL_SNAP_LAZY);
        /* V1 is the default (KFC-1340, 20-nov): the guard is equal by
         * construction, the call is a no-op where .z is 0. 0/off: the old
         * unconditional call. */
        int off = e && (e[0] == '0' || !strcasecmp(e, "off"));
        mode = !e ? 1 : off ? 0 : !strcasecmp(e, "alt") ? 2 : !strcasecmp(e, "alt2") ? 4
             : e[0] == '1' ? 1 : e[0] == '2' ? 3 : 0;
        if (e && !mode && !off)
            fprintf(stderr, LOGP "metal_snap_lazy=%s: not 1, alt, 2 or alt2, off\n", e);
        else if (mode >= 3)
            fprintf(stderr, LOGP "metal_snap_lazy=%s: nv2a_snap_uv only on stages with"
                    " tex_scale.z != 0, chosen per draw\n", e);
        else if (mode && e)
            fprintf(stderr, LOGP "metal_snap_lazy=%s: nv2a_snap_uv only where tex_scale.z != 0\n", e);
    }
    if (mode == 1 || (mode == 2 && !(s_presents & 1)))
        return 1;
    return mode == 3 || (mode == 4 && !(s_presents & 1)) ? 2 : 0;
}

/* metal_snap_lazy=2: bit k for a stage whose tex_scale.z is not 0, the
 * only ones nv2a_snap_uv can move. It depends on nothing else, so the
 * draws of one state share a function (fs_select_body's key[36] 6..9). */
static unsigned s_snap_mask;

static unsigned snap_mask(const struct nv2a_ps_consts *pc)
{
    unsigned m = 0;
    int k;
    for (k = 0; k < 4; k++)
        if (pc->tex_scale[k][2] != 0.0f)
            m |= 1u << k;
    return m;
}

/* fs_select returned the combiners' W variant (nv2a_w_depth_on): the
 * vertex side's W mode 1 (vc.u[7]) and the WDepth buffer go with it, and with
 * nothing else, so a fallback to fs_basic keeps the stock z -- or, on a
 * buffer in RECOMP_W_EARLYZ's mode, takes the vertex E (metal_on_draw's
 * wz_vs 2). */
static int s_fs_w;

/* RECOMP_TRACE=tex_mag (nv2a_tex_dump.h): this batch's combiners take the
 * trace's variant: a program batch. A pre-transformed one has its figure from
 * its vertices (mag_pretransformed), exact, with no atomics to pay. As on
 * D3D11, in the same terms. */
static int s_fs_mag;

static id<MTLFunction> fs_select_body(const struct nv2a_pb_vsh *v, int w);

/* The fragment function for the batch: its register combiners when they
 * are programmed (v->rc_set and a stage count), else fs_basic. As the D3D11
 * setup_pixel, the texture shader's modes are nv2a_stage_ps_mode's: cube
 * maps where a stage binds one, the dot-product family, BUMPENVMAP, and 2D
 * for every other sampling mode. */
static id<MTLFunction> fs_select(const struct nv2a_pb_vsh *v, int w)
{
    id<MTLFunction> fn = fs_select_body(v, w);
    s_fs_comb = !fs_is_basic(fn);
    s_fs_w = w && s_fs_comb;
    return fn;
}

/* metal_nodiscard's counts for an encoded draw (pt_judge's line): blend,
 * the pipeline blends. */
static void nd_count(int blend)
{
    NdCount *c = &s_ptf[s_presents & 1].nd;

    c->draws++;
    c->elig += (uint64_t)(s_fs_nd_elig != 0);
    c->win += (uint64_t)(s_fs_nd_elig && !blend && !s_fs_w);   /* the ceiling, either arm */
    c->fast += (uint64_t)(s_fs_sel_fast != 0);
    if (!s_fs_sel_nd)
        return;
    c->taken++;
    if (blend)
        c->blend++;
    if (s_fs_w)
        c->fdepth++;
}

static id<MTLFunction> fs_select_body(const struct nv2a_pb_vsh *v, int w)
{
    static char msl[65536];
    static struct { int valid, nd, fast; uint32_t r[38]; id<MTLFunction> fn; } memo;
    NV2ACombinerState st;
    uint32_t key[38], modes = 0, inputs = 0, hash;
    FsEntry *e = NULL;
    char what[32];
    int n, fast;

    if (!(v->rc_set && (v->rc_ctl & 0xFF)))
        return fs_basic_pick();
    nv2a_ps_stage_key(s_stage, v->other_stage_input, v->dot_rgbmapping, &modes, &inputs);
    memcpy(key, v->rc_cicw, 32);
    memcpy(key + 8, v->rc_aicw, 32);
    memcpy(key + 16, v->rc_cocw, 32);
    memcpy(key + 24, v->rc_aocw, 32);
    key[32] = v->rc_ctl;
    key[33] = v->rc_fcw0;
    key[34] = v->rc_fcw1;
    key[35] = modes;
    /* render.scale changes at run time: each scale class has its own text */
    /* RECOMP_DEBUG=metal_rect_snap0=1: timing probe, no nv2a_snap_uv (its
     * dfdx/dfdy per stage) at any scale; misplaces 2D texels */
    static int no_snap = -1;
    if (no_snap < 0)
        no_snap = recomp_env_on(RENV_METAL_RECT_SNAP0);
    key[36] = (nv2a_host_render_scale() > 1 && !no_snap) | (nv2a_host_opts()->aspect_w ? 2u : 0u)
            | (nv2a_snap_2d_on(nv2a_host_render_scale()) ? 4u : 0u)
            | (w ? 8u : 0u)    /* W-buffered: the w_depth variant */
            | (s_fs_mag ? 16u : 0u)    /* the trace's variant */
            | (snap_lazy_now() ? 32u : 0u);   /* RECOMP_DEBUG=metal_snap_lazy */
    if ((key[36] & 1u) && snap_lazy_now() == 2)   /* V2: bit 10, the stages in 6..9 */
        key[36] |= 1024u | (s_snap_mask & 15u) << 6;
    /* RECOMP_DEBUG=metal_nodiscard (KFC-0846): bit 11, no discard_fragment
     * (nd_decide); metal_fs_fastmath: bit 12, fast math, not for the W
     * variant, whose depth comes from the fragment */
    if (s_fs_nd)
        key[36] |= 2048u;
    if (!w && ff_now())
        key[36] |= 4096u;
    if (null_now())
        key[36] |= 8192u;   /* metal_ab_null: bit 13, the same text compiled again */
    key[37] = inputs;
    if (memo.valid && !memcmp(memo.r, key, sizeof key)) {
        s_fs_sel_nd = memo.nd;
        s_fs_sel_fast = memo.fast;
        return memo.fn;
    }

    memset(&st, 0, sizeof st);
    d3d8_combiners_from_regs(v->rc_cicw, v->rc_aicw, v->rc_cocw, v->rc_aocw,
                             v->rc_ctl, v->rc_fcw0, v->rc_fcw1, modes, &st);
    st.fog_input = 1;   /* FOG.a is the interpolated vertex fog factor */
    st.rect_snap = (int)(key[36] & 1u);   /* nv2a_snap_uv; in the hash via st */
    if (st.rect_snap && (key[36] & 32u))
        st.rect_snap = (key[36] & 1024u) ? 3 | (int)(key[36] >> 6 & 15u) << 4
                     : 2;   /* metal_snap_lazy: the call only where .z != 0 */
    st.rt_offset = (int)(key[36] >> 1 & 1u);   /* hor+: tex_scale.w, nv2a_aspect_place */
    st.snap_2d = (int)(key[36] >> 2 & 1u);     /* render.snap_2d: nv2a_snap_uv2d */
    st.w_depth = (int)(key[36] >> 3 & 1u);     /* Z_PERSPECTIVE: nv2a_w_depth_on */
    st.tex_mag = (int)(key[36] >> 4 & 1u);     /* RECOMP_TRACE=tex_mag */
    st.no_discard = (int)(key[36] >> 11 & 1u); /* RECOMP_DEBUG=metal_nodiscard */
    fast = (int)(key[36] >> 12 & 1u);          /* metal_fs_fastmath: in FsEntry, not st */
    if (key[36] & 8192u)
        fast |= 2;                              /* metal_ab_null: its own entry */
    for (n = 0; n < 4; n++) {
        st.tex_input[n] = (int)((inputs >> (4 * n)) & 0xF);
        st.dot_map[n] = n ? (int)((inputs >> (16 + 4 * (n - 1))) & 0xF) : 0;
        st.tex_cube[n] = (int)((inputs >> (28 + n)) & 1u);
    }
    st.bump_on = 1;
    /* no_discard is last: while it is 0 the hash (each "combiners %08X"
     * name) is the one before it was added */
    hash = fnv(&st, st.no_discard ? sizeof st : offsetof(NV2ACombinerState, no_discard));
    if (!s_fs && !(s_fs = (FsEntry *)calloc(FS_MAX, sizeof *s_fs)))
        return fs_basic_pick();
    for (n = 0; n < s_fs_n; n++)
        if (s_fs[n].hash == hash && s_fs[n].fast == fast
                && !memcmp(&s_fs[n].st, &st, sizeof st)) {
            e = &s_fs[n];
            break;
        }
    if (!e) {
        if (s_fs_n == FS_MAX)
            return fs_basic_pick();
        e = &s_fs[s_fs_n++];
        e->hash = hash;
        e->fast = fast;
        e->st = st;
        e->bad = 1;
        snprintf(what, sizeof what, "combiners %08X%s%s", hash, (fast & 1) ? " fast" : "",
                 (fast & 2) ? " null" : "");
        if (d3d8_combiners_generate_msl(&st, msl, (int)sizeof msl) <= 0) {
            s_compile_errors++;
            fprintf(stderr, LOGP "%s: no MSL\n", what);
        } else if ((e->fn = compile_fn_opts(msl, @"fs_main", what, fast & 1)) != nil) {
            e->bad = 0;
            s_fs_compiled++;
        }
    }
    memcpy(memo.r, key, sizeof key);
    memo.valid = 1;
    if (e->bad) {
        memo.fn = fs_basic_pick();   /* as D3D11: the basic PS */
    } else {
        memo.fn = e->fn;
        s_fs_sel_nd = st.no_discard;
        s_fs_sel_fast = e->fast & 1;
    }
    memo.nd = s_fs_sel_nd;
    memo.fast = s_fs_sel_fast;
    return memo.fn;
}

/* ---- the preload list --------------------------------------------------------
 *
 * macOS caches compiled shaders and pipeline states itself, but only once a
 * run has built them: each effect's first sight still compiles on the ack
 * thread (RECOMP_TRACE=metal_shaders, the [metal] build lines). This keeps
 * a list of what runs built, <hdd dir>/ShaderCache/metal/preload.bin
 * (RECOMP_METAL_PRELOAD: off, or another file), and at the next start
 * builds it all again on a background queue, so macOS's caches are warm
 * when the game asks. What the preload builds is thrown away: the game's
 * own build still runs, and hits.
 *
 * The file: PL_MAGIC and a version word, then records, each a payload
 * length and the payload's FNV-1a 32 ahead of the payload, appended and
 * flushed one at a time. A load stops at the first record that does not
 * check (a run cut short mid-write) and cuts the file there, so later
 * appends stay readable; a wrong magic or version starts the file over.
 *   L  a compile_fn library: what, the entry name and the MSL source as
 *      generated (before metal_shader_nonce's rename)
 *   P  a pipeline() state: its two functions by identity and its key[6]
 * A function's identity is the FNV-1a 64 of its entry name and source (two
 * entries of one source are two functions). The base library's vs_pass and
 * fs_basic have one too, so their pipelines are listed; the preload uses
 * the functions init built for them.
 *
 * Threads: the record set, the function map and the FILE are the ack
 * thread's. The preload gets its own copy of the records init read, and
 * the device (thread-safe); nothing waits on it. */

#define PL_MAGIC       "XRMTLPL\x1a"
#define PL_VERSION     2u               /* 2: metal_snap_lazy=1 by default (20-nov) */
#define PL_HEAD        12u               /* magic, version */
#define PL_MAX_RECORDS 4096u
#define PL_MAX_PAYLOAD (1u << 20)
#define PL_L_HEAD      20u               /* 'L', pad, u16 what, u16 name, u32 src, u64 ident */
#define PL_P_SIZE      44u               /* 'P', pad, u64 vs, u64 fs, u32 key[6] */
#define PL_SET_SLOTS   8192u             /* > 2 x PL_MAX_RECORDS */
#define PL_FN_SLOTS    4096u             /* > VS_MAX + FS_MAX + the rest */
#define FNV64_INIT     0xcbf29ce484222325ull

static int      s_pl_on;                 /* recording: the list is open */
static FILE    *s_pl_file;
static char     s_pl_path[1100];
static unsigned s_pl_n;                  /* records in the file */
static uint64_t s_pl_set[PL_SET_SLOTS];  /* their identities, 0 = empty slot */
static struct { const void *fn; uint64_t ident; } s_pl_fn[PL_FN_SLOTS];

static uint64_t fnv64(uint64_t h, const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    while (n--) {
        h ^= *b++;
        h *= 0x100000001b3ull;
    }
    return h;
}

static uint64_t pl_fn_identity(const char *name, const char *src, size_t src_len)
{
    uint64_t h = fnv64(FNV64_INIT, name, strlen(name) + 1);
    h = fnv64(h, src, src_len);
    return h ? h : 1;
}

/* 1 when ident was not in the set (it is now). */
static int pl_set_add(uint64_t ident)
{
    unsigned i = (unsigned)(ident ^ ident >> 32) & (PL_SET_SLOTS - 1);
    while (s_pl_set[i]) {
        if (s_pl_set[i] == ident)
            return 0;
        i = (i + 1) & (PL_SET_SLOTS - 1);
    }
    s_pl_set[i] = ident;
    return 1;
}

static unsigned pl_fn_slot(const void *fn)
{
    uintptr_t p = (uintptr_t)fn;
    return (unsigned)(((p >> 4) ^ (p >> 20)) * 0x9E3779B1u) & (PL_FN_SLOTS - 1);
}

/* The backend keeps every function it builds for the life of the process,
 * so a pointer names one function. */
static void pl_fn_note(id<MTLFunction> fn, uint64_t ident)
{
    unsigned i = pl_fn_slot(fn), k;
    for (k = 0; k < PL_FN_SLOTS; k++, i = (i + 1) & (PL_FN_SLOTS - 1))
        if (!s_pl_fn[i].fn || s_pl_fn[i].fn == fn) {
            s_pl_fn[i].fn = fn;
            s_pl_fn[i].ident = ident;
            return;
        }
}

static uint64_t pl_fn_id(id<MTLFunction> fn)
{
    unsigned i = pl_fn_slot(fn), k;
    for (k = 0; k < PL_FN_SLOTS && s_pl_fn[i].fn; k++, i = (i + 1) & (PL_FN_SLOTS - 1))
        if (s_pl_fn[i].fn == fn)
            return s_pl_fn[i].ident;
    return 0;
}

static void pl_append(const uint8_t *payload, uint32_t len)
{
    uint8_t hd[8];
    uint32_t ck = fnv(payload, len);
    memcpy(hd, &len, 4);
    memcpy(hd + 4, &ck, 4);
    if (fwrite(hd, 1, 8, s_pl_file) != 8 || fwrite(payload, 1, len, s_pl_file) != len
     || fflush(s_pl_file) != 0) {
        fprintf(stderr, LOGP "preload: cannot write %s; the list stops here\n", s_pl_path);
        fclose(s_pl_file);
        s_pl_file = NULL;
        s_pl_on = 0;
        return;
    }
    s_pl_n++;
}

static void pl_note_library(id<MTLFunction> fn, const char *what, const char *name,
                            const char *src)
{
    size_t wl, nl, sl;
    uint32_t len;
    uint16_t w16, n16;
    uint32_t s32;
    uint64_t ident;
    uint8_t *p;

    if (!s_pl_on || !name || !src)
        return;
    sl = strlen(src);
    ident = pl_fn_identity(name, src, sl);
    pl_fn_note(fn, ident);
    if (s_pl_n >= PL_MAX_RECORDS || !pl_set_add(ident))   /* the set stays half empty */
        return;
    wl = strlen(what);
    if (wl > 255)
        wl = 255;
    nl = strlen(name);
    if (nl > 255 || sl + wl + nl + PL_L_HEAD > PL_MAX_PAYLOAD)
        return;
    len = (uint32_t)(PL_L_HEAD + wl + nl + sl);
    if (!(p = (uint8_t *)calloc(1, len)))
        return;
    w16 = (uint16_t)wl;
    n16 = (uint16_t)nl;
    s32 = (uint32_t)sl;
    p[0] = 'L';
    memcpy(p + 4, &w16, 2);
    memcpy(p + 6, &n16, 2);
    memcpy(p + 8, &s32, 4);
    memcpy(p + 12, &ident, 8);
    memcpy(p + PL_L_HEAD, what, wl);
    memcpy(p + PL_L_HEAD + wl, name, nl);
    memcpy(p + PL_L_HEAD + wl + nl, src, sl);
    pl_append(p, len);
    free(p);
}

static void pl_note_pso(id<MTLFunction> vs, id<MTLFunction> fs, const uint32_t key[6])
{
    uint8_t p[PL_P_SIZE];
    uint64_t vid, fid, ident;

    if (!s_pl_on)
        return;
    /* A function no compile_fn built (and not vs_pass or fs_basic): not
     * listed, nor its pipelines. */
    if (!(vid = pl_fn_id(vs)) || !(fid = pl_fn_id(fs)))
        return;
    memset(p, 0, sizeof p);
    p[0] = 'P';
    memcpy(p + 4, &vid, 8);
    memcpy(p + 12, &fid, 8);
    memcpy(p + 20, key, 24);
    ident = fnv64(FNV64_INIT, p, sizeof p);
    if (!ident)
        ident = 1;
    if (s_pl_n >= PL_MAX_RECORDS || !pl_set_add(ident))   /* the set stays half empty */
        return;
    pl_append(p, PL_P_SIZE);
}

/* A record the parser can take: its type and its lengths add up. */
static int pl_record_ok(const uint8_t *p, uint32_t len)
{
    uint16_t wl, nl;
    uint32_t sl;
    if (p[0] == 'P')
        return len == PL_P_SIZE;
    if (p[0] != 'L' || len < PL_L_HEAD)
        return 0;
    memcpy(&wl, p + 4, 2);
    memcpy(&nl, p + 6, 2);
    memcpy(&sl, p + 8, 4);
    /* pl_note_library writes what and the name at 255 bytes at most. */
    return wl <= 255 && nl && nl <= 255 && sl && (uint64_t)PL_L_HEAD + wl + nl + sl == len;
}

/* The identity a record adds to the set. */
static uint64_t pl_record_id(const uint8_t *p, uint32_t len)
{
    uint64_t ident;
    if (p[0] == 'L') {
        memcpy(&ident, p + 12, 8);
    } else {
        ident = fnv64(FNV64_INIT, p, len);
        if (!ident)
            ident = 1;
    }
    return ident;
}

typedef struct {
    id<MTLDevice>   dev;
    id<MTLFunction> vs_pass, fs_basic;   /* the base library's, as init built them */
    uint64_t        vs_pass_id, fs_basic_id;
    uint8_t        *data;                /* the records (validated), malloc'd */
    size_t          len;
    char            path[1100];
} PreloadJob;

/* The preload's queue: build every listed library with compile_fn's rename
 * and options, then every listed pipeline from them with pso_build; keep
 * nothing. Nothing here touches the ack thread's state. */
static void preload_run(void *ctx)
{
    PreloadJob *j = (PreloadJob *)ctx;
    uint64_t t0 = prof_now();
    unsigned nl = 0, np = 0, fails = 0, skipped = 0;
    size_t off = 0;

    @autoreleasepool {
        NSMutableDictionary *fns = [[NSMutableDictionary alloc] init];
        [fns setObject:j->vs_pass forKey:@(j->vs_pass_id)];
        [fns setObject:j->fs_basic forKey:@(j->fs_basic_id)];
        while (off + 8 <= j->len) {
            const uint8_t *p = j->data + off + 8;
            uint32_t len;
            memcpy(&len, j->data + off, 4);
            off += 8 + (size_t)len;
            @autoreleasepool {
                if (p[0] == 'L') {
                    uint16_t wl, nlen;
                    uint32_t sl;
                    uint64_t ident;
                    char what[256];
                    NSString *name, *text;
                    NSError *err = nil;
                    id<MTLLibrary> lib;
                    id<MTLFunction> fn = nil;
                    memcpy(&wl, p + 4, 2);
                    memcpy(&nlen, p + 6, 2);
                    memcpy(&sl, p + 8, 4);
                    memcpy(&ident, p + 12, 8);
                    memcpy(what, p + PL_L_HEAD, wl);
                    what[wl] = 0;
                    name = [[[NSString alloc] initWithBytes:p + PL_L_HEAD + wl length:nlen
                                                   encoding:NSUTF8StringEncoding] autorelease];
                    text = [[[NSString alloc] initWithBytes:p + PL_L_HEAD + wl + nlen length:sl
                                                   encoding:NSUTF8StringEncoding] autorelease];
                    if (name && text) {
                        text = sh_nonce_rename(text, &name);
                        lib = metal_compile_library(j->dev, text, what, &err);
                        if (lib) {
                            fn = [lib newFunctionWithName:name];
                            [lib release];
                        }
                    }
                    if (fn) {
                        [fns setObject:fn forKey:@(ident)];
                        [fn release];
                        nl++;
                    } else {
                        fails++;
                    }
                } else {
                    uint64_t vid, fid;
                    uint32_t key[6];
                    id<MTLFunction> vs, fs;
                    id<MTLRenderPipelineState> pso;
                    memcpy(&vid, p + 4, 8);
                    memcpy(&fid, p + 12, 8);
                    memcpy(key, p + 20, 24);
                    vs = [fns objectForKey:@(vid)];
                    fs = [fns objectForKey:@(fid)];
                    if (vs && fs) {
                        pso = pso_build(j->dev, vs, fs, key, NULL);
                        if (pso) {
                            [pso release];
                            np++;
                        } else {
                            fails++;
                        }
                    } else {
                        skipped++;
                    }
                }
            }
        }
        [fns release];
    }
    fprintf(stderr, LOGP "preload: %u libraries, %u pipelines in %.0f ms (%s)",
            nl, np, (prof_now() - t0) / 1e6, j->path);
    if (fails || skipped)
        fprintf(stderr, "; %u failed, %u pipelines without their functions", fails, skipped);
    fputc('\n', stderr);
    [j->dev release];
    [j->vs_pass release];
    [j->fs_basic release];
    free(j->data);
    free(j);
}

/* Where the list lives, per RECOMP_METAL_PRELOAD; 0 = off. */
static int pl_resolve(char *path, size_t n)
{
    const char *e = recomp_env(RENV_METAL_PRELOAD);
    shader_cache_env_kind kind = shader_cache_env(e);
    shader_cache probe;
    char dir[1024];

    if (kind == SHADER_CACHE_ENV_OFF)
        return 0;
    if (kind == SHADER_CACHE_ENV_RELATIVE) {
        fprintf(stderr, LOGP "preload: off (RECOMP_METAL_PRELOAD=%s is not an absolute"
                " path)\n", e);
        return 0;
    }
    if (kind == SHADER_CACHE_ENV_DIR) {
        const char *slash = strrchr(e, '/');
        size_t dl = slash ? (size_t)(slash - e) : 0;
        if (snprintf(path, n, "%s", e) >= (int)n || dl >= sizeof dir)
            return 0;
        memcpy(dir, e, dl);
        dir[dl ? dl : 1] = 0;
        if (!dl)
            dir[0] = '/';
    } else {
        /* Beside the D3D11 backend's dxbc, under no name a guest path maps to. */
        if (!xbox_path_shader_cache(dir, sizeof dir - 16))
            return 0;
        strcat(dir, "/metal");
        if (snprintf(path, n, "%s/preload.bin", dir) >= (int)n)
            return 0;
    }
    if (!shader_cache_init(&probe, dir)) {   /* mkdir -p, and a write probe */
        fprintf(stderr, LOGP "preload: off (cannot make or write %s)\n", dir);
        return 0;
    }
    return 1;
}

/* On the ack thread, from init once the device and the base library are
 * up: open the list (made or repaired), take its records into the set and
 * queue the preload with a copy of them. */
static void preload_start(void)
{
    uint8_t *buf = NULL;
    long size;
    size_t off, end = 0;
    FILE *f;
    uint64_t vs_pass_id, fs_basic_id;
    uint32_t ver;
    int fresh = 0;

    sh_env();
    if (!pl_resolve(s_pl_path, sizeof s_pl_path))
        return;
    if (!(f = fopen(s_pl_path, "r+b")) && !(f = fopen(s_pl_path, "w+b"))) {
        fprintf(stderr, LOGP "preload: off (cannot open %s)\n", s_pl_path);
        return;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) < 0)
        size = 0;
    if (size >= (long)PL_HEAD && size <= (long)(PL_MAX_RECORDS * (8u + 64u * 1024u))
     && (buf = (uint8_t *)malloc((size_t)size)) != NULL) {
        rewind(f);
        if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
            free(buf);
            buf = NULL;
        }
    }
    if (buf) {
        memcpy(&ver, buf + 8, 4);
        fresh = memcmp(buf, PL_MAGIC, 8) != 0 || ver != PL_VERSION;
    } else {
        fresh = 1;
    }
    if (!fresh) {
        for (off = PL_HEAD; off + 8 <= (size_t)size && s_pl_n < PL_MAX_RECORDS; ) {
            uint32_t len, ck;
            memcpy(&len, buf + off, 4);
            memcpy(&ck, buf + off + 4, 4);
            if (len < 1 || len > PL_MAX_PAYLOAD || len > (size_t)size - off - 8
             || fnv(buf + off + 8, len) != ck || !pl_record_ok(buf + off + 8, len))
                break;
            pl_set_add(pl_record_id(buf + off + 8, len));
            s_pl_n++;
            off += 8 + (size_t)len;
        }
        end = off;
        if (end < (size_t)size) {   /* a cut-short tail: appends go after the last good record */
            fflush(f);
            if (ftruncate(fileno(f), (off_t)end) != 0)
                fresh = 1;
        }
    }
    if (fresh) {
        uint8_t head[PL_HEAD];
        s_pl_n = 0;
        memset(s_pl_set, 0, sizeof s_pl_set);
        end = 0;
        memcpy(head, PL_MAGIC, 8);
        ver = PL_VERSION;
        memcpy(head + 8, &ver, 4);
        fflush(f);
        rewind(f);
        if (ftruncate(fileno(f), 0) != 0 || fwrite(head, 1, PL_HEAD, f) != PL_HEAD
         || fflush(f) != 0) {
            fprintf(stderr, LOGP "preload: off (cannot write %s)\n", s_pl_path);
            fclose(f);
            free(buf);
            return;
        }
    }
    fseek(f, 0, SEEK_END);
    s_pl_file = f;
    s_pl_on = 1;

    /* The base library's two: their pipelines are listed like the others. */
    vs_pass_id = pl_fn_identity("vs_pass", s_msl, sizeof s_msl - 1);
    fs_basic_id = pl_fn_identity("fs_basic", s_msl, sizeof s_msl - 1);
    pl_fn_note(s_vs_pass, vs_pass_id);
    pl_fn_note(s_fs_basic, fs_basic_id);

    if (end > PL_HEAD) {
        PreloadJob *j = (PreloadJob *)calloc(1, sizeof *j);
        if (j && (j->data = (uint8_t *)malloc(end - PL_HEAD)) != NULL) {
            dispatch_queue_attr_t at = dispatch_queue_attr_make_with_qos_class(
                DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0);
            dispatch_queue_t q = dispatch_queue_create("xboxrecomp.metal.preload", at);
            memcpy(j->data, buf + PL_HEAD, end - PL_HEAD);
            j->len = end - PL_HEAD;
            j->dev = [s_dev retain];
            j->vs_pass = [s_vs_pass retain];
            j->fs_basic = [s_fs_basic retain];
            j->vs_pass_id = vs_pass_id;
            j->fs_basic_id = fs_basic_id;
            snprintf(j->path, sizeof j->path, "%s", s_pl_path);
            dispatch_async_f(q, j, preload_run);
            dispatch_release(q);
        } else {
            free(j);
        }
    }
    free(buf);
}

/* ---- depth / stencil state ---------------------------------------------------
 *
 * As the D3D11 backend's depth_state: SET_DEPTH_FUNC and SET_STENCIL_FUNC
 * are GL compare codes in MTLCompareFunction's order (nv2a_cmp_from_gl), the
 * stencil ops in MTLStencilOperation's (nv2a_stencil_op_from_gl). Depth and
 * stencil each only when enabled; one state per distinct key. ez: the
 * buffer stores RECOMP_W_EARLYZ's E, the test reversed (nv2a_cmp_reverse),
 * the key holding the reversed code. */
#define DSS_MAX 64
static struct { uint64_t key; id<MTLDepthStencilState> dss; } s_dss[DSS_MAX];
static int s_dss_n;

static id<MTLDepthStencilState> depth_state(int depth, int stencil, int ez)
{
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    uint32_t f = ez ? nv2a_cmp_reverse(nv2a_cmp_from_gl(v->depth_func))
                    : nv2a_cmp_from_gl(v->depth_func);
    uint64_t key = 0;
    MTLDepthStencilDescriptor *dd;
    id<MTLDepthStencilState> st;
    int i;

    if (depth)
        key |= 1u | f << 1 | (v->depth_mask ? 1u : 0u) << 4;
    if (stencil)
        key |= (uint64_t)(1u | (v->stencil_func & 7u) << 1
                          | nv2a_stencil_op_from_gl(v->stencil_op[0]) << 4
                          | nv2a_stencil_op_from_gl(v->stencil_op[1]) << 8
                          | nv2a_stencil_op_from_gl(v->stencil_op[2]) << 12
                          | (v->stencil_rmask & 0xFFu) << 16
                          | (v->stencil_wmask & 0xFFu) << 24) << 8;
    for (i = 0; i < s_dss_n; i++)
        if (s_dss[i].key == key)
            return s_dss[i].dss;
    dd = [[MTLDepthStencilDescriptor alloc] init];
    dd.depthCompareFunction = depth ? (MTLCompareFunction)f : MTLCompareFunctionAlways;
    dd.depthWriteEnabled = depth && v->depth_mask ? YES : NO;
    if (stencil) {
        MTLStencilDescriptor *sd = [[MTLStencilDescriptor alloc] init];
        sd.stencilCompareFunction = (MTLCompareFunction)(v->stencil_func & 7u);
        sd.stencilFailureOperation = (MTLStencilOperation)nv2a_stencil_op_from_gl(v->stencil_op[0]);
        sd.depthFailureOperation = (MTLStencilOperation)nv2a_stencil_op_from_gl(v->stencil_op[1]);
        sd.depthStencilPassOperation = (MTLStencilOperation)nv2a_stencil_op_from_gl(v->stencil_op[2]);
        sd.readMask = v->stencil_rmask & 0xFFu;
        sd.writeMask = v->stencil_wmask & 0xFFu;
        dd.frontFaceStencil = sd;               /* GL one-sided stencil */
        dd.backFaceStencil = sd;
        [sd release];
    }
    st = [s_dev newDepthStencilStateWithDescriptor:dd];
    [dd release];
    if (!st)
        return nil;
    if (s_dss_n >= DSS_MAX)
        return st;   /* uncached: a title with this many wants LRU (leaks) */
    s_dss[s_dss_n].key = key;
    s_dss[s_dss_n].dss = st;
    s_dss_n++;
    return st;
}

/* ---- render-to-texture ---------------------------------------------------------
 *
 * A stage whose texture is a surface this backend draws into samples the
 * render target itself, as the D3D11 backend's rt_texture: guest memory
 * under it only changes at a flip's write-back (and then only for the
 * present surface). nv2a_rt_tex_reads says which reads are the target's,
 * 16-bit ones included: a write-back keeps only the stock columns, so under
 * hor+ a 16-bit frame copy decoded from it was a centred 4:3 picture.
 * The low 27 bits of the address compare, so either memory
 * window matches. The target being
 * drawn into is sampled through a copy (s_self), made by a blit between
 * passes. RECOMP_METAL_NO_RTT=1 decodes guest memory instead (A/B). */
static RenderTarget s_self;   /* tex, w, h: the copy; offset: whose */

static int aspect_trace(void);

/* ---- depth surfaces as textures ---------------------------------------------
 *
 * A stage whose texture lies where a depth surface this backend draws into
 * starts reads that surface's depth, as the console's memory holds it, as
 * the D3D11 backend's zt_texture. Kung Fu Chaos's sci-fi stage draws its
 * ground haze twice: once with depth func ALWAYS and colour writes off (the
 * haze's own depth into the W-buffer), then again sampling the zeta surface
 * as a 640x480 A8R8G8B8 rect texture, the Z24S8 bytes taken as colour, to
 * fade the haze by the distance between the two. The depth lives only in the
 * Depth32Float_Stencil8 buffer here, the guest bytes are never written, and
 * the haze came out opaque over the street. A compute pass between passes
 * (textures are bound before the batch's pass, as the self copy's blit)
 * writes the guest's bytes into an RGBA8 texture the size of the depth's
 * host buffer, render.scale and hor+ pads included, served as a colour
 * target is. Converted again only after a write.
 *
 * The depth holds the NV2A's 24-bit value over 0xFFFFFF in both modes (the
 * W-buffer's w / zmax, the z-buffer's normalised z), and Z24S8 is the dword
 * (z << 8) | stencil: read as A8R8G8B8, B is the stencil, G z's low byte, R
 * its middle one, A its top one. Z16, a float zeta, a swizzled or another
 * texture format, a texture starting inside the surface, or a buffer no
 * pass has cleared yet fall back to the guest bytes, said once.
 * RECOMP_METAL_ZTEX=0: off (A/B; depth buffers are then made without
 * ShaderRead, as before).
 *
 * A buffer in RECOMP_W_EARLYZ's mode holds E = B (1/w - 1/zmax) instead,
 * decoded first as D3D11's: w = B / (E + B / zmax), the guest's value
 * w / zmax. zk = (mode, B, B / zmax, 1 / zmax) at buffer 0, from the
 * buffer's own earlyz. */
static RenderTarget s_zt_view[DT_MAX];   /* s_dt[i].ztex as rt_texture serves it */
static id<MTLComputePipelineState> s_zt_pso;
static int      s_zt_bad;                /* no compute pipeline: never again */
static uint64_t s_zt_converts, s_zt_binds;

static const char s_zt_msl[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void nv2a_zt(depth2d<float, access::read> zd [[texture(0)]],\n"
    "                    texture2d<uint, access::read> zs [[texture(1)]],\n"
    "                    texture2d<float, access::write> o [[texture(2)]],\n"
    "                    constant float4 &zk [[buffer(0)]],\n"
    "                    uint2 id [[thread_position_in_grid]]) {\n"
    "    if (id.x >= o.get_width() || id.y >= o.get_height()) return;\n"
    "    float e = saturate(zd.read(id));\n"
    "    if (zk.x != 0.0f) e = saturate(zk.y / (e + zk.z) * zk.w);\n"
    /* min: 1.0 (the clear) * 16777215 rounds to 2^24 in float, which would
     * carry into a ninth bit and zero the low bytes */
    "    uint z = min((uint)round(e * 16777215.0f), 16777215u);\n"
    "    uint s = zs.read(id).r & 255u;\n"
    "    o.write(float4(float((z >> 8) & 255u), float(z & 255u), float(s),\n"
    "                   float(z >> 16)) / 255.0f, id);\n"
    "}\n";

static int zt_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("RECOMP_METAL_ZTEX");
        on = !(e && e[0] == '0');
    }
    return on;
}

static void zt_say(const DepthTarget *d, const Stage *t, const char *why)
{
    static int said;
    if (said++ < 6)
        fprintf(stderr, LOGP "zeta 0x%08X sampled as a texture (format 0x%02X, %ux%u, pitch %u):"
                " %s; it reads guest memory\n", d->zeta, nv2a_tex_linear_twin(t->color),
                t->width, t->height, t->pitch, why);
}

/* The depth surface a texture reads, if its bytes are the depth's: one in
 * use within the last 2 flips (a surface left in s_dt after the title moved
 * on is not what the memory holds), written since anything else wrote over
 * its memory (a 2D blit in guest memory, a target written back there), and
 * since any colour target at its address was drawn or filled by a blit --
 * the title's match copies frames to its menus' background address, which
 * is also a zeta surface; whichever wrote last is the memory's. */
static DepthTarget *zt_find(const Stage *t)
{
    int i, k;
    if (!zt_on() || s_zt_bad || !t->offset)
        return NULL;
    for (i = 0; i < DT_MAX; i++) {
        DepthTarget *d = &s_dt[i];
        if (!d->tex || s_presents - d->use_flip > 2
                || !nv2a_rt_overlap(d->addr, d->cpitch * d->h, t->offset, 1))
            continue;
        if (d->mseq > d->wseq)
            continue;   /* written over in memory after its last depth write */
        /* Only a texture at its first byte (as D3D11's zt_find). */
        if ((d->addr ^ t->offset) & 0x07FFFFFFu) {
            uint32_t lin = nv2a_tex_linear_twin(t->color);
            if ((lin == 0x12 || lin == 0x1E) && !d3d8_format_is_swizzled(t->color))
                zt_say(d, t, "the texture starts inside the surface");
            continue;
        }
        for (k = 0; k < RT_MAX; k++) {
            const RenderTarget *rt = &s_rt[k];
            if (!rt->tex || ((rt->addr ^ d->addr) & 0x07FFFFFFu) || rt->zseq <= d->wseq)
                continue;
            {
                static int said;
                if (said++ < 4)
                    fprintf(stderr, LOGP "texture 0x%08X: surface 0x%08X (seq %llu) and zeta"
                            " 0x%08X (seq %llu) both there, the colour target wins"
                            " (flip %llu)\n", t->offset, rt->offset,
                            (unsigned long long)rt->zseq, d->zeta,
                            (unsigned long long)d->wseq, (unsigned long long)s_presents);
            }
            break;
        }
        if (k < RT_MAX)
            continue;
        return d;
    }
    return NULL;
}

static int zt_make(DepthTarget *d)
{
    MTLTextureDescriptor *td;

    if (!s_zt_pso) {
        NSError *err = nil;
        id<MTLLibrary> lib = metal_compile_library(s_dev, [NSString stringWithUTF8String:s_zt_msl],
                                                   "nv2a_zt", &err);
        id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"nv2a_zt"] : nil;
        if (fn)
            s_zt_pso = [s_dev newComputePipelineStateWithFunction:fn error:&err];
        [fn release];
        [lib release];
        if (!s_zt_pso) {
            s_zt_bad = 1;
            fprintf(stderr, LOGP "no compute pipeline for depth surfaces (%s): one sampled"
                    " as a texture reads guest memory\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            return 0;
        }
    }
    if (!d->zst) {
        /* The stencil half: a view of the same texture (ShaderRead and
         * PixelFormatView were asked for at depth_target). */
        d->zst = [d->tex newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
        if (!d->zst) {
            static int said;
            if (!said++)
                fprintf(stderr, LOGP "zeta 0x%08X: no stencil view of its depth buffer: a depth"
                        " surface sampled as a texture reads guest memory\n", d->zeta);
            return 0;
        }
    }
    if (d->ztex)
        return 1;
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                            width:d->hw height:d->hh mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModePrivate;
    d->ztex = [s_dev newTextureWithDescriptor:td];
    return d->ztex != nil;
}

/* d's depth, as it stands before the batch being set up draws, as the
 * texture t reads it; NULL to decode guest memory as before. Ends the open
 * pass when it converts: the draw opens a new one (Load). */
static RenderTarget *zt_texture(DepthTarget *d, const Stage *t)
{
    uint32_t lin = nv2a_tex_linear_twin(t->color);
    RenderTarget *r = &s_zt_view[d - s_dt];

    if (d->zfmt != 2) {
        zt_say(d, t, "a Z16 surface (not converted)");
        return NULL;
    }
    if (nv2a_pb_vsh_state()->control0 & 0x1000u) {
        zt_say(d, t, "a float depth format (not converted)");
        return NULL;
    }
    if ((lin != 0x12 && lin != 0x1E) || d3d8_format_is_swizzled(t->color)
            || t->pitch != d->cpitch || t->width > d->w || t->height > d->h) {
        zt_say(d, t, "not a linear 32-bit colour texture of its layout");
        return NULL;
    }
    if (d->fresh) {
        zt_say(d, t, "no pass has cleared the buffer yet");
        return NULL;
    }
    if (!zt_make(d))
        return NULL;
    if (d->zconv != d->wseq) {
        id<MTLComputeCommandEncoder> c;
        float zmax = nv2a_zmax(d->zfmt << 4), b = nv2a_w_earlyz_b();
        float zk[4] = { d->earlyz ? 1.0f : 0.0f, b, b / zmax, 1.0f / zmax };
        PT_WHY(PT_ZTEX);
        end_encoder();
        ensure_cmd();
        if (pt_on()) {
            /* metal_passes: the depth and stencil stores' reader */
            c = pt_compute(PT_ZTEX, NV2A_GCAT_ZTEX);
            if (s_pt_track) {
                pt_read(d->tex, 1, PTR_ZTEX, s_pt_aux);
                pt_read(d->tex, 2, PTR_ZTEX, s_pt_aux);
            }
        } else {
            c = [s_cmd computeCommandEncoder];
        }
        [c setComputePipelineState:s_zt_pso];
        [c setTexture:d->tex atIndex:0];
        [c setTexture:d->zst atIndex:1];
        [c setTexture:d->ztex atIndex:2];
        [c setBytes:zk length:sizeof zk atIndex:0];
        [c dispatchThreadgroups:MTLSizeMake((d->hw + 7u) / 8u, (d->hh + 7u) / 8u, 1)
          threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [c endEncoding];
        d->zconv = d->wseq;
        if (s_zt_converts++ == 0)
            fprintf(stderr, LOGP "zeta 0x%08X sampled as a texture (format 0x%02X, %ux%u):"
                    " its depth converted on the GPU (flip %llu)\n", d->zeta, lin,
                    t->width, t->height, (unsigned long long)s_presents);
    }
    /* What the draw, the hor+ placement and the trace read of a render
     * target. Never written back, freed or drawn into: it is no s_rt. */
    memset(r, 0, sizeof *r);
    r->offset = d->zeta;
    r->addr = d->addr;
    r->w = d->w;
    r->h = d->h;
    r->hw = d->hw;
    r->hh = d->hh;
    r->pad = d->pad;
    r->n = d->n;
    r->tex = d->ztex;
    r->pitch = d->cpitch;
    r->last_use = s_tick;
    s_zt_binds++;
    return r;
}

static RenderTarget *rt_texture(const Stage *t, RenderTarget *cur)
{
    static int off = -1;
    RenderTarget *self = NULL;
    DepthTarget *zd;
    int i;

    if (off < 0)
        off = recomp_env(RENV_METAL_NO_RTT) != NULL;
    if (off || !t->offset)
        return NULL;
    /* The depth first: a colour target at its address drawn later wins
     * there (zt_find), else the depth is what the memory holds. */
    zd = zt_find(t);
    if (zd) {
        RenderTarget *z = zt_texture(zd, t);
        if (z)
            return z;
    }
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (!rt->tex || ((rt->addr ^ t->offset) & 0x07FFFFFFu))
            continue;
        if (!nv2a_rt_tex_reads(t->color, t->pitch, rt->fmt, rt->pitch)) {
            /* Decoded from guest memory instead: the write-back's stock
             * columns, or what the title left there. The format
             * word is the last draw's: reuse matches offset and size only. */
            if (aspect_trace() && nv2a_rt_reject_new(rt->fmt, rt->pitch, t->color, t->pitch))
                fprintf(stderr, "[aspect.rt-reject] flip %llu: target 0x%08X surface fmt %X"
                        " pitch %u, texture fmt %X pitch %u\n",
                        (unsigned long long)s_presents, rt->addr, rt->fmt, rt->pitch, t->color,
                        t->pitch);
            continue;
        }
        if (rt_drop_if_stale(rt, "bind"))
            continue;   /* the texture decodes from the title's bytes */
        if (rt == cur) {
            self = rt;
            continue;
        }
        cf_drop(rt, NULL, PT_FOLD_READ);   /* metal_clear_fold: its held clear, before the draw */
        return rt;
    }
    if (!self)
        return NULL;
    /* copyFromTexture:toTexture: wants equal sizes: the source's host size. */
    if (!s_self.tex || s_self.w != self->w || s_self.h != self->h
            || s_self.hw != self->hw || s_self.hh != self->hh) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                         width:self->hw height:self->hh mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        [s_self.tex release];
        s_self.tex = [s_dev newTextureWithDescriptor:td];
        if (!s_self.tex)
            return NULL;
        s_self.w = self->w;
        s_self.h = self->h;
        s_self.hw = self->hw;
        s_self.hh = self->hh;
    }
    {
        id<MTLBlitCommandEncoder> b;
        PT_WHY(PT_SELF_COPY);
        end_encoder();
        ensure_cmd();
        if (pt_on()) {
            b = pt_blit(PT_SELF_COPY, NV2A_GCAT_OTHER);
            if (s_pt_track)
                pt_read(self->tex, 0, PTR_BLIT, s_pt_aux);
        } else {
            b = [s_cmd blitCommandEncoder];
        }
        [b copyFromTexture:self->tex toTexture:s_self.tex];
        [b endEncoding];
        rt_read_mark(self);
    }
    s_self.offset = self->offset;
    s_self.pad = self->pad;
    s_self.n = self->n;
    s_self_copies++;
    {
        static int said;
        if (!said++)
            fprintf(stderr, LOGP "surface 0x%08X sampled while drawn into:"
                    " bound as a copy\n", self->offset);
    }
    return &s_self;
}

/* ---- draws ------------------------------------------------------------------- */

/* RECOMP_TRACE=aspect: one line per vertex program (0: pass-through) and
 * placement the first time a 2D batch of it is placed, with what it
 * samples: the inventory display.aspect's rules and any per-program
 * override are judged by. */
static int aspect_trace(void)
{
    static int on = -1;
    if (on < 0)
        on = recomp_env(RENV_ASPECT_TRACE) != NULL;
    return on;
}

/* Hor+: whether stage k of a 2D batch reads src as a frame copy where the
 * batch lies (nv2a_aspect_place_copy): src is padded, a blit filled it in
 * this flip and nothing has drawn into it since, and the batch's texcoords
 * track its screen position (nv2a_aspect_uv_tracks). A copy from an
 * earlier flip is a still frame: BLiNX 2 copies its back buffer to the
 * display buffer at a few flips (764, 890, 892, 3959 and 3971 of its
 * golden), and that copy must not make a later HUD batch over it a lens. */
static int aspect_copy_tracks(const RenderTarget *src, const Stage *t, unsigned k,
                              const uint32_t (*prog)[4], uint32_t start, float (*c)[4],
                              const float *vf, unsigned nf4, const int *slot, uint32_t count,
                              uint32_t hash)
{
    float xy[NV2A_ASPECT_UV_MAX][2], uv[NV2A_ASPECT_UV_MAX][2];
    uint32_t n;
    int norm, ok;

    if (!src->copy || !src->pad || src->copy_flip != (uint32_t)s_presents)
        return 0;
    n = nv2a_aspect_uv_sample(prog, start, c, vf, nf4, slot, count, k, xy, uv,
                              NV2A_ASPECT_UV_MAX);
    /* A swizzled texture's coordinates are normalised, a linear one's are
     * texels (the NV2A's rect textures). */
    norm = nv2a_tex_size_from_format(t->color);
    ok = nv2a_aspect_uv_tracks((const float (*)[2])xy, (const float (*)[2])uv, n,
                               norm ? (float)t->width : 1.0f, norm ? (float)t->height : 1.0f,
                               src->w, src->h);
    if (aspect_trace() && n) {
        static int said;
        if (said++ < 64)
            fprintf(stderr, LOGP "aspect: vs %08X stage %u reads copy 0x%08X: %u verts,"
                    " xy (%.1f,%.1f) uv (%.3f,%.3f), xy (%.1f,%.1f) uv (%.3f,%.3f): %s\n",
                    hash, k, src->offset, count, xy[0][0], xy[0][1], uv[0][0], uv[0][1],
                    xy[n - 1][0], xy[n - 1][1], uv[n - 1][0], uv[n - 1][1],
                    ok ? "tracks, stretched" : "does not track");
    }
    return ok;
}

static void aspect_log(uint32_t hash, uint32_t count, uint32_t ntri, float lo_x,
                       float hi_x, const RenderTarget *rt, int place, int lift, int wide)
{
    static const char *const names[] = { "centre", "stretch", "cover" };
    /* Said once per program, stage-0 texture and placement: the XDK's
     * fixed function is one program for most of the 2D, so the texture
     * tells the pieces apart. An open-addressed set; full, it stops. */
    enum { SET = 4096 };
    static uint64_t set[SET];
    static unsigned used;
    const Stage *t0 = &s_stage[0];
    RenderTarget *src;
    uint64_t key = ((uint64_t)hash << 32 | (t0->mode ? t0->offset : 0)) * 0x9E3779B97F4A7C15ull
                   + (uint64_t)(place | lift << 4 | wide << 5) + 1u;
    unsigned h = (unsigned)(key >> 52) & (SET - 1u);
    while (set[h] && set[h] != key)
        h = (h + 1u) & (SET - 1u);
    if (set[h] || used >= SET / 2u)
        return;
    set[h] = key;
    used++;
    src = !t0->mode || t0->mode == 4 ? NULL : rt_texture(t0, (RenderTarget *)rt);
    fprintf(stderr, "[aspect.program] %08X flip %llu: %u vertices %u tris x %g..%g on %ux%u"
            " pad %u at 0x%08X pitch %u, stage 0 %s 0x%08X fmt %X %ux%u pitch %u -> %s%s%s\n", hash,
            (unsigned long long)s_presents, count, ntri, lo_x, hi_x, rt->w, rt->h, rt->pad,
            rt->addr, rt->pitch,
            !t0->mode || t0->mode == 4 ? "none" : src ? "rt" : "tex", t0->offset,
            t0->color, t0->width, t0->height, t0->pitch, names[place], lift ? " lift" : "",
            wide ? " wide" : "");
}

/* RECOMP_TRACE=aspect: one line per program and w the first time a flat
 * near layer of it is clipped to the guest columns (nv2a_aspect_flat). */
static void aspect_log_flat(uint32_t hash, uint32_t count, float w, const RenderTarget *rt)
{
    enum { SET = 256 };
    static uint64_t set[SET];
    static unsigned used;
    union { float f; uint32_t u; } wb = { w };
    uint64_t key = ((uint64_t)hash << 32 | wb.u) * 0x9E3779B97F4A7C15ull + 1u;
    unsigned h = (unsigned)(key >> 56) & (SET - 1u);
    while (set[h] && set[h] != key)
        h = (h + 1u) & (SET - 1u);
    if (set[h] || used >= SET / 2u)
        return;
    set[h] = key;
    used++;
    fprintf(stderr, "[aspect.flat] %08X flip %llu: %u vertices at w %g on %ux%u pad %u at"
            " 0x%08X\n", hash, (unsigned long long)s_presents, count, w, rt->w, rt->h, rt->pad,
            rt->addr);
}

/* RECOMP_ASPECT_TRACE: each program and target size the first time
 * nv2a_aspect_place_tex drops a guest picture: a full-width 2D pass that
 * samples a padded target and a picture, placed as a post pass, not a
 * movie (D3D11's aspect_log_post, same line). */
static void aspect_log_post(uint32_t hash, const RenderTarget *rt, int place)
{
    static const char *const names[] = { "centre", "stretch", "cover" };
    enum { SET = 256 };
    static uint64_t set[SET];
    static unsigned used;
    uint64_t key = ((uint64_t)hash << 32 | rt->w << 16 | rt->h) * 0x9E3779B97F4A7C15ull + 1u;
    unsigned h = (unsigned)(key >> 56) & (SET - 1u);
    while (set[h] && set[h] != key)
        h = (h + 1u) & (SET - 1u);
    if (set[h] || used >= SET / 2u)
        return;
    set[h] = key;
    used++;
    fprintf(stderr, "[aspect.post] %08X flip %llu: picture beside a padded target on %ux%u"
            " pad %u at 0x%08X -> %s\n", hash, (unsigned long long)s_presents, rt->w, rt->h,
            rt->pad, rt->addr, names[place]);
}

/* metal_passes, metal_gpu_blocks: the draw about to go into the open pass,
 * by nv2a_gpu_cat_draw's rule from what it samples (the smallest colour
 * target; a depth view is ZTEX), and the targets it samples as readers of
 * the passes that stored them (a depth view's reader is its conversion,
 * the self copy's its blit). The load heuristic's "the first draw covers
 * the whole colour target": no blend, every channel written, at most two
 * triangles, the surface clip the whole target. */
static int pt_draw(const RenderTarget *rt, const RenderTarget *const srcs[4], int depth_test,
                   int zuse, uint32_t nidx)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    uint32_t mw = 0, mh = 0, b[4];
    int zt = 0, k, full, cat;
    for (k = 0; k < 4; k++) {
        const RenderTarget *s = srcs[k];
        if (!s)
            continue;
        if (s >= s_zt_view && s < s_zt_view + DT_MAX) {
            zt = 1;
            continue;
        }
        if (s != &s_self && s_pt_track)
            pt_read(s->tex, 0, PTR_TEXTURE, s_pt_cur);
        if (!mw || s->w * s->h < mw * mh) {
            mw = s->w;
            mh = s->h;
        }
    }
    full = !g->blend_enable && g->color_keep == 0 && nidx <= 6
        && nv2a_draw_box(g, rt->w, rt->h, b) && !b[0] && !b[1] && b[2] == rt->w && b[3] == rt->h;
    cat = nv2a_gpu_cat_draw(rt->w, rt->h, depth_test, zt, mw, mh);
    pt_note_draw(cat, full);
    if (zuse) {
        PtPass *p = pt_cur_pass();
        if (p)
            p->zdraws++;   /* metal_depth_dontcare's guess reads it */
    }
    /* RECOMP_DEBUG=metal_flat_half: a whole-target flat draw gets half the
     * pixels (the caller sets the scissor); =alt, even flips only */
    return s_fh_mode && cat == NV2A_GCAT_FLAT && full && !(s_fh_mode == 2 && (s_presents & 1));
}

/* metal_passes_at: the shader of the pass's first draw (see PtPass). stex:
 * the batch's stage textures, s_black where a stage samples nothing; snap:
 * the combiners' function with render.scale above 1 (nv2a_snap_uv, key[36]). */
static void pt_shader(id<MTLFunction> ffn, id<MTLTexture> const stex[4], int snap)
{
    PtPass *p = pt_cur_pass();
    uintptr_t fp = (uintptr_t)ffn;
    int k;
    if (!p || p->sh_set || !p->draws)
        return;
    p->sh_set = 1;
    p->fs_id = fnv(&fp, sizeof fp);
    p->snap = (uint8_t)(snap != 0);
    for (k = 0; k < 4; k++)
        if (stex[k] && stex[k] != s_black) {
            p->nstages++;
            p->tw[k] = (uint32_t)[stex[k] width];
            p->th[k] = (uint32_t)[stex[k] height];
        }
}

static void metal_on_draw(const struct nv2a_pb_draw *d)
{
    /* The expanded list: at most three entries per index (a fan or a quad
     * strip), grown to the largest batch so far rather than sized to
     * NV_MAX_INDICES. */
    static uint32_t *idx;
    static uint32_t idx_cap;
    static float (*hxy)[2];        /* hor+: a 2D batch's vertex x, y (full width) */
    static uint32_t hxy_cap;
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    enum nv2a_topology topo;
    MTLPrimitiveType ptype;
    RenderTarget *rt;
    id<MTLRenderPipelineState> pso;
    id<MTLFunction> vfn, ffn;
    struct nv2a_vp_consts vc;
    MsPsConsts mpc;
    NSUInteger vb_off, ib_off, vbytes;
    uint32_t lo = 0xFFFF, hi = 0, count, nidx, i, k, nf4 = 0;
    int slot[NV2A_VS_MAX_INPUTS];
    float *vf;
    void *ip;
    int cull = 0, cull_all = 0;
    unsigned bump_src;
    int full_w = 0, guest_tex = 0, flat_rt = 0, two_d = 0, have_x = 0, tris = 0, flat = 0;
    int padded_rt = 0;
    int lift = 0, wide = 0;
    int vary_z = 0;     /* a program's 2D batch off one oPos.z (nv2a_aspect_gate) */
    int w_on = 0, wz_ez = 0, wz_fast = 0, wz_vs = 0;   /* RECOMP_W_EARLYZ (below) */
    int scan;
    int place = NV2A_PLACE_CENTRE;
    float lo_x = 0, hi_x = 0;   /* hor+: a 2D batch's x range, surface pixels */
    const RenderTarget *srcs[4] = { NULL, NULL, NULL, NULL };
    VsEntry *ve = NULL;
    /* Stencil only on a Z24S8 zeta surface, as on the CPU. */
    int use_s = v->stencil_enable && ((g->format >> 4) & 0xF) == 2;
    int use_z = v->depth_enable || use_s;
    id<MTLTexture> stex[4];

    if (!init())
        return;
    /* A mod's batch filter (nv2a_backend_common.h); D3D11 asks it at the
     * same point. Nothing without the enhancements layer. */
    if (nv2a_batch_skip(d, g, v))
        return;
    /* A batch that writes nothing is dropped, unless a visibility test is
     * counting it: a test's bounding draws typically mask every write. */
    if (g->color_keep == 0xFFFFFFFFu && !(v->depth_enable && v->depth_mask) && !use_s
            && !s_occ_q)
        return;
    if (!d->idx_count) {
        s_draws_skipped++;
        return;
    }
    @autoreleasepool {
        /* Vertex function and the float4 inputs it reads, in order. */
        if (d->program) {
            VsEntry *e = vs_cache();
            s_draws_prog += d->batches;
            if (!e) {
                s_draws_skipped++;
                return;
            }
            ve = e;
            vfn = e->fn;
            for (i = 0; i < NV2A_VS_MAX_INPUTS; i++)
                if (e->inputs & (1u << i))
                    slot[nf4++] = (int)i;
        } else {
            vfn = s_vs_pass;
            for (k = 0; k < VTX_F4; k++)
                slot[nf4++] = s_vtx_attr[k];
        }
        rt = surface();
        if (!rt) {
            s_draws_skipped++;
            return;
        }
        build_stages();
        stages_cube_rt();
        bump_src = nv2a_bump_sources(s_stage, v->rc_set && (v->rc_ctl & 0xFF),
                                     v->other_stage_input);
        {
            static int bump_trace = -1;
            if (bump_trace < 0)
                bump_trace = recomp_env(RENV_BUMP_TRACE) != NULL;
            /* Only a combiner draw samples a bump stage (the basic shader
             * reads every stage as plain 2D), so only those are logged. */
            if (bump_trace && v->rc_set && (v->rc_ctl & 0xFF))
                nv2a_bump_trace(v, s_stage, g->flips);
        }
        nv2a_ps_consts_fill(v, s_stage, &mpc.b);
        for (k = 0; k < 4; k++) {
            /* As the D3D11 MipLODBias: only a mipmapped stage has one. */
            const Stage *t = &s_stage[k];
            float b = t->levels > 1 && filter_mip(t->filter) ? lod_bias(t->filter) : 0.0f;
            mpc.tex_bias[k] = b > 15.99f ? 15.99f : b < -16.0f ? -16.0f : b;
        }
        /* The variant declares the buffer: no buffer, no variant. */
        s_fs_mag = nv2a_tex_mag_on() && d->program && mag_init();
        /* RECOMP_W_EARLYZ: the batch's one decision, as d3d11_on_draw's. A
         * buffer in the mode reverses every depth test; a W batch on it
         * takes the fast path (vertex E, stock fragment function, depth
         * clip on) when nv2a_w_earlyz_fast says so, else the W variant
         * writing E. */
        w_on = nv2a_w_depth_on(v, d->program != 0, v->rc_set && (v->rc_ctl & 0xFF));
        wz_ez = v->depth_enable && dt_earlyz_peek(rt->w, rt->h);
        wz_fast = wz_ez && w_on && nv2a_w_earlyz_fast(v, d->prim);
        s_snap_mask = snap_mask(&mpc.b);   /* metal_snap_lazy=2; final below */
        /* RECOMP_DEBUG=metal_nodiscard: the batch's one choice, which the
         * second fs_select below keeps */
        nd_decide(v, &mpc.b, w_on && !wz_fast);
        ffn = fs_select(v, w_on && !wz_fast);
        /* spec_ctl.w: 1 the W variant writes the depth; 2 the vertex z is
         * E, on the fast path or for a W batch on such a buffer whose W
         * variant fell back to fs_basic (it still has to store E) */
        wz_vs = s_fs_w ? 1 : (wz_ez && w_on) ? 2 : 0;
        for (i = 0; i < d->idx_count; i++) {
            if (d->idx[i] < lo) lo = d->idx[i];
            if (d->idx[i] > hi) hi = d->idx[i];
        }
        count = hi - lo + 1;
        if (d->idx_count * 3u > idx_cap) {
            uint32_t *grown = (uint32_t *)realloc(idx, (size_t)d->idx_count * 12u);
            if (!grown) {
                s_draws_skipped++;
                return;
            }
            idx = grown;
            idx_cap = d->idx_count * 3u;
        }
        nidx = nv2a_prim_to_list(d->prim, d->idx, d->idx_count, lo, idx, &topo);
        if (!nidx) {
            s_draws_skipped++;
            return;
        }
        ptype = topo == NV2A_TOPO_POINTS ? MTLPrimitiveTypePoint
              : topo == NV2A_TOPO_LINES  ? MTLPrimitiveTypeLine
              :                            MTLPrimitiveTypeTriangle;
        if (v->cull_enable) {     /* as the D3D11 backend's rasterizer_state */
            if (v->cull_face == 0x405)      cull = 1;   /* BACK */
            else if (v->cull_face == 0x404) cull = 2;   /* FRONT */
            else if (v->cull_face == 0x408) cull_all = 1;
        }
        if (cull_all && ptype == MTLPrimitiveTypeTriangle) {
            s_draws_culled++;
            return;
        }

        /* Rings first, the whole draw at once: an overflow flushes, which
         * ends the open pass. */
        vbytes = (NSUInteger)count * nf4 * 16u;
        if (vbytes < 16)
            vbytes = 16;
        if (!ring_reserve(vbytes, (NSUInteger)nidx * 4u, &vf, &vb_off, &ip, &ib_off)) {
            s_draws_skipped++;
            return;
        }
        /* Each input as float4, fetched as the CPU path does; a disabled
         * one (or one without data) is its SET_VERTEX_DATA value. Stride 0
         * is an array of one element (fetch_attr). */
        for (k = 0; k < nf4; k++) {
            const VertexAttr *a = &d->attr[slot[k]];
            int on = a->size != 0;
            for (i = 0; i < count; i++) {
                float *o = vf + ((size_t)i * nf4 + k) * 4;
                if (!on || !fetch(d, a, lo + i, o))
                    memcpy(o, v->inl[slot[k]], 16);
            }
        }
        memcpy(ip, idx, (size_t)nidx * 4u);

        /* Hor+: which batches are 2D, and their x range (nv2a_aspect_place
         * decides from it). A pass-through batch is 2D, its positions in
         * surface pixels; a program batch is 2D when its own output is
         * orthographic (nv2a_aspect_scan). With render.snap_2d above
         * render scale 1 the 2D test runs without hor+ too: a 2D batch's
         * stages sample at their 1x points (nv2a_snap_flat2d). */
        full_w = 0;
        scan = rt->pad || nv2a_snap_2d_on(rt->n);
        tris = rt->pad && topo == NV2A_TOPO_TRIANGLES;
        if (tris && count > hxy_cap) {
            float (*grown)[2] = (float (*)[2])realloc(hxy, (size_t)count * sizeof *hxy);
            if (grown) {
                hxy = grown;
                hxy_cap = count;
            } else {
                tris = 0;
            }
        }
        if (rt->pad && !d->program && nf4 > 0 && s_vtx_attr[0] == 0 && count) {
            lo_x = hi_x = vf[0];
            for (i = 0; i < count; i++) {
                float x = vf[(size_t)i * nf4 * 4u];
                if (x < lo_x) lo_x = x;
                if (x > hi_x) hi_x = x;
                if (tris) {
                    hxy[i][0] = x;
                    hxy[i][1] = vf[(size_t)i * nf4 * 4u + 1];
                }
            }
            two_d = have_x = 1;
        } else if (!rt->pad && scan && !d->program) {
            two_d = 1;
        } else if (scan && d->program && ve && count) {
            two_d = have_x = nv2a_aspect_scan(&ve->hp, (const uint32_t (*)[4])v->prog, v->start,
                                              (float (*)[4])(uintptr_t)v->c, vf, nf4, slot,
                                              count, NULL, &lo_x, &hi_x);
            vary_z = have_x && ve->hp.vary_z;
            /* Full-width bounds: every vertex, for the per-triangle test
             * (a 2D program is scanned every batch, so the gate passes). */
            if (have_x && tris && nv2a_aspect_full_width(lo_x, hi_x, rt->w))
                nv2a_aspect_scan(&ve->hp, (const uint32_t (*)[4])v->prog, v->start,
                                 (float (*)[4])(uintptr_t)v->c, vf, nf4, slot, count, hxy,
                                 &lo_x, &hi_x);
        }
        if (have_x)
            full_w = nv2a_aspect_full(lo_x, hi_x, tris ? (const float (*)[2])hxy : NULL,
                                      tris ? idx : NULL, nidx / 3u, rt->w, rt->h, &wide);

        /* Depth: the zeta surface's buffer, attached to this target's
         * passes from now on (a new pass if the open one lacks it). */
        if (use_z) {
            DepthTarget *dt = depth_target(rt->w, rt->h);
            if (dt && rt->dt != dt) {
                rt->dt = dt;
                if (s_enc_rt == rt) {
                    PT_WHY(PT_DEPTH);
                    end_encoder();
                }
            } else if (s_dc_pass && s_enc_rt == rt) {
                /* metal_depth_dontcare guessed no depth draw in this pass:
                 * it ends, and the next opens loading the depth (only ever
                 * set with the switch on) */
                pt_dc_miss("a depth or stencil draw in its pass");
                PT_WHY(PT_DEPTH);
                end_encoder();
            }
            /* A plain-z depth batch on a buffer in RECOMP_W_EARLYZ's mode
             * has no fix this frame: the mode goes off from the next depth
             * clear. */
            if (dt && wz_ez) {
                if (w_on)
                    nv2a_w_earlyz_note(nv2a_zmax(g->format));
                else
                    nv2a_w_earlyz_latch();
            }
        }
        /* Textures before the pass: a self copy is a blit between passes. */
        {
            int textured = 0;
            for (k = 0; k < 4; k++) {
                const Stage *t = &s_stage[k];
                RenderTarget *src;
                stex[k] = s_black;
                s_stage_pack[k][0] = s_stage_pack[k][1] = 1;
                s_stage_hash[k] = 0;
                if (!t->mode || t->mode == 4)
                    continue;
                textured = 1;
                /* A cube map's faces are decoded from guest memory: one
                 * over a live target is not bound as a cube (cube_rt).
                 * Only the combiners' shader declares one (s_fs_comb);
                 * fs_basic samples face 0. A dot stage that reads no
                 * texture keeps the black one: nothing to decode. */
                if (s_fs_comb && !nv2a_texmode_samples((int)nv2a_stage_ps_mode(t, k))) {
                    continue;
                } else if (s_fs_comb && nv2a_stage_cube_bind(t, nv2a_stage_ps_mode(t, k))) {
                    stex[k] = tex_bind(t, 1);
                    stage_from_cache(k);
                } else if ((src = rt_texture(t, rt)) != NULL) {
                    stex[k] = src->tex;
                    rt_read_mark(src);   /* sampled: a GPU read (use_sub) */
                    nv2a_ps_consts_rt(&mpc.b, k, t, src->w, src->h);
                    if (nv2a_rt_no_alpha(nv2a_tex_linear_twin(t->color)))
                        mpc.b.alpha_only[k] = 2;
                    srcs[k] = src;
                    if (!src->pad)
                        flat_rt = 1;
                    else
                        padded_rt = 1;
                    s_rtt_binds++;
                    nv2a_tex_dump_rt(t);
                } else {
                    stex[k] = tex_bind(t, 0);
                    stage_from_cache(k);
                    if (nv2a_aspect_picture(t->width, rt->w, bump_src >> k & 1))
                        guest_tex = 1;
                }
            }
            if (textured)
                s_draws_tex++;
        }
        /* RECOMP_DOF_SKIP: Kung Fu Chaos blurs a 320x240 copy of the scene and
         * repaints it over the 640x480 screen with the depth test on and depth
         * writes off, so the blur lands on what lies behind the fighters. A
         * draw into a larger target that samples such a copy (a render
         * target stage, never a depth view or the self copy) under that
         * state is that repaint: dropped, the far scene stays sharp. The
         * full-screen passes that read the same copies with the depth test
         * off (tint, brightness) are not touched. Nothing is encoded yet
         * for this draw, so there is no state to undo. */
        if (v->depth_enable && !v->depth_mask && !nv2a_dof_skip_size(rt->w, rt->h)) {
            int dof = 0;
            for (k = 0; k < 4; k++)
                if (srcs[k] && srcs[k] != &s_self
                    && !(srcs[k] >= s_zt_view && srcs[k] < s_zt_view + DT_MAX)
                    && nv2a_dof_skip_size(srcs[k]->w, srcs[k]->h))
                    dof = 1;
            if (dof) {
                s_dof_skipped++;
                return;
            }
        }
        /* Hor+: a perspective screen quad over padded targets is a
         * full-width pass too (nv2a_aspect_screen_quad) and marks the third
         * it lies in as holding a bar; one drawn without depth in such a
         * third is stretched as a piece of a letterbox band
         * (nv2a_aspect_band). */
        if (rt->pad && d->program && ve && topo == NV2A_TOPO_TRIANGLES && !two_d) {
            uint32_t flips = nv2a_pb_gpu_state()->flips;
            unsigned bars = nv2a_aspect_bars(rt->bar_flip, flips);
            int quad = nv2a_aspect_quad_ok(srcs[0] || srcs[1] || srcs[2] || srcs[3], flat_rt,
                                           guest_tex);
            int band = bars && !v->depth_enable && !flat_rt && !guest_tex;
            float qxy[NV2A_ASPECT_QUAD_MAX][2];
            if ((quad || band)
                && nv2a_aspect_screen_quad((const uint32_t (*)[4])v->prog, v->start,
                                           (float (*)[4])(uintptr_t)v->c, vf, nf4, slot, count,
                                           qxy, &lo_x, &hi_x)) {
                if (quad && nv2a_aspect_quad_span(lo_x, hi_x, rt->w)
                    && nv2a_aspect_full_tris((const float (*)[2])qxy, idx, nidx / 3u, rt->w)) {
                    two_d = full_w = 1;
                    nv2a_aspect_bar_mark(rt->bar_flip,
                                         nv2a_aspect_third((const float (*)[2])qxy, count, rt->h),
                                         flips);
                } else if (band && nv2a_aspect_band((const float (*)[2])qxy, count, rt->w, rt->h,
                                                    bars)) {
                    two_d = full_w = 1;
                }
            }
        }
        /* Hor+: a flat near layer drawn without depth keeps its place but
         * is clipped to the guest columns like the 2D (nv2a_aspect_flat). */
        flat = 0;
        if (rt->pad && d->program && ve && !two_d && !v->depth_enable && count) {
            float fw;
            flat = nv2a_aspect_flat((const uint32_t (*)[4])v->prog, v->start,
                                    (float (*)[4])(uintptr_t)v->c, vf, nf4, slot, count, &fw);
            if (flat && aspect_trace())
                aspect_log_flat(ve->hash, count, fw, rt);
        }
        /* A square target is a reflection, glow or shadow map, never a
         * movie: display.movie applies to the frame only. */
        place = rt->pad ? nv2a_aspect_place(two_d, full_w,
                                            nv2a_aspect_place_tex(guest_tex, padded_rt), flat_rt,
                                            rt->w == rt->h ? NV2A_MOVIE_BLACK
                                                           : nv2a_host_opts()->movie_mode)
                        : NV2A_PLACE_CENTRE;
        if (rt->pad && two_d && place == NV2A_PLACE_CENTRE) {
            int copy_rt = 0;
            for (k = 0; k < 4 && !copy_rt; k++)
                copy_rt = srcs[k]
                          && aspect_copy_tracks(srcs[k], &s_stage[k], k,
                                                d->program ? (const uint32_t (*)[4])v->prog
                                                           : NULL,
                                                v->start, (float (*)[4])(uintptr_t)v->c, vf,
                                                nf4, slot, count, ve ? ve->hash : 0);
            place = nv2a_aspect_place_copy(place, two_d, copy_rt);
        }
        if (nv2a_snap_2d_on(rt->n)) {
            /* render.snap_2d: a centred 2D batch samples guest textures at
             * its guest pixels' 1x points (nv2a_uv_snap), as setup_pixel. */
            unsigned rt_mask = 0;
            for (k = 0; k < 4; k++)
                if (srcs[k])
                    rt_mask |= 1u << k;
            nv2a_ps_consts_snap(&mpc.b, s_stage, rt_mask, rt->n,
                                nv2a_snap_flat2d(two_d && !vary_z, place));
        }
        if (place != NV2A_PLACE_CENTRE)
            s_draws_stretch++;
        /* A HUD mod clipped its elements in guest memory: lift the 4:3
         * clip of the 2D it moved (nv2a_aspect_hud_lift). */
        lift = rt->pad && nv2a_aspect_hud_lift(rt->w, rt->h, two_d, flat, full_w,
                                               have_x ? lo_x : 0.0f,
                                               have_x ? hi_x : (float)rt->w);
        if (two_d && rt->pad && aspect_trace())
            aspect_log(ve ? ve->hash : 0, count, nidx / 3u, lo_x, hi_x, rt, place, lift, wide);
        if (rt->pad && two_d && full_w && guest_tex && padded_rt && aspect_trace())
            aspect_log_post(ve ? ve->hash : 0, rt, place);
        /* A batch sampling a padded target the stock way reads it whole,
         * pads and all, which is what a stretched pass wants, short of a
         * square's overscan (nv2a_rt_stretch_read). Any other reads its
         * guest columns (nv2a_rt_guest_read; tex_scale.w,
         * rt_offset), on an unpadded target too: Burnout 3's 640x464 frame
         * has no pad, but the squares it samples do. */
        for (k = 0; k < 4; k++) {
            const RenderTarget *src = srcs[k];
            float sc, off;
            if (!src || !src->pad)
                continue;
            if (place != NV2A_PLACE_STRETCH && place != NV2A_PLACE_COVER)
                nv2a_rt_guest_read(src->w, src->pad, &sc, &off);
            /* Not a square's overscan: a stretch spans the ratio's pad. */
            else if (!nv2a_rt_stretch_read(src->w, src->h, src->pad, &sc, &off))
                continue;
            mpc.b.tex_scale[k][0] *= sc;
            mpc.b.tex_scale[k][3] = off;
        }
        nv2a_vp_consts_fill(v, g->format, rt->w, rt->h, rt->pad, &vc);
        nv2a_aspect_map(&vc, rt->w, rt->h, rt->pad, place);
        nv2a_vp_tie_fill(&vc, rt->hw, rt->hh);
        vc.u[7] = (uint32_t)wz_vs;   /* spec_ctl.w: the epilogue's W mode */
        if (wz_vs == 2) {
            /* RECOMP_W_EARLYZ's E = B - B w / zmax: B and B / zmax in the
             * two spare lanes, vp_scale.w and fog_param.w */
            float b = nv2a_w_earlyz_b();
            vc.c[3] = b;
            vc.c[11] = b / nv2a_zmax(g->format);
        }

        if (!s_enc || s_enc_rt != rt) {
            PT_WHY(s_enc ? PT_TARGET : PT_RESUME);
            /* metal_depth_dontcare: a pass opened for a draw using neither
             * depth nor stencil may leave them out of tile memory */
            s_dc_opener = !use_z && pt_on() && s_dc_mode;
            begin_encoder(rt, 0, NULL);
            s_dc_opener = 0;
        }
        /* metal_snap_lazy=2: .z is final only here, after the render-target
         * stages (nv2a_ps_consts_rt) and render.snap_2d; a changed mask picks
         * the function again. The texture loop and wz_vs read the first
         * call's s_fs_comb and s_fs_w, so a fallback (FS_MAX, a compile
         * failure) keeps the first function, whose extra calls on the
         * render-target stages are no-ops. */
        if (snap_lazy_now() == 2 && s_fs_comb && snap_mask(&mpc.b) != s_snap_mask) {
            int comb = s_fs_comb, fw = s_fs_w, snd = s_fs_sel_nd, sff = s_fs_sel_fast;
            id<MTLFunction> fn;
            s_snap_mask = snap_mask(&mpc.b);
            fn = fs_select(v, w_on && !wz_fast);
            if (s_fs_comb) {
                ffn = fn;
            } else {   /* the first function stays, and what it is */
                s_fs_sel_nd = snd;
                s_fs_sel_fast = sff;
            }
            s_fs_comb = comb;
            s_fs_w = fw;
        }
        pso = pipeline(vfn, ffn);
        if (!pso) {
            s_draws_skipped++;
            return;
        }
        if (s_pt_on > 0)
            nd_count(g->blend_enable != 0);   /* metal_nodiscard's line */
        [s_enc setRenderPipelineState:pso];
        if (s_enc_dt) {
            id<MTLDepthStencilState> dss = use_z ? depth_state(v->depth_enable, use_s, wz_ez)
                                                 : depth_state(0, 0, 0);
            [s_enc setDepthStencilState:dss];
            [s_enc setStencilReferenceValue:use_s ? (v->stencil_ref & 0xFFu) : 0u];
            if (use_z)
                s_draws_z++;
        }
        {
            MTLViewport vp = {0.0, 0.0, (double)rt->hw, (double)rt->hh, 0.0, 1.0};
            MTLScissorRect sr;
            uint32_t b[4], hb[4];
            [s_enc setViewport:vp];
            /* The surface clip, as the hardware scissors (nv2a_draw_box). */
            if (!nv2a_draw_box(g, rt->w, rt->h, b)) {
                s_draws_skipped++;
                return;
            }
            nv2a_aspect_box(b, rt->w, rt->h, rt->pad, rt->n, place, two_d || flat, lift, hb);
            sr.x = hb[0];
            sr.y = hb[1];
            sr.width = hb[2] - hb[0];
            sr.height = hb[3] - hb[1];
            [s_enc setScissorRect:sr];
            s_draw_sr = sr;
        }
        [s_enc setFrontFacingWinding:v->front_face != 0x900
                                     ? MTLWindingCounterClockwise : MTLWindingClockwise];
        [s_enc setCullMode:cull == 1 ? MTLCullModeBack
                         : cull == 2 ? MTLCullModeFront : MTLCullModeNone];
        {
            /* A W batch: the depth tested and stored is the pixel's w (the W
             * variant writes it), so the program's z means nothing here, and
             * clipping on it threw away whatever it put past 1 (Kung Fu
             * Chaos's street under the flying saucer, as on D3D11). xemu
             * turns the GPU's depth clipping off and clips or clamps the
             * depth per pixel instead; the W shader's clip range is that.
             * Clamp rather than clip, per encoder as the bias (a new one
             * starts at clip). RECOMP_METAL_W_ZCLIP=1: clip as before (A/B). */
            static int keep = -1;
            static uint64_t clip_enc;
            static int clip_set;
            int clamp;
            if (keep < 0) {
                const char *e = getenv("RECOMP_METAL_W_ZCLIP");
                keep = e && e[0] == '1';
            }
            /* RECOMP_W_EARLYZ's vertex E (wz_vs 2) keeps the clip: E outside
             * [0, 1] is w past zmax or nearer than about B. */
            clamp = wz_vs == 1 && !keep;
            if (clip_enc != s_enc_serial) {
                clip_enc = s_enc_serial;
                clip_set = 0;
            }
            if (clamp != clip_set) {
                [s_enc setDepthClipMode:clamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip];
                clip_set = clamp;
            }
        }
        {
            /* Polygon offset (nv2a_depth_bias); a W batch takes it in its
             * shader instead, the hardware bias not reaching a written
             * depth. A new encoder starts at 0, so a batch without offset
             * sets nothing. Nor does RECOMP_W_EARLYZ's vertex E (wz_vs 2):
             * the fast path has no offset, and a fallback W batch's is in w
             * units, which a bias on E cannot give. */
            static uint64_t bias_enc;
            static float bias_set[2];
            float zb = 0.0f, zs = 0.0f;
            if (wz_vs == 0)
                nv2a_depth_bias(v, d->prim, g->format, nv2a_host_render_scale(), &zb, &zs);
            if (bias_enc != s_enc_serial) {
                bias_enc = s_enc_serial;
                bias_set[0] = bias_set[1] = 0.0f;
            }
            if (zb != bias_set[0] || zs != bias_set[1]) {
                [s_enc setDepthBias:zb slopeScale:zs clamp:0.0f];
                bias_set[0] = zb;
                bias_set[1] = zs;
            }
        }
        {
            float bf[4];
            (void)nv2a_blend_constant(g->blend_enable, g->blend_sfactor,
                                      g->blend_dfactor, g->blend_color, bf, &(int){0});
            [s_enc setBlendColorRed:bf[0] green:bf[1] blue:bf[2] alpha:bf[3]];
        }
        [s_enc setVertexBuffer:s_vb offset:vb_off atIndex:0];
        s_ring_bound = 1;   /* s_cmd's commit is this copy's last_sub */
        if (d->program) {
            /* c[192] is 3 KB: under setVertexBytes' 4 KB. An encoder keeps
             * its bindings from draw to draw, so while it is the same
             * encoder, buffer 1 still holds c[] and the generation says
             * nothing wrote c[] since (nv2a_pb_vsh_consts_gen), the 3 KB
             * copy is skipped: most batches of a busy frame. */
            uint32_t cg = nv2a_pb_vsh_consts_gen();
            if (!memo_on() || !s_vc1.ok || s_vc1.enc != s_enc_serial || s_vc1.gen != cg) {
                [s_enc setVertexBytes:v->c length:sizeof v->c atIndex:1];
                s_vc1.enc = s_enc_serial;
                s_vc1.gen = cg;
                s_vc1.ok = 1;
                s_c_uploads++;
            }
            [s_enc setVertexBytes:&vc length:sizeof vc atIndex:2];
        } else {
            [s_enc setVertexBytes:&vc length:sizeof vc atIndex:1];
            s_vc1.ok = 0;              /* buffer 1 is vc now, not c[] */
        }
        [s_enc setFragmentBytes:&mpc length:sizeof mpc atIndex:0];
        if (s_fs_w) {
            struct nv2a_wdepth_consts wc;
            nv2a_wdepth_consts_fill(v, d->prim, g->format, nv2a_host_render_scale(), &wc);
            if (wz_ez) {
                /* RECOMP_W_EARLYZ, off the fast path: the same E (d3d8_msl.h) */
                wc.w1[2] = nv2a_w_earlyz_b();
                wc.w1[3] = 1.0f;
            }
            [s_enc setFragmentBytes:&wc length:sizeof wc atIndex:1];
        }
        for (k = 0; k < 4; k++) {
            [s_enc setFragmentTexture:stex[k] atIndex:k];
            [s_enc setFragmentSamplerState:sampler(&s_stage[k], bump_src >> k & 1,
                                                   s_stage_pack[k][0], s_stage_pack[k][1])
                                     atIndex:k];
        }
        /* A read of this batch's inputs only: the dump must not move a frame. */
        if (nv2a_tex_dump_on()) {
            int dump_2d = !d->program;
            if (d->program && nv2a_tex_dump_draw_open(s_stage_hash))
                dump_2d = nv2a_aspect_batch_2d((const uint32_t (*)[4])v->prog, v->start,
                                               (float (*)[4])(uintptr_t)v->c, vf, nf4, slot,
                                               count, v->depth_enable);
            nv2a_tex_dump_draw(s_stage_hash, bump_src, dump_2d);
        }
        if (s_fs_mag && s_fs_comb)
            mag_draw();
        else if (nv2a_tex_mag_on() && !d->program && topo == NV2A_TOPO_TRIANGLES)
            mag_pretransformed(vf, nf4, slot, idx, nidx);
        if (pt_on()) {
            int half = pt_draw(rt, srcs, v->depth_enable && s_enc_dt, use_z, nidx);   /* metal_passes */
            pt_shader(ffn, stex, s_fs_comb && nv2a_host_render_scale() > 1);
            if (half) {
                /* metal_flat_half: this draw only; the next sets its own scissor */
                MTLScissorRect hs = s_draw_sr;
                NSUInteger lim = rt->hw / 2u;
                if (hs.x >= lim)
                    hs.x = 0;
                if (hs.x + hs.width > lim)
                    hs.width = lim - hs.x > 0 ? lim - hs.x : 1;
                [s_enc setScissorRect:hs];
                s_pts.fh_draws++;
            }
        }
        [s_enc drawIndexedPrimitives:ptype indexCount:nidx indexType:MTLIndexTypeUInt32
                         indexBuffer:s_ib indexBufferOffset:ib_off];
        /* After the draw: a later batch sampling this depth converts again,
         * this one converted what came before it (zt_texture). */
        rt->zseq = ++s_zseq;
        if (s_enc_dt && ((v->depth_enable && v->depth_mask)
                         || (use_s && (v->stencil_wmask & 0xFFu))))
            s_enc_dt->wseq = ++s_zseq;
        s_draws += d->batches;
    }
}

/* ---- flip ---------------------------------------------------------------------- */

/* The render target for the surface the walker picked: the one at its
 * offset and pitch (a field buffer and a frame can share an offset),
 * whatever its size. A larger one covers a smaller frame; a smaller
 * one (a buffer drawn only below the display mode's frame) is grown by
 * present_extent. NULL when this backend never drew there; the guest's own
 * bytes are presented then. */
static RenderTarget *present_target(uint32_t surface_offset)
{
    const struct nv2a_pb_present *p = nv2a_pb_present_state();
    int i;

    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex && surface_offset && rt->offset == surface_offset
                && (!p->offset || rt->pitch == p->pitch)) {
            rt->presented = 1;
            return rt;
        }
    }
    return NULL;
}

/* The part of rt a flip shows, in guest pixels: the walker's present extent
 * when it names this surface (D3D11's present_extent). A target grown for a
 * larger clip (rt_grow) can be bigger than the frame; the rest of it is not
 * the frame. */
static void present_extent(RenderTarget *rt, uint32_t *w, uint32_t *h)
{
    const struct nv2a_pb_present *p = nv2a_pb_present_state();
    /* The walker publishes the display mode's frame, which a title that only
     * ever drew this buffer at a smaller clip has no target that big for:
     * grow it (as a larger clip would) rather than show less than the CPU
     * path, which reads the whole frame from guest memory. A failed grow
     * shows what there is. */
    if (p->offset && p->offset == rt->offset && p->pitch == rt->pitch && p->w && p->h
            && (p->w > rt->w || p->h > rt->h))
        rt_grow(rt, rt->w > p->w ? rt->w : p->w, rt->h > p->h ? rt->h : p->h);
    *w = rt->w;
    *h = rt->h;
    if (p->offset && p->offset == rt->offset && p->pitch == rt->pitch && p->w && p->h) {
        if (p->w < *w) *w = p->w;
        if (p->h < *h) *h = p->h;
    }
}

static uint32_t s_shown_w, s_shown_h;   /* the last flip's present extent */

/* One row of rt's BGRA8 pixels into a 16-bit guest row: in the target's
 * X1R5G5B5 or R5G6B5 format with its constant bits (nv2a_surf_pack_row),
 * R5G6B5 for any other format seen at a 16-bit pitch. */
static void pack16(const RenderTarget *rt, const uint32_t *row, uint16_t *o, uint32_t w)
{
    uint32_t f = rt->fmt & 0xFu, x;

    if (f == 0x1 || f == 0x2 || f == 0x3) {
        nv2a_surf_pack_row(f, (const uint8_t *)row, w, (uint8_t *)o);
        return;
    }
    for (x = 0; x < w; x++) {
        uint32_t c = row[x];
        o[x] = (uint16_t)(((c >> 8) & 0xF800) | ((c >> 5) & 0x07E0) | ((c >> 3) & 0x001F));
    }
}

/* The X8R8G8B8 and X1A7R8G8B8 formats' constant alpha bits over the w x h
 * 32-bit guest pixels just written at dst (nv2a_surf_pack_row, in place):
 * the host alpha is what the combiners made, the guest's what the surface
 * stores. */
static void fix32(const RenderTarget *rt, uint8_t *dst, uint32_t pitch, uint32_t h)
{
    uint32_t f = rt->fmt & 0xFu, y;

    if (f < 0x4 || f > 0x7)
        return;
    for (y = 0; y < h; y++)
        nv2a_surf_pack_row(f, dst + (size_t)y * pitch, rt->w, dst + (size_t)y * pitch);
}

/* rt's pixels into the guest surface at `addr`, `pitch` bytes a row: 32-bit
 * as they are (BGRA8 is X8R8G8B8's byte order) but for the constant alpha
 * bits (fix32), 16-bit packed by pack16. Runs after the wait, on the
 * executor's thread. */
static void write_back(RenderTarget *rt, uint32_t addr, uint32_t pitch)
{
    uint8_t *dst = (uint8_t *)xbox_GetMemoryOffset() + addr;
    uint32_t bpp = pitch / rt->w, w = rt->w, y;

    if (bpp >= 4) {
        rt_get(rt, dst, pitch, MTLRegionMake2D(0, 0, w, rt->h));
        fix32(rt, dst, pitch, rt->h);
    } else if (bpp == 2) {
        static uint32_t *row;
        static uint32_t row_cap;
        if (row_cap < w) {
            uint32_t *n = (uint32_t *)realloc(row, (size_t)w * 4);
            if (!n)
                return;
            row = n;
            row_cap = w;
        }
        for (y = 0; y < rt->h; y++) {
            uint16_t *o = (uint16_t *)(dst + (size_t)y * pitch);
            rt_get(rt, row, w * 4, MTLRegionMake2D(0, y, w, 1));
            pack16(rt, row, o, w);
        }
    } else {
        return;
    }
    tex_cache_stale(addr, pitch * rt->h);
    s_writebacks++;
}

/* A scaled rt (render.scale > 1) read back once at the flip, hw x hh BGRA8
 * tightly packed: the window shows it as it is and the write-back boxes it
 * down to guest size, so one getBytes serves both. NULL on no memory. */
static const uint8_t *host_readback(RenderTarget *rt)
{
    static uint8_t *px;
    static size_t cap;
    size_t need = (size_t)rt->hw * rt->hh * 4u;
    if (cap < need) {
        uint8_t *n = (uint8_t *)realloc(px, need);
        if (!n)
            return NULL;
        px = n;
        cap = need;
    }
    rt_get(rt, px, rt->hw * 4u, MTLRegionMake2D(0, 0, rt->hw, rt->hh));
    return px;
}

/* write_back for a scaled rt: the host pixels (host_readback) averaged over
 * each n x n block, so the guest -- RECOMP_FB_DUMP, fb_dump_at, the title --
 * sees a w x h frame at its pitch, packed as write_back packs. */
static void write_back_scaled(RenderTarget *rt, const uint8_t *px, uint32_t addr,
                              uint32_t pitch)
{
    uint8_t *dst = (uint8_t *)xbox_GetMemoryOffset() + addr;
    uint32_t bpp = pitch / rt->w, w = rt->w, n = rt->n, y;

    px += (size_t)rt->pad * n * 4u;   /* hor+: the guest columns, centred */
    if (bpp >= 4) {
        nv2a_downscale_box32(px, rt->hw * 4u, w, rt->h, n, dst, pitch);
        fix32(rt, dst, pitch, rt->h);
    } else if (bpp == 2) {
        static uint32_t *row;
        static uint32_t row_cap;
        if (row_cap < w) {
            uint32_t *nr = (uint32_t *)realloc(row, (size_t)w * 4);
            if (!nr)
                return;
            row = nr;
            row_cap = w;
        }
        for (y = 0; y < rt->h; y++) {
            uint16_t *o = (uint16_t *)(dst + (size_t)y * pitch);
            nv2a_downscale_box32(px + (size_t)y * n * rt->hw * 4u, rt->hw * 4u, w, 1, n,
                                 (uint8_t *)row, w * 4u);
            pack16(rt, row, o, w);
        }
    } else {
        return;
    }
    tex_cache_stale(addr, pitch * rt->h);
    s_writebacks++;
}

/* ---- write-back on demand ---------------------------------------------------
 *
 * Guest memory under a render target is written only when something is about
 * to read it: the executor's dumps (sync_guest), a texture decode over it
 * (tex_bind), the eviction of a target (surface). The title itself
 * was never seen reading its framebuffer (metal-zero-copy-present, design),
 * and D3D11 never writes back at all. metal_writeback=always, or
 * metal_no_rtt, writes the present surface at every flip as before. */
static int s_guard_always;   /* metal_fb_guard saw a title read */

static int wb_always(void)
{
    const char *e = recomp_env(RENV_METAL_WRITEBACK);
    return (e && !strcasecmp(e, "always")) || recomp_env(RENV_METAL_NO_RTT) != NULL
           || s_guard_always;
}

/* rt into guest memory now, at `pitch`, if drawn since the last write-back.
 * Waits for the GPU; the ring positions survive the wait, since a draw may
 * have placed its vertices already (tex_bind runs inside on_draw). */
/* Managed storage (a discrete GPU): queue the copy back of the GPU's
 * contents, which the next wait completes. Nothing on unified memory. */
static void rt_managed_sync(RenderTarget *rt)
{
    id<MTLBlitCommandEncoder> b;

    if (s_unified && !s_rtp)
        return;
    if (s_rtp && !stage_fit((size_t)rt->hw * rt->hh * 4u))
        return;   /* s_stage_tex NULL: rt_get skips the read */
    end_encoder();   /* for the caller's reason (metal_passes): readback, flip, host-pixels */
    ensure_cmd();
    b = pt_on() ? pt_blit(PT_SYNC, NV2A_GCAT_OTHER) : [s_cmd blitCommandEncoder];
    if (s_rtp) {
        /* metal_rt_private: the whole texture into s_rtp_buf, after every pass
         * already encoded; the caller's wait completes it, then rt_get. */
        [b copyFromTexture:rt->tex sourceSlice:0 sourceLevel:0
              sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(rt->hw, rt->hh, 1)
                  toBuffer:s_rtp_buf destinationOffset:0
    destinationBytesPerRow:(NSUInteger)rt->hw * 4u
  destinationBytesPerImage:(NSUInteger)rt->hw * 4u * rt->hh];
        [b endEncoding];
        s_stage_tex = (const void *)rt->tex;
        s_stage_reads++;
        rt->spend = 1;   /* the CPU's copy: rt_read_ready waits for it too */
        s_rtp_pend = 1;  /* it writes s_rtp_buf: rt_seed's fill waits for it */
        return;
    }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [b synchronizeResource:rt->tex];
#pragma clang diagnostic pop
    [b endEncoding];
    rt->spend = 1;   /* the CPU's copy: rt_read_ready waits for it too */
}

/* Before a CPU read of rt's bytes (getBytes, or metal_rt_private's staging
 * copy): the submissions that make them visible, its last write (a pass, a
 * blit) and the copy for the CPU rt_managed_sync queued (managed storage,
 * metal_rt_private). The open s_cmd is committed first when it holds either
 * (`why` tags its pass end, metal_passes), not otherwise; then exactly those
 * submissions are waited for, not the frame. 0 when either ended in Error:
 * the caller publishes nothing (counted), since the bytes are not that
 * write's. KFC-1720 H1b step 3a. */
static int rt_read_ready(RenderTarget *rt, int why)
{
    if (rt->wpend || rt->spend) {
        PT_WHY(why);
        flush(0);
    }
    sub_wait(rt->last_write_sub, SUBW_READ);   /* sub_observe marks wbad */
    if (rt->sync_sub)
        sub_wait(rt->sync_sub, SUBW_READ);
    if (rt->wbad || (rt->sync_sub && sub_poll(rt->sync_sub) < 0)) {
        static int said;
        s_rb_skipped++;
        if (!said++)
            fprintf(stderr, LOGP "surface 0x%08X: its last write (submission %llu) ended in"
                    " error; the CPU read of it is skipped (later ones counted)\n",
                    rt->offset, (unsigned long long)rt->last_write_sub);
        return 0;
    }
    return 1;
}

static void rt_sync(RenderTarget *rt, uint32_t pitch)
{
    NSUInteger vb = s_vb_pos, ib = s_ib_pos;
    const uint8_t *px;

    if (!rt->tex || !rt->dirty || !pitch || pitch < rt->w * 2u)
        return;
    guard_open();
    @autoreleasepool {
        int ok;
        if (pt_on() && s_pt_track)
            pt_read(rt->tex, 0, PTR_READBACK, -1);   /* metal_passes */
        PT_WHY(PT_SYNC);
        rt_managed_sync(rt);
        /* Commits s_cmd only when it writes rt, and waits for that write's
         * submission; a commit ends the pass, the rings keep their place. */
        ok = rt_read_ready(rt, PT_SYNC);
        s_vb_pos = vb;
        s_ib_pos = ib;
        if (!ok)
            return;   /* still dirty: the guest keeps its bytes */
        if (rt->hw == rt->w)
            write_back(rt, rt->addr, pitch);
        else if ((px = host_readback(rt)) != NULL)
            write_back_scaled(rt, px, rt->addr, pitch);
        /* The memory holds these pixels now, as of their last write. */
        dt_mem_written(rt->addr, pitch * rt->h, rt->zseq);
        rt->dirty = 0;
        rt_own_reset(rt);
    }
}

/* nv2a_pb_backend.sync_guest: the target present_target would pick for this
 * surface (offset, covering guest w x h), else the most recently used one at its
 * address (a surface dump of a target drawn at another clip). */
static void metal_sync_guest(uint32_t offset, uint32_t pitch, uint32_t w, uint32_t h)
{
    RenderTarget *rt = NULL;
    uint32_t addr = nv2a_pb_dma_resolve(offset) & 0x07FFFFFFu;
    int i;

    guard_open();
    if (s_failed)
        return;
    for (i = 0; i < RT_MAX && !rt; i++)
        if (s_rt[i].tex && s_rt[i].offset == offset && s_rt[i].w >= w && s_rt[i].h >= h)
            rt = &s_rt[i];
    if (!rt) {
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *c = &s_rt[i];
            if (c->tex && (c->addr & 0x07FFFFFFu) == addr
                    && (!rt || c->last_use > rt->last_use))
                rt = c;
        }
    }
    if (rt && !rt_drop_if_stale(rt, "sync"))
        rt_sync(rt, pitch);
}

/* For tests/nv2a_backend_smoke: write-backs so far, and the host pixels
 * (hw x hh BGRA8, tightly packed) of the target at a colour offset, read
 * back now, so a test can box-filter them itself. */
uint64_t nv2a_pb_metal_writebacks(void) { return s_writebacks; }

/* For tests/nv2a_backend_smoke: blits copied on the GPU and in guest memory,
 * and the live targets. */
void nv2a_pb_metal_blit_stats(uint64_t *gpu, uint64_t *guest, unsigned *targets)
{
    int i;
    *gpu = s_blits_gpu;
    *guest = s_blits_guest;
    *targets = 0;
    for (i = 0; i < RT_MAX; i++)
        *targets += s_rt[i].tex != nil;
}

/* For tests/nv2a_backend_smoke: the target at a colour offset as the Hor+
 * copy rule sees it: -1 none, 0 not a blit copy (or drawn into since), 1
 * a copy a blit filled this flip, 2 one from an earlier flip. */
int nv2a_pb_metal_copy_state(uint32_t offset)
{
    int i;
    for (i = 0; i < RT_MAX; i++)
        if (s_rt[i].tex && s_rt[i].offset == offset)
            return !s_rt[i].copy ? 0 : s_rt[i].copy_flip == (uint32_t)s_presents ? 1 : 2;
    return -1;
}

/* For tests/nv2a_backend_smoke: the MSL version the runtime shaders
 * compile at (major << 16 | minor), as the compile options report it: with
 * metal_msl_version=0 that is the OS default (0x40001 on macOS 27). 0 before
 * init and before macOS 13, where the options have no version. */
unsigned nv2a_pb_metal_msl_version(void)
{
    if (!s_dev)
        return 0;
    if (@available(macOS 13.0, iOS 16.0, *))
        return (unsigned)metal_compile_options().languageVersion;
    return 0;
}

/* For tests/nv2a_backend_smoke: the size of the target at a colour offset
 * and the part of it the last flip showed, 0 when there is none (the shape
 * of nv2a_pb_d3d11_shown). */
int nv2a_pb_metal_shown(uint32_t offset, uint32_t *tw, uint32_t *th,
                        uint32_t *sw, uint32_t *sh)
{
    int i;
    for (i = 0; i < RT_MAX; i++)
        if (s_rt[i].tex && s_rt[i].offset == offset) {
            *tw = s_rt[i].w;
            *th = s_rt[i].h;
            *sw = s_shown_w;
            *sh = s_shown_h;
            return 1;
        }
    return 0;
}
const uint8_t *nv2a_pb_metal_host_pixels(uint32_t offset, uint32_t *hw, uint32_t *hh)
{
    int i;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex && rt->offset == offset) {
            PT_WHY(PT_HOST_PIXELS);
            rt_managed_sync(rt);
            if (!rt_read_ready(rt, PT_HOST_PIXELS))
                return NULL;
            *hw = rt->hw;
            *hh = rt->hh;
            return host_readback(rt);
        }
    }
    return NULL;
}

/* ---- metal_fb_guard ----------------------------------------------------------
 *
 * After each lazy flip the present surface's pages are PROT_NONE; the first
 * access from any thread is recorded by the handler, which opens the pages
 * again and lets it proceed. A title READ then switches the run to
 * metal_writeback=always (that read saw the last write-back, or the seed); a
 * write needs nothing from the GPU (the allocator's bzero on re-allocation is
 * one) and is logged once. Debug only: PROT_NONE turns a kernel-side access to
 * those pages (read() into re-allocated memory) into EFAULT. */
static struct {
    uintptr_t plo, phi;          /* the page span made PROT_NONE */
    uint32_t addr;
    volatile int armed, hit;
    uintptr_t far, pc;
    uint32_t write;
    char th[24];
    struct sigaction prev_bus, prev_segv;
} s_guard;

static void guard_handler(int sig, siginfo_t *si, void *ucv)
{
    uintptr_t f = (uintptr_t)si->si_addr;
    struct sigaction *prev;

    if (s_guard.armed && f >= s_guard.plo && f < s_guard.phi) {
        uint32_t esr = 0;
        uintptr_t pc = 0;
#if defined(__APPLE__) && defined(__aarch64__)
        {
            ucontext_t *uc = (ucontext_t *)ucv;
            esr = uc->uc_mcontext->__es.__esr;
            pc = (uintptr_t)uc->uc_mcontext->__ss.__pc;
        }
#endif
        if (!s_guard.hit) {
            s_guard.far = f;
            s_guard.pc = pc;
            s_guard.write = (esr >> 6) & 1u;   /* ESR WnR */
            s_guard.th[0] = 0;
            pthread_getname_np(pthread_self(), s_guard.th, sizeof s_guard.th);
            s_guard.hit = 1;
        }
        mprotect((void *)s_guard.plo, s_guard.phi - s_guard.plo, PROT_READ | PROT_WRITE);
        s_guard.armed = 0;
        return;   /* re-executes the access */
    }
    prev = sig == SIGBUS ? &s_guard.prev_bus : &s_guard.prev_segv;
    if (prev->sa_flags & SA_SIGINFO) {
        if (prev->sa_sigaction) {
            prev->sa_sigaction(sig, si, ucv);
            return;
        }
    } else if (prev->sa_handler != SIG_DFL && prev->sa_handler != SIG_IGN) {
        prev->sa_handler(sig);
        return;
    }
    signal(sig, SIG_DFL);
}

/* Open the pages before this thread reads or writes them itself. */
static void guard_open(void)
{
    if (s_guard.armed) {
        s_guard.armed = 0;
        mprotect((void *)s_guard.plo, s_guard.phi - s_guard.plo, PROT_READ | PROT_WRITE);
    }
}

/* At the flip: report what the last span saw, then close the new one. */
static void guard_flip(uint32_t addr, uint32_t len)
{
    static int on = -1, said_write;
    uintptr_t pg = (uintptr_t)getpagesize(), lo;

    if (on < 0) {
        on = recomp_env(RENV_METAL_FB_GUARD) != NULL;
        if (on) {
            /* Installed now, after the title's own fault handlers, and
             * chained to them for every fault outside the span. */
            struct sigaction sa;
            memset(&sa, 0, sizeof sa);
            sa.sa_sigaction = guard_handler;
            sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
            sigemptyset(&sa.sa_mask);
            sigaction(SIGBUS, &sa, &s_guard.prev_bus);
            sigaction(SIGSEGV, &sa, &s_guard.prev_segv);
            fprintf(stderr, LOGP "fb_guard: armed; a title read of the present surface"
                    " switches to metal_writeback=always\n");
        }
    }
    if (!on)
        return;
    guard_open();
    if (s_guard.hit) {
        Dl_info di;
        const char *sym = dladdr((void *)s_guard.pc, &di) && di.dli_sname ? di.dli_sname : "?";
        unsigned long off = sym[0] != '?' ? (unsigned long)(s_guard.pc - (uintptr_t)di.dli_saddr)
                                          : (unsigned long)s_guard.pc;
        uint32_t ga = (uint32_t)(s_guard.far - (uintptr_t)xbox_GetMemoryOffset());
        if (!s_guard.write) {
            fprintf(stderr, LOGP "fb_guard: guest READ of present surface 0x%08X (at 0x%08X)"
                    " by '%s' pc %s+0x%lX; switching to metal_writeback=always\n",
                    s_guard.addr, ga, s_guard.th, sym, off);
            s_guard_always = 1;
        } else if (!said_write++) {
            fprintf(stderr, LOGP "fb_guard: guest WRITE of present surface 0x%08X (at 0x%08X)"
                    " by '%s' pc %s+0x%lX; ignored\n", s_guard.addr, ga, s_guard.th, sym, off);
        }
        s_guard.hit = 0;
    }
    if (s_guard_always || !addr || !len)
        return;
    lo = (uintptr_t)xbox_GetMemoryOffset() + addr;
    s_guard.addr = addr;
    s_guard.plo = lo & ~(pg - 1);
    s_guard.phi = (lo + len + pg - 1) & ~(pg - 1);
    __atomic_store_n(&s_guard.armed, 1, __ATOMIC_SEQ_CST);
    if (mprotect((void *)s_guard.plo, s_guard.phi - s_guard.plo, PROT_NONE) != 0)
        s_guard.armed = 0;
}

/* ---- Visibility tests: Metal visibility-result counting ---------------------
 *
 * The shared tracker (nv2a_occ, nv2a_backend_common.h) as the D3D11 backend
 * drives it: BeginVisibilityTest is CLEAR_REPORT_VALUE +
 * SET_ZPASS_PIXEL_COUNT_ENABLE(1), EndVisibilityTest is ENABLE(0) +
 * GET_REPORT; each stretch with counting on is one query, and a report sums
 * the stretches since the last clear, completing later (poll at every batch
 * and flip, with a flush on the ack thread's idle tick) or as visible after
 * 250 ms. The count is samples passing depth and stencil, alpha-test discard
 * applied.
 *
 * A query here is a list of slots in s_vis, one per render pass it spans:
 * Metal resets, at the start of a pass, every offset that pass names (even
 * in setVisibilityResultMode:Disabled), so a slot never outlives its pass
 * and a stretch over a pass change takes a new one. Slots accumulate within
 * the pass. A result is in once every slot's command buffer has completed,
 * so GET_REPORT commits the open command buffer: a count held to the flip
 * lands a frame later than D3D11's, and light shafts whose brightness
 * comes from the count come out too bright.
 *
 * Each slot, from its allocation to its release, is written by one
 * submission (KFC-1720 H1b step 3a): s_vis_sub[slot], 0 while it is in the
 * open s_cmd, numbered when flush commits it (occ_committed). The allocation
 * is the slot's generation: a slot is allocated again only once released,
 * and released (here, on the backend's thread) only once that submission is
 * observed done, so the CPU's zeroing never races a GPU write; one freed
 * before that goes on s_retire, and occ_retire releases it later, oldest
 * first. A query's result is in when the submissions of all its slots are
 * done (sub_poll), whatever the passes and ring cuts it spans; one that ended
 * in Error answers visible, as an overflow does. The completed handlers only
 * record completion (sub_commit); nothing here runs in them.
 *
 * At the end of each walk, before a semaphore release that follows a
 * report (both on_poll) and at the flip after its wait, nv2a_occ_settle
 * waits for the results while the frame's RECOMP_DEBUG=occ_settle_us budget
 * lasts (8000 by default; a query still in the open s_cmd is committed by
 * occ_result's flush), then settles whatever is still pending with the
 * report address's last count (visible the first time), so the title never
 * reads a test as incomplete (nv2a_backend_common.h). At stock
 * almost nothing is left to settle: the reports' command buffers were
 * committed at GET_REPORT. occ_delay_us holds every result back (a test
 * hook for a slow GPU).
 *
 * RECOMP_METAL_OCC=sync waits at GET_REPORT: the report's slots still in
 * s_cmd are committed (flush(0)) and exactly their submissions waited for
 * (SUBW_REPORT); =fixed leaves reports to the walker (RECOMP_ZPASS_FIXED,
 * every test visible). */
#define OCC_QSLOTS 16
/* Pending and retired reports, a query each at least, and the open ones. */
#define OCC_POOL   (NV2A_OCC_PENDING * 3 + NV2A_OCC_SEG)
typedef struct OccQuery {
    int                   slot[OCC_QSLOTS];   /* its submission: s_vis_sub[slot] */
    int                   n, ovf;
    struct OccQuery      *next;              /* free list */
} OccQuery;
static OccQuery          s_qpool[OCC_POOL];
static OccQuery         *s_qfree;
static int               s_qmade;
static unsigned char     s_vis_used[VIS_SLOTS];
static int               s_vis_next;
static uint64_t          s_vis_sub[VIS_SLOTS];   /* the submission writing it, 0: the open s_cmd */
static int               s_vis_open[VIS_SLOTS], s_nvis_open;   /* slots the open s_cmd writes */
static int               s_retire[VIS_SLOTS], s_nretire;
static struct nv2a_occ   s_occ;
static int               s_occ_mode = -1;  /* 0 defer, 1 sync, 2 fixed */
static int               s_occ_sync;       /* inside a RECOMP_METAL_OCC=sync GET_REPORT */
static uint64_t          s_occ_ovf;
static uint64_t          s_occ_bad;        /* results whose submission ended in Error */
static uint64_t          s_occ_settle_waits;   /* settles that waited for results */

/* flush committed s_cmd as submission n: the slots it counts into. */
static void occ_committed(uint64_t n)
{
    while (s_nvis_open)
        s_vis_sub[s_vis_open[--s_nvis_open]] = n;
}

/* A released slot's submission is done (an Error one too): it can go. */
static int vis_done(int i)
{
    return s_vis_sub[i] && sub_poll(s_vis_sub[i]) != 0;
}

/* Release, oldest first, every retired slot whose submission is done. */
static void occ_retire(void)
{
    int k, j;
    for (k = j = 0; k < s_nretire; k++) {
        if (vis_done(s_retire[k]))
            s_vis_used[s_retire[k]] = 0;
        else
            s_retire[j++] = s_retire[k];
    }
    s_nretire = j;
}

static int occ_mode(void)
{
    if (s_occ_mode < 0) {
        const char *e = recomp_env(RENV_METAL_OCC);
        s_occ_mode = !e ? 0 : !strcasecmp(e, "sync") ? 1 : !strcasecmp(e, "fixed") ? 2 : 0;
    }
    return s_occ_mode;
}

/* A free slot for the open s_cmd's pass; when none is, the retired ones
 * whose submissions are done are released first. -1 if still none. */
static int vis_alloc(void)
{
    int i, k, tries;
    for (tries = 0; tries < 2; tries++) {
        for (k = 0; k < VIS_SLOTS; k++) {
            i = (s_vis_next + k) % VIS_SLOTS;
            if (!s_vis_used[i]) {
                s_vis_used[i] = 1;
                s_vis_next = (i + 1) % VIS_SLOTS;
                ((uint64_t *)[s_vis contents])[i] = 0;   /* no submission writes it now */
                s_vis_sub[i] = 0;
                s_vis_open[s_nvis_open++] = i;
                return i;
            }
        }
        if (!s_nretire)
            break;
        occ_retire();
    }
    return -1;
}

static void occ_pass_begin(void)
{
    OccQuery *q = s_occ_q;
    int i;
    if (!q || !s_enc || s_occ_slot >= 0)
        return;
    if (q->n >= OCC_QSLOTS || (i = vis_alloc()) < 0) {
        q->ovf = 1;
        s_occ_ovf++;
        return;
    }
    q->slot[q->n] = i;
    q->n++;
    [s_enc setVisibilityResultMode:MTLVisibilityResultModeCounting offset:(NSUInteger)i * 8u];
    s_occ_slot = i;
}

static void occ_pass_end(void)
{
    if (s_occ_slot >= 0 && s_enc)
        [s_enc setVisibilityResultMode:MTLVisibilityResultModeDisabled
                                offset:(NSUInteger)s_occ_slot * 8u];
    s_occ_slot = -1;
}

static uint64_t occ_now_us(void *ctx)
{
    struct timespec ts;
    (void)ctx;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

static void *occ_get(void *ctx)
{
    OccQuery *q;
    (void)ctx;
    if (s_qfree) {
        q = s_qfree;
        s_qfree = q->next;
    } else if (s_qmade < OCC_POOL) {
        q = &s_qpool[s_qmade++];
    } else {
        return NULL;
    }
    q->n = 0;
    q->ovf = 0;
    return q;
}

static void occ_put(void *ctx, void *qv)
{
    OccQuery *q = (OccQuery *)qv;
    int i;
    (void)ctx;
    if (!q)
        return;
    if (q == s_occ_q) {     /* not expected: the tracker ends a query first */
        occ_pass_end();
        s_occ_q = NULL;
    }
    for (i = 0; i < q->n; i++) {
        if (vis_done(q->slot[i]))
            s_vis_used[q->slot[i]] = 0;
        else
            s_retire[s_nretire++] = q->slot[i];   /* in s_cmd or in flight */
    }
    q->n = 0;
    q->next = s_qfree;
    s_qfree = q;
}

static void occ_begin(void *ctx, void *qv)
{
    (void)ctx;
    occ_pass_end();
    s_occ_q = (OccQuery *)qv;
    occ_pass_begin();
}

static void occ_end(void *ctx, void *qv)
{
    (void)ctx;
    if (s_occ_q == (OccQuery *)qv) {
        occ_pass_end();
        s_occ_q = NULL;
    }
}

/* Every piece of q (a slot per pass it spans) is in once the submission
 * writing it is done. With flush_gpu a piece still in s_cmd is committed;
 * inside a RECOMP_METAL_OCC=sync GET_REPORT the piece's own submission is
 * then waited for. Never polls the tracker (it calls this from its poll). */
static int occ_result(void *ctx, void *qv, int flush_gpu, uint64_t *count)
{
    OccQuery *q = (OccQuery *)qv;
    const uint64_t *v = (const uint64_t *)[s_vis contents];
    uint64_t sum = 0;
    int i, bad = 0, r;
    (void)ctx;
    for (i = 0; i < q->n; i++) {
        int s = q->slot[i];
        if (!s_vis_sub[s]) {            /* not committed yet */
            if (!flush_gpu)
                return 0;
            PT_WHY(PT_QUERY_READ);
            flush(0);
        }
        if (flush_gpu && s_occ_sync)
            sub_wait(s_vis_sub[s], SUBW_REPORT);
        if (!(r = sub_poll(s_vis_sub[s])))
            return 0;
        bad |= r < 0;
    }
    for (i = 0; i < q->n; i++)
        sum += v[q->slot[i]];
    if (bad)
        s_occ_bad++;
    *count = q->ovf || bad ? NV2A_OCC_VISIBLE : sum;
    return 1;
}

static const struct nv2a_occ_ops s_occ_ops = {
    occ_get, occ_put, occ_begin, occ_end, occ_result, occ_now_us, NULL
};

static uint64_t occ_env_us(recomp_env_id id, uint64_t def)
{
    long v = recomp_env_int(id, (long)def);
    return v > 0 ? (uint64_t)v : 0;
}

static int occ_ready_on(void)
{
    if (occ_mode() == 2 || !init())
        return 0;
    if (!s_occ.ops) {
        nv2a_occ_init(&s_occ, &s_occ_ops, (uint8_t *)xbox_GetMemoryOffset());
        nv2a_occ_set_delay(&s_occ, occ_env_us(RENV_OCC_DELAY_US, 0));
        nv2a_occ_set_settle(&s_occ, occ_env_us(RENV_OCC_SETTLE_US,
                                               NV2A_OCC_SETTLE_US));
        nv2a_occ_set_trace(&s_occ, recomp_env(RENV_ZPASS_TRACE) != NULL);
    }
    return 1;
}

static void metal_on_zpass(int enable)
{
    if (occ_ready_on()) {
        @autoreleasepool {
            nv2a_occ_zpass(&s_occ, enable);
        }
    }
}

static void metal_on_zpass_clear(void)
{
    if (occ_ready_on()) {
        @autoreleasepool {
            nv2a_occ_zpass_clear(&s_occ);
        }
    }
}

static uint64_t s_report_commits;    /* flushes at GET_REPORT */

/* Does the report just pushed at va count in the open command buffer? */
static int report_in_cmd(uint32_t va)
{
    const struct nv2a_occ_report *p;
    int k, i;
    if (!s_occ.npend || s_occ.pend[s_occ.npend - 1].va != va)
        return 0;
    p = &s_occ.pend[s_occ.npend - 1];
    for (k = 0; k < p->n; k++) {
        const OccQuery *q = (const OccQuery *)p->q[k];
        for (i = 0; i < q->n; i++)
            if (!s_vis_sub[q->slot[i]])   /* not committed yet */
                return 1;
    }
    return 0;
}

static int metal_on_report(uint32_t va)
{
    if (!occ_ready_on())
        return 0;
    @autoreleasepool {
        /* sync: occ_result commits what the report still has in s_cmd and
         * waits for exactly its slots' submissions (SUBW_REPORT). */
        s_occ_sync = occ_mode() == 1;
        nv2a_occ_report(&s_occ, va, s_occ_sync);
        s_occ_sync = 0;
        /* Commit now, so the count can land within the frame, as a D3D11
         * query's does; held to the flip, it lands a frame late. Only when
         * the report counts in s_cmd: each commit ends the pass (a tile
         * store and load), and a report whose queries drew nothing, or
         * drew in a buffer already committed, gains nothing from it. */
        if (occ_mode() == 0 && s_cmd && report_in_cmd(va)) {
            s_report_commits++;
            PT_WHY(PT_REPORT);
            flush(0);
        }
    }
    return 1;
}

static void occ_settle(int flip);

/* At the end of a walk, before a semaphore release after a report, and
 * every ack pass (retired queries only, unflushed, when nothing is
 * pending). */
static void metal_on_poll(void)
{
    if (s_occ.npend || s_occ.nret) {
        @autoreleasepool {
            occ_settle(0);
        }
    }
}

/* After a flush(1): its submission and every earlier one are done, so the
 * retired slots go (occ_retire checks each one's own submission), and the
 * results that are in are taken. */
static void occ_after_wait(void)
{
    occ_retire();
    if (s_occ.ops)
        nv2a_occ_poll(&s_occ, 0);
}

/* At the end of a walk, before a semaphore release after a report, and at
 * the flip after its wait. Not in occ_after_wait, which also runs at the
 * waits inside a walk (a full ring), where the title cannot read yet. Also
 * releases the retired slots that are done, wait or no wait. */
static void occ_settle(int flip)
{
    uint64_t w0;
    if (!s_occ.ops)
        return;
    w0 = s_occ.settle_wait_us;
    if (flip)
        nv2a_occ_flip(&s_occ);
    else
        nv2a_occ_settle(&s_occ);
    if (s_occ.settle_wait_us != w0)
        s_occ_settle_waits++;
    occ_retire();
}

static void flip_log(uint32_t surface_offset, RenderTarget *rt)
{
    static int on = -1;
    static uint64_t p_draws, p_prog, p_skip, p_cull, p_clears, p_tex, p_up,
                    p_chg, p_re, p_fl, p_occ, p_vis, p_late, p_super, p_sync, p_ovf,
                    p_set, p_stale, p_first, p_cup;
    static int zt = -1;
    if (sh_trace() && (s_sh_flip.lib || s_sh_flip.pso || s_sh_flip.tex_ns >= 5000000))
        fprintf(stderr, LOGP "build flip %llu: %u libraries %.1f ms, %u pipelines %.1f ms,"
                " %u texture uploads %.1f ms\n", (unsigned long long)s_presents,
                s_sh_flip.lib, s_sh_flip.lib_ns / 1e6, s_sh_flip.pso, s_sh_flip.pso_ns / 1e6,
                s_sh_flip.tex, s_sh_flip.tex_ns / 1e6);
    memset(&s_sh_flip, 0, sizeof s_sh_flip);
    mag_flip();
    nv2a_tex_dump_flip(s_presents);
    /* With a pack, its hit count every 600 flips, flip trace or not. */
    if (nv2a_tex_pack_count() && s_presents % 600 == 0) {
        fprintf(stderr, LOGP "textures.pack: %llu hits so far\n",
                (unsigned long long)nv2a_tex_pack_hits());
        if (recomp_env(RENV_TEX_PACK_TRACE))
            nv2a_tex_pack_print_misses();
    }
    if (on < 0) {
        on = recomp_env(RENV_FLIP_LOG) != NULL;
        zt = recomp_env(RENV_ZPASS_TRACE) != NULL;
    }
    /* The visibility totals so far: a run killed by its timeout has no
     * exit to print them at, so the last of these is the run's. */
    if ((on || zt) && s_occ.reports && s_presents % 300 == 0)
        nv2a_occ_summary(&s_occ, "[METAL]");
    if (!on)
        return;
    fprintf(stderr, "[METAL] flip %llu batches %llu prog %llu skipped %llu"
            " culled %llu clears %llu tex %llu uploads %llu changed %llu rehash %llu"
            " flushes %llu consts %llu occ %llu (visible %llu, pending %d, late %llu, superseded %llu,"
            " settled %llu (stale %llu, first %llu), sync %llu us, slot overflow %llu)"
            " present 0x%08X walker 0x%08X\n",
            (unsigned long long)s_presents,
            (unsigned long long)(s_draws - p_draws),
            (unsigned long long)(s_draws_prog - p_prog),
            (unsigned long long)(s_draws_skipped - p_skip),
            (unsigned long long)(s_draws_culled - p_cull),
            (unsigned long long)(s_clears - p_clears),
            (unsigned long long)(s_draws_tex - p_tex),
            (unsigned long long)(s_tex_uploads - p_up),
            (unsigned long long)(s_tex_changed - p_chg),
            (unsigned long long)(s_tex_rehash - p_re),
            (unsigned long long)(s_flushes - p_fl),
            (unsigned long long)(s_c_uploads - p_cup),
            (unsigned long long)(s_occ.reports - p_occ),
            (unsigned long long)(s_occ.vis - p_vis), s_occ.npend,
            (unsigned long long)(s_occ.late - p_late),
            (unsigned long long)(s_occ.super - p_super),
            (unsigned long long)(s_occ.settled - p_set),
            (unsigned long long)(s_occ.stale - p_stale),
            (unsigned long long)(s_occ.first - p_first),
            (unsigned long long)(s_occ.sync_us - p_sync),
            (unsigned long long)(s_occ_ovf - p_ovf),
            rt ? rt->offset : 0, surface_offset);
    p_occ = s_occ.reports; p_vis = s_occ.vis; p_late = s_occ.late;
    p_super = s_occ.super; p_sync = s_occ.sync_us; p_ovf = s_occ_ovf;
    p_set = s_occ.settled; p_stale = s_occ.stale; p_first = s_occ.first;
    p_draws = s_draws; p_prog = s_draws_prog; p_skip = s_draws_skipped;
    p_cull = s_draws_culled; p_clears = s_clears; p_tex = s_draws_tex;
    p_up = s_tex_uploads; p_chg = s_tex_changed; p_re = s_tex_rehash;
    p_fl = s_flushes; p_cup = s_c_uploads;
    if (rt && rt->pad && aspect_trace()) {
        static uint64_t p_stretch;
        fprintf(stderr, LOGP "flip %llu hor+: target %ux%u pad %u (%ux%u host),"
                " %llu batches off-centre\n", (unsigned long long)s_presents,
                rt->w, rt->h, rt->pad, rt->hw, rt->hh,
                (unsigned long long)(s_draws_stretch - p_stretch));
        p_stretch = s_draws_stretch;
    }
}

static void prof_flip_end(uint64_t t_flip0)
{
    static uint64_t p_wb, p_ds, p_pa;
    int i, j, done = 0;
    uint64_t g0;
    if (!prof_on()) {
        if (pt_on())
            pt_flip();   /* metal_passes without metal_prof */
        return;
    }
    g0 = s_pa.gpu;
    /* One still running (metal_inflight=2's frame, the layer's slot copy)
     * stays for a later flip, which adds its GPU time: never a wait. */
    for (i = j = 0; i < s_prof_ncmd; i++) {
        id<MTLCommandBuffer> c = s_prof_cmd[i];
        MTLCommandBufferStatus st = [c status];
        if (st < MTLCommandBufferStatusCompleted) {
            s_prof_cmd[j++] = c;
            continue;
        }
        if (st == MTLCommandBufferStatusCompleted) {
            CFTimeInterval a = [c GPUStartTime], b = [c GPUEndTime];
            if (b > a)
                s_pa.gpu += (uint64_t)((b - a) * 1e9);
        }
        [c release];
        done++;
    }
    s_pa_ez_gpu[s_prof_ez ? 1 : 0] += s_pa.gpu - g0;
    s_pa_ez_n[s_prof_ez ? 1 : 0]++;
    if (pt_on()) {
        s_pts.cmd_ns += s_pa.gpu - g0;   /* metal_gpu_blocks' total to compare */
        pt_flip();
    }
    s_pa.ncmd += (uint64_t)done;
    s_prof_ncmd = j;
    s_pa.flip += prof_now() - t_flip0;
    if (++s_pa.n == 300) {
        double n = (double)s_pa.n;
        fprintf(stderr, LOGP "prof %llu flips: on_flip %.3f ms = wait %.3f + hostrb %.3f"
                " + wb %.3f + winrb %.3f + winpush %.3f + blit %.3f (+rest); gpu %.3f ms"
                " over %.1f cmd buffers/flip; %llu write-backs, %llu decode syncs,"
                " %llu pitch-alias binds; rt ownership hashes %.3f ms/flip\n",
                (unsigned long long)s_presents,
                s_pa.flip / n / 1e6, s_pa.wait / n / 1e6, s_pa.hostrb / n / 1e6,
                s_pa.wb / n / 1e6, s_pa.winrb / n / 1e6, s_pa.winpush / n / 1e6,
                s_pa.blit / n / 1e6, s_pa.gpu / n / 1e6, s_pa.ncmd / n,
                (unsigned long long)(s_writebacks - p_wb),
                (unsigned long long)(s_decode_syncs - p_ds),
                (unsigned long long)(s_pitch_alias_binds - p_pa), s_pa.hash / n / 1e6);
        if (s_pa_ez_n[0] && s_pa_ez_n[1])
            fprintf(stderr, LOGP "prof w_earlyz: gpu %.3f ms old (%llu flips), %.3f ms early-Z (%llu flips)\n",
                    s_pa_ez_gpu[0] / (double)s_pa_ez_n[0] / 1e6, (unsigned long long)s_pa_ez_n[0],
                    s_pa_ez_gpu[1] / (double)s_pa_ez_n[1] / 1e6, (unsigned long long)s_pa_ez_n[1]);
        memset(s_pa_ez_gpu, 0, sizeof s_pa_ez_gpu);
        memset(s_pa_ez_n, 0, sizeof s_pa_ez_n);
        p_wb = s_writebacks;
        p_ds = s_decode_syncs;
        p_pa = s_pitch_alias_binds;
        memset(&s_pa, 0, sizeof s_pa);
    }
}

/* RECOMP_DEBUG=metal_host_dump=<prefix>: the frame as drawn, host size,
 * every fb_dump_flips flip (20), <prefix>host_NNNNN.bmp by present count. A
 * guest-size dump (fb_dump) shows only the centre of a padded target. */
static int host_dump_due(void)
{
    static const char *prefix;
    static int every = -1;
    if (every < 0) {
        const char *e = recomp_env(RENV_FB_DUMP_FLIPS);
        prefix = recomp_env(RENV_METAL_HOST_DUMP);
        every = e ? atoi(e) : 0;
        if (every <= 0)
            every = 20;
    }
    return prefix && ((s_presents + 1) % (uint64_t)every) == 0;
}

/* w x h BGRA8 pixels, rows `stride` bytes apart, into a 24-bit bottom-up
 * BMP at path: host_dump's writer, and metal_capture_at's, so the same pixels
 * give the same file. 0 when the file cannot be opened. */
static int bmp_write(const char *path, const uint8_t *px, size_t stride, uint32_t w,
                     uint32_t h)
{
    uint8_t hdr[54];
    uint32_t row = w * 3u, padb = (4u - (row & 3u)) & 3u, size, y, x;
    FILE *f = fopen(path, "wb");

    if (!f)
        return 0;
    size = 54u + (row + padb) * h;
    memset(hdr, 0, sizeof hdr);
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &size, 4);
    hdr[10] = 54; hdr[14] = 40;
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    hdr[26] = 1; hdr[28] = 24;
    fwrite(hdr, 1, sizeof hdr, f);
    for (y = h; y-- > 0; ) {
        const uint8_t *s = px + (size_t)y * stride;
        for (x = 0; x < w; x++)
            fwrite(s + (size_t)x * 4u, 1, 3, f);
        if (padb)
            fwrite(hdr + 1, 1, padb, f);   /* zero bytes of the header */
    }
    fclose(f);
    return 1;
}

static void host_dump(RenderTarget *rt, uint32_t hcw, uint32_t hch)
{
    const uint8_t *px = host_readback(rt);
    char path[1024];

    if (!px || !hcw || !hch)
        return;
    snprintf(path, sizeof path, "%shost_%05llu.bmp", recomp_env(RENV_METAL_HOST_DUMP),
             (unsigned long long)(s_presents + 1));
    (void)bmp_write(path, px, (size_t)rt->hw * 4u, hcw, hch);
}

/* RECOMP_DEBUG=metal_capture_at=<flip>[,<flip>...][,<prefix>] (KFC-1720 H1b
 * step 4, the image check without a forced wait): the presented target of
 * each listed flip, hcw x hch from its top left as host_dump writes it, is
 * copied on the GPU into a Shared buffer of its own inside the frame's own
 * command buffer, after the debug draw and the readback sync, just before
 * the flip's commit (cap_encode); nothing waits for it. A later flip that
 * sees its submission done (sub_poll, cap_poll) writes it to
 * <prefix>cap-NNNNN.bmp with host_dump's writer, so metal_inflight=1 and =2
 * captures of one flip compare byte for byte (and with host_dump's
 * host_NNNNN.bmp of that flip). NNNNN is the flip as host_dump numbers it,
 * s_presents + 1 while the frame is built. Items are separated by ',' (the key is
 * in recomp_env.c's s_comma_keys) or ':'; the first item that does not start with a
 * digit is the prefix, the rest of the value (default: metal_host_dump's,
 * else fb_dump's, else none: the working directory). Up to CAP_AT_MAX flips;
 * CAP_PEND captures on the GPU at once, past that one is skipped and logged.
 * One whose submission ended in Error is not written. Captures still on the
 * GPU at exit are not written. */
#define CAP_AT_MAX 32
#define CAP_PEND   4
static struct {
    int         n;              /* -1: not read yet */
    uint64_t    at[CAP_AT_MAX];
    const char *prefix;
} s_capat = { -1, {0}, NULL };
static struct {
    id<MTLBuffer> buf;          /* nil: free */
    uint64_t      flip, sub;    /* sub: 0 while in the open s_cmd */
    int           pend;         /* in the open s_cmd: flush numbers it */
    uint32_t      w, h;
} s_capq[CAP_PEND];
static uint64_t s_cap_written, s_cap_skipped;

static void cap_parse(void)
{
    const char *e = recomp_env(RENV_METAL_CAPTURE_AT);
    s_capat.n = 0;
    while (e && *e) {
        char *end;
        unsigned long long v;
        if (*e < '0' || *e > '9') {
            s_capat.prefix = e;
            break;
        }
        v = strtoull(e, &end, 10);
        if (s_capat.n < CAP_AT_MAX)
            s_capat.at[s_capat.n++] = (uint64_t)v;
        if (*end != ',' && *end != ':')
            break;
        e = end + 1;
    }
    if (!s_capat.prefix)
        s_capat.prefix = recomp_env(RENV_METAL_HOST_DUMP);
    if (!s_capat.prefix)
        s_capat.prefix = recomp_env(RENV_FB_DUMP);
    if (!s_capat.prefix)
        s_capat.prefix = "";
    if (s_capat.n)
        fprintf(stderr, LOGP "metal_capture_at: %d flips, first %llu, into %scap-NNNNN.bmp\n",
                s_capat.n, (unsigned long long)s_capat.at[0], s_capat.prefix);
}

/* flush committed s_cmd as submission n: the captures it copies into. */
static void cap_committed(uint64_t n)
{
    int k;
    for (k = 0; k < CAP_PEND; k++)
        if (s_capq[k].buf && s_capq[k].pend) {
            s_capq[k].sub = n;
            s_capq[k].pend = 0;
        }
}

/* At the flip, before its commit: rt's presented part into a fresh Shared
 * buffer, in the open s_cmd, when this flip is listed. */
static void cap_encode(RenderTarget *rt, uint32_t hcw, uint32_t hch)
{
    uint64_t flip = s_presents + 1;
    id<MTLBlitCommandEncoder> b;
    int i, k;

    if (s_capat.n < 0)
        cap_parse();
    for (i = 0; i < s_capat.n && s_capat.at[i] != flip; i++)
        ;
    if (i == s_capat.n || !rt->tex || !hcw || !hch)
        return;
    for (k = 0; k < CAP_PEND && s_capq[k].buf; k++)
        ;
    if (k == CAP_PEND) {
        s_cap_skipped++;
        fprintf(stderr, LOGP "metal_capture_at: flip %llu skipped, %d captures still on the"
                " GPU\n", (unsigned long long)flip, CAP_PEND);
        return;
    }
    s_capq[k].buf = [s_dev newBufferWithLength:(NSUInteger)hcw * hch * 4u
                                       options:MTLResourceStorageModeShared];
    if (!s_capq[k].buf) {
        s_cap_skipped++;
        fprintf(stderr, LOGP "metal_capture_at: flip %llu skipped, no %ux%u buffer\n",
                (unsigned long long)flip, hcw, hch);
        return;
    }
    end_encoder();   /* the flip's reason (metal_passes), set by the caller */
    ensure_cmd();
    b = pt_on() ? pt_blit(PT_FLIP, NV2A_GCAT_OTHER) : [s_cmd blitCommandEncoder];
    [b copyFromTexture:rt->tex sourceSlice:0 sourceLevel:0
          sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(hcw, hch, 1)
              toBuffer:s_capq[k].buf destinationOffset:0
    destinationBytesPerRow:(NSUInteger)hcw * 4u
  destinationBytesPerImage:(NSUInteger)hcw * 4u * hch];
    [b endEncoding];
    rt_read_mark(rt);   /* a GPU read of rt (use_sub) */
    s_capq[k].flip = flip;
    s_capq[k].sub = 0;
    s_capq[k].pend = 1;
    s_capq[k].w = hcw;
    s_capq[k].h = hch;
}

/* Each flip: the captures whose submission is done are written, never
 * waited for. */
static void cap_poll(void)
{
    int k, r = 0;
    for (k = 0; k < CAP_PEND; k++) {
        char path[1024];
        if (!s_capq[k].buf || s_capq[k].pend || !(r = sub_poll(s_capq[k].sub)))
            continue;
        snprintf(path, sizeof path, "%scap-%05llu.bmp", s_capat.prefix,
                 (unsigned long long)s_capq[k].flip);
        if (r < 0)
            fprintf(stderr, LOGP "metal_capture_at: flip %llu not written, its submission"
                    " %llu ended in error\n", (unsigned long long)s_capq[k].flip,
                    (unsigned long long)s_capq[k].sub);
        else if (bmp_write(path, (const uint8_t *)[s_capq[k].buf contents],
                           (size_t)s_capq[k].w * 4u, s_capq[k].w, s_capq[k].h)) {
            s_cap_written++;
            fprintf(stderr, LOGP "metal_capture_at: flip %llu (%ux%u) -> %s, written at flip"
                    " %llu\n", (unsigned long long)s_capq[k].flip, s_capq[k].w, s_capq[k].h,
                    path, (unsigned long long)(s_presents + 1));
        } else
            fprintf(stderr, LOGP "metal_capture_at: cannot open %s\n", path);
        [s_capq[k].buf release];
        s_capq[k].buf = nil;
    }
}

#ifdef XBOXRECOMP_ENHANCE
/* ── The mods' 3D debug draw (nv2a_dbg_draw.h, mod API version 8) ─────────
 *
 * As the D3D11 backend's: at the flip, into the target about to be shown,
 * before the sync, the readbacks and the dumps, in a loading pass on the
 * target that attaches its own depth buffer (rt->dt), whose z is the
 * frame's normalised z, the clip z / w of the game's projection. The
 * vertices are clip space already; the shader only narrows x to the
 * guest's columns of a hor+ target. The vertex buffer is reused: the
 * flip's flush(1) waits for the pass that read it. */
static const char s_dbg_msl[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct DV { packed_float4 p; uchar4 c; };\n"
    "struct VO { float4 p [[position]]; float4 c; };\n"
    "vertex VO dbg_vs(uint id [[vertex_id]], device const DV *v [[buffer(0)]],\n"
    "                 constant float &xs [[buffer(1)]]) {\n"
    "    VO o;\n"
    "    float4 p = float4(v[id].p);\n"
    "    o.p = float4(p.x * xs, p.yzw);\n"
    "    o.c = float4(v[id].c) / 255.0;\n"
    "    return o;\n"
    "}\n"
    "fragment float4 dbg_fs(VO i [[stage_in]]) { return i.c; }\n";

static id<MTLRenderPipelineState> s_dbg_pso[2];     /* without, with a depth attachment */
static id<MTLDepthStencilState> s_dbg_dss[3];       /* test and write, test, off */
static id<MTLBuffer> s_dbg_vb;
static size_t s_dbg_vb_cap;
static int s_dbg_failed;

static int dbg_init(void)
{
    id<MTLFunction> vs, fs;
    int k;

    /* s_dbg_failed, not s_dbg_pso[0], says whether the whole pipeline is
     * there: a depth state can fail after both pipelines were made. */
    if (s_dbg_pso[0] || s_dbg_failed)
        return !s_dbg_failed;
    s_dbg_failed = 1;
    vs = compile_fn(s_dbg_msl, @"dbg_vs", "nv2a_dbg_vs");
    fs = compile_fn(s_dbg_msl, @"dbg_fs", "nv2a_dbg_fs");
    if (!vs || !fs)
        return 0;
    for (k = 0; k < 2; k++) {
        MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
        NSError *err = nil;
        pd.vertexFunction = vs;
        pd.fragmentFunction = fs;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        /* Straight alpha: the mods' colours are 0xAARRGGBB as written. */
        pd.colorAttachments[0].blendingEnabled = YES;
        pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        if (k) {
            pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
            pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        }
        s_dbg_pso[k] = [s_dev newRenderPipelineStateWithDescriptor:pd error:&err];
        [pd release];
        if (!s_dbg_pso[k]) {
            fprintf(stderr, LOGP "debug draw pipeline failed (%s); no 3D debug drawing\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            if (s_dbg_pso[0]) {
                [s_dbg_pso[0] release];
                s_dbg_pso[0] = nil;
            }
            return 0;
        }
    }
    for (k = 0; k < 3; k++) {
        MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor alloc] init];
        dd.depthCompareFunction = k < 2 ? MTLCompareFunctionLessEqual : MTLCompareFunctionAlways;
        dd.depthWriteEnabled = k == 0 ? YES : NO;
        s_dbg_dss[k] = [s_dev newDepthStencilStateWithDescriptor:dd];
        [dd release];
        if (!s_dbg_dss[k]) {
            fprintf(stderr, LOGP "debug draw depth state failed; no 3D debug drawing\n");
            return 0;
        }
    }
    s_dbg_failed = 0;
    return 1;
}

static void dbg_present(RenderTarget *rt)
{
    const struct nv2a_dbg_list *l = nv2a_dbg_take();
    const size_t stride = sizeof(struct nv2a_dbg_vtx);
    uint32_t first[NV2A_DBG_SEGS], at = 0;
    float xs;
    int s, z;

    if (!l || !l->total || !rt || !rt->tex || !dbg_init())
        return;
    for (s = 0; s < NV2A_DBG_SEGS; s++) {
        first[s] = at;
        at += l->n[s];
    }
    if ((size_t)at * stride > s_dbg_vb_cap) {
        size_t c = s_dbg_vb_cap ? s_dbg_vb_cap : 65536;
        while (c < (size_t)at * stride)
            c *= 2;
        [s_dbg_vb release];   /* a command buffer using it keeps its own reference */
        s_dbg_vb = [s_dev newBufferWithLength:c options:MTLResourceStorageModeShared];
        s_dbg_vb_cap = s_dbg_vb ? c : 0;
        s_dbg_vb_pend = 0;   /* a new buffer: no submission uses it */
        s_dbg_vb_sub = 0;
        if (!s_dbg_vb)
            return;
    }
    /* The last flip's draw may still read it (KFC-1720 H1b step 3b). */
    cpu_write_wait(&s_dbg_vb_pend, &s_dbg_vb_sub, PT_DEBUG, SUBW_DBG);
    for (s = 0; s < NV2A_DBG_SEGS; s++)
        if (l->n[s])
            memcpy((uint8_t *)[s_dbg_vb contents] + (size_t)first[s] * stride, l->v[s],
                   (size_t)l->n[s] * stride);
    PT_WHY(PT_DEBUG);
    begin_pass(rt, 0, NULL, 0, 1.0f, 0);
    if (!s_enc)
        return;
    z = s_enc_dt != NULL;
    xs = nv2a_dbg_xscale(rt->w, rt->pad);
    {
        MTLViewport vp = {0.0, 0.0, (double)rt->hw, (double)rt->hh, 0.0, 1.0};
        [s_enc setRenderPipelineState:s_dbg_pso[z]];
        [s_enc setViewport:vp];
        [s_enc setCullMode:MTLCullModeNone];
        [s_enc setVertexBuffer:s_dbg_vb offset:0 atIndex:0];
        s_dbg_vb_pend = 1;
        [s_enc setVertexBytes:&xs length:sizeof xs atIndex:1];
    }
    for (s = 0; s < NV2A_DBG_SEGS; s++) {
        int xray = s == NV2A_DBG_TRI_XRAY || s == NV2A_DBG_LINE_XRAY;
        int line = s == NV2A_DBG_LINE || s == NV2A_DBG_LINE_XRAY;
        if (!l->n[s] || (!xray && !z))
            continue;
        if (z)
            [s_enc setDepthStencilState:s_dbg_dss[xray ? 2 : s == NV2A_DBG_TRI_OPAQUE ? 0 : 1]];
        [s_enc drawPrimitives:line ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle
                  vertexStart:first[s]
                  vertexCount:l->n[s]];
    }
    PT_WHY(PT_DEBUG);
    end_encoder();
}
#endif /* XBOXRECOMP_ENHANCE */

/* RECOMP_DEBUG=metal_gpu_spin=1|alt|<us> (probe): Apple GPUs set their
 * clock by how busy they are, and the flip waits for the frame, so the GPU
 * idles while the CPU builds the next one. When on, the flip, after its
 * wait, commits on a queue of its own one small dispatch (one threadgroup,
 * a dependent FMA loop) that keeps the GPU busy for about that gap: 4 ms,
 * or <us> when the value is 100 or more. Its length follows the last
 * spin's GPU time, read at the next flip without waiting; a spin still
 * running then skips this one. alt: only before even flips (the parity
 * snap_lazy_now() sees in that frame), an in-run A/B. Never in s_prof_cmd
 * and never timestamped: metal_prof and metal_gpu_blocks time only the
 * frame's command buffers. Off: no queue, no kernel. */
static const char s_spin_msl[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void nv2a_gpu_spin(device float *out [[buffer(0)]],\n"
    "                          constant uint &n [[buffer(1)]],\n"
    "                          uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "    float a = 1.0f + float(tid) * 1e-6f;\n"
    "    for (uint i = 0; i < n; i++)\n"
    "        a = fma(a, 0.9999999f, 1e-7f);\n"
    "    out[tid] = a;\n"
    "}\n";
#define SPIN_THREADS 32u
#define SPIN_N_MIN   (1u << 10)
#define SPIN_N_MAX   (1u << 26)

static void gpu_spin_flip(void)
{
    static int mode = -1;   /* 0 off, 1 every flip, 2 alt, 3 auto: while fullscreen */
    static double target_us = 4000.0;
    static uint32_t iters = 1u << 16;
    static id<MTLCommandQueue> q;
    static id<MTLComputePipelineState> pso;
    static id<MTLBuffer> out;
    static id<MTLCommandBuffer> last;
    static struct { int flips, n, busy; double ms; } w;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> ce;

    if (mode < 0) {
        const char *e = recomp_env(RENV_METAL_GPU_SPIN);
        long v = e ? strtol(e, NULL, 10) : 0;
        int off = e && ((e[0] == '0' && !v) || !strcasecmp(e, "off"));
        /* Default auto (hub, KFC-1650, 20-nov: 13.5 -> 9.8 ms of frame GPU
         * time and 60 FPS at scale 4 on an M4 Air): spin while the window is
         * fullscreen (present.fullscreen, live), a provisional fix until the
         * flip stops waiting (KFC-1720). 0/off: never. */
        mode = !e ? 3 : off ? 0 : !strcasecmp(e, "alt") ? 2 : v == 1 || v >= 100 ? 1 : 0;
        if (v >= 100)
            target_us = (double)v;
        if (e && !mode && !off)
            fprintf(stderr, LOGP "metal_gpu_spin=%s: not 1, alt or microseconds >= 100, off\n", e);
        if (mode) {
            NSError *err = nil;
            id<MTLLibrary> lib = metal_compile_library(s_dev, [NSString stringWithUTF8String:s_spin_msl],
                                                       "nv2a_gpu_spin", &err);
            id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"nv2a_gpu_spin"] : nil;
            if (fn)
                pso = [s_dev newComputePipelineStateWithFunction:fn error:&err];
            [fn release];
            [lib release];
            if (pso) {
                q = [s_dev newCommandQueue];
                out = [s_dev newBufferWithLength:SPIN_THREADS * sizeof(float)
                                         options:MTLResourceStorageModePrivate];
            }
            if (!pso || !q || !out) {
                fprintf(stderr, LOGP "metal_gpu_spin=%s: no spin kernel (%s), off\n", e ? e : "auto",
                        err ? [[err localizedDescription] UTF8String] : "?");
                mode = 0;
            } else {
                fprintf(stderr, LOGP "metal_gpu_spin=%s: after each flip's wait a %u-thread"
                        " spin on its own queue, target %.0f us of GPU time%s\n", e ? e : "auto",
                        SPIN_THREADS, target_us, mode == 2 ? " (alt: before even flips only)"
                        : mode == 3 ? " (while fullscreen)" : "");
            }
        }
    }
    if (!mode)
        return;
    if (++w.flips == 300) {
        fprintf(stderr, LOGP "gpu spin over 300 flips: target %.0f us, mean spin %.3f ms, %u"
                " iterations (%d spins timed, %d skipped while the last still ran)\n", target_us,
                w.n ? w.ms / w.n : 0.0, iters, w.n, w.busy);
        memset(&w, 0, sizeof w);
    }
    if (last) {
        MTLCommandBufferStatus st = [last status];
        if (st < MTLCommandBufferStatusCompleted) {
            w.busy++;   /* still spinning: never stack spins, never wait */
            return;
        }
        if (st == MTLCommandBufferStatusCompleted) {
            double us = ([last GPUEndTime] - [last GPUStartTime]) * 1e6, f, nn;
            if (us > 0) {
                f = target_us / us;
                f = f < 0.5 ? 0.5 : f > 2.0 ? 2.0 : f;   /* damped: the clock moves with it */
                nn = (double)iters * f;
                iters = nn < SPIN_N_MIN ? SPIN_N_MIN : nn > SPIN_N_MAX ? SPIN_N_MAX : (uint32_t)nn;
                w.n++;
                w.ms += us / 1e3;
            }
        }
        [last release];
        last = nil;
    }
    if (mode == 2 && ((s_presents + 1) & 1))   /* the frame built next is odd */
        return;
    if (mode == 3 && !nv2a_host_opts()->fullscreen)
        return;
    cb = [q commandBuffer];
    ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:pso];
    [ce setBuffer:out offset:0 atIndex:0];
    [ce setBytes:&iters length:sizeof iters atIndex:1];
    [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(SPIN_THREADS, 1, 1)];
    [ce endEncoding];
    [cb commit];
    last = [cb retain];
}

/* RECOMP_DEBUG=metal_cal_passes=N, metal_cal_tex=1 (KFC-1105): a probe to
 * calibrate metal_gpu_blocks. Before the flip's commit, N extra render
 * passes in the frame's own command buffer, each its own pass: a triangle
 * over a private 2560x1920 BGRA8 target (load DontCare, store Store) whose
 * fragment returns a constant, or with metal_cal_tex=1 samples (linear) a
 * 2560x1920 texture filled with a pattern. They are timed as any pass (the
 * "other" block of metal_gpu_blocks, so in its busy time); with
 * metal_ab_alt only in arm B. Off (N = 0): no texture, no pipeline. */
#define XCAL_W 2560u
#define XCAL_H 1920u
static const char s_xcal_msl[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct XcalOut { float4 pos [[position]]; };\n"
    "vertex XcalOut xcal_vs(uint vid [[vertex_id]])\n"
    "{\n"
    "    float2 p = float2(float((vid << 1) & 2u), float(vid & 2u));\n"
    "    XcalOut o;\n"
    "    o.pos = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);\n"
    "    return o;\n"
    "}\n"
    "fragment float4 xcal_fs(XcalOut in [[stage_in]])\n"
    "{\n"
    "    return float4(0.25f, 0.5f, 0.75f, 1.0f);\n"
    "}\n"
    "fragment float4 xcal_fs_tex(XcalOut in [[stage_in]],\n"
    "                            texture2d<float> t [[texture(0)]])\n"
    "{\n"
    "    constexpr sampler s(filter::linear, address::clamp_to_edge);\n"
    "    return t.sample(s, in.pos.xy / float2(2560.0f, 1920.0f));\n"
    "}\n";

static void xcal_frame(void)
{
    static int n = -1, use_tex;
    static id<MTLTexture> dst, src;
    static id<MTLRenderPipelineState> pso;
    int i;

    if (n < 0) {
        NSError *err = nil;
        id<MTLLibrary> lib;
        id<MTLFunction> vs, fs;
        long v = recomp_env_int(RENV_METAL_CAL_PASSES, 0);
        n = v > 0 ? (int)(v > 64 ? 64 : v) : 0;
        use_tex = n && recomp_env_on(RENV_METAL_CAL_TEX);
        if (!n || !s_dev)
            n = 0;
        if (n) {
            MTLTextureDescriptor *td = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                             width:XCAL_W height:XCAL_H mipmapped:NO];
            td.storageMode = MTLStorageModePrivate;
            td.usage = MTLTextureUsageRenderTarget;
            dst = [s_dev newTextureWithDescriptor:td];
            if (use_tex) {
                uint32_t *px = (uint32_t *)malloc((size_t)XCAL_W * XCAL_H * 4u);
                td = [MTLTextureDescriptor
                    texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                 width:XCAL_W height:XCAL_H mipmapped:NO];
                td.storageMode = MTLStorageModeShared;
                td.usage = MTLTextureUsageShaderRead;
                src = px ? [s_dev newTextureWithDescriptor:td] : nil;
                if (src) {
                    uint32_t x, y;
                    for (y = 0; y < XCAL_H; y++)
                        for (x = 0; x < XCAL_W; x++)
                            px[(size_t)y * XCAL_W + x] = 0xFF000000u | ((x ^ y) & 0xFFu) << 16
                                                         | (x & 0xFFu) << 8 | (y & 0xFFu);
                    [src replaceRegion:MTLRegionMake2D(0, 0, XCAL_W, XCAL_H) mipmapLevel:0
                             withBytes:px bytesPerRow:XCAL_W * 4u];
                }
                free(px);
            }
            lib = metal_compile_library(s_dev, [NSString stringWithUTF8String:s_xcal_msl],
                                        "xcal", &err);
            vs = lib ? [lib newFunctionWithName:@"xcal_vs"] : nil;
            fs = lib ? [lib newFunctionWithName:use_tex ? @"xcal_fs_tex" : @"xcal_fs"] : nil;
            if (vs && fs && dst && (!use_tex || src)) {
                MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
                pd.vertexFunction = vs;
                pd.fragmentFunction = fs;
                pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
                pso = [s_dev newRenderPipelineStateWithDescriptor:pd error:&err];
                [pd release];
            }
            [vs release];
            [fs release];
            [lib release];
            if (!pso) {
                fprintf(stderr, LOGP "metal_cal_passes: no pipeline or target (%s), off\n",
                        err ? [[err localizedDescription] UTF8String] : "?");
                n = 0;
            } else {
                fprintf(stderr, LOGP "KFC-1105: metal_cal_passes=%d: %d extra full-screen passes"
                        " a flip on a %ux%u BGRA8 target, %s%s\n", n, n, XCAL_W, XCAL_H,
                        use_tex ? "each samples a texture (metal_cal_tex)" : "constant colour",
                        s_ab_blk ? "; arm B only (metal_ab_alt)" : "");
            }
        }
    }
    if (n <= 0 || !nd_arm_b())
        return;
    PT_WHY(PT_FLIP);
    end_encoder();   /* the scene's last pass closes as flush would have closed it */
    ensure_cmd();
    for (i = 0; i < n; i++) {
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        id<MTLRenderCommandEncoder> e;
        rp.colorAttachments[0].texture = dst;
        rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        if (pt_on()) {
            /* a pass with a draw, in the "other" block (no block of its own: the
             * categories are the backends' shared enum) */
            int idx;
            PtPass *p = pt_new(PTK_RENDER, PT_FLIP, &idx);
            if (p) {
                p->cat = NV2A_GCAT_OTHER;
                p->close_why = PT_FLIP;
                p->draws = 1;
                p->first_seen = 1;
                p->cat_draws[NV2A_GCAT_OTHER] = 1;
                s_pts.close[PT_FLIP]++;
                if (gb_on()) {
                    if (@available(macOS 11.0, iOS 14.0, *)) {
                        int base = gb_take(4);
                        if (base >= 0) {
                            MTLRenderPassSampleBufferAttachmentDescriptor *a =
                                rp.sampleBufferAttachments[0];
                            a.sampleBuffer = (id<MTLCounterSampleBuffer>)s_gb_sb[s_presents & 1];
                            a.startOfVertexSampleIndex = (NSUInteger)base;
                            a.endOfVertexSampleIndex = (NSUInteger)base + 1;
                            a.startOfFragmentSampleIndex = (NSUInteger)base + 2;
                            a.endOfFragmentSampleIndex = (NSUInteger)base + 3;
                            p->gb = base;
                            p->g_open = gb_now();
                        }
                    }
                }
            }
        }
        e = [s_cmd renderCommandEncoderWithDescriptor:rp];
        [e setRenderPipelineState:pso];
        if (use_tex)
            [e setFragmentTexture:src atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e endEncoding];
    }
}

/* The end of each flip's GPU work, before s_presents moves on (KFC-1720 H1b
 * step 4): the flip's last submission noted (s_flip_sub), the frames with
 * work committed and not done counted (s_ifl_*: this one and the one before
 * at most), then, with metal_inflight=2, the wait one frame back: flip N-1's
 * last submission (its slot copy included), SUBW_FRAME. After it N+1 is
 * built while only N runs: its ring copy (N-1's), the pass tracer's frame
 * table and timestamp buffer by flip parity (pt_flip judges N-1, gb_resolve
 * reads it), the debug draws' buffer and every other last use from N-1 are
 * done. metal_inflight=1 only notes the submission. */
static void flip_bound(void)
{
    uint64_t f = s_presents;   /* this flip, before s_presents++ */
    uint64_t k;
    int busy = 0;

    s_flip_sub[f % 4] = s_sub_last;
    if (s_inflight < 2)
        return;
    /* flips N-2, N-1, N: one with a submission of its own still running */
    for (k = f >= 2 ? f - 2 : 0; k <= f; k++) {
        uint64_t v = s_flip_sub[k % 4], pv = k ? s_flip_sub[(k + 3) % 4] : 0;
        if (v > pv && !sub_done_upto(v))
            busy++;
    }
    s_ifl_sum += (uint64_t)busy;
    s_ifl_n++;
    s_ifl_hist[busy]++;
    if (busy > s_ifl_max)
        s_ifl_max = busy;
    if (f)
        sub_wait_upto(s_flip_sub[(f + 3) % 4], SUBW_FRAME);
}

static void metal_flip(uint32_t surface_offset, uint32_t pitch)
{
    RenderTarget *rt = NULL;
    uint32_t addr = surface_offset ? nv2a_pb_dma_resolve(surface_offset) : 0;
    const uint8_t *hpx = NULL;   /* a scaled rt's host pixels (host_readback) */
    uint64_t t_flip0 = prof_on() ? prof_now() : 0, t;
    int always = wb_always(), win = 0, layer = 0, hdump = 0;
    int rb_bad = 0;   /* rt's last write ended in Error: nothing read back */
    uint32_t cw = 0, ch = 0, hcw = 0, hch = 0;   /* the frame, guest and host */

    guard_open();
    if (init()) {
        @autoreleasepool {
            int wb;
            id<MTLTexture> slot = nil;
            rt = present_target(surface_offset);
            if (rt) {
                /* The frame's target size, for a HUD canvas on "frame". */
                nv2a_aspect_frame_publish(rt->w, rt->h);
                present_extent(rt, &cw, &ch);
                hcw = (cw + 2u * rt->pad) * rt->n;
                hch = ch * rt->n;
                s_shown_w = cw;
                s_shown_h = ch;
                if (pt_on() && s_pt_track)
                    pt_read(rt->tex, 0, PTR_FLIP, -1);   /* metal_passes */
#ifdef XBOXRECOMP_ENHANCE
                dbg_present(rt);
#endif
            }
            win = rt && xbox_FramebufferWindowRunning();
            /* Layer mode: the frame goes to the window as a GPU copy into the
             * presenter's back slot (fb_present_metal.m), below. */
            layer = win && s_layer;
            wb = rt && addr && pitch && pitch >= rt->w * 2u && always;
            hdump = rt && host_dump_due();
            PT_WHY(PT_FLIP);
            if (rt && (wb || hdump || (win && !layer)))
                rt_managed_sync(rt);
            if (rt) {
                /* RECOMP_DEBUG=metal_capture_at: in this command buffer */
                PT_WHY(PT_FLIP);
                cap_encode(rt, hcw, hch);
            }
            xcal_frame();   /* RECOMP_DEBUG=metal_cal_passes: off, returns */
            t = prof_t();
            PT_WHY(PT_FLIP);
            /* metal_inflight=2 (KFC-1720 H1b step 4): the frame is committed
             * and not waited for; flip_bound below waits one frame back. */
            flush(s_inflight < 2);
            if (s_inflight > 1 && s_occ.ops) {
                /* This frame's results are not in: the flip settles without
                 * the spin (occ_settle_us), from the last counts, and the
                 * results amend them as they land (a frame late). */
                uint64_t us = s_occ.settle_us, set = s_occ.settled;
                s_occ.settle_us = 0;
                occ_settle(1);
                s_occ.settle_us = us;
                s_occ_flip_early += s_occ.settled - set;
            } else {
                occ_settle(1);
            }
            ring_switch();   /* the frame boundary: the other ring copy */
            prof_add(&s_pa.wait, t);
            /* The CPU reads below (host dump, host readback, write-back,
             * readback window) need rt's last write, its sync blit
             * included: exactly that submission (KFC-1720 H1b step 3a; the
             * serial flush(1) above has observed it already, with
             * metal_inflight=2 it is waited for here). One that
             * ended in Error publishes nothing: no dump, no write-back, the
             * window keeps the image it has. */
            if (rt && (wb || hdump || (win && !layer)) && !rt_read_ready(rt, PT_FLIP)) {
                rb_bad = 1;
                wb = 0;
                hdump = 0;
            }
            if (hdump)
                host_dump(rt, hcw, hch);
            if (layer) {
                /* After the wait, in a command buffer of its own that nobody
                 * waits for: the flip's wait never includes the copy, nor a
                 * present pass still reading the slot (tracked hazards order
                 * the two on the GPU). Published once committed, so the main
                 * thread's pass that shows it is committed after it. */
                t = prof_t();
                slot = (id<MTLTexture>)xbox_FramebufferWindowBackTexture(hcw, hch, rt->n,
                                                                         rt->pad);
                if (slot) {
                    id<MTLBlitCommandEncoder> b;
                    ensure_cmd();
                    /* metal_gpu_blocks: the present, as far as this backend's
                     * command buffers see it (the presenter's own pass is
                     * fb_present_metal.m's) */
                    b = pt_on() ? pt_blit(PT_FLIP, NV2A_GCAT_PRESENT) : [s_cmd blitCommandEncoder];
                    [b copyFromTexture:rt->tex sourceSlice:0 sourceLevel:0
                          sourceOrigin:MTLOriginMake(0, 0, 0)
                            sourceSize:MTLSizeMake(hcw, hch, 1)
                             toTexture:slot destinationSlice:0 destinationLevel:0
                     destinationOrigin:MTLOriginMake(0, 0, 0)];
                    [b endEncoding];
                    /* KFC-1720 H1b step 3b: the copy reads rt (use_sub; a
                     * later pass that writes rt is on this queue after it,
                     * ordered by tracked hazards), and writes the slot, which
                     * a later CPU upload into a slot waits for (s_win_blit_sub). */
                    rt_read_mark(rt);
                    flush(0);
                    s_win_blit_sub = s_sub_last;   /* flush(0)'s commit of the copy */
                    xbox_FramebufferWindowPublishTexture();
                    s_window_presents++;
                }
                prof_add(&s_pa.blit, t);
            }
            /* The host-size readback only when something reads it: the
             * write-back below or the readback-mode window. */
            t = prof_t();
            if (rt && rt->hw != rt->w && !rb_bad && (wb || (win && !layer)))
                hpx = host_readback(rt);
            prof_add(&s_pa.hostrb, t);
            if (wb) {
                t = prof_t();
                if (rt->hw == rt->w)
                    write_back(rt, addr, pitch);
                else if (hpx)
                    write_back_scaled(rt, hpx, addr, pitch);
                rt->dirty = 0;
                prof_add(&s_pa.wb, t);
                if (addr == rt->addr && pitch == rt->pitch)
                    rt_own_reset(rt);
            }
            /* metal_inflight=2: the wait one frame back (metal_prof: the
             * flip's wait, as flush(1)'s is with metal_inflight=1) */
            t = prof_t();
            flip_bound();
            prof_add(&s_pa.wait, t);
            cap_poll();   /* RECOMP_DEBUG=metal_capture_at: the ones done */
            /* after the flip's own GPU work (present blit, readbacks) */
            gpu_spin_flip();   /* RECOMP_DEBUG=metal_gpu_spin; off: returns */
        }
    }
    s_presents++;
    nv2a_w_earlyz_flip();
    /* The [MEM] exit line's GPU figure: every allocation this device made
     * for the process (the presenter's too, on the shared device). Unified
     * memory has no separate non-local pool. */
    if (s_dev && (s_presents == 1 || s_presents % 60 == 0))
        mem_stats_gpu_note("metal", (uint64_t)[s_dev currentAllocatedSize], MEM_STATS_NA);
    flip_log(surface_offset, rt);
    /* Running totals every 300 presents only under the present_stats trace
     * (as in nv2a_pb_d3d11.c: printed by default they cost the frame). */
    static int stats_on = -1;
    if (stats_on < 0)
        stats_on = recomp_env_on(RENV_PRESENT_STATS);
    const int periodic = s_presents == 1 || (stats_on && (s_presents % 300) == 0);
    if (periodic)
        fprintf(stderr, LOGP "present #%llu: surface 0x%08X (%s), %llu clears,"
                " %llu draws (%llu vertex-program), %llu skipped, %llu culled,"
                " %llu texture uploads, %llu rehashes, %llu write-backs (%s),"
                " %llu decode syncs, %llu pitch-alias binds, %llu stale drops,"
                " %llu evictions lost, %llu grows, %llu partial clears;"
                " shaders %llu vs + %llu fs, %llu compile errors;"
                " %llu z-clears, %llu depth draws, %llu rt binds (%llu self copies);"
                " %llu window presents (%s); %llu flushes,"
                " %llu at GET_REPORT\n",
                (unsigned long long)s_presents, surface_offset,
                rt ? "drawn here" : "guest bytes",
                (unsigned long long)s_clears, (unsigned long long)s_draws,
                (unsigned long long)s_draws_prog, (unsigned long long)s_draws_skipped,
                (unsigned long long)s_draws_culled,
                (unsigned long long)s_tex_uploads, (unsigned long long)s_tex_rehash,
                (unsigned long long)s_writebacks, always ? "always" : "lazy",
                (unsigned long long)s_decode_syncs, (unsigned long long)s_pitch_alias_binds,
                (unsigned long long)s_stale_drops, (unsigned long long)s_evict_lost,
                (unsigned long long)s_grows,
                (unsigned long long)s_partial_clears,
                (unsigned long long)s_vs_compiled, (unsigned long long)s_fs_compiled,
                (unsigned long long)s_compile_errors,
                (unsigned long long)s_zclears, (unsigned long long)s_draws_z,
                (unsigned long long)s_rtt_binds, (unsigned long long)s_self_copies,
                (unsigned long long)s_window_presents,
                s_layer ? "CAMetalLayer slots" : "texture readback",
                (unsigned long long)s_flushes, (unsigned long long)s_report_commits);
    if (periodic && (s_blits_gpu || s_blits_guest))
        fprintf(stderr, LOGP "present #%llu: 2D blits %llu target copies, %llu in guest"
                " memory\n", (unsigned long long)s_presents, (unsigned long long)s_blits_gpu,
                (unsigned long long)s_blits_guest);
    /* KFC-1720 H1b step 2: submissions, ring copies and waits, running totals.
     * Step 4: the back-pressure, ring guard and frame bound (running totals),
     * and metal_inflight=2's frames in flight at the end of each flip over
     * the last 300 flips (max, mean, how many flips saw 0 / 1 / 2 / 3). */
    if ((stats_on || s_ring_small || s_inflight > 1) && (s_presents % 300) == 0) {
        fprintf(stderr, LOGP "present #%llu: %llu submissions (%llu errors), %llu ring switches,"
                " %llu ring cuts, %llu oversize draws; blocked waits: flush %llu (%.1f ms),"
                " seed %llu (%.1f ms), ring %llu (%.1f ms), table full %llu (%.1f ms),"
                " ring guard %llu (%.1f ms), frame bound %llu (%.1f ms); metal_inflight=%d,"
                " last 300 flips: frames in flight at the flip max %d mean %.2f"
                " (0/1/2/3: %llu/%llu/%llu/%llu), reports the flip settled early %llu;"
                " captures written %llu, skipped %llu\n",
                (unsigned long long)s_presents, (unsigned long long)s_sub_last,
                (unsigned long long)atomic_load_explicit(&s_sub_errs, memory_order_relaxed),
                (unsigned long long)s_ring_switches, (unsigned long long)s_ring_cuts,
                (unsigned long long)s_ring_oversize,
                (unsigned long long)s_subw_blocked[SUBW_FLUSH], s_subw_ns[SUBW_FLUSH] / 1e6,
                (unsigned long long)s_subw_blocked[SUBW_STAGE], s_subw_ns[SUBW_STAGE] / 1e6,
                (unsigned long long)s_subw_blocked[SUBW_RING], s_subw_ns[SUBW_RING] / 1e6,
                (unsigned long long)s_subw_blocked[SUBW_FULL], s_subw_ns[SUBW_FULL] / 1e6,
                (unsigned long long)s_subw_blocked[SUBW_GUARD], s_subw_ns[SUBW_GUARD] / 1e6,
                (unsigned long long)s_subw_blocked[SUBW_FRAME], s_subw_ns[SUBW_FRAME] / 1e6,
                s_inflight, s_ifl_max, s_ifl_n ? (double)s_ifl_sum / (double)s_ifl_n : 0.0,
                (unsigned long long)s_ifl_hist[0], (unsigned long long)s_ifl_hist[1],
                (unsigned long long)s_ifl_hist[2], (unsigned long long)s_ifl_hist[3],
                (unsigned long long)s_occ_flip_early,
                (unsigned long long)s_cap_written, (unsigned long long)s_cap_skipped);
        s_ifl_sum = s_ifl_n = 0;
        s_ifl_max = 0;
        memset(s_ifl_hist, 0, sizeof s_ifl_hist);
        s_occ_flip_early = 0;
    }
    /* KFC-1720 H1b step 3a: visibility and CPU reads, over the last 300
     * flips. Sync report waits: RECOMP_METAL_OCC=sync's blocked waits on a
     * report's own submissions; settle waits: settles that spun for results
     * within occ_settle_us; settled: answered before the result landed (the
     * address's last count), amended: the late result written over it after.
     * Step 3b: the blocked waits of CPU writes, by site (cpu_write_wait, the
     * slot hand-off, mag_flip); texture uploads never wait (new textures). */
    if ((stats_on || s_ring_small) && (s_presents % 300) == 0) {
        static uint64_t p_rep, p_rep_ns, p_rd, p_rd_ns, p_skip, p_sw, p_sw_us, p_set, p_am,
                        p_late, p_bad, p_rtp, p_rtp_ns, p_dbg, p_dbg_ns, p_slot, p_slot_ns,
                        p_mag, p_mag_ns;
        fprintf(stderr, LOGP "present #%llu, last 300 flips: visibility: sync report waits"
                " %llu (%.1f ms), settle waits %llu (%.1f ms), settled before the result %llu,"
                " amended late %llu, late as visible %llu, error results %llu; readback waits"
                " %llu (%.1f ms), readbacks skipped on error %llu; CPU-write waits: staging"
                " fill %llu (%.1f ms), debug draws %llu (%.1f ms), slot upload %llu (%.1f ms),"
                " tex_mag %llu (%.1f ms)\n",
                (unsigned long long)s_presents,
                (unsigned long long)(s_subw_blocked[SUBW_REPORT] - p_rep),
                (s_subw_ns[SUBW_REPORT] - p_rep_ns) / 1e6,
                (unsigned long long)(s_occ_settle_waits - p_sw),
                (s_occ.settle_wait_us - p_sw_us) / 1e3,
                (unsigned long long)(s_occ.settled - p_set),
                (unsigned long long)(s_occ.amended - p_am),
                (unsigned long long)(s_occ.late - p_late),
                (unsigned long long)(s_occ_bad - p_bad),
                (unsigned long long)(s_subw_blocked[SUBW_READ] - p_rd),
                (s_subw_ns[SUBW_READ] - p_rd_ns) / 1e6,
                (unsigned long long)(s_rb_skipped - p_skip),
                (unsigned long long)(s_subw_blocked[SUBW_RTP] - p_rtp),
                (s_subw_ns[SUBW_RTP] - p_rtp_ns) / 1e6,
                (unsigned long long)(s_subw_blocked[SUBW_DBG] - p_dbg),
                (s_subw_ns[SUBW_DBG] - p_dbg_ns) / 1e6,
                (unsigned long long)(s_subw_blocked[SUBW_SLOT] - p_slot),
                (s_subw_ns[SUBW_SLOT] - p_slot_ns) / 1e6,
                (unsigned long long)(s_subw_blocked[SUBW_MAG] - p_mag),
                (s_subw_ns[SUBW_MAG] - p_mag_ns) / 1e6);
        p_rep = s_subw_blocked[SUBW_REPORT]; p_rep_ns = s_subw_ns[SUBW_REPORT];
        p_rd = s_subw_blocked[SUBW_READ]; p_rd_ns = s_subw_ns[SUBW_READ];
        p_rtp = s_subw_blocked[SUBW_RTP]; p_rtp_ns = s_subw_ns[SUBW_RTP];
        p_dbg = s_subw_blocked[SUBW_DBG]; p_dbg_ns = s_subw_ns[SUBW_DBG];
        p_slot = s_subw_blocked[SUBW_SLOT]; p_slot_ns = s_subw_ns[SUBW_SLOT];
        p_mag = s_subw_blocked[SUBW_MAG]; p_mag_ns = s_subw_ns[SUBW_MAG];
        p_skip = s_rb_skipped; p_sw = s_occ_settle_waits; p_sw_us = s_occ.settle_wait_us;
        p_set = s_occ.settled; p_am = s_occ.amended; p_late = s_occ.late; p_bad = s_occ_bad;
    }
    if (stats_on && (s_presents % 300) == 0 && s_dof_skipped)
        fprintf(stderr, LOGP "dof_skip: %llu draws skipped\n", (unsigned long long)s_dof_skipped);
    if (periodic && s_rtp)
        fprintf(stderr, LOGP "metal_rt_private: staging reads %llu, writes %llu\n",
                (unsigned long long)s_stage_reads, (unsigned long long)s_stage_writes);
    if (s_zt_binds && stats_on && (s_presents % 300) == 0)
        fprintf(stderr, LOGP "depth surfaces as textures: %llu binds, %llu conversions\n",
                (unsigned long long)s_zt_binds, (unsigned long long)s_zt_converts);
    if (!always && rt && addr && pitch)
        guard_flip(addr, pitch * rt->h);
    if (!pitch || !addr || layer) {
        prof_flip_end(t_flip0);
        return;
    }
    xbox_FramebufferWindowSet(addr, pitch);
    /* Readback mode (metal_present=readback, or no CAMetalLayer window): a
     * surface drawn here reaches the window by readback, getBytes of rt->tex
     * after the flip's wait, a copy into the window's slot, then
     * SDL_UpdateTexture. It reads the texture, not the guest bytes. A surface
     * the backend never drew (guest bytes) goes as the CPU path presents it,
     * in either mode. */
    /* With the layer, the hand-offs below upload into the presenter's back
     * slot (fbm_upload, a CPU write), which waits there only for the
     * presenter's own pass over it: the copies of earlier frames into slots
     * come first (KFC-1720 H1b step 3b). 0: none ever, nothing to wait for. */
    sub_wait_upto(s_win_blit_sub, SUBW_SLOT);
    if (rt && win && rb_bad) {
        /* No readback (rt_read_ready): not the guest bytes either, which
         * are an older frame; the window shows what it has. */
        prof_flip_end(t_flip0);
        return;
    }
    if (rt && hpx && win) {
        /* Scaled: the frame's part of the host-size readback above. */
        t = prof_t();
        xbox_FramebufferWindowPresentPixels(hpx, hcw, hch, rt->hw * 4u, 4u);
        prof_add(&s_pa.winpush, t);
        s_window_presents++;
        prof_flip_end(t_flip0);
        return;
    }
    if (rt && rt->hw == rt->w && win) {
        static uint8_t *px;
        static size_t cap;
        size_t need = (size_t)cw * ch * 4u;
        if (cap < need) {
            uint8_t *n = (uint8_t *)realloc(px, need);
            if (n) {
                px = n;
                cap = need;
            }
        }
        if (cap >= need) {
            t = prof_t();
            rt_get(rt, px, cw * 4u, MTLRegionMake2D(0, 0, cw, ch));
            prof_add(&s_pa.winrb, t);
            t = prof_t();
            xbox_FramebufferWindowPresentPixels(px, cw, ch, cw * 4u, 4u);
            prof_add(&s_pa.winpush, t);
            s_window_presents++;
            prof_flip_end(t_flip0);
            return;
        }
    }
    /* Guest bytes go to the window: our read, not the title's (the guard
     * may have armed this surface above). Headless, nothing reads them. */
    if (xbox_FramebufferWindowRunning())
        guard_open();
    xbox_FramebufferWindowPresent(addr, pitch);
    prof_flip_end(t_flip0);
}

/* render.scale changed at this flip: every target and depth buffer was made
 * at the old factor. Targets go with the eviction's write-back, so the next
 * one at that surface seeds from what was drawn at the new scale. After the
 * present, so the frame being shown keeps the targets it was drawn in.
 *
 * The texture cache stays. Its entries are guest texels at guest size, which
 * no scale changes; a render-to-texture input binds the target itself
 * (rt_texture, s_self resized by its own check), and the write-back marks
 * every entry over a target's memory stale (tex_cache_stale). Dropping it
 * made the next flip decode and upload every texture again: 180-550 ms on
 * Jet Set Radio Future's title screen, the freeze seen at each change. */
static void scale_drop(void)
{
    int i;
    @autoreleasepool {
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *rt = &s_rt[i];
            if (!rt->tex)
                continue;
            if (!rt_drop_if_stale(rt, "scale")) {
                rt_sync(rt, rt->pitch);
                rt_free(rt);
            }
        }
        for (i = 0; i < DT_MAX; i++) {
            DepthTarget *d = &s_dt[i];
            if (!d->tex)
                continue;
            cf_drop(NULL, d, PT_SCALE);   /* metal_clear_fold: its held clear, as before */
            if (s_enc_dt == d) {
                PT_WHY(PT_SCALE);
                end_encoder();
            }
            dt_free(d);
        }
    }
    /* Queries issued at the old scale are read at it: a count taken at 2x
     * divided by 1 would report four times the samples. Retired results go
     * unread (nv2a_occ_set_scale). */
    if (s_occ.ops) {
        @autoreleasepool {
            occ_settle(0);
        }
        nv2a_occ_set_scale(&s_occ, nv2a_host_render_scale());
    }
    fprintf(stderr, LOGP "render.scale=%u from this flip: targets dropped\n",
            nv2a_host_render_scale());
}

/* Live options (nv2a_host_opts_request) are applied after the present:
 * the one point where no target is bound and no upload is in flight, and
 * the frame shown was drawn at the old render.scale, which the present
 * sizes its source by (applied first, a 2x frame lowered to 1x would show
 * its top-left quarter for one flip). */
/* Filter, fullscreen and overscan are the SDL presenter's (fb_present_sdl.c
 * reads them each present, on its own thread); aniso and movie are read at
 * each draw, so storing them is the change. */
static void metal_on_flip(uint32_t surface_offset, uint32_t pitch)
{
    unsigned changed;
    metal_flip(surface_offset, pitch);
    changed = nv2a_host_opts_apply();
    if (changed & NV2A_OPT_SCALE)
        scale_drop();
}

/* ---- the 2D engine ---------------------------------------------------------- */

/* nv2a_pb_backend.on_blit. A blit out of a render target copies the
 * target's texture, pads and render scale included, into a target at the
 * destination (the "twin": the source's geometry, made on first use, kept
 * while each blit refills it), which a later draw samples like any target
 * (rt_texture). The guest bytes under the source are stale while it is
 * drawn into, so a guest copy would hand the next pass an old frame; and
 * under Hor+ the wings exist only on the GPU. Quantum Redshift copies its
 * 640x480 R5G6B5 back buffer each frame and refracts the copy through its
 * lens raindrops, which drew black with no copy at all. Anything
 * else (a guest-memory source, a partial copy with no twin to land in,
 * BLEND_AND) runs in guest memory: the source is written back first, the
 * targets over the destination are written back and retired, and the
 * texture cache rehashes the destination. */

static RenderTarget *blit_source(const struct nv2a_pb_blit *b)
{
    int i;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex && !((rt->addr ^ b->src) & 0x07FFFFFFu) && !rt_drop_if_stale(rt, "blit"))
            return rt;
    }
    return NULL;
}

static int blit_gpu(const struct nv2a_pb_blit *b, RenderTarget *src)
{
    RenderTarget *dst = NULL, *free_rt = NULL, *lru = NULL;
    id<MTLBlitCommandEncoder> e;
    uint32_t box[6];
    int i, whole;

    /* The source rectangle names the target's pixels only at its pitch. The
     * destination may have another (a copy into a wider surface): the twin
     * takes the destination's pitch, so its guest span and any write-back
     * are the blit's. */
    if (b->operation != NV09F_OP_SRCCOPY || b->src_pitch != src->pitch
            || b->dst_pitch < src->w * b->bpp || b->bpp != nv2a_surf_color_bytes(src->fmt)
            || nv2a_rt_overlap(b->src, src->pitch * src->h, b->dst, b->dst_pitch * src->h)
            || !nv2a_2d_host_box(b, src->w, src->h, src->pad, src->n, src->hw, src->hh, box))
        return 0;
    whole = !b->in_x && !b->in_y && !b->out_x && !b->out_y && b->w == src->w && b->h == src->h;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex && !((rt->addr ^ b->dst) & 0x07FFFFFFu) && rt->w == src->w
                && rt->h == src->h && rt->hw == src->hw && rt->hh == src->hh
                && rt->pad == src->pad && rt->n == src->n && rt->pitch == b->dst_pitch
                && rt->fmt == src->fmt && !rt_drop_if_stale(rt, "blit")) {
            dst = rt;
            break;
        }
    }
    /* A partial copy lands in what the destination held: with no twin to
     * hold that, guest memory has it. */
    if (!dst && !whole)
        return 0;
    /* No two live targets share memory (surface()'s rule): what else lies
     * over the destination is written back, then retired. */
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt == dst || rt == src || !rt->tex
                || !nv2a_rt_overlap(rt->addr, rt->pitch * rt->h, b->dst, b->dst_pitch * src->h))
            continue;
        if (!rt_drop_if_stale(rt, "overlap")) {
            rt_sync(rt, rt->pitch);
            rt_free(rt);
        }
    }
    if (!dst) {
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *rt = &s_rt[i];
            if (!rt->tex && !free_rt)
                free_rt = rt;
            if (rt != src && (!lru || rt->last_use < lru->last_use))
                lru = rt;
        }
        if (!free_rt) {
            free_rt = lru;
            if (!rt_drop_if_stale(free_rt, "evict")) {
                rt_sync(free_rt, free_rt->pitch);
                rt_free(free_rt);
            }
        }
        free_rt->tex = rt_texture_new(src->hw, src->hh);
        if (!free_rt->tex)
            return 0;
        dst = free_rt;
        dst->offset = b->dst_offset;
        dst->addr = b->dst;
        dst->w = src->w;
        dst->h = src->h;
        dst->hw = src->hw;
        dst->hh = src->hh;
        dst->pad = src->pad;
        dst->n = src->n;
        dst->pitch = b->dst_pitch;
        dst->fmt = src->fmt;
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target (%ux%u host), the 2D"
                " blit copy of 0x%08X\n", dst->offset, dst->w, dst->h, dst->hw, dst->hh,
                src->offset);
    }
    PT_WHY(PT_BLIT);
    end_encoder();
    ensure_cmd();
    if (pt_on()) {
        /* metal_passes: the source's store read, the copy a write of the
         * destination (a partial one keeps the rest: a load) */
        e = pt_blit(PT_BLIT, NV2A_GCAT_OTHER);
        if (s_pt_track) {
            pt_read(src->tex, 0, PTR_BLIT, s_pt_aux);
            pt_write(dst->tex, 0, whole, s_pt_aux);
        }
    } else {
        e = [s_cmd blitCommandEncoder];
    }
    [e copyFromTexture:src->tex sourceSlice:0 sourceLevel:0
          sourceOrigin:MTLOriginMake(box[0], box[1], 0)
            sourceSize:MTLSizeMake(box[4], box[5], 1)
             toTexture:dst->tex destinationSlice:0 destinationLevel:0
     destinationOrigin:MTLOriginMake(box[2], box[3], 0)];
    [e endEncoding];
    s_tick++;
    dst->last_use = s_tick;
    dst->dirty = 1;   /* the copy is on the GPU only: an eviction writes it back */
    dst->wpend = 1;   /* s_cmd's number is its last write */
    rt_read_mark(src);
    dst->zseq = ++s_zseq;
    dt_mem_written(dst->addr, dst->pitch * dst->h, dst->zseq);   /* zt_find */
    dst->copy = 1;
    dst->copy_flip = (uint32_t)s_presents;
    rt_own_reset(dst);
    return 1;
}

static void metal_on_blit(const struct nv2a_pb_blit *b)
{
    RenderTarget *src;
    uint64_t span = (uint64_t)(b->out_y + b->h) * b->dst_pitch;
    int i;

    guard_open();
    if (!init()) {
        nv2a_pb_blit_guest(b);
        return;
    }
    @autoreleasepool {
        src = blit_source(b);
        if (src && blit_gpu(b, src)) {
            s_blits_gpu++;
            return;
        }
        if (src)
            rt_sync(src, src->pitch);
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *rt = &s_rt[i];
            if (rt == src || !rt->tex
                    || !nv2a_rt_overlap(rt->addr, rt->pitch * rt->h, b->dst, (uint32_t)span))
                continue;
            if (!rt_drop_if_stale(rt, "overlap")) {
                rt_sync(rt, rt->pitch);   /* a partial blit keeps the rest */
                rt_free(rt);
            }
        }
        nv2a_pb_blit_guest(b);
        tex_cache_stale(b->dst, (uint32_t)span);
        dt_mem_written(b->dst, (uint32_t)span, ++s_zseq);   /* zt_find */
        s_blits_guest++;
        {
            static int said;
            if (!said++)
                fprintf(stderr, LOGP "2D blit 0x%08X -> 0x%08X %ux%u in guest memory%s\n",
                        b->src_offset, b->dst_offset, b->w, b->h,
                        src ? " (the source target written back)" : "");
        }
    }
}

/* ---- registration ---------------------------------------------------------- */

/* Under Hor+ or render.snap_2d, at a batch's END: whether the walker may
 * hold it for a merged list (nv2a_merge_placed), as d3d11_merge_placed.
 * The program and stages are the ones its draw would look up (both
 * memoised); a stage counts as reading a target when its offset is one,
 * whatever rt_texture would then make of the format. */
static int metal_merge_placed(int program)
{
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    const VsEntry *ve;
    int n, i, rt = 0;

    if (!program || !v->depth_enable || !init())
        return 0;
    @autoreleasepool {
        ve = vs_cache();
    }
    if (!ve)
        return 0;
    build_stages();
    for (n = 0; n < 4 && !rt; n++) {
        const Stage *t = &s_stage[n];
        if (!t->mode || t->mode == 4 || !t->offset)
            continue;
        for (i = 0; i < RT_MAX && !rt; i++)
            rt = s_rt[i].tex && !((s_rt[i].addr ^ t->offset) & 0x07FFFFFFu);
    }
    return nv2a_merge_placed(program, v->depth_enable, &ve->hp, rt);
}

static const struct nv2a_pb_backend s_metal_backend = {
    metal_on_clear,
    metal_on_zclear,
    metal_on_draw,
    metal_on_flip,
    metal_on_zpass,
    metal_on_zpass_clear,
    metal_on_report,
    metal_on_poll,
    1,                  /* writes_back: dumps read guest memory, after sync_guest */
    metal_sync_guest,
    1,                  /* merge_draws */
    metal_merge_placed,
    metal_on_blit,
};

const struct nv2a_pb_backend *nv2a_pb_metal_backend(void)
{
    if (!s_black && init()) {
        @autoreleasepool {
            s_black = solid_texture(0xFF000000u);
            s_magenta = solid_texture(NV2A_TEX_MAGENTA);
            s_black_cube = solid_cube(0xFF000000u);
        }
    }
    return &s_metal_backend;
}

/* RECOMP_PB_BACKEND=metal routes the executor here. Called once from the
 * executor's first method, on the ack thread. */
void nv2a_pb_metal_register_from_env(void)
{
    const char *b = recomp_env(RENV_PB_BACKEND);
    if (b && !strcasecmp(b, "metal")) {
        const struct nv2a_pb_backend *mb = nv2a_pb_metal_backend();
        if (s_failed) {
            fprintf(stderr, LOGP "RECOMP_PB_BACKEND=metal: no device, staying on"
                    " the CPU rasteriser\n");
            return;
        }
        nv2a_pb_set_backend(mb);
        fprintf(stderr, LOGP "RECOMP_PB_BACKEND=metal: executor draws through Metal\n");
    }
}

#endif /* __APPLE__ */
