/* fb_present_metal.m: the CAMetalLayer present of the SDL window (macOS).
 * Called by fb_present_sdl.c only; see that file and fb_present_metal.m. */
#ifndef FB_PRESENT_METAL_H
#define FB_PRESENT_METAL_H

#include <stddef.h>
#include <stdint.h>
#include <SDL3/SDL.h>

/* Device, queue, pipeline and the window's layer; 0 with a reason in err. */
int   fbm_create(SDL_Window *win, int vsync, int shots, char *err, size_t errn);
void  fbm_destroy(void);
void  fbm_get(void **device, void **queue);
/* Slot k (0..2) at w x h: its id<MTLTexture>, or NULL. */
void *fbm_slot(int k, uint32_t w, uint32_t h);
int   fbm_upload(int k, const uint8_t *src, uint32_t w, uint32_t h, uint32_t pitch,
                 uint32_t bpp);
/* RECOMP_TRACE=pacing: the frame a present shows, for the presenter's
 * [PRESENT] cadence and age line. seq is the hand-off's publish count of
 * the slot (0 before the first frame: not counted), pub its publish time on
 * fbm_now's clock. */
struct fbm_pace_frame {
    uint64_t seq;
    double   pub;
};
/* CACurrentMediaTime(): seconds on the host clock presentedTime is in
 * (mach_absolute_time's, which stops while the machine sleeps). Any thread. */
double fbm_now(void);
/* layers: the overlay's visible lists (nv2a_overlay.h), drawn over the
 * frame in the same pass; NULL and 0 when none is visible. pace: NULL when
 * the pacing trace is off (no handlers, no extra work). */
struct nv2a_ovl_list;
int   fbm_present(SDL_Window *win, int k, uint32_t fw, uint32_t fh, unsigned n,
                  uint32_t pad, int filter, const char *shot_path, uint32_t flip,
                  const struct nv2a_ovl_list *const *layers, int nlayers,
                  const struct fbm_pace_frame *pace);

/* The window's NSWindow level: above the menu bar and the Dock
 * (kCGMainMenuWindowLevel + 1, where SDL puts a fullscreen window without
 * Spaces) when top, else the normal level. Any SDL window, on either
 * present path. 1 when the level changed. */
int   fbm_window_level(SDL_Window *win, int top);

/* present.fullscreen_mode=native (fb_present_sdl.c): the desktop's display
 * mode, kept in a file while a fullscreen window holds another, so a run
 * that never puts it back is undone at the next start. save: the mode of
 * win's display now, called before the switch; a save still outstanding is
 * kept, since it holds the desktop's and the display may be native by now.
 * 1 when the file holds it, with the native pixel size the switch takes.
 * restore: reads the file and deletes it; only when the display is still at
 * that native size, sets the saved mode with Core Graphics for the session
 * (never kCGConfigurePermanently). why is for the log. */
int   fbm_display_save(SDL_Window *win, unsigned long native_pw, unsigned long native_ph);
void  fbm_display_restore(const char *why);

#endif
