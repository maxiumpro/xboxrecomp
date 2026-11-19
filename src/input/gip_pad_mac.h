/**
 * xbox_input -- wired USB Xbox One (GIP) pads on macOS, read in userspace.
 *
 * Some wired Xbox One pads (third-party GIP pads such as the Hyperkin
 * "Xenon", 2e24:072c) are seen neither by GameController nor by SDL on
 * macOS: no kernel driver matches their vendor-class interface. This reader
 * opens that interface through IOKit (IOUSBLib), sends the GIP startup
 * sequence and keeps the latest report (gip_pad_mac.m). xinput_device.c
 * offers the pad on the first port that has no SDL pad open.
 *
 * Only built on macOS (src/input/CMakeLists.txt). RECOMP_GIP=0 turns it off:
 * xbox_GipPadStart then starts nothing and xbox_GipPadRead always returns 0.
 *
 * This header stays free of pad_map.h on purpose: pad_map.h pulls in
 * platform/xbox_winnt.h, whose REFIID and BOOL clash with IOKit's
 * CFPlugInCOM.h and objc.h in the .m. The struct is the one pad_map.h
 * defines; xinput_device.c checks the two counts below against it.
 */
#ifndef XBOX_GIP_PAD_MAC_H
#define XBOX_GIP_PAD_MAC_H

#ifdef __cplusplus
extern "C" {
#endif

/* pad_map.h's XBOX_HOST_SDL_GAMEPAD (button[] and axis[], SDL3's numbering). */
struct XBOX_HOST_SDL_GAMEPAD;
#define XBOX_GIP_BUTTON_COUNT 15   /* XBOX_SDL_BUTTON_COUNT */
#define XBOX_GIP_AXIS_COUNT   6    /* XBOX_SDL_AXIS_COUNT */

/* Idempotent, any thread, never blocks: starts device matching and a reader
 * thread. Logs "[INPUT] gip: ..." to stderr when a pad opens or is lost. */
void xbox_GipPadStart(void);

/* Copies the latest state of the first connected GIP pad into *out the way
 * SDL fills it: buttons 0/1, sticks -32768..32767 with Y positive down,
 * triggers 0..32767; Guide reads 0. Returns 1 when a pad is connected and
 * initialised, else 0 (*out untouched). Any thread: only a copy under the
 * reader's own mutex, no IOKit call. */
int xbox_GipPadRead(struct XBOX_HOST_SDL_GAMEPAD *out);

/* Releases the pad's interface (a pad whose process died with it open can
 * stay dark until replugged). Also registered with atexit by the first
 * start; a second call does nothing. */
void xbox_GipPadStop(void);

#ifdef __cplusplus
}
#endif

#endif /* XBOX_GIP_PAD_MAC_H */
