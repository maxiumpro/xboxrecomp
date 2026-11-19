// SPDX-License-Identifier: GPL-2.0-or-later
//
// Wired USB Xbox One (GIP) pads read in userspace on macOS, for pads that neither GameController
// nor SDL sees (third-party GIP pads such as the Hyperkin Xenon, 2e24:072c). gip_pad_mac.h has
// the API; xinput_device.c offers the pad on the first port without an SDL pad.
//
// Facts:
// - A GIP pad exposes a vendor-class interface (class 0xFF, subclass 0x47, protocol 0xD0) that no
//   macOS kernel driver matches, so a non-sandboxed process opens it through the IOKit USB user
//   client (IOUSBLib) without entitlements. libusb uses the same path on macOS.
// - Such a device may stay unconfigured (no driver sets a configuration on a vendor-class device):
//   then no interface exists until the device is opened and its first configuration is set, as
//   libusb does in darwin_claim_interface.
// - The pad only reports after the host sends the startup sequence; a pad opened long after it
//   announced itself (or left dark by a process that died with the pad open) gets the sequence on
//   every open, and again whenever it announces.
// - Protocol values (init packets, header layout, acknowledge format, input offsets) follow SDL's
//   src/joystick/hidapi/SDL_hidapi_xboxone.c (zlib licence; its GIP parts are adapted there from
//   medusalix/xone): xboxone_init_packets and SendControllerStartup, HIDAPI_GIP_DecodeHeader,
//   HIDAPI_GIP_AcknowledgePacket, HIDAPI_DriverXboxOne_HandleStatePacket. The code is our own.
// - The state is handed out as SDL's gamepad API would read it (XBOX_HOST_SDL_GAMEPAD), with
//   SDL's conversions: sticks raw, Y inverted as ~v; 10-bit triggers scaled to 0..32767. No
//   deadzone: SDL's wired GIP path passes the axes raw, and pad_map.c applies its own.
// - Manual retain/release (no -fobjc-arc), like the other .m files of the runtime.

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOCFPlugIn.h>
#include <IOKit/IOMessage.h>
#include <IOKit/usb/IOUSBLib.h>
#include <dispatch/dispatch.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SDL3/SDL_gamepad.h>

#include "gip_pad_mac.h"

// pad_map.h's layout (see gip_pad_mac.h for why it is not included): indexed by SDL3's
// SDL_GamepadButton and SDL_GamepadAxis, which xinput_device.c checks against XBOX_SDL_*.
struct XBOX_HOST_SDL_GAMEPAD {
  uint8_t button[XBOX_GIP_BUTTON_COUNT];
  int16_t axis[XBOX_GIP_AXIS_COUNT];
};
typedef struct XBOX_HOST_SDL_GAMEPAD GipPadState;
_Static_assert(SDL_GAMEPAD_BUTTON_DPAD_RIGHT + 1 == XBOX_GIP_BUTTON_COUNT, "button count");
_Static_assert(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER + 1 == XBOX_GIP_AXIS_COUNT, "axis count");

#define GIP_IF_CLASS 0xFF
#define GIP_IF_SUBCLASS 0x47
#define GIP_IF_PROTOCOL 0xD0

#define GIP_CMD_ACKNOWLEDGE 0x01
#define GIP_CMD_ANNOUNCE 0x02
#define GIP_CMD_INPUT 0x20
#define GIP_OPT_ACKNOWLEDGE 0x10
#define GIP_OPT_INTERNAL 0x20
#define GIP_OPT_CHUNK 0x80

#define GIP_IN_MAX 512   // largest interrupt IN packet we accept (full speed pads send 64)
#define GIP_OUT_MAX 64
#define GIP_OUT_SLOTS 16 // buffers of the writes in flight (startup sequence plus acks)
#define GIP_READ_ERRORS_MAX 8

// Startup sequence, in order (SDL xboxone_init_packets). Byte 2 is the sequence number, filled
// at send time. vid/pid 0 = every pad.
static const uint8_t k_power_on[] = {0x05, 0x20, 0x00, 0x01, 0x00};
static const uint8_t k_enable_led[] = {0x0A, 0x20, 0x00, 0x03, 0x00, 0x01, 0x14};
static const uint8_t k_security_passed[] = {0x06, 0x20, 0x00, 0x02, 0x01, 0x00};
// Some PowerA pads only report after their rumble motors were started.
static const uint8_t k_powera_rumble[] = {0x09, 0x00, 0x00, 0x09, 0x00, 0x0F, 0x00,
                                          0x00, 0x1D, 0x1D, 0xFF, 0x00, 0x00};
static const uint8_t k_rumble_setup[] = {0x09, 0x00, 0x00, 0x09, 0x00, 0x0F, 0x00,
                                         0x00, 0x00, 0x00, 0xFF, 0x00, 0xEB};
typedef struct {
  uint16_t vid, pid;
  const uint8_t *data;
  uint8_t size;
} GipInitPacket;
static const GipInitPacket k_init[] = {
    {0, 0, k_power_on, sizeof(k_power_on)},
    {0, 0, k_enable_led, sizeof(k_enable_led)},
    {0, 0, k_security_passed, sizeof(k_security_passed)},
    {0x24c6, 0x541a, k_powera_rumble, sizeof(k_powera_rumble)},
    {0x24c6, 0x542a, k_powera_rumble, sizeof(k_powera_rumble)},
    {0x24c6, 0x543a, k_powera_rumble, sizeof(k_powera_rumble)},
    {0, 0, k_rumble_setup, sizeof(k_rumble_setup)},
};

// The open pad. Only the reader thread touches it. gen changes on every open and close, so an
// async completion of a closed pad (its refcon carries the gen) is ignored.
typedef struct {
  int open;
  uint32_t gen;
  io_service_t device;
  io_object_t interest;
  IOUSBDeviceInterface320 **dev;
  int dev_opened; // USBDeviceOpen held: only when the device had to be configured
  IOUSBInterfaceInterface300 **intf;
  CFRunLoopSourceRef async_src;
  uint8_t pipe_in, pipe_out;
  uint32_t in_size;
  uint16_t vid, pid;
  uint8_t seq;
  int startup_left;  // startup writes not completed yet
  int startup_failed;
  int read_errors;
  int got_data;  // a read completed: the pad works, so the retry backoff and log dedupe reset
  int announced; // "connected" was printed, so "disconnected" is too
  char name[64];
  int writes_in_flight;
  unsigned out_next;
  CFRunLoopTimerRef ready_timer;
  uint8_t in_buf[GIP_IN_MAX];
  uint8_t out_buf[GIP_OUT_SLOTS][GIP_OUT_MAX];
} GipPad;

static GipPad g_pad;
static IONotificationPortRef g_port;
static io_iterator_t g_added_iter;
static CFRunLoopTimerRef g_rescan_timer;
static double g_backoff = 1.0;
static int g_exiting;

// Shared with the game threads.
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static CFRunLoopRef g_loop;
static GipPadState g_state;
static int g_ready;
static int g_stopped; // xbox_GipPadStop ran

// One line per distinct failure: a pad that keeps failing the same step is retried quietly.
static int g_last_step = -1;
static IOReturn g_last_err;

enum { OPEN_NOT_GIP, OPEN_OK, OPEN_FAILED };

static void rescan_now(void);
static void close_pad(const char *why);

static void log_failure(int step, const char *what, IOReturn err) {
  if (step == g_last_step && err == g_last_err) return;
  g_last_step = step;
  g_last_err = err;
  fprintf(stderr, "[INPUT] gip: %04x:%04x %s failed: 0x%08x\n", g_pad.vid, g_pad.pid, what, (unsigned)err);
}

static void set_state(const GipPadState *s, int ready) {
  pthread_mutex_lock(&g_lock);
  if (s) g_state = *s;
  g_ready = ready;
  pthread_mutex_unlock(&g_lock);
}

// SDL's idle pad: no button, sticks and triggers at 0.
static GipPadState neutral_state(void) {
  GipPadState s;
  memset(&s, 0, sizeof s);
  return s;
}

static void schedule_rescan(double delay) {
  if (g_exiting) return;
  if (g_rescan_timer) {
    CFRunLoopTimerInvalidate(g_rescan_timer);
    CFRelease(g_rescan_timer);
  }
  g_rescan_timer = CFRunLoopTimerCreateWithHandler(NULL, CFAbsoluteTimeGetCurrent() + delay, 0, 0, 0,
                                                   ^(CFRunLoopTimerRef t) {
                                                     (void)t;
                                                     rescan_now();
                                                   });
  CFRunLoopAddTimer(CFRunLoopGetCurrent(), g_rescan_timer, kCFRunLoopDefaultMode);
}

// Retry later with a growing delay: the failure may clear (another process releases the pad).
static void schedule_retry(void) {
  schedule_rescan(g_backoff);
  g_backoff = g_backoff * 2 > 30 ? 30 : g_backoff * 2;
}

// The pad is usable: neutral until its first real report. Logged here, not at open, so a pad
// that opens but never initialises stays quiet on every retry.
static void mark_ready(void) {
  GipPadState s = neutral_state();
  set_state(&s, 1);
  if (!g_pad.announced) {
    g_pad.announced = 1;
    fprintf(stderr, "[INPUT] gip: %04x:%04x '%s' connected\n", g_pad.vid, g_pad.pid, g_pad.name);
  }
}

// --- Writes ---------------------------------------------------------------------------------

static void write_done(void *refcon, IOReturn result, void *arg0) {
  (void)arg0;
  uintptr_t r = (uintptr_t)refcon;
  if ((uint32_t)(r >> 1) != g_pad.gen || !g_pad.open) return;
  g_pad.writes_in_flight--;
  int startup = (int)(r & 1);
  if (result != kIOReturnSuccess) {
    log_failure(3, startup ? "init write" : "write", result);
    if (result == kIOUSBPipeStalled) (*g_pad.intf)->ClearPipeStallBothEnds(g_pad.intf, g_pad.pipe_out);
    if (startup) g_pad.startup_failed = 1;
  }
  if (!startup || --g_pad.startup_left > 0) return;
  if (g_pad.startup_failed) {
    close_pad(NULL);
    schedule_retry();
    return;
  }
  // SDL waits 50 ms after the startup sequence before it takes input (PREPARE_INPUT).
  if (g_pad.ready_timer) return;
  uint32_t gen = g_pad.gen;
  g_pad.ready_timer = CFRunLoopTimerCreateWithHandler(NULL, CFAbsoluteTimeGetCurrent() + 0.05, 0, 0, 0,
                                                      ^(CFRunLoopTimerRef t) {
                                                        (void)t;
                                                        if (gen != g_pad.gen || !g_pad.open) return;
                                                        pthread_mutex_lock(&g_lock);
                                                        int was = g_ready;
                                                        pthread_mutex_unlock(&g_lock);
                                                        if (!was) mark_ready();
                                                      });
  CFRunLoopAddTimer(CFRunLoopGetCurrent(), g_pad.ready_timer, kCFRunLoopDefaultMode);
}

static IOReturn send_packet(const uint8_t *data, uint32_t size, int startup) {
  if (size > GIP_OUT_MAX) return kIOReturnBadArgument;
  if (g_pad.writes_in_flight >= GIP_OUT_SLOTS) return kIOReturnNoResources; // the pad resends what it needs
  uint8_t *buf = g_pad.out_buf[g_pad.out_next++ % GIP_OUT_SLOTS];
  memcpy(buf, data, size);
  void *refcon = (void *)(((uintptr_t)g_pad.gen << 1) | (uintptr_t)(startup != 0));
  IOReturn kr = (*g_pad.intf)->WritePipeAsync(g_pad.intf, g_pad.pipe_out, buf, size, write_done, refcon);
  if (kr == kIOReturnSuccess) g_pad.writes_in_flight++;
  return kr;
}

static uint8_t next_seq(void) {
  if (++g_pad.seq == 0) g_pad.seq = 1;
  return g_pad.seq;
}

static IOReturn send_startup(void) {
  g_pad.startup_failed = 0;
  for (size_t i = 0; i < sizeof(k_init) / sizeof(k_init[0]); i++) {
    const GipInitPacket *p = &k_init[i];
    if ((p->vid && p->vid != g_pad.vid) || (p->pid && p->pid != g_pad.pid)) continue;
    uint8_t pkt[GIP_OUT_MAX];
    memcpy(pkt, p->data, p->size);
    pkt[2] = next_seq();
    IOReturn kr = send_packet(pkt, p->size, 1);
    if (kr != kIOReturnSuccess) return kr;
    g_pad.startup_left++;
  }
  return kIOReturnSuccess;
}

// Acknowledge (SDL HIDAPI_GIP_AcknowledgePacket): header 01 20 <seq of the packet> 09, then
// unknown 0, the command, options 0x20, length LE16, 2 bytes padding, remaining LE16.
static void send_ack(uint8_t cmd, uint8_t seq, uint32_t length) {
  uint8_t pkt[13] = {GIP_CMD_ACKNOWLEDGE, GIP_OPT_INTERNAL, seq, 9, 0, cmd, GIP_OPT_INTERNAL,
                     (uint8_t)length, (uint8_t)(length >> 8), 0, 0, 0, 0};
  IOReturn kr = send_packet(pkt, sizeof(pkt), 0);
  if (kr != kIOReturnSuccess) log_failure(4, "ack", kr);
}

// --- Input ----------------------------------------------------------------------------------

// 10-bit trigger to SDL's 0..32767: 0 -> 0, 1023 -> 32767 (SDL scales by 64 into the joystick
// range and maps that to the gamepad trigger, the same endpoints).
static int16_t trigger_to_sdl(unsigned t) { return (int16_t)((t << 5) | (t >> 5)); }

// Input report 0x20 payload (after the header): buttons at 0-1, LT/RT 10 bit at 2-3/4-5, sticks
// int16 LE at 6-13 with Y up positive. SDL's Y points down: ~v (-1 - v, no overflow at -32768),
// as SDL_hidapi_xboxone.c does.
static void decode_input(const uint8_t *p, GipPadState *s) {
  memset(s, 0, sizeof *s);
  uint8_t *b = s->button;
  b[SDL_GAMEPAD_BUTTON_START] = (p[0] & 0x04) != 0; // Menu
  b[SDL_GAMEPAD_BUTTON_BACK] = (p[0] & 0x08) != 0;  // View
  b[SDL_GAMEPAD_BUTTON_SOUTH] = (p[0] & 0x10) != 0; // A
  b[SDL_GAMEPAD_BUTTON_EAST] = (p[0] & 0x20) != 0;  // B
  b[SDL_GAMEPAD_BUTTON_WEST] = (p[0] & 0x40) != 0;  // X
  b[SDL_GAMEPAD_BUTTON_NORTH] = (p[0] & 0x80) != 0; // Y
  b[SDL_GAMEPAD_BUTTON_DPAD_UP] = (p[1] & 0x01) != 0;
  b[SDL_GAMEPAD_BUTTON_DPAD_DOWN] = (p[1] & 0x02) != 0;
  b[SDL_GAMEPAD_BUTTON_DPAD_LEFT] = (p[1] & 0x04) != 0;
  b[SDL_GAMEPAD_BUTTON_DPAD_RIGHT] = (p[1] & 0x08) != 0;
  // The Razer Atrox has the shoulder bits swapped (SDL).
  int swap = g_pad.vid == 0x1532 && g_pad.pid == 0x0a00;
  b[SDL_GAMEPAD_BUTTON_LEFT_SHOULDER] = (p[1] & (swap ? 0x20 : 0x10)) != 0;
  b[SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER] = (p[1] & (swap ? 0x10 : 0x20)) != 0;
  b[SDL_GAMEPAD_BUTTON_LEFT_STICK] = (p[1] & 0x40) != 0;
  b[SDL_GAMEPAD_BUTTON_RIGHT_STICK] = (p[1] & 0x80) != 0;
  s->axis[SDL_GAMEPAD_AXIS_LEFT_TRIGGER] = trigger_to_sdl((unsigned)(p[2] | p[3] << 8) & 0x3FF);
  s->axis[SDL_GAMEPAD_AXIS_RIGHT_TRIGGER] = trigger_to_sdl((unsigned)(p[4] | p[5] << 8) & 0x3FF);
  s->axis[SDL_GAMEPAD_AXIS_LEFTX] = (int16_t)(p[6] | p[7] << 8);
  s->axis[SDL_GAMEPAD_AXIS_LEFTY] = (int16_t)~(int16_t)(p[8] | p[9] << 8);
  s->axis[SDL_GAMEPAD_AXIS_RIGHTX] = (int16_t)(p[10] | p[11] << 8);
  s->axis[SDL_GAMEPAD_AXIS_RIGHTY] = (int16_t)~(int16_t)(p[12] | p[13] << 8);
}

static int decode_varint(const uint8_t *d, uint32_t avail, uint32_t *out) {
  uint32_t v = 0;
  int i = 0;
  while (i < 4 && (uint32_t)i < avail) {
    v |= (uint32_t)(d[i] & 0x7F) << (7 * i);
    if (!(d[i++] & 0x80)) break;
  }
  *out = v;
  return i;
}

static void dispatch_packet(uint8_t cmd, uint8_t opt, const uint8_t *data, uint32_t len) {
  if (opt & GIP_OPT_INTERNAL) {
    // A pad that announces (again) waits for the startup sequence. The guide button (0x07) is
    // not mapped here; it only needed its acknowledge.
    if (cmd == GIP_CMD_ANNOUNCE) {
      IOReturn kr = send_startup();
      if (kr != kIOReturnSuccess) log_failure(5, "init after announce", kr);
    }
    return;
  }
  if (cmd != GIP_CMD_INPUT || len < 14) return;
  pthread_mutex_lock(&g_lock);
  int ready = g_ready;
  pthread_mutex_unlock(&g_lock);
  if (!ready) {
    // Input before the startup settled may be spurious (SDL drops it); it proves the pad is up.
    mark_ready();
    return;
  }
  GipPadState s;
  decode_input(data, &s);
  set_state(&s, 1);
}

static void process_data(const uint8_t *d, uint32_t size) {
  while (size > 3) {
    uint8_t cmd = d[0], opt = d[1], seq = d[2];
    uint32_t len = 0, chunk = 0;
    uint32_t hdr = 3;
    hdr += (uint32_t)decode_varint(d + hdr, size - hdr, &len);
    if ((opt & GIP_OPT_CHUNK) && hdr < size) hdr += (uint32_t)decode_varint(d + hdr, size - hdr, &chunk);
    if (hdr > size) break;
    if (len > size - hdr) len = size - hdr;
    if (opt & GIP_OPT_ACKNOWLEDGE) send_ack(cmd, seq, chunk + len);
    // Low option bits address a device plugged into the pad (headset): not ours.
    if ((opt & 0x0F) == 0) dispatch_packet(cmd, opt, d + hdr, len);
    if (!g_pad.open) return;
    d += hdr + len;
    size -= hdr + len;
  }
}

static void read_done(void *refcon, IOReturn result, void *arg0);

static IOReturn issue_read(void) {
  return (*g_pad.intf)->ReadPipeAsync(g_pad.intf, g_pad.pipe_in, g_pad.in_buf, g_pad.in_size, read_done,
                                      (void *)(uintptr_t)g_pad.gen);
}

static void read_done(void *refcon, IOReturn result, void *arg0) {
  if ((uint32_t)(uintptr_t)refcon != g_pad.gen || !g_pad.open) return;
  if (result == kIOReturnSuccess) {
    g_pad.read_errors = 0;
    if (!g_pad.got_data) {
      g_pad.got_data = 1;
      g_last_step = -1;
      g_backoff = 1.0;
    }
    uint32_t n = (uint32_t)(uintptr_t)arg0;
    process_data(g_pad.in_buf, n > g_pad.in_size ? g_pad.in_size : n);
    if (!g_pad.open) return;
  } else if (result == kIOUSBPipeStalled || result == kIOReturnOverrun) {
    if (result == kIOUSBPipeStalled) (*g_pad.intf)->ClearPipeStallBothEnds(g_pad.intf, g_pad.pipe_in);
    if (++g_pad.read_errors >= GIP_READ_ERRORS_MAX) {
      log_failure(6, "read", result);
      close_pad(NULL);
      schedule_retry();
      return;
    }
  } else {
    // Unplugged (no device, aborted by the kernel) or not responding. Retried with backoff: a
    // pad that stays enumerated but keeps failing must not be reopened in a tight loop.
    if (!g_pad.announced) log_failure(6, "read", result);
    close_pad("disconnected");
    schedule_retry();
    return;
  }
  IOReturn kr = issue_read();
  if (kr != kIOReturnSuccess) {
    log_failure(6, "read", kr);
    close_pad(NULL);
    schedule_retry();
  }
}

// --- Open and close -------------------------------------------------------------------------

static void *query(io_service_t svc, CFUUIDRef type, CFUUIDRef iid, IOReturn *err) {
  IOCFPlugInInterface **plug = NULL;
  SInt32 score = 0;
  void *obj = NULL;
  IOReturn kr = IOCreatePlugInInterfaceForService(svc, type, kIOCFPlugInInterfaceID, &plug, &score);
  if (kr != kIOReturnSuccess || !plug) {
    *err = kr != kIOReturnSuccess ? kr : kIOReturnError;
    return NULL;
  }
  HRESULT hr = (*plug)->QueryInterface(plug, CFUUIDGetUUIDBytes(iid), (LPVOID *)&obj);
  IODestroyPlugInInterface(plug);
  if (hr != S_OK || !obj) {
    *err = kIOReturnUnsupported;
    return NULL;
  }
  *err = kIOReturnSuccess;
  return obj;
}

// The first configuration holds an interface of the GIP class.
static int config_has_gip(IOUSBDeviceInterface320 **dev, uint8_t *config_value) {
  IOUSBConfigurationDescriptorPtr cd = NULL;
  if ((*dev)->GetConfigurationDescriptorPtr(dev, 0, &cd) != kIOReturnSuccess || !cd) return 0;
  const uint8_t *p = (const uint8_t *)cd;
  uint32_t total = (uint32_t)(p[2] | p[3] << 8);
  *config_value = p[5];
  for (uint32_t i = 0; i + 2 <= total;) {
    uint8_t l = p[i];
    if (l < 2 || i + l > total) break;
    if (p[i + 1] == kUSBInterfaceDesc && l >= 9 && p[i + 5] == GIP_IF_CLASS && p[i + 6] == GIP_IF_SUBCLASS &&
        p[i + 7] == GIP_IF_PROTOCOL)
      return 1;
    i += l;
  }
  return 0;
}

// The GIP interface with the lowest number (interface 0 carries input), or 0.
static io_service_t find_interface(IOUSBDeviceInterface320 **dev) {
  IOUSBFindInterfaceRequest req = {GIP_IF_CLASS, GIP_IF_SUBCLASS, GIP_IF_PROTOCOL, kIOUSBFindInterfaceDontCare};
  io_iterator_t it = 0;
  if ((*dev)->CreateInterfaceIterator(dev, &req, &it) != kIOReturnSuccess) return 0;
  io_service_t best = 0, s;
  int best_num = 256;
  while ((s = IOIteratorNext(it))) {
    int num = 255;
    CFTypeRef v = IORegistryEntryCreateCFProperty(s, CFSTR("bInterfaceNumber"), kCFAllocatorDefault, 0);
    if (v) {
      if (CFGetTypeID(v) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef)v, kCFNumberIntType, &num);
      CFRelease(v);
    }
    if (num < best_num) {
      if (best) IOObjectRelease(best);
      best = s;
      best_num = num;
    } else {
      IOObjectRelease(s);
    }
  }
  IOObjectRelease(it);
  return best;
}

static void device_interest(void *refcon, io_service_t svc, uint32_t type, void *arg) {
  (void)svc;
  (void)arg;
  if (type != kIOMessageServiceIsTerminated) return;
  if ((uint32_t)(uintptr_t)refcon != g_pad.gen || !g_pad.open) return;
  close_pad("disconnected");
  schedule_rescan(0.5);
}

static void close_pad(const char *why) {
  if (!g_pad.open) return;
  g_pad.open = 0;
  g_pad.gen++;
  set_state(NULL, 0);
  if (why && g_pad.announced) fprintf(stderr, "[INPUT] gip: %04x:%04x %s\n", g_pad.vid, g_pad.pid, why);
  if (g_pad.ready_timer) {
    CFRunLoopTimerInvalidate(g_pad.ready_timer);
    CFRelease(g_pad.ready_timer);
    g_pad.ready_timer = NULL;
  }
  // Same order as libusb's darwin_release_interface: the async source goes first, then the
  // interface is closed and released.
  if (g_pad.intf) {
    if (g_pad.pipe_in) (*g_pad.intf)->AbortPipe(g_pad.intf, g_pad.pipe_in);
    if (g_pad.pipe_out) (*g_pad.intf)->AbortPipe(g_pad.intf, g_pad.pipe_out);
  }
  if (g_pad.async_src) {
    CFRunLoopRemoveSource(CFRunLoopGetCurrent(), g_pad.async_src, kCFRunLoopDefaultMode);
    CFRelease(g_pad.async_src);
    g_pad.async_src = NULL;
  }
  if (g_pad.intf) {
    (*g_pad.intf)->USBInterfaceClose(g_pad.intf);
    (*g_pad.intf)->Release(g_pad.intf);
    g_pad.intf = NULL;
  }
  if (g_pad.dev) {
    if (g_pad.dev_opened) (*g_pad.dev)->USBDeviceClose(g_pad.dev);
    (*g_pad.dev)->Release(g_pad.dev);
    g_pad.dev = NULL;
    g_pad.dev_opened = 0;
  }
  if (g_pad.interest) {
    IOObjectRelease(g_pad.interest);
    g_pad.interest = 0;
  }
  if (g_pad.device) {
    IOObjectRelease(g_pad.device);
    g_pad.device = 0;
  }
}

// Fails after logging; the caller retries later. On failure everything opened here is released.
static int open_pad(io_service_t svc) {
  IOReturn kr;
  IOUSBDeviceInterface320 **dev = query(svc, kIOUSBDeviceUserClientTypeID, kIOUSBDeviceInterfaceID320, &kr);
  if (!dev) return OPEN_NOT_GIP; // not a device we can ask for descriptors: not one we can open either
  uint8_t config_value = 1;
  if (!config_has_gip(dev, &config_value)) {
    (*dev)->Release(dev);
    return OPEN_NOT_GIP;
  }
  uint32_t gen = g_pad.gen + 1; // monotonic: no completion of an earlier pad matches
  memset(&g_pad, 0, offsetof(GipPad, in_buf));
  g_pad.gen = gen;
  g_pad.dev = dev;
  (*dev)->GetDeviceVendor(dev, &g_pad.vid);
  (*dev)->GetDeviceProduct(dev, &g_pad.pid);

  io_service_t ifsvc = find_interface(dev);
  if (!ifsvc) {
    // Unconfigured: open the device and set its first configuration (libusb does the same).
    kr = (*dev)->USBDeviceOpen(dev);
    if (kr != kIOReturnSuccess) {
      log_failure(0, "USBDeviceOpen (to configure)", kr);
      goto fail;
    }
    g_pad.dev_opened = 1;
    kr = (*dev)->SetConfiguration(dev, config_value);
    if (kr != kIOReturnSuccess) {
      log_failure(1, "SetConfiguration", kr);
      goto fail;
    }
    ifsvc = find_interface(dev);
    if (!ifsvc) {
      log_failure(1, "interface lookup after SetConfiguration", kIOReturnNotFound);
      goto fail;
    }
  }
  g_pad.intf = query(ifsvc, kIOUSBInterfaceUserClientTypeID, kIOUSBInterfaceInterfaceID300, &kr);
  IOObjectRelease(ifsvc);
  if (!g_pad.intf) {
    log_failure(2, "interface plug-in", kr);
    goto fail;
  }
  kr = (*g_pad.intf)->USBInterfaceOpen(g_pad.intf);
  if (kr != kIOReturnSuccess) {
    (*g_pad.intf)->Release(g_pad.intf);
    g_pad.intf = NULL;
    log_failure(2, "USBInterfaceOpen", kr);
    goto fail;
  }

  UInt8 n = 0;
  (*g_pad.intf)->GetNumEndpoints(g_pad.intf, &n);
  for (UInt8 ref = 1; ref <= n; ref++) {
    UInt8 dir = 0, num = 0, type = 0, interval = 0;
    UInt16 mps = 0;
    if ((*g_pad.intf)->GetPipeProperties(g_pad.intf, ref, &dir, &num, &type, &mps, &interval) != kIOReturnSuccess)
      continue;
    if (type != kUSBInterrupt) continue;
    if (dir == kUSBIn && !g_pad.pipe_in) {
      g_pad.pipe_in = ref;
      g_pad.in_size = (mps > GIP_IN_MAX || mps == 0) ? GIP_IN_MAX : mps;
    } else if (dir == kUSBOut && !g_pad.pipe_out) {
      g_pad.pipe_out = ref;
    }
  }
  if (!g_pad.pipe_in || !g_pad.pipe_out) {
    log_failure(2, "interrupt IN/OUT endpoint lookup", kIOReturnNotFound);
    goto fail;
  }
  // A pad left mid-transfer by a process that died may have halted endpoints.
  (*g_pad.intf)->ClearPipeStallBothEnds(g_pad.intf, g_pad.pipe_in);
  (*g_pad.intf)->ClearPipeStallBothEnds(g_pad.intf, g_pad.pipe_out);

  kr = (*g_pad.intf)->CreateInterfaceAsyncEventSource(g_pad.intf, &g_pad.async_src);
  if (kr != kIOReturnSuccess || !g_pad.async_src) {
    g_pad.async_src = NULL;
    log_failure(2, "CreateInterfaceAsyncEventSource", kr);
    goto fail;
  }
  CFRunLoopAddSource(CFRunLoopGetCurrent(), g_pad.async_src, kCFRunLoopDefaultMode);
  kr = IOServiceAddInterestNotification(g_port, svc, kIOGeneralInterest, device_interest,
                                        (void *)(uintptr_t)g_pad.gen, &g_pad.interest);
  if (kr != kIOReturnSuccess) {
    g_pad.interest = 0;
    log_failure(2, "IOServiceAddInterestNotification", kr);
    goto fail;
  }
  IOObjectRetain(svc);
  g_pad.device = svc;
  g_pad.open = 1;

  kr = issue_read();
  if (kr != kIOReturnSuccess) {
    log_failure(3, "first read", kr);
    close_pad(NULL);
    return OPEN_FAILED;
  }
  kr = send_startup();
  if (kr != kIOReturnSuccess) {
    log_failure(3, "init write", kr);
    close_pad(NULL);
    return OPEN_FAILED;
  }
  CFTypeRef pn = IORegistryEntryCreateCFProperty(svc, CFSTR("kUSBProductString"), kCFAllocatorDefault, 0);
  if (!pn) pn = IORegistryEntryCreateCFProperty(svc, CFSTR("USB Product Name"), kCFAllocatorDefault, 0);
  if (pn) {
    if (CFGetTypeID(pn) == CFStringGetTypeID())
      CFStringGetCString((CFStringRef)pn, g_pad.name, sizeof(g_pad.name), kCFStringEncodingUTF8);
    CFRelease(pn);
  }
  return OPEN_OK;

fail:
  g_pad.open = 1; // close_pad releases what is set
  close_pad(NULL);
  return OPEN_FAILED;
}

// Opens the first GIP pad among svcs (all released). Returns OPEN_OK, OPEN_FAILED (a pad is
// there but did not open) or OPEN_NOT_GIP.
static int open_from(io_iterator_t it) {
  int res = OPEN_NOT_GIP;
  io_service_t s;
  while ((s = IOIteratorNext(it))) {
    if (!g_pad.open && !g_exiting) {
      int r = open_pad(s);
      if (r == OPEN_OK) res = OPEN_OK;
      else if (r == OPEN_FAILED && res != OPEN_OK) res = OPEN_FAILED;
    }
    IOObjectRelease(s);
  }
  return res;
}

static void device_added(void *refcon, io_iterator_t it) {
  (void)refcon;
  if (open_from(it) == OPEN_FAILED) schedule_retry();
}

static void rescan_now(void) {
  if (g_pad.open || g_exiting) return;
  io_iterator_t it = 0;
  if (IOServiceGetMatchingServices(MACH_PORT_NULL, IOServiceMatching("IOUSBHostDevice"), &it) != kIOReturnSuccess) {
    schedule_retry();
    return;
  }
  int r = open_from(it);
  IOObjectRelease(it);
  if (r == OPEN_FAILED) schedule_retry();
  else if (!g_added_iter) schedule_rescan(5.0); // no arrival notification: poll
}

// --- Thread and API -------------------------------------------------------------------------

static void *gip_thread(void *arg) {
  (void)arg;
  pthread_setname_np("gip-pad");
  CFRunLoopRef loop = CFRunLoopGetCurrent();
  g_port = IONotificationPortCreate(MACH_PORT_NULL);
  if (!g_port) {
    fprintf(stderr, "[INPUT] gip: IONotificationPortCreate failed; wired Xbox One pads are not read\n");
    return NULL;
  }
  CFRunLoopAddSource(loop, IONotificationPortGetRunLoopSource(g_port), kCFRunLoopDefaultMode);
  pthread_mutex_lock(&g_lock);
  if (g_stopped) { // stopped before the loop existed: nothing to open
    pthread_mutex_unlock(&g_lock);
    return NULL;
  }
  g_loop = (CFRunLoopRef)CFRetain(loop);
  pthread_mutex_unlock(&g_lock);
  kern_return_t kr = IOServiceAddMatchingNotification(g_port, kIOFirstMatchNotification, IOServiceMatching("IOUSBHostDevice"),
                                                      device_added, NULL, &g_added_iter);
  if (kr != KERN_SUCCESS) {
    fprintf(stderr, "[INPUT] gip: IOServiceAddMatchingNotification failed: 0x%08x; polling instead\n", (unsigned)kr);
    g_added_iter = 0;
    rescan_now();
  } else {
    device_added(NULL, g_added_iter); // the devices already present; also arms the notification
  }
  while (!g_exiting) CFRunLoopRun();
  return NULL;
}

// Release the interface on a normal exit: a pad whose process died with it open can stay dark
// until it is replugged. Runs once (atexit and an explicit xbox_GipPadStop).
static void gip_stop(void) {
  pthread_mutex_lock(&g_lock);
  CFRunLoopRef loop = (g_loop && !g_stopped) ? (CFRunLoopRef)CFRetain(g_loop) : NULL;
  g_stopped = 1;
  pthread_mutex_unlock(&g_lock);
  if (!loop) return;
  if (CFRunLoopGetCurrent() == loop) {
    g_exiting = 1;
    close_pad(NULL);
  } else {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    CFRunLoopPerformBlock(loop, kCFRunLoopDefaultMode, ^{
      g_exiting = 1;
      close_pad(NULL);
      CFRunLoopStop(loop);
      dispatch_semaphore_signal(done);
    });
    CFRunLoopWakeUp(loop);
    dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 300 * NSEC_PER_MSEC));
#if !__has_feature(objc_arc)
    dispatch_release(done); // the copied block holds its own reference if it has not run yet
#endif
  }
  CFRelease(loop);
}

static void start_once(void) {
  const char *env = getenv("RECOMP_GIP");
  if (env && strcmp(env, "0") == 0) {
    fprintf(stderr, "[INPUT] gip: off (RECOMP_GIP=0)\n");
    return;
  }
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
  pthread_t t;
  if (pthread_create(&t, &attr, gip_thread, NULL) != 0) {
    fprintf(stderr, "[INPUT] gip: reader thread not started; wired Xbox One pads are not read\n");
  } else {
    atexit(gip_stop);
  }
  pthread_attr_destroy(&attr);
}

void xbox_GipPadStop(void) { gip_stop(); }

void xbox_GipPadStart(void) {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, start_once);
}

int xbox_GipPadRead(struct XBOX_HOST_SDL_GAMEPAD *out) {
  pthread_mutex_lock(&g_lock);
  int ready = g_ready;
  if (ready && out) *out = g_state;
  pthread_mutex_unlock(&g_lock);
  return ready;
}
