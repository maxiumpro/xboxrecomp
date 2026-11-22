/*
 * kernel_pacing.h - frame pacing: the vblank schedule's arithmetic, the
 * sleeping wait that lowered guest spin loops call, and RECOMP_TRACE=pacing.
 *
 * A title that waits for its next frame in a busy loop on a counter the
 * vblank DPC moves (a title's main loop, typically) holds a host core for
 * nothing. With `tools.recomp --spin-waits`, the generator lowers such a
 * loop's back edge to RECOMP_SPIN_WAIT(va) (templates/runtime/recomp_types.h),
 * which calls xbox_SpinWait while the pacing mode is "sleep". The kernel
 * signals xbox_SpinWake whenever guest-visible state may have changed (a
 * vblank pass, any host-run ISR or DPC, a fence write), and every wait is
 * bounded at 1 ms, so a wake the kernel never sends costs a millisecond and
 * never hangs the title. Mode "spin" (the default) returns at once: the loop
 * busy-waits exactly as the console does.
 */
#ifndef XBOX_KERNEL_PACING_H
#define XBOX_KERNEL_PACING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── The vblank schedule (pure; tests/vblank_schedule drives it) ──
 *
 * Vblank N is due at t0 + N * 10^9 / 60 ns, so the rate is exact whatever
 * the caller's sleep granularity. A caller more than 100 ms behind (a
 * debugger stop, a stalled timer thread) restarts the grid at now instead of
 * firing the missed vblanks in a burst. */

/* RECOMP_DEBUG=vblank_anchor=<us> (probe; unset or 0: off, and nothing but
 * one cached load per vblank). The grid above runs on the host clock from
 * t0, so it slides slowly against the display's real refresh, and the age
 * of a frame (commit to screen) slides with it. With the probe, a refresh
 * estimate from the present path (xbox_VblankPhaseSetSource) puts each
 * emulated vblank <us> after an estimated refresh: after each vblank the
 * grid's epoch t0 moves a quarter of the way there, at most
 * XBOX_VBLANK_ANCHOR_STEP_NS a vblank, so no vblank is lost or doubled and
 * the rate is the display's. A refresh period further than
 * XBOX_VBLANK_ANCHOR_TOL_HZ from the vblank rate (59.94 Hz, VRR,
 * RECOMP_VBLANK_HZ other than the display's), a stale estimate or none
 * (D3D11: no source) leave the grid as it is. State of the timer thread's
 * one schedule, so it lives there. */
typedef struct {
    int      read;            /* the probe's value was read */
    long     offset_us;       /* 0: off */
    uint64_t win_t0_ns;       /* the report window's start */
    uint64_t win_vblanks, win_anchored;
    int64_t  total_ns;        /* t0's correction since the start */
    int64_t  win_ns;          /* ... in this window */
    double   win_abs_err_ns;  /* sum of |error| over the anchored vblanks */
    double   win_max_err_ns;
    uint64_t period_ns;       /* the last estimate seen, for the report */
    int64_t  edge_ns;         /* the next edge after its refresh, after the move */
    int      why;             /* the last free vblank's reason (pacing.c) */
} XboxVblankAnchor;

typedef struct {
    int      started;
    uint64_t t0_ns;      /* the grid's epoch */
    uint64_t count;      /* vblanks fired since t0 */
    XboxVblankAnchor anchor;
} XboxVblankSchedule;

#define XBOX_VBLANK_HZ          60u
#define XBOX_VBLANK_RESTART_NS  100000000ull   /* 100 ms behind: restart */

/* Advance the schedule to now_ns. Returns 1 when a vblank is due now (and
 * counts it), 0 when not. *next_due_ns is the next edge on the grid (in the
 * past while catching up); *restarted_late_ns is how far behind a restart
 * was, 0 when none. */
int xbox_VblankSchedule(XboxVblankSchedule *s, uint64_t now_ns,
                        uint64_t *next_due_ns, uint64_t *restarted_late_ns);

/* The edge of the vblank the last call that returned 1 fired: now minus
 * this is how far behind the grid it was raised (0 after a restart). */
uint64_t xbox_VblankScheduleLastDue(const XboxVblankSchedule *s);

#define XBOX_VBLANK_ANCHOR_STEP_NS   500000       /* the most t0 moves a vblank */
#define XBOX_VBLANK_ANCHOR_TOL_HZ    0.02         /* refresh vs vblank rate */
#define XBOX_VBLANK_ANCHOR_REPORT_NS 10000000000ull

/* The refresh estimate: 1 with the time of a recent real refresh (*phase_ns,
 * on xbox_HostNowNs's clock) and the refresh period, 0 without a valid one.
 * The Metal present (fb_present_metal.m, fb_vblank_phase_ns) registers its
 * own at window creation when vblank_anchor is set; nothing else does, so
 * without one (D3D11, the tests) xbox_VblankPhaseNs returns 0. */
typedef int (*XboxVblankPhaseFn)(uint64_t *phase_ns, uint64_t *period_ns);
void xbox_VblankPhaseSetSource(XboxVblankPhaseFn fn);
int  xbox_VblankPhaseNs(uint64_t *phase_ns, uint64_t *period_ns);

/* After a call to xbox_VblankSchedule that returned 1 (the timer thread):
 * with vblank_anchor set, move the grid toward the refresh estimate and
 * set *next_due_ns to the moved next edge; a [PACING] vblank_anchor: line
 * every 10 s. With the probe off, one load and nothing else. */
void xbox_VblankAnchorStep(XboxVblankSchedule *s, uint64_t now_ns,
                           uint64_t *next_due_ns);

/* ── The pacing mode ── */
enum { XBOX_PACING_SPIN = 0, XBOX_PACING_SLEEP = 1 };

/* Set before the guest starts (the enhancements layer's present.pacing),
 * and again from any thread while it runs (xbox_enhance_set_pacing): every
 * pass reads the mode afresh and a sleeping pass is bounded at 1 ms, so a
 * switch shows from the next pass. Without a call the mode is spin. */
void xbox_PacingSetMode(int mode);
int  xbox_PacingMode(void);

/* Read by RECOMP_SPIN_WAIT on every pass of a lowered loop, with
 * g_xbox_irq_pending (kernel_hal.c): a pass with an interrupt posted calls
 * the wait at any pacing, as the gate holder's safe point. */
extern volatile int g_xbox_spin_sleep;
extern volatile int g_xbox_irq_pending;
/* Raised while the NV2A walker sleeps for the guest's next kick
 * (nv2a_kick_wake.h): a pass of a lowered loop then calls the wait at any
 * pacing, and the wait pokes the walker first. */
extern volatile int g_xbox_kick_armed;

/* One pass of a lowered loop whose condition was still true. With the
 * walker asleep, first a poke (xbox_Nv2aKickPoke), which is all a pass
 * that came only for that does. Then a safe
 * point (xbox_IrqSafePoint): the gate holder runs what was posted. Mode spin
 * returns there. Mode sleep:
 * return at once if the wake generation moved since this thread last looked,
 * else block until it moves or 1 ms passes. Never sleeps at DISPATCH level
 * or above, or with the one-CPU gate held: the DPC the loop waits for could
 * not run. */
void xbox_SpinWait(uint32_t va);
/* The same pass for a site lowered with "sleep": "kick_event"
 * (RECOMP_SPIN_WAIT_KICK): it sleeps only while the walker's kick event
 * wait is on (xbox_Nv2aKickEventOn), and passes as at "spin" otherwise. */
void xbox_SpinWaitKick(uint32_t va);

/* Guest-visible state may have changed: wake every xbox_SpinWait. One load
 * when the mode is spin. */
void xbox_SpinWake(void);

/* Per-site counters, for the trace and the tests. */
typedef struct {
    uint32_t va;
    long waits, immediate, wakes, timeouts, dispatch;
    /* Loops that left on a signal (or at once) or on the timeout; a loop's
     * exit is counted when its thread next waits, more than 0.5 ms later. */
    long exit_wake, exit_timeout;
    /* At spin pacing: passes the macro made for a waiting guest thread, and
     * how many of them handed the guest CPU over. */
    long spin_passes, spin_yields;
} XboxSpinSiteStats;
/* Copy the site's counters since the last reset; 0 if the site never waited
 * or made a spin pass. */
int  xbox_SpinSiteStats(uint32_t va, XboxSpinSiteStats *out);
void xbox_SpinStatsReset(void);

/* ── RECOMP_TRACE=pacing ── */

/* The executor's flip, on every backend (nv2a_pb_exec.c). */
void xbox_PacingFlip(uint32_t flip, uint32_t batches);
/* A vblank raised at now_ns (the timer thread), for the gap range. */
void xbox_PacingVblank(uint64_t now_ns);

/* CPU time of the calling thread, and of the process, in ns. */
uint64_t xbox_ThreadCpuNs(void);
int      xbox_PacingTraceOn(void);  /* RECOMP_PACING_TRACE set: summaries print */
uint64_t xbox_ProcessCpuNs(void);

#ifdef __cplusplus
}
#endif

#endif /* XBOX_KERNEL_PACING_H */
