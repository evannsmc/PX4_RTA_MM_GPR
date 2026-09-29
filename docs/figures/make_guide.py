"""Architecture figure for docs/00_package_guide.qmd: threads (callback groups), the shared snapshots they exchange,
the rollout worker process and PX4. Run from docs/:  python3 figures/make_guide.py"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8.5})

GROUP_COLORS = {'px4_io': '#6a51a3', 'state': '#2171b5', 'control': '#cb181d', 'rollout': '#238b45',
                'estimation': '#d94801', 'autosave': '#636363'}


def box(ax, x, y, w, h, text, fc='white', ec='k', lw=1.0, bold_first=True, fontsize=8.5, alpha=1.0, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=lw,
                                alpha=alpha, ls=ls))
    lines = text.split('\n')
    if bold_first:
        ax.text(x + w / 2, y + h - 0.2, lines[0], ha='center', va='top', fontsize=fontsize, weight='bold')
        if len(lines) > 1:
            ax.text(x + w / 2, y + h - 0.52, '\n'.join(lines[1:]), ha='center', va='top', fontsize=fontsize - 1,
                    linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p, q, color='k', ls='-', lw=1.2, both=False, rad=0.0, label=None, label_pos=0.5, label_dy=0.12):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='<|-|>' if both else '-|>', mutation_scale=9, color=color, lw=lw,
                                 ls=ls, connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))
    if label:
        x = p[0] + label_pos * (q[0] - p[0])
        y = p[1] + label_pos * (q[1] - p[1]) + label_dy
        ax.text(x, y, label, ha='center', va='bottom', fontsize=7, color=color,
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=0.5))


fig, ax = plt.subplots(figsize=(11, 7.2))
ax.set_xlim(0, 17.2)
ax.set_ylim(-0.3, 10.4)
ax.axis('off')

# outside the node
PX4 = box(ax, 0.1, 8.0, 2.9, 1.7, 'PX4\n(SITL or hardware)\nuXRCE-DDS bridge', fc='#f0f0f0')

# node process
ax.add_patch(FancyBboxPatch((3.5, -0.1), 9.3, 10.3, boxstyle='round,pad=0.02,rounding_size=0.2', fc='#fbfbfb',
                            ec='0.3', lw=1.3))
ax.text(3.7, 10.0, 'node process: rta_mm_gpr_node (rclpy, ONE Python interpreter, one GIL)', fontsize=9,
        weight='bold', va='top')
ax.text(3.9, 9.45, 'executor threads (one per callback group)', fontsize=8, style='italic', va='top')
ax.text(8.55, 9.45, 'shared snapshots (replaced, never mutated)', fontsize=8, style='italic', va='top')

G = {}
G['px4_io'] = box(ax, 3.8, 7.75, 4.1, 1.4, 'px4_io group\nheartbeat 10 Hz, vehicle_status, RC',
                  ec=GROUP_COLORS['px4_io'], lw=1.6)
G['state'] = box(ax, 3.8, 6.05, 4.1, 1.4, 'state group\nodometry 100 Hz: state + wind EKF',
                 ec=GROUP_COLORS['state'], lw=1.6)
G['control'] = box(ax, 3.8, 4.35, 4.1, 1.4, 'control group\ncontrol law 100 Hz (AOT control_step)',
                   ec=GROUP_COLORS['control'], lw=1.6)
G['rollout'] = box(ax, 3.8, 2.65, 4.1, 1.4, 'rollout group\nreplan check 100 Hz, install plans',
                   ec=GROUP_COLORS['rollout'], lw=1.6)
G['estimation'] = box(ax, 3.8, 0.95, 4.1, 1.4, 'estimation group\nGP buffers 10 Hz, LQR check 20 Hz',
                      ec=GROUP_COLORS['estimation'], lw=1.6)
G['autosave'] = box(ax, 3.8, 0.05, 4.1, 0.7, 'flight_recorder autosave thread (5 s)', ec=GROUP_COLORS['autosave'],
                    lw=1.2, bold_first=False, fontsize=7.5, ls='--')

S = {}
S['flags'] = box(ax, 8.6, 8.35, 3.9, 0.8, 'in_offboard_mode, armed, RC switch (bools)', bold_first=False,
                 fontsize=6.8)
S['state'] = box(ax, 8.6, 7.1, 3.9, 1.0, 'self.state\nVehicleState (frozen dataclass)', fontsize=8)
S['wind'] = box(ax, 8.6, 6.05, 3.9, 0.8, 'self.wy, self.wz, self.last_input', bold_first=False, fontsize=7.5)
S['gains'] = box(ax, 8.6, 4.8, 3.9, 1.0, 'self.gains\n(K_feedback, K_reference) tuple', fontsize=8)
S['plan'] = box(ax, 8.6, 3.3, 3.9, 1.25, 'self.plan\nRolloutPlan (frozen dataclass)\nplan_lock: install bookkeeping',
                fontsize=8)
S['gp'] = box(ax, 8.6, 2.05, 3.9, 1.0, 'GP data buffers\nobs_wy / obs_wz, copy-on-write', fontsize=8)
S['rec'] = box(ax, 8.6, 0.15, 3.9, 1.55, 'FlightRecorder\nColumnBuffers: ticks | wind | gains\nplans stored once, by reference',
               fontsize=8)

# worker process
W = box(ax, 13.3, 2.35, 3.8, 2.3, 'rollout worker process\nspawned: own interpreter, GIL, JAX\nRolloutEngine: AOT rollout\n'
        '(--rollout-backend process)', fc='#eef7ee', ec=GROUP_COLORS['rollout'], lw=1.6)
box(ax, 13.3, 5.2, 3.8, 1.5, 'XLA thread pool\nruns every compiled kernel;\nreleases the GIL while it runs',
    fc='#f7f7f7', ec='0.5')


def right(b, dy=0.0):
    return (b[0] + b[2], b[1] + b[3] / 2 + dy)


def left(b, dy=0.0):
    return (b[0], b[1] + b[3] / 2 + dy)


# PX4 <-> node
arrow(ax, right(PX4, 0.35), left(G['px4_io'], 0.35), color='0.25',
      label='status / RC', label_dy=0.05)
arrow(ax, left(G['px4_io'], -0.3), right(PX4, -0.3), color=GROUP_COLORS['px4_io'],
      label='heartbeat, mode, arm', label_dy=-0.38)
arrow(ax, (2.2, 8.0), left(G['state']), color='0.25', label='odometry 100 Hz,\nlocal position 50 Hz', label_pos=0.45, label_dy=-0.55)
arrow(ax, left(G['control']), (2.75, 8.0), color=GROUP_COLORS['control'])
ax.text(2.15, 4.95, 'body-rate\nsetpoints 100 Hz', color=GROUP_COLORS['control'], fontsize=7, ha='center', va='center')

# writes (solid, group color)
arrow(ax, right(G['px4_io']), left(S['flags']), color=GROUP_COLORS['px4_io'])
arrow(ax, right(G['state'], 0.3), left(S['state']), color=GROUP_COLORS['state'])
arrow(ax, right(G['state'], -0.2), left(S['wind'], 0.15), color=GROUP_COLORS['state'])
arrow(ax, right(G['control'], 0.4), left(S['wind'], -0.2), color=GROUP_COLORS['control'])
arrow(ax, right(G['control'], -0.45), left(S['rec'], 0.5), color=GROUP_COLORS['control'], rad=0.12)
arrow(ax, right(G['rollout'], 0.25), left(S['plan']), color=GROUP_COLORS['rollout'])
arrow(ax, right(G['estimation'], 0.45), left(S['gp']), color=GROUP_COLORS['estimation'])
arrow(ax, right(G['estimation'], 0.2), left(S['gains'], -0.3), color=GROUP_COLORS['estimation'], rad=-0.15)
arrow(ax, left(S['rec'], -0.5), right(G['autosave']), color=GROUP_COLORS['autosave'], ls='--')

# reads of the control thread (dashed): the hot path
for key, dy in (('state', 0.0), ('gains', 0.2), ('plan', 0.2)):
    arrow(ax, left(S[key], dy - 0.25), right(G['control'], 0.15 if key != 'plan' else -0.1),
          color=GROUP_COLORS['control'], ls=(0, (3, 2)), lw=0.9)

# worker
arrow(ax, (12.5, 3.9), left(W, 0.35), color=GROUP_COLORS['rollout'], label='RolloutRequest', label_dy=0.03)
arrow(ax, left(W, -0.45), (12.5, 3.05), color=GROUP_COLORS['rollout'], label='RolloutResult', label_dy=-0.33)
ax.text(12.9, 2.1, 'multiprocessing Pipe\n(pickled snapshots)', fontsize=7, ha='center', va='top',
        color=GROUP_COLORS['rollout'])

# legend
ax.plot([13.4, 14.1], [1.2, 1.2], color='k', lw=1.2)
ax.text(14.2, 1.2, 'writes / publishes (one assignment)', va='center', fontsize=7.5)
ax.plot([13.4, 14.1], [0.75, 0.75], color='k', lw=0.9, ls=(0, (3, 2)))
ax.text(14.2, 0.75, 'reads a snapshot (control: once per tick)', va='center', fontsize=7.5)
ax.text(13.4, 0.3, 'arrow color = the thread that does it', fontsize=7.5)

fig.tight_layout()
fig.savefig('figures/package_architecture.pdf')
print('figures/package_architecture.pdf')
