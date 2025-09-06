from collections import defaultdict

import numpy as np
import itertools

from autopilot import state_def as sd


class Flag:
    """System to denote when some special event occurs in the simulation"""

    def __init__(self, text, time, object=None, end_sim=False):
        self.text = text
        self.time = time
        self.object = object
        self.end_sim = end_sim

    def __str__(self):
        return f'Flag({self.text}, time={self.time:.2f}, object={self.object}, terminate={self.end_sim})'


class History(dict):
    """Dictionary tracking the history of specified arbitrary floats or numpy arrays.
    self[name] returns a numpy array with num_steps added as the first dimension."""

    def __init__(self, num_steps, values_to_track: dict):
        super().__init__()
        self.num_steps = num_steps
        self.refs = {}
        self.history = {}
        self.flags: dict[int, list[Flag]] = defaultdict(list)
        for name, info in values_to_track.items():
            if isinstance(info, tuple):
                ref, shape = info
            else:
                ref = info
                try:
                    value = ref() if callable(ref) else ref
                except AttributeError:
                    print(
                        f'Could not access shape to use for {ref} when creating history tracker'
                        ' - specify it in values_to_track as "name": (item, shape)'
                    )
                    raise
                shape = np.shape(value)  # works for non-numpy numbers too
                if len(shape) == 0:
                    shape = (1,)

            self.refs[name] = (ref, callable(ref))
            self[name] = np.full((num_steps,) + shape, np.nan)

    def save_timestep(self, i, time):
        """Access each of the tracked items, and write their current values into timestep i."""
        self.last_saved = i
        for name, tracker_array in self.items():
            thing_to_be_written, should_be_called = self.refs[name]
            try:
                if should_be_called:
                    tracker_array[i] = thing_to_be_written()
                else:
                    tracker_array[i] = thing_to_be_written
            except (ValueError, ZeroDivisionError, sd.ReferenceFrameError) as e:
                tracker_array[i] = np.nan
                self.add_flag(Flag(f'Non-fatal exception: {e}', time, e), i)
            except Exception as e:
                tracker_array[i] = np.nan
                self.add_flag(Flag(f'Fatal exception: {e}', time, e, end_sim=True), i)

    def add_flag(self, flag, i=None):
        if i is None:
            i = self.last_saved
        self.flags[i].append(flag)

    def print_flags(self):
        for i, flags in self.flags.items():
            for flag in flags:
                print(f'[Step {i}] {flag}')


def run_simulation(time_arr, update_func, hist: History):
    """Run a simulation across the given timestamps.

    Parameters:
        time_arr: Sequence of timestamps (in seconds, for example)
        update_func: Function matching the signature update_func(start_time, dt) to update simulation at each timestep, with argument dt being the length of the timestep
        hist: Optional History object to be updated after each timestep

    """

    hist.save_timestep(0, time_arr[0])
    for i, (t_old, t_new) in enumerate(itertools.pairwise(time_arr)):
        dt = t_new - t_old
        update_func(t_old, dt)
        hist.save_timestep(i + 1, t_new)
        for flag_list in hist.flags.values():
            for flag in flag_list:
                if flag.end_sim:
                    return
